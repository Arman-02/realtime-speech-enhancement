"""Export the single-frame streaming graph, contract v0.2 (PLAN.md §3.2, §4.7).

    uv run python -m ans.export_onnx --ckpt checkpoints/crn_small/best.pt --variant benchmark \\
        --out ../web/public/models --int8 --int8-static --quant-eval

Graph: ``(spec[1,1,161,2], state_in, atten_lim_db[1]) -> (spec_out[1,1,161,2], state_out)``. The attenuation
limit floors the mask magnitude at ``10^(-lim/20)`` (phase kept); 100 dB is effectively unlimited.

* ``--int8``: dynamic quantization (MatMul/Gemm weights only; the GRU stays fp32).
* ``--int8-static``: QDQ static quantization, per-channel, Conv/ConvTranspose/MatMul/Gemm, calibrated on
  ~100 validation clips. Both print an op-type x dtype table of the quantized graph.
* ``--quant-eval``: fp32 vs int8-dynamic vs int8-static size, ms/frame and PESQ/STOI on VB-D test (+OOD
  sets with ``--quant-sets``) through the ORT streaming loop -> ``results.json -> quantization``. An int8
  model is listed in the manifest (``onnx_int8``) only if its VB-D PESQ drop is < 0.05 and its SI-SDR drop
  < 0.5 dB; the fastest such variant is chosen.

Without ``--ckpt`` a seeded random-init model is exported (``"trained": false``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import Tensor, nn

from ans.benchmark import budget, crn_macs_per_frame
from ans.dsp import HOP, N_BINS, N_FFT, SAMPLE_RATE, stft_np
from ans.models import build_model, count_params
from ans.stream import StreamingEnhancer, enhance_streaming, ort_frame_fn

OPSET = 17
FORBIDDEN_OPS = {"If", "Loop", "Scan"}
VERSION = "0.2.0"
ATTEN = {"default": 100, "min": 0, "max": 100}
ATTEN_CASE_DB = 12.0
PESQ_GATE = 0.05  # ship int8 only if VB-D PESQ drops by less than this (§4.7)...
SISDR_GATE = 0.5  # ...and SI-SDR by less than this many dB; the fastest passing variant ships


class StreamGraph(nn.Module):
    """``(spec[1,1,F,2], state_in, atten_lim_db[1]) -> (spec_out[1,1,F,2], state_out)``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, spec: Tensor, state_in: Tensor, atten_lim_db: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(spec, state_in, atten_lim_db=atten_lim_db)


def chirp_signal(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """Deterministic test-vector input: linear chirp 100 Hz -> 4 kHz (0.5 amp) + white noise (0.05 std)."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f0, f1 = 100.0, 4000.0
    chirp = 0.5 * np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t**2))
    noise = 0.05 * np.random.default_rng(seed).standard_normal(n)
    return (chirp + noise).astype(np.float32)


def export_graph(model: nn.Module, path: Path, opset: int = OPSET) -> None:
    if getattr(getattr(model, "cfg", None), "bidirectional", False):
        raise ValueError("the non-causal teacher cannot be exported as a streaming graph")
    graph = StreamGraph(model).eval()
    spec = torch.zeros(1, 1, N_BINS, 2)
    state = model.init_state(1)
    lim = torch.tensor([100.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            graph,
            (spec, state, lim),
            str(path),
            input_names=["spec", "state_in", "atten_lim_db"],
            output_names=["spec_out", "state_out"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    proto = onnx.load(str(path))
    try:
        from onnxsim import simplify

        proto, ok = simplify(proto)
        if ok:
            onnx.save(proto, str(path))
    except ImportError:
        pass
    onnx.checker.check_model(proto)
    ops = {n.op_type for n in proto.graph.node}
    if bad := ops & FORBIDDEN_OPS:
        raise RuntimeError(f"exported graph contains forbidden ops: {bad}")


def ort_session(path: Path, threads: int = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


@torch.no_grad()
def validate(model: nn.Module, sess, n_frames: int = 100, seed: int = 0, atten: float = 100.0) -> float:
    """Max abs diff (spec_out and state_out) between ORT and torch over a stateful frame sequence."""
    rng = np.random.default_rng(seed)
    specs = (rng.standard_normal((n_frames, 1, 1, N_BINS, 2)) * 3).astype(np.float32)
    st_t = model.init_state(1)
    st_o = st_t.numpy()
    lim = np.array([atten], np.float32)
    has_atten = any(i.name == "atten_lim_db" for i in sess.get_inputs())
    worst = 0.0
    for s in specs:
        out_t, st_t = model(torch.from_numpy(s), st_t, atten_lim_db=torch.from_numpy(lim))
        feeds = {"spec": s, "state_in": st_o, **({"atten_lim_db": lim} if has_atten else {})}
        out_o, st_o = sess.run(["spec_out", "state_out"], feeds)
        worst = max(worst, float(np.abs(out_t.numpy() - out_o).max()), float(np.abs(st_t.numpy() - st_o).max()))
    return worst


def _r(a: np.ndarray) -> list:
    return np.round(a.astype(np.float64), 7).tolist()


def _vector_case(sess, state_shape: list[int], x: np.ndarray, frames: np.ndarray, atten: float) -> dict:
    fn = ort_frame_fn(sess, atten)
    state = np.zeros(state_shape, np.float32)
    outs = []
    for f in frames:
        o, state = fn(f[None, None], state)
        outs.append(o[0, 0])
    y = StreamingEnhancer(ort_frame_fn(sess, atten), state_shape).push(x)
    return {"model_out_frames": _r(np.stack(outs)), "output_wav": _r(y)}


def make_test_vectors(sess, state_shape: list[int]) -> dict:
    x = chirp_signal()
    frames = stft_np(x)[:8]
    main = _vector_case(sess, state_shape, x, frames, 100.0)
    atten = _vector_case(sess, state_shape, x, frames, ATTEN_CASE_DB)
    return {
        "description": (
            "1 s linear chirp 100->4000 Hz (amp 0.5) + white noise (std 0.05, numpy default_rng seed 0), 16 kHz"
        ),
        "conventions": {
            "stft_frames": "frame t = sqrt_hann * input_wav[t*160 : t*160+320] -> rfft (center=False, no padding)",
            "model_out_frames": "model run on stft_frames[0..7] in order, state starting at zeros",
            "output_wav": (
                "streaming loop: 160-sample history and 160-sample OLA tail start at zeros; for each "
                "160-sample hop: frame=[history, hop] -> STFT -> model -> iSTFT*window -> out = tail + "
                "frame[:160], tail = frame[160:]. One output hop per input hop (len == len(input_wav)); "
                "output is delayed by 160 samples relative to the input."
            ),
            "atten_case": "same input and conventions, with atten_lim_db = 12 fed on every frame",
        },
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop": HOP,
        "state_shape": state_shape,
        "tolerances": {"stft": 1e-4, "model": 1e-3, "waveform": 1e-3},
        "atten_lim_db": 100,
        "input_wav": _r(x),
        "stft_frames": _r(frames),
        **main,
        "atten_case": {"atten_lim_db": ATTEN_CASE_DB, **atten},
    }


# ------------------------------------------------------------------------------------------------
# Quantization
# ------------------------------------------------------------------------------------------------

INT8_TYPES = {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}


def op_dtype_table(path: Path) -> dict[str, Counter]:
    """op_type -> Counter of 'int8' (integer tensors in/out), 'qdq' (fed by DequantizeLinear, i.e. run in
    int8 by ORT's QDQ fusion), 'fp32'."""
    proto = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    types = {i.name: i.data_type for i in proto.graph.initializer}
    for v in [*proto.graph.value_info, *proto.graph.input, *proto.graph.output]:
        types[v.name] = v.type.tensor_type.elem_type
    producer = {o: n.op_type for n in proto.graph.node for o in n.output}
    table: dict[str, Counter] = defaultdict(Counter)
    for n in proto.graph.node:
        tensors = [t for t in [*n.input, *n.output] if t]
        if any(types.get(t) in INT8_TYPES for t in tensors) and n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            kind = "int8"
        elif n.op_type in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"):
            kind = "quant"
        elif any(producer.get(t) == "DequantizeLinear" for t in n.input):
            kind = "qdq"
        else:
            kind = "fp32"
        table[n.op_type][kind] += 1
    return dict(table)


def print_op_table(path: Path) -> dict:
    table = op_dtype_table(path)
    kinds = ["int8", "qdq", "quant", "fp32"]
    print(f"\nop-type x dtype: {path.name}")
    print(f"  {'op':<24s}" + "".join(f"{k:>7s}" for k in kinds))
    for op in sorted(table):
        print(f"  {op:<24s}" + "".join(f"{table[op].get(k, 0):>7d}" for k in kinds))
    return {op: dict(c) for op, c in table.items()}


def quantize_dynamic_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])


def calibration_frames(model: nn.Module, n_clips: int = 100, stride: int = 4, seed: int = 0) -> list[dict]:
    """(spec, state_in, atten) feeds from ~``n_clips`` validation clips, every ``stride``-th frame, with the
    true recurrent state (from running the torch model through each clip)."""
    from ans.data import VBDDataset, manifest_path

    path = manifest_path("val")
    if path.exists():
        ds = VBDDataset(path, train=False, max_items=n_clips)
        wavs = [ds[i]["noisy"].numpy() for i in range(len(ds))]
    else:  # synthetic fallback (tests / no data)
        from ans.data import SyntheticDataset

        ds = SyntheticDataset(min(n_clips, 8), train=False, seed=seed)
        wavs = [ds[i]["noisy"].numpy() for i in range(len(ds))]
    feeds = []
    lim = np.array([100.0], np.float32)
    with torch.no_grad():
        for w in wavs:
            spec = model.stft(torch.from_numpy(w)[None])  # [1, T, F, 2] (streaming framing)
            st = model.init_state(1)
            for t in range(spec.shape[1]):
                frame = spec[:, t : t + 1]
                if t % stride == 0:
                    feeds.append({"spec": frame.numpy().copy(), "state_in": st.numpy().copy(), "atten_lim_db": lim})
                _, st = model(frame, st)
    return feeds


def quantize_static_int8(src: Path, dst: Path, model: nn.Module, n_clips: int = 100) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    feeds = calibration_frames(model, n_clips)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    pre = dst.with_name(dst.stem + "_pre.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        source = pre
    except Exception as e:  # pre-processing is optional
        print(f"quant_pre_process skipped: {e}")
        source = src
    try:
        quantize_static(
            str(source),
            str(dst),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"],
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"CalibMaxIntermediateOutputs": 256, "ActivationSymmetric": True},
        )
    finally:
        pre.unlink(missing_ok=True)
    print(f"static int8: calibrated on {len(feeds)} frames from <= {n_clips} validation clips")


def quant_eval(
    variants: dict[str, Path], state_shape: list[int], sets: list[str], limit: int | None, workers: int
) -> list:
    """Size, ms/frame (ORT, 1 thread) and PESQ/STOI per variant through the streaming loop."""
    import soundfile as sf

    from ans.benchmark import bench_ort
    from ans.metrics import compute_metrics_batch, mean_ci
    from ans.testsets import available_sets, read_set

    rows = []
    data = {}
    for s in available_sets(sets):
        items = [it for it in read_set(s, limit) if it.clean_path is not None]
        data[s] = [
            (sf.read(str(it.clean_path), dtype="float32")[0], sf.read(str(it.noisy_path), dtype="float32")[0])
            for it in items
        ]
    for variant, path in variants.items():
        sess = ort_session(path)
        row = {
            "variant": variant,
            "file": path.name,
            "size_kb": round(path.stat().st_size / 1024, 1),
            "ms_per_frame": round(bench_ort(path, n_frames=1500).ms_per_frame, 4),
        }
        for s, pairs in data.items():
            jobs = []
            for clean, noisy in pairs:
                n = min(len(clean), len(noisy))
                y = enhance_streaming(noisy[:n], ort_frame_fn(sess), state_shape, align=True)
                jobs.append((clean[:n], y))
            res = compute_metrics_batch(jobs, workers=workers, which=("pesq", "stoi", "si_sdr"))
            for m in ("pesq", "stoi", "si_sdr"):
                val = round(mean_ci([r[m] for r in res])[0], 4)
                row[m if s == "vbd" else f"{m}_{s}"] = val
            row["n" if s == "vbd" else f"n_{s}"] = len(jobs)
        rows.append(row)
        print(
            f"  {variant:<13s} {row['size_kb']:>8.1f} KiB  {row['ms_per_frame']:.4f} ms/frame  PESQ {row.get('pesq')}"
        )
    ref = next((r for r in rows if r["variant"] == "fp32"), None)
    for r in rows:
        if ref and r.get("pesq") is not None and ref.get("pesq") is not None:
            r["pesq_drop"] = round(ref["pesq"] - r["pesq"], 4)
            r["si_sdr_drop"] = round(ref["si_sdr"] - r["si_sdr"], 4)
            r["ships"] = r["variant"] == "fp32" or (r["pesq_drop"] < PESQ_GATE and r["si_sdr_drop"] < SISDR_GATE)
    return rows


def write_quantization(rows: list[dict], results: Path, web_copy: Path | None) -> None:
    from ans.evaluate import locked

    with locked(results):
        res = json.loads(results.read_text()) if results.exists() else {"version": "0.2", "rows": []}
        res["quantization"] = rows
        tmp = results.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
        tmp.replace(results)
        if web_copy is not None:
            shutil.copyfile(results, web_copy)


# ------------------------------------------------------------------------------------------------
# Export
# ------------------------------------------------------------------------------------------------


def _test_metrics(display: str) -> dict | None:
    """VB-D test metrics of this model from ml/exports/results.json, if it has been evaluated."""
    from ans.data import ML_ROOT

    p = ML_ROOT / "exports" / "results.json"
    try:
        res = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for r in res.get("rows", []):
        if r.get("model") == display and r.get("set", "vbd") == "vbd":
            keys = ("pesq", "stoi", "estoi", "si_sdr", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")
            return {k: r.get(k) for k in keys if r.get(k) is not None}
    return None


def export(
    ckpt: str | None,
    out_dir: Path,
    int8: bool = False,
    int8_static: bool = False,
    seed: int = 0,
    name: str | None = None,
    variant: str = "benchmark",
    calib_clips: int = 100,
    quant_eval_sets: list[str] | None = None,
    quant_limit: int | None = None,
    workers: int = 4,
    results: Path | None = None,
    web_results: Path | None = None,
) -> dict:
    from ans.registry import display_name
    from ans.runmeta import git_hash, read_meta

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, cfg, training = None, {}, {"distilled": False, "git_hash": git_hash()}
    if ckpt:
        from ans.models import load_checkpoint

        model = load_checkpoint(ckpt)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        disp = display_name(cfg) if cfg else Path(ckpt).parent.name
        metrics = _test_metrics(disp) or ck.get("best_metrics")
        meta = read_meta(Path(ckpt).parent) or {}
        aug = (cfg.get("data") or {}).get("augment") or {}
        extra = [k for k in ("extra_noise", "reverb", "mic", "level") if (aug.get(k) or {}).get("prob", 0) > 0]
        training = {
            "run": cfg.get("run_name"),
            "display_name": disp,
            "data": "VoiceBank+DEMAND" + (" + " + ", ".join(extra) if extra else "") + " (dynamic remixing)",
            "distilled": bool((cfg.get("distill") or {}).get("teacher_ckpt")),
            "epochs": (cfg.get("schedule") or {}).get("epochs"),
            "seed": cfg.get("seed"),
            "best_val": ck.get("best_metrics"),
            "git_hash": (meta.get("git") or {}).get("commit", "uncommitted")
            + (f"+{meta['git']['source_hash'][:12]}" if (meta.get("git") or {}).get("source_hash") else ""),
            "exported_with": git_hash(),
        }
    else:
        torch.manual_seed(seed)
        model = build_model("ans_crn").eval()
    model = model.cpu().eval()
    state_shape = list(model.state_shape)
    lookahead = int(getattr(model.cfg, "lookahead", 0) or 0)
    name = name or (f"crn-{variant}" if ckpt else "ans-crn-untrained")

    onnx_path = out_dir / "ans_stream.onnx"
    export_graph(model, onnx_path)
    sess = ort_session(onnx_path)
    err = validate(model, sess)
    err12 = validate(model, sess, atten=ATTEN_CASE_DB, seed=1)
    print(f"ORT vs torch max abs diff: {err:.2e} (unlimited), {err12:.2e} (atten_lim_db = 12)")
    # Floored bins keep the phase of a mask that may be far below the floor, which amplifies float
    # differences by floor/|M| (inherent to "floor magnitude, keep phase"); hence the looser tolerance.
    if err > 1e-4 or err12 > 5e-3:
        raise RuntimeError(f"ORT/torch mismatch {err:.2e} (tol 1e-4) / {err12:.2e} (tol 5e-3)")

    macs = crn_macs_per_frame(model)
    manifest = {
        "name": name,
        "version": VERSION,
        "variant": variant,
        "trained": bool(ckpt),
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop": HOP,
        "window": "sqrt_hann_periodic",
        "lookahead_frames": lookahead,
        "onnx": onnx_path.name,
        "inputs": {"spec": [1, 1, N_BINS, 2], "state_in": state_shape, "atten_lim_db": [1]},
        "outputs": {"spec_out": [1, 1, N_BINS, 2], "state_out": state_shape},
        "state_shape": state_shape,
        "atten_lim_db": dict(ATTEN),
        "params": count_params(model),
        "macs_per_frame": macs,
        "budget": budget(model, macs),
        "training": training,
        "metrics": metrics,
    }
    variants = {"fp32": onnx_path}
    if int8:
        q_path = out_dir / "ans_stream_int8.onnx"
        quantize_dynamic_int8(onnx_path, q_path)
        print(f"int8-dynamic ORT vs torch max abs diff: {validate(model, ort_session(q_path)):.2e}")
        manifest.setdefault("quantization", {})["int8-dynamic"] = print_op_table(q_path)
        variants["int8-dynamic"] = q_path
    if int8_static:
        s_path = out_dir / "ans_stream_int8_static.onnx"
        quantize_static_int8(onnx_path, s_path, model, calib_clips)
        print(f"int8-static ORT vs torch max abs diff: {validate(model, ort_session(s_path)):.2e}")
        manifest.setdefault("quantization", {})["int8-static"] = print_op_table(s_path)
        variants["int8-static"] = s_path

    bud = manifest["budget"]
    from ans.benchmark import bench_ort, bench_torch_stream

    bud["ms_per_frame_torch_cpu"] = round(bench_torch_stream(model, n_frames=500).ms_per_frame, 4)
    bud["ms_per_frame_ort_fp32"] = round(bench_ort(onnx_path, n_frames=1500).ms_per_frame, 4)
    for v, p in variants.items():
        if v != "fp32":
            bud[f"ms_per_frame_ort_{v.replace('-', '_')}"] = round(bench_ort(p, n_frames=1500).ms_per_frame, 4)
        bud[f"file_kb_{v.replace('-', '_')}"] = round(p.stat().st_size / 1024, 1)
    torch.set_num_threads(4)

    if quant_eval_sets and len(variants) > 1:
        print("\nquantization eval (ORT streaming loop):")
        rows = quant_eval(variants, state_shape, quant_eval_sets, quant_limit, workers)
        for r in rows:
            r["model"] = training.get("display_name", name)
        shippable = [r for r in rows if r["variant"] != "fp32" and r.get("ships")]
        best = min(shippable, key=lambda r: r["ms_per_frame"]) if shippable else None
        if best:
            manifest["onnx_int8"] = variants[best["variant"]].name
            manifest["int8_variant"] = best["variant"]
        else:
            manifest["onnx_int8_note"] = (
                f"no int8 variant within {PESQ_GATE} PESQ / {SISDR_GATE} dB SI-SDR of fp32; int8 not shipped"
            )
        if results is not None:
            write_quantization(rows, results, web_results)
    elif int8:
        manifest["onnx_int8"] = variants["int8-dynamic"].name  # unvalidated (no --quant-eval)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "test_vectors.json").write_text(json.dumps(make_test_vectors(sess, state_shape)) + "\n")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB), manifest.json, test_vectors.json")
    return manifest


def main() -> None:
    from ans.data import ML_ROOT, REPO_ROOT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="train.py checkpoint; omit for a seeded random-init model")
    ap.add_argument("--out", default="../web/public/models")
    ap.add_argument("--variant", default="benchmark", choices=["benchmark", "demo"])
    ap.add_argument("--int8", action="store_true", help="also write a dynamic-quantized int8 model")
    ap.add_argument("--int8-static", action="store_true", help="also write a static QDQ per-channel int8 model")
    ap.add_argument("--calib-clips", type=int, default=100)
    ap.add_argument("--quant-eval", action="store_true", help="score fp32/int8 variants -> results.json")
    ap.add_argument("--quant-sets", nargs="+", default=["vbd"], help="sets for --quant-eval")
    ap.add_argument("--quant-limit", type=int, help="subset size per set for --quant-eval")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results", default=str(ML_ROOT / "exports" / "results.json"))
    ap.add_argument("--web-results", default=str(REPO_ROOT / "web" / "public" / "results.json"))
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    args = ap.parse_args()
    export(
        args.ckpt,
        Path(args.out),
        int8=args.int8,
        int8_static=args.int8_static,
        seed=args.seed,
        name=args.name,
        variant=args.variant,
        calib_clips=args.calib_clips,
        quant_eval_sets=args.quant_sets if args.quant_eval else None,
        quant_limit=args.quant_limit,
        workers=args.workers,
        results=Path(args.results),
        web_results=None if args.no_web else Path(args.web_results),
    )


if __name__ == "__main__":
    main()
