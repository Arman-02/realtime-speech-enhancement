"""Params, MACs/frame, ms/frame and RTF for torch CPU, ORT CPU (1 thread) and ORT int8, plus the
deployment budget (PLAN.md §4.7).

uv run python -m ans.benchmark --onnx ../web/public/models/ans_stream.onnx [--ckpt best.pt] [--budget]

``--budget``: weight memory (fp32, and int8 with the GRU kept in fp32 as ORT quantization does), peak
activation memory per frame (largest live tensor set, analytic via forward hooks), recurrent state memory,
and MMAC/s (MACs per frame x 100).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ans.dsp import HOP, N_BINS, SAMPLE_RATE
from ans.models import build_model, count_params, load_checkpoint

FRAME_MS = 1000.0 * HOP / SAMPLE_RATE


def count_macs(model: nn.Module, *inputs: torch.Tensor) -> int:
    """MACs of conv / linear / GRU layers for one forward pass (elementwise ops ignored)."""
    total = 0

    def conv_hook(m: nn.Module, inp, out) -> None:
        nonlocal total
        k = int(np.prod(m.kernel_size))
        if isinstance(m, nn.ConvTranspose1d | nn.ConvTranspose2d):
            total += inp[0].numel() * (m.out_channels // m.groups) * k
        else:
            total += out.numel() * (m.in_channels // m.groups) * k

    def linear_hook(m: nn.Linear, inp, out) -> None:
        nonlocal total
        total += out.numel() * m.in_features

    def gru_hook(m: nn.GRU, inp, out) -> None:
        nonlocal total
        x = inp[0]
        steps = x.shape[0] * x.shape[1]
        dirs = 2 if m.bidirectional else 1
        for layer in range(m.num_layers):
            in_size = m.input_size if layer == 0 else m.hidden_size * dirs
            total += steps * dirs * 3 * (in_size + m.hidden_size) * m.hidden_size

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv1d | nn.Conv2d | nn.ConvTranspose1d | nn.ConvTranspose2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.GRU):
            handles.append(mod.register_forward_hook(gru_hook))
    with torch.no_grad():
        model(*inputs)
    for h in handles:
        h.remove()
    return int(total)


def crn_macs_per_frame(model: nn.Module) -> int:
    spec = torch.zeros(1, 1, N_BINS, 2)
    return count_macs(model, spec, model.init_state(1))


def macs_per_frame(model: nn.Module) -> int:
    """MACs per 10 ms hop for any registered model."""
    if hasattr(model, "init_state"):
        return crn_macs_per_frame(model)
    seconds = 1.0
    wav = torch.zeros(1, int(SAMPLE_RATE * seconds))
    return int(count_macs(model, wav) / (seconds * 1000.0 / FRAME_MS))


QUANTIZABLE = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear)


def weight_bytes(model: nn.Module) -> dict[str, int]:
    """fp32 bytes, and int8 bytes where conv/linear weights are int8 (+ fp32 scales) and everything else
    (biases, norms, PReLU, the GRU) stays fp32 — what ORT dynamic/static quantization produces."""
    q = 0
    for mod in model.modules():
        if isinstance(mod, QUANTIZABLE):
            q += mod.weight.numel()
    total = count_params(model)
    out_channels = sum(m.weight.shape[0] for m in model.modules() if isinstance(m, QUANTIZABLE))
    return {"fp32": 4 * total, "int8": q + 4 * (total - q) + 4 * out_channels}


@torch.no_grad()
def peak_activation_bytes(model: nn.Module) -> int:
    """Largest live float32 tensor set during one streaming frame (analytic, via forward hooks).

    Live set at a layer = its input + output + tensors held for later (U-Net skips, GRU state)."""
    events: list[tuple[str, int, int]] = []

    def hook(name):
        def fn(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            i = sum(t.numel() for t in inp if isinstance(t, torch.Tensor))
            events.append((name, i, o.numel()))

        return fn

    handles = []
    for name, mod in model.named_modules():
        if name and not any(True for _ in mod.children()):
            handles.append(mod.register_forward_hook(hook(name)))
    spec = torch.zeros(1, 1, N_BINS, 2)
    model(spec, model.init_state(1))
    for h in handles:
        h.remove()
    state = int(np.prod(model.state_shape))
    skips = 0
    peak = 0
    enc_out = {}
    for name, i, o in events:
        if name.startswith("encoder.") and name.endswith(".act"):
            enc_out[name.split(".")[1]] = o
            skips += o
        live = i + o + skips + 2 * state
        peak = max(peak, live)
        if name.startswith("decoder.") and name.endswith(".conv"):
            idx = str(len(enc_out) - 1 - int(name.split(".")[1]))
            skips -= enc_out.pop(idx, 0)
    return 4 * peak


def budget(model: nn.Module, macs: int | None = None) -> dict:
    """Deployment budget for the manifest / model card (KiB, MMAC/s)."""
    wb = weight_bytes(model)
    macs = macs if macs is not None else macs_per_frame(model)
    return {
        "weights_kb_fp32": round(wb["fp32"] / 1024, 1),
        "weights_kb_int8": round(wb["int8"] / 1024, 1),
        "peak_activation_kb": round(peak_activation_bytes(model) / 1024, 1),
        "state_kb": round(4 * int(np.prod(model.state_shape)) / 1024, 2),
        "mmacs_per_s": round(macs * 1000.0 / FRAME_MS / 1e6, 1),
        "notes": "int8 keeps the GRU, biases and norms in fp32; peak activation is analytic (float32)",
    }


@dataclass
class BenchResult:
    backend: str
    ms_per_frame: float
    rtf: float


def _time_loop(step, n_frames: int, warmup: int) -> float:
    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(n_frames):
        step()
    return (time.perf_counter() - t0) * 1000.0 / n_frames


def bench_torch_stream(model: nn.Module, n_frames: int = 1000, threads: int = 1) -> BenchResult:
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    spec = torch.randn(1, 1, N_BINS, 2)
    state = [model.init_state(1)]

    @torch.inference_mode()
    def step() -> None:
        _, state[0] = model(spec, state[0])

    ms = _time_loop(step, n_frames, 50)
    return BenchResult(f"torch-cpu-{threads}t", ms, ms / FRAME_MS)


def bench_torch_offline(model: nn.Module, seconds: float = 4.0, threads: int = 1) -> BenchResult:
    """Offline models (e.g. Conv-TasNet): time a whole clip, report per-10 ms-hop cost."""
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    wav = torch.randn(1, int(SAMPLE_RATE * seconds)) * 0.1
    with torch.inference_mode():
        model.enhance(wav)
        t0 = time.perf_counter()
        model.enhance(wav)
        dt = time.perf_counter() - t0
    ms = dt * 1000.0 / (seconds * 1000.0 / FRAME_MS)
    return BenchResult(f"torch-cpu-{threads}t-offline", ms, ms / FRAME_MS)


def bench_ort(onnx_path: str | Path, n_frames: int = 2000, threads: int = 1) -> BenchResult:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    shape = [d if isinstance(d, int) else 1 for d in inputs["state_in"].shape]
    spec = np.random.default_rng(0).standard_normal((1, 1, N_BINS, 2)).astype(np.float32)
    state = [np.zeros(shape, np.float32)]
    extra = {"atten_lim_db": np.array([100.0], np.float32)} if "atten_lim_db" in inputs else {}

    def step() -> None:
        _, state[0] = sess.run(None, {"spec": spec, "state_in": state[0], **extra})

    ms = _time_loop(step, n_frames, 100)
    return BenchResult(f"ort-cpu-{threads}t:{Path(onnx_path).name}", ms, ms / FRAME_MS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="checkpoint (default: random-init ans_crn)")
    ap.add_argument("--model", default="ans_crn")
    ap.add_argument("--onnx", default="../web/public/models/ans_stream.onnx")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--json", help="write results to this JSON file")
    ap.add_argument("--budget", action="store_true", help="print the deployment budget table (§4.7)")
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt) if args.ckpt else build_model(args.model).eval()
    params = count_params(model)
    macs = macs_per_frame(model)
    print(f"params: {params:,}   MACs/frame: {macs:,}   ({macs * 100 / 1e9:.3f} GMAC/s)")

    results: list[BenchResult] = []
    if hasattr(model, "init_state"):
        results.append(bench_torch_stream(model, n_frames=min(args.frames, 1000)))
    else:
        results.append(bench_torch_offline(model))
    onnx_path = Path(args.onnx)
    variants = [onnx_path, onnx_path.with_name(onnx_path.stem + "_int8.onnx")]
    variants.append(onnx_path.with_name(onnx_path.stem + "_int8_static.onnx"))
    for p in variants:
        if p.exists():
            results.append(bench_ort(p, n_frames=args.frames))
    for r in results:
        print(f"{r.backend:<40s} {r.ms_per_frame:8.4f} ms/frame   RTF {r.rtf:.4f}")
    bud = None
    if args.budget and hasattr(model, "init_state"):
        bud = budget(model, macs)
        print("\ndeployment budget")
        for k, v in bud.items():
            print(f"  {k:<20s} {v}")
        for p in variants:
            if p.exists():
                print(f"  {'file ' + p.name:<20s} {p.stat().st_size / 1024:.1f} KiB")
    if args.json:
        out = {"params": params, "macs_per_frame": macs, "results": [asdict(r) for r in results], "budget": bud}
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
