"""Evaluate models on the test sets -> ``results.json`` v0.2 + per-utterance CSV (PLAN.md §3.3, §4.6, §4.10).

uv run python -m ans.evaluate --sets vbd ood_esc talker real \\
    --models noisy classical rnnoise deepfilternet crn:checkpoints/crn_small/best.pt \\
    --decompose --atten-sweep --merge

Models: ``noisy``, ``classical`` (Wiener), ``rnnoise``, ``deepfilternet`` (isolated envs, cached outputs),
``crn:<ckpt>`` (any ``ans_crn``/``crn_teacher`` checkpoint; display name from its config), ``tasnet:<ckpt>``.
Sets whose manifests are missing are skipped. DNSMOS P.835 is computed on every set (``--no-dnsmos`` to
skip). ``--decompose`` adds speech distortion / noise reduction for mask-based models (CRN family, Wiener).
``--atten-sweep`` scores the benchmark CRN at ``atten_lim_db`` ∈ {6, 12, 18, 24, ∞}.

Runs on the CPU by default (``--device cpu``) so it never competes with MPS training. ``--merge`` keeps
rows of models/sets not evaluated in this run (v0.1 rows without ``set`` are read as ``vbd``); the
per-utterance store ``exports/per_utterance.csv`` is merged the same way and feeds the significance tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import contextlib
import csv
import fcntl
import json
import math
import multiprocessing as mp
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from ans.benchmark import FRAME_MS, bench_torch_offline, bench_torch_stream, macs_per_frame
from ans.data import ML_ROOT, REPO_ROOT, SyntheticDataset
from ans.dsp import HOP
from ans.metrics import compute_metrics, mean_ci
from ans.models import ClassicalEnhancer, count_params, load_checkpoint
from ans.registry import BENCHMARK_RUN, DEMO_RUN, ckpt_config, display_name, run_info
from ans.testsets import SET_ORDER
from ans.train import pick_device

VERSION = "0.2"
INTRUSIVE = ("pesq", "stoi", "estoi", "si_sdr")
DNSMOS_KEYS = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")
DECOMP = ("speech_distortion_sisdr", "noise_reduction_db")
METRIC_KEYS = (*INTRUSIVE, *DNSMOS_KEYS, *DECOMP)
ATTEN_LIMS: tuple[float | None, ...] = (6.0, 12.0, 18.0, 24.0, None)
SAMPLES_JSON = REPO_ROOT / "web" / "public" / "samples" / "samples.json"

FIXED = {
    "noisy": ("Noisy (unprocessed)", "input"),
    "classical": ("Spectral subtraction (Wiener)", "baseline"),
    "rnnoise": ("RNNoise (reference)", "reference"),
    "deepfilternet": ("DeepFilterNet3 (reference)", "reference"),
}
LEGACY_NAMES = {"ANS-CRN (ours)": "CRN-256 (benchmark)"}


# ------------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------------


@dataclass
class ModelSpec:
    key: str
    ckpt: str | None = None
    name: str = ""
    family: str = "ours"
    info: dict = field(default_factory=dict)


def parse_model_arg(arg: str) -> ModelSpec:
    key, _, ckpt = arg.partition(":")
    key = {"wiener": "classical", "dfn": "deepfilternet", "df3": "deepfilternet"}.get(key, key)
    if key in FIXED:
        name, family = FIXED[key]
        return ModelSpec(key, None, name, family)
    if not ckpt:
        raise ValueError(f"model {key!r} needs a checkpoint: {key}:<path>")
    path = Path(ckpt)
    if not path.is_absolute() and not path.exists():
        path = ML_ROOT / path
    cfg = ckpt_config(path)
    info = run_info(cfg) if cfg else {}
    if key in ("tasnet", "conv_tasnet"):
        return ModelSpec("tasnet", str(path), "Conv-TasNet (causal, small)", "baseline", info)
    family = "teacher" if info.get("teacher") else "ours"
    return ModelSpec("crn", str(path), display_name(cfg) if cfg else path.parent.name, family, info)


class Runner:
    """Uniform enhancement (+ optional mask decomposition) and row metadata for one model."""

    def __init__(self, spec: ModelSpec, device: torch.device) -> None:
        self.spec, self.device = spec, device
        self.model = None
        self.mask_based = False
        self.timing: dict = {}
        info = spec.info
        self.meta: dict = {
            "family": spec.family,
            "causal": info.get("causal", True),
            "distilled": info.get("distilled", False),
            "params": None,
            "macs_per_frame": None,
            "ms_per_frame_cpu": None,
            "rtf": None,
        }
        for k in ("run", "hidden", "seed", "beta", "lookahead", "augment", "epochs"):
            if k in info:
                self.meta[k] = info[k]
        if spec.key == "noisy":
            return
        if spec.key == "classical":
            self.model = ClassicalEnhancer()
            self.meta["params"] = 0
            self.mask_based = True
            return
        if spec.key in ("rnnoise", "deepfilternet"):
            from ans.models.reference import INFO

            ref = INFO[spec.key]
            self.meta.update({"params": ref["params"], "macs_per_frame": ref["macs_per_frame"]})
            self.meta["native_sr"] = ref["native_sr"]
            self.meta["note"] = "trained on far more data, full band (48 kHz); run via 16->48->16 kHz resampling"
            return
        self.model = load_checkpoint(spec.ckpt, device)
        cpu_model = load_checkpoint(spec.ckpt, "cpu")
        self.mask_based = hasattr(cpu_model, "masks")
        self.meta["params"] = count_params(cpu_model)
        self.meta["macs_per_frame"] = macs_per_frame(cpu_model)
        threads = torch.get_num_threads()
        streaming = hasattr(cpu_model, "init_state") and getattr(cpu_model, "causal", True)
        bench = bench_torch_stream(cpu_model, n_frames=500) if streaming else bench_torch_offline(cpu_model)
        torch.set_num_threads(threads)
        self.meta["ms_per_frame_cpu"], self.meta["rtf"] = round(bench.ms_per_frame, 4), round(bench.rtf, 4)
        if getattr(getattr(cpu_model, "cfg", None), "lookahead", 0):
            self.meta["latency_ms"] = 20 + 10 * cpu_model.cfg.lookahead

    @property
    def name(self) -> str:
        return self.spec.name

    def prepare(self, set_name: str, items: list[dict]) -> None:
        if self.spec.key in ("rnnoise", "deepfilternet"):
            from ans.models.reference import ensure_outputs

            if any(it.get("noisy_path") is None for it in items):
                raise RuntimeError("reference systems need file-backed items")
            self.timing = ensure_outputs(self.spec.key, set_name, [(it["id"], it["noisy_path"]) for it in items])
            if self.timing.get("rtf"):
                self.meta["rtf"] = round(self.timing["rtf"], 4)
                self.meta["ms_per_frame_cpu"] = round(self.timing["rtf"] * FRAME_MS, 4)
                self.meta["timing_note"] = "includes 16<->48 kHz resampling"

    @torch.no_grad()
    def run(
        self, set_name: str, item: dict, decompose: bool, atten: float | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        noisy, clean = item["noisy"], item["clean"]
        can_dec = decompose and self.mask_based and clean is not None
        if self.spec.key == "noisy":
            return noisy, None, None
        if self.spec.key in ("rnnoise", "deepfilternet"):
            from ans.models.reference import load_output

            return load_output(self.spec.key, set_name, item["id"], len(noisy)), None, None
        if isinstance(self.model, ClassicalEnhancer):
            if can_dec:
                from ans.decompose import wiener_decompose

                return wiener_decompose(self.model, noisy, clean)
            return self.model.enhance_np(noisy), None, None
        if can_dec:
            from ans.decompose import crn_decompose

            return crn_decompose(self.model, noisy, clean, atten)
        x = torch.from_numpy(noisy).to(self.device)[None]
        y = self.model.enhance(x) if atten is None else self.model.enhance(x, atten_lim_db=atten)
        return y[0].float().cpu().numpy(), None, None

    def time_classical(self, wav: np.ndarray) -> None:
        if isinstance(self.model, ClassicalEnhancer):
            t0 = time.perf_counter()
            self.model.enhance_np(wav)
            ms = (time.perf_counter() - t0) * 1000 / (len(wav) / HOP)
            self.meta["ms_per_frame_cpu"], self.meta["rtf"] = round(ms, 4), round(ms / FRAME_MS, 4)


# ------------------------------------------------------------------------------------------------
# Scoring (process pool)
# ------------------------------------------------------------------------------------------------

_DNSMOS_OK = True


def _score(job: dict) -> dict:
    global _DNSMOS_OK
    out: dict = {}
    deg = job["deg"]
    if job.get("ref") is not None:
        out.update(compute_metrics(job["ref"], deg, which=INTRUSIVE))
    if job.get("dnsmos") and _DNSMOS_OK:
        try:
            from ans.dnsmos import dnsmos

            out.update(dnsmos(deg))
        except Exception as e:  # download blocked etc.
            _DNSMOS_OK = False
            print(f"[evaluate] DNSMOS disabled in this worker: {type(e).__name__}: {e}", flush=True)
    if job.get("s_hat") is not None and job.get("n_hat") is not None:
        from ans.decompose import decomposition_metrics

        out.update(decomposition_metrics(job["ref"], job["noise"], job["s_hat"], job["n_hat"]))
    return out


class Scorer:
    def __init__(self, workers: int) -> None:
        self.pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None

    def map(self, jobs: list[dict], desc: str) -> list[dict]:
        it = self.pool.imap(_score, jobs, chunksize=4) if self.pool else map(_score, jobs)
        return list(tqdm(it, total=len(jobs), desc=desc, leave=False))

    def close(self) -> None:
        if self.pool:
            self.pool.close()
            self.pool.join()


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------


def load_items(set_name: str, limit: int | None = None) -> list[dict]:
    if set_name == "synthetic":
        ds = SyntheticDataset(limit or 16, train=False, seed=123)
        out = []
        for i in range(len(ds)):
            it = ds[i]
            out.append(
                {
                    "id": it["id"],
                    "noise_type": it["noise_type"],
                    "snr_db": it["snr_db"],
                    "noisy": it["noisy"].numpy(),
                    "clean": it["clean"].numpy(),
                    "noisy_path": None,
                }
            )
        return out
    from ans.testsets import read_set

    out = []
    for e in read_set(set_name, limit):
        noisy, _ = sf.read(str(e.noisy_path), dtype="float32", always_2d=True)
        noisy = noisy.mean(axis=1)
        clean = None
        if e.clean_path is not None:
            clean, _ = sf.read(str(e.clean_path), dtype="float32", always_2d=True)
            clean = clean.mean(axis=1)
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]
        out.append(
            {
                "id": e.id,
                "noise_type": e.noise_type,
                "snr_db": e.snr_db,
                "noisy": noisy,
                "clean": clean,
                "noisy_path": e.noisy_path,
            }
        )
    return out


# ------------------------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------------------------


def _clean(x, nd: int = 4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def _means(recs: list[dict], keys=METRIC_KEYS) -> tuple[dict, dict]:
    vals, cis = {}, {}
    for m in keys:
        mean, half, n = mean_ci([r.get(m) for r in recs if r.get(m) is not None])
        vals[m], cis[m] = (_clean(mean), _clean(half)) if n else (None, None)
    return vals, cis


def snr_bucket(snr: float) -> str:
    return "unknown" if snr is None or math.isnan(snr) else f"{snr:g}"


def aggregate(records: list[dict], metas: dict[str, dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["model"], r["set"])].append(r)
    rows = []
    for (model, set_name), recs in groups.items():
        vals, cis = _means(recs)
        row = {"model": model, "set": set_name, **metas.get(model, {}), **vals}
        row["n"] = len(recs)
        row["ci95"] = {k: v for k, v in cis.items() if v is not None}
        rows.append(row)

    def breakdown(field_name: str, fmt, keys) -> dict:
        out: dict = defaultdict(lambda: defaultdict(dict))
        sub: dict = defaultdict(list)
        for r in records:
            sub[(r["set"], fmt(r[field_name]), r["model"])].append(r)
        for (s, g, model), recs in sorted(sub.items()):
            vals, _ = _means(recs, keys)
            out[s][g][model] = {**{k: v for k, v in vals.items() if v is not None}, "n": len(recs)}
        return {s: dict(d) for s, d in out.items()}

    return {
        "rows": rows,
        "per_noise_type": breakdown("noise_type", str, ("pesq", "stoi", "si_sdr", "dnsmos_ovrl", *DECOMP)),
        "per_snr": breakdown("snr_db", snr_bucket, ("pesq", "stoi", "si_sdr", "dnsmos_ovrl", *DECOMP)),
    }


def sweep_table(sweep_records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in sweep_records:
        groups[(r["model"], r["set"], r["atten_lim_db"])].append(r)
    out = []
    for (model, set_name, lim), recs in groups.items():
        vals, cis = _means(recs, ("pesq", "stoi", "si_sdr", *DECOMP, *DNSMOS_KEYS))
        out.append(
            {
                "model": model,
                "set": set_name,
                "atten_lim_db": lim,
                **vals,
                "n": len(recs),
                "ci95": {k: v for k, v in cis.items() if v is not None},
            }
        )
    return sorted(
        out, key=lambda e: (e["model"], e["set"], math.inf if e["atten_lim_db"] is None else e["atten_lim_db"])
    )


# ------------------------------------------------------------------------------------------------
# Derived tables (from merged rows / per-utterance store)
# ------------------------------------------------------------------------------------------------


def _is_plain_registry(r: dict) -> bool:
    return (
        r.get("family") == "ours"
        and not r.get("lookahead")
        and not r.get("augment")
        and r.get("epochs", 100) == 100
        and r.get("hidden") is not None
    )


def pareto_table(rows: list[dict]) -> list[dict]:
    out = []
    by = {(r["model"], r["set"]): r for r in rows}
    for r in rows:
        if not _is_plain_registry(r) or r.get("set") not in SET_ORDER:
            continue
        e = {
            "model": r["model"],
            "set": r["set"],
            "run": r.get("run"),
            "hidden": r.get("hidden"),
            "distilled": bool(r.get("distilled")),
            "beta": r.get("beta"),
            "seed": r.get("seed", 0),
            "macs_per_frame": r.get("macs_per_frame"),
            "params": r.get("params"),
        }
        for k in ("pesq", "stoi", "si_sdr", *DNSMOS_KEYS):
            e[k] = r.get(k)
        if r["set"] == "vbd":  # convenience: OOD numbers alongside the VB-D point
            for s in ("ood_esc", "talker", "reverb"):
                o = by.get((r["model"], s))
                if o:
                    e[f"pesq_{s}"], e[f"dnsmos_ovrl_{s}"] = o.get("pesq"), o.get("dnsmos_ovrl")
        out.append(e)
    return sorted(out, key=lambda e: (e["set"], e["distilled"], e["hidden"] or 0, e["seed"] or 0))


def lookahead_table(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if (
            r.get("family") == "ours"
            and r.get("hidden") == 256
            and not r.get("distilled")
            and not r.get("augment")
            and r.get("epochs", 100) == 100
            and not r.get("seed")
        ):
            k = int(r.get("lookahead") or 0)
            out.append(
                {
                    "lookahead_frames": k,
                    "latency_ms": 20 + 10 * k,
                    "model": r["model"],
                    "set": r["set"],
                    "pesq": r.get("pesq"),
                    "stoi": r.get("stoi"),
                    "si_sdr": r.get("si_sdr"),
                    "dnsmos_ovrl": r.get("dnsmos_ovrl"),
                }
            )
    return sorted(out, key=lambda e: (e["set"], e["lookahead_frames"]))


def significance_table(rows: list[dict], store: list[dict]) -> list[dict]:
    from ans.stats import compare_pairs

    by_set: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for rec in store:
        for m in (*INTRUSIVE, "dnsmos_ovrl"):
            v = rec.get(m)
            if v not in (None, ""):
                by_set[rec["set"]][m][rec["model"]][rec["id"]] = float(v)
    rows_by_set: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_set[r["set"]].append(r)
    out = []
    def arm(r: dict) -> tuple:  # everything that defines a training arm except distillation
        aug = r.get("augment") or ()
        return (r.get("hidden"), r.get("seed", 0), tuple(aug), r.get("epochs", 100), r.get("lookahead") or 0)

    for set_name, srows in rows_by_set.items():
        names = {r["model"] for r in srows}
        ours = [r for r in srows if r.get("family") == "ours" and r.get("hidden") is not None]
        plain = {arm(r): r["model"] for r in ours if not r.get("distilled")}
        kd_pairs = [(r["model"], plain[arm(r)]) for r in ours if r.get("distilled") and arm(r) in plain]
        bench = next((r["model"] for r in srows if r.get("run") == BENCHMARK_RUN), None)
        demo = next((r["model"] for r in srows if r.get("run") == DEMO_RUN), None)
        fixed = [FIXED[k][0] for k in FIXED if FIXED[k][0] in names]
        base_pairs = [(bench, b) for b in fixed if bench]
        demo_pairs = [(demo, b) for b in [bench, *fixed] if demo and b]
        metrics = ("pesq", "stoi", "si_sdr", "dnsmos_ovrl") if set_name != "real" else ("dnsmos_ovrl",)
        for metric in metrics:
            scores = by_set[set_name][metric]
            out += compare_pairs(scores, kd_pairs, metric, set_name, "kd_vs_plain")
            out += compare_pairs(scores, base_pairs, metric, set_name, "benchmark_vs_baselines")
            out += compare_pairs(scores, demo_pairs, metric, set_name, "demo_vs_others")
    for r in out:
        for k in ("median_diff", "mean_diff", "statistic", "p_wilcoxon", "p_holm"):
            r[k] = None if r.get(k) is None or not math.isfinite(r[k]) else float(f"{r[k]:.6g}")
    return out


def per_sample_table(store: list[dict], rows: list[dict]) -> dict:
    try:
        ids = {s["id"] for s in json.loads(SAMPLES_JSON.read_text())}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}
    # §3.3: {sample id: {model: metrics}}, limited to the rows Studio shows (inputs, baselines,
    # reference systems, the benchmark and the shipped demo model), in that order.
    rank = {"input": 0, "baseline": 1, "reference": 2}
    shown = {
        r["model"]: rank.get(r.get("family"), 3 if r.get("run") == BENCHMARK_RUN else 4)
        for r in rows
        if r.get("set") == "vbd" and (r.get("family") in rank or r.get("run") in (BENCHMARK_RUN, DEMO_RUN))
    }
    out: dict = defaultdict(dict)
    for rec in sorted(store, key=lambda s: shown.get(s["model"], 99)):
        if rec["set"] == "vbd" and rec["id"] in ids and rec["model"] in shown:
            out[rec["id"]][rec["model"]] = {
                k: _clean(float(rec[k]) if rec.get(k) not in (None, "") else None)
                for k in ("pesq", "stoi", "si_sdr", "dnsmos_ovrl")
            }
    return dict(out)


def test_sets_meta(rows: list[dict]) -> dict:
    from ans.testsets import SET_META, set_metadata

    out = {}
    for s in sorted({r["set"] for r in rows}, key=lambda s: (list(SET_META).index(s) if s in SET_META else 99, s)):
        n = max((r.get("n") or 0) for r in rows if r["set"] == s)
        out[s] = set_metadata(s, n) if s in SET_META else {"label": f"{s} ({n})", "intrusive": True, "n": n}
    return out


# ------------------------------------------------------------------------------------------------
# Merge / store
# ------------------------------------------------------------------------------------------------


def upgrade_v01(old: dict) -> dict:
    """v0.1 -> v0.2 shapes: rows get ``set: vbd``; breakdowns get a set level; legacy names renamed."""
    old = json.loads(json.dumps(old))
    for r in old.get("rows", []):
        r.setdefault("set", "vbd")
        r["model"] = LEGACY_NAMES.get(r["model"], r["model"])
    for key in ("per_noise_type", "per_snr"):
        d = old.get(key) or {}
        # v0.1: {group: {model: {..., "n"}}}  ->  v0.2: {set: {group: {model: {..., "n"}}}}
        is_v01 = any(isinstance(leaf, dict) and "n" in leaf for g in d.values() for leaf in g.values())
        if is_v01:
            d = {"vbd": {g: {LEGACY_NAMES.get(m, m): v for m, v in models.items()} for g, models in d.items()}}
        old[key] = d
    return old


def merge_results(old: dict, new: dict) -> dict:
    old = upgrade_v01(old)
    done = {(r["model"], r["set"]) for r in new["rows"]}
    rows = [r for r in old.get("rows", []) if (r["model"], r["set"]) not in done] + new["rows"]
    out = {**old, **new, "rows": rows}
    for key in ("per_noise_type", "per_snr"):
        merged: dict = {}
        for s, groups in (old.get(key) or {}).items():
            for g, models in groups.items():
                keep = {m: v for m, v in models.items() if (m, s) not in done}
                if keep:
                    merged.setdefault(s, {}).setdefault(g, {}).update(keep)
        for s, groups in new.get(key, {}).items():
            for g, models in groups.items():
                merged.setdefault(s, {}).setdefault(g, {}).update(models)
        out[key] = merged
    sweep_done = {(e["model"], e["set"]) for e in new.get("atten_sweep") or []}
    out["atten_sweep"] = [e for e in old.get("atten_sweep") or [] if (e["model"], e["set"]) not in sweep_done] + (
        new.get("atten_sweep") or []
    )
    for key in ("quantization", "mos"):
        if new.get(key) is None and old.get(key) is not None:
            out[key] = old[key]
    return out


STORE_FIELDS = ["set", "id", "model", "noise_type", "snr_db", *METRIC_KEYS]


def read_store(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("set", "vbd")
        if not r.get("set"):
            r["set"] = "vbd"
        r["model"] = LEGACY_NAMES.get(r["model"], r["model"])
    return rows


def write_store(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STORE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in STORE_FIELDS})
    tmp.replace(path)


@contextlib.contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


def evaluate(
    models: list[str],
    sets: list[str] | None = None,
    limit: int | None = None,
    synthetic: bool = False,
    device: str = "cpu",
    workers: int | None = 4,
    out_json: Path | None = None,
    out_csv: Path | None = None,
    web_copy: Path | None = None,
    merge: bool = False,
    decompose: bool = False,
    atten_sweep: list[float | None] | None = None,
    atten_model: str | None = None,
    atten_sets: list[str] | None = None,
    dnsmos: bool = True,
    manifest: str | None = None,
) -> dict:
    from ans.runmeta import git_hash
    from ans.testsets import available_sets

    dev = pick_device(device)
    if synthetic:
        sets = ["synthetic"]
    else:
        wanted = sets or ["vbd"]
        sets = available_sets(wanted)
        if skipped := [s for s in wanted if s not in sets]:
            print(f"[evaluate] skipping sets without a manifest: {skipped}")
    if manifest:
        raise SystemExit("--manifest was replaced by --sets (manifests are discovered, PLAN.md §3.5)")
    specs = [parse_model_arg(m) for m in models]
    seen: dict[str, int] = defaultdict(int)
    for s in specs:
        seen[s.name] += 1
    for s in specs:
        if seen[s.name] > 1 and s.ckpt:
            s.name = f"{s.name} [{Path(s.ckpt).parent.name}]"
    runners = [Runner(s, dev) for s in specs]
    if atten_sweep:
        cands = [r for r in runners if r.spec.key == "crn" and r.mask_based]
        target = next((r for r in cands if atten_model and atten_model in (r.name, r.meta.get("run"))), None)
        target = target or next((r for r in cands if r.meta.get("run") == BENCHMARK_RUN), None)
        target = target or (cands[0] if cands else None)
    else:
        target = None
    atten_sets = atten_sets or ["vbd", "synthetic"]

    scorer = Scorer(workers if workers is not None else 4)
    records: list[dict] = []
    sweep_records: list[dict] = []
    try:
        for set_name in sets:
            items = load_items(set_name, limit)
            print(f"[evaluate] {set_name}: {len(items)} items, {len(runners)} models")
            for r in runners:
                r.prepare(set_name, items)
                lims: list[float | None] = [None]
                if r is target and set_name in atten_sets:
                    lims += [lim for lim in atten_sweep if lim is not None]
                for lim in lims:
                    jobs = []
                    for it in tqdm(items, desc=f"{set_name}:{r.name}" + (f"@{lim:g}dB" if lim else ""), leave=False):
                        dec = decompose or (r is target and set_name in atten_sets)
                        y, s_hat, n_hat = r.run(set_name, it, dec, lim)
                        jobs.append(
                            {
                                "ref": it["clean"],
                                "deg": np.asarray(y, np.float32),
                                "noise": None if it["clean"] is None else it["noisy"] - it["clean"],
                                "s_hat": s_hat,
                                "n_hat": n_hat,
                                "dnsmos": dnsmos,
                            }
                        )
                    scores = scorer.map(jobs, f"score {set_name}:{r.name}")
                    for it, m in zip(items, scores, strict=True):
                        rec = {
                            "set": set_name,
                            "id": it["id"],
                            "model": r.name,
                            "noise_type": it["noise_type"],
                            "snr_db": it["snr_db"],
                            **m,
                        }
                        if lim is None:
                            records.append(rec)
                        if r is target and set_name in atten_sets:
                            sweep_records.append({**rec, "atten_lim_db": lim})
            if items:
                for r in runners:
                    r.time_classical(items[0]["noisy"])
    finally:
        scorer.close()

    out_json = out_json or ML_ROOT / "exports" / "results.json"
    out_csv = out_csv or out_json.with_name("per_utterance.csv")
    agg = aggregate(records, {r.name: r.meta for r in runners})
    new = {
        "version": VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        **agg,
        "atten_sweep": sweep_table(sweep_records) if sweep_records else [],
        "dnsmos": {"enabled": dnsmos, "citation": "DNSMOS P.835, Reddy et al., ICASSP 2022 (microsoft/DNS-Challenge)"},
    }
    with locked(out_json):
        old = json.loads(out_json.read_text()) if merge and out_json.exists() else {}
        result = merge_results(old, new) if merge and old else new
        store_old = read_store(out_csv) if merge else []
        done = {(r["model"], r["set"]) for r in new["rows"]}
        store = [s for s in store_old if (s["model"], s["set"]) not in done] + records
        result["test_sets"] = test_sets_meta(result["rows"])
        result["pareto"] = pareto_table(result["rows"])
        result["lookahead"] = lookahead_table(result["rows"])
        result["significance"] = significance_table(result["rows"], store)
        result["per_sample"] = per_sample_table(store, result["rows"])
        result.setdefault("quantization", None)
        result.setdefault("mos", None)
        result.pop("dataset", None)
        order = [
            "version",
            "generated_at",
            "git_hash",
            "test_sets",
            "rows",
            "per_noise_type",
            "per_snr",
            "pareto",
            "atten_sweep",
            "lookahead",
            "quantization",
            "significance",
            "per_sample",
            "mos",
            "dnsmos",
        ]
        result = {k: result[k] for k in order if k in result} | {k: v for k, v in result.items() if k not in order}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        write_store(out_csv, store)
        tmp = out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(out_json)
        if web_copy is not None:
            web_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_json, web_copy)
    print_tables(result, sets)
    print(f"wrote {out_json}, {out_csv}" + (f", {web_copy}" if web_copy else ""))
    return result


def print_tables(result: dict, sets: list[str] | None = None) -> None:
    def f(v, fmt):
        return format(v, fmt) if isinstance(v, int | float) else "—"

    for s in sets or sorted({r["set"] for r in result["rows"]}):
        rows = [r for r in result["rows"] if r["set"] == s]
        if not rows:
            continue
        print(f"\n[{s}] {result.get('test_sets', {}).get(s, {}).get('label', s)}")
        print(
            f"{'model':<34s} {'PESQ':>6s} {'STOI':>6s} {'SI-SDR':>7s} {'SIG':>5s} {'BAK':>5s} {'OVRL':>5s} "
            f"{'SD':>6s} {'NR':>6s}"
        )
        for r in rows:
            print(
                f"{r['model']:<34s} {f(r.get('pesq'), '6.3f')} {f(r.get('stoi'), '6.3f')} {f(r.get('si_sdr'), '7.2f')} "
                f"{f(r.get('dnsmos_sig'), '5.2f')} {f(r.get('dnsmos_bak'), '5.2f')} {f(r.get('dnsmos_ovrl'), '5.2f')} "
                f"{f(r.get('speech_distortion_sisdr'), '6.2f')} {f(r.get('noise_reduction_db'), '6.2f')}"
            )


def _lim(s: str) -> float | None:
    return None if s.lower() in ("inf", "none", "null", "∞") else float(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["noisy", "classical"],
        help="noisy classical rnnoise deepfilternet crn:<ckpt> ...",
    )
    ap.add_argument("--sets", nargs="+", default=["vbd"], help="vbd ood_esc talker reverb real (missing are skipped)")
    ap.add_argument("--decompose", action="store_true", help="speech distortion / noise reduction (mask-based models)")
    ap.add_argument("--atten-sweep", nargs="*", type=_lim, help="atten_lim_db values (default 6 12 18 24 inf)")
    ap.add_argument("--atten-model", help="model (display name or run) for the sweep; default the benchmark CRN")
    ap.add_argument("--atten-sets", nargs="+", default=["vbd"], help="sets for the attenuation sweep")
    ap.add_argument("--no-dnsmos", action="store_true")
    ap.add_argument("--limit", type=int, help="evaluate an evenly spaced subset of each set")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu (default) | auto | mps | cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ML_ROOT / "exports" / "results.json"))
    ap.add_argument("--csv", help="per-utterance store (default: next to --out)")
    ap.add_argument("--web", default=str(REPO_ROOT / "web" / "public" / "results.json"), help="copy target")
    ap.add_argument("--no-web", action="store_true", help="don't copy results.json into web/public")
    ap.add_argument("--merge", action="store_true", help="keep rows of models/sets not evaluated in this run")
    ap.add_argument("--manifest", help=argparse.SUPPRESS)
    args = ap.parse_args()
    sweep = None
    if args.atten_sweep is not None:
        sweep = list(args.atten_sweep) or list(ATTEN_LIMS)
        if None not in sweep:
            sweep.append(None)
    evaluate(
        args.models,
        sets=args.sets,
        limit=args.limit,
        synthetic=args.synthetic,
        device=args.device,
        workers=args.workers,
        out_json=Path(args.out),
        out_csv=Path(args.csv) if args.csv else None,
        web_copy=None if (args.no_web or args.synthetic) else Path(args.web),
        merge=args.merge,
        decompose=args.decompose,
        atten_sweep=sweep,
        atten_model=args.atten_model,
        atten_sets=args.atten_sets,
        dnsmos=not args.no_dnsmos,
        manifest=args.manifest,
    )


if __name__ == "__main__":
    main()
