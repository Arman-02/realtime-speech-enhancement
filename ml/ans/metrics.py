"""Objective metrics (PLAN.md §4.6): PESQ-WB, STOI, ESTOI, SI-SDR, plus a multiprocessing batch helper."""

from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Iterable, Sequence

import numpy as np

from ans.dsp import SAMPLE_RATE

METRICS = ("pesq", "stoi", "estoi", "si_sdr")


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    ref = np.asarray(ref, np.float64) - np.mean(ref)
    est = np.asarray(est, np.float64) - np.mean(est)
    target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def pesq_wb(ref: np.ndarray, deg: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    from pesq import pesq

    try:
        return float(pesq(sr, np.asarray(ref, np.float32), np.asarray(deg, np.float32), "wb"))
    except Exception:
        return math.nan


def stoi(ref: np.ndarray, deg: np.ndarray, sr: int = SAMPLE_RATE, extended: bool = False) -> float:
    from pystoi import stoi as _stoi

    try:
        return float(_stoi(np.asarray(ref, np.float64), np.asarray(deg, np.float64), sr, extended=extended))
    except Exception:
        return math.nan


def compute_metrics(
    ref: np.ndarray, deg: np.ndarray, sr: int = SAMPLE_RATE, which: Sequence[str] = METRICS
) -> dict[str, float]:
    n = min(len(ref), len(deg))
    ref, deg = np.asarray(ref[:n]), np.asarray(deg[:n])
    fns = {
        "pesq": lambda: pesq_wb(ref, deg, sr),
        "stoi": lambda: stoi(ref, deg, sr),
        "estoi": lambda: stoi(ref, deg, sr, extended=True),
        "si_sdr": lambda: si_sdr(ref, deg),
    }
    return {k: fns[k]() for k in which}


def _worker(args: tuple) -> dict[str, float]:
    return compute_metrics(*args)


def compute_metrics_batch(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    workers: int | None = None,
    which: Sequence[str] = METRICS,
    sr: int = SAMPLE_RATE,
    progress: bool = False,
) -> list[dict[str, float]]:
    """Metrics for many ``(ref, deg)`` pairs in a process pool (PESQ is CPU-bound)."""
    jobs = [(r, d, sr, tuple(which)) for r, d in pairs]
    workers = workers if workers is not None else max(1, (mp.cpu_count() or 2) - 2)
    if workers <= 1 or len(jobs) <= 1:
        return [_worker(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        it = pool.imap(_worker, jobs, chunksize=4)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(jobs), desc="metrics")
        return list(it)


def mean_ci(values: Sequence[float]) -> tuple[float, float, int]:
    """Mean and 95 % CI half-width (normal approx.), ignoring NaNs."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], np.float64)
    if len(v) == 0:
        return math.nan, math.nan, 0
    ci = 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(ci), len(v)
