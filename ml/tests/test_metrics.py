from __future__ import annotations

import numpy as np

from ans.data import SyntheticDataset
from ans.metrics import compute_metrics, compute_metrics_batch, mean_ci, pesq_wb, si_sdr, stoi


def _clean() -> np.ndarray:
    return SyntheticDataset(2, train=False, seed=0)[0]["clean"].numpy()


def test_pesq_identical_is_max():
    x = _clean()
    assert pesq_wb(x, x) > 4.4


def test_si_sdr_identical_and_scaled():
    x = _clean()
    assert si_sdr(x, x) > 60
    assert si_sdr(x, 0.3 * x) > 60
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(x)) * np.std(x)
    assert abs(si_sdr(x, x + noise)) < 1.0  # ~0 dB


def test_stoi_identical():
    x = _clean()
    assert stoi(x, x) > 0.99
    assert stoi(x, x, extended=True) > 0.99


def test_batch_matches_serial():
    item = SyntheticDataset(2, train=False, seed=0)[1]
    pairs = [(item["clean"].numpy(), item["noisy"].numpy())] * 3
    serial = [compute_metrics(r, d) for r, d in pairs]
    parallel = compute_metrics_batch(pairs, workers=2)
    for a, b in zip(serial, parallel, strict=True):
        for k in a:
            assert np.isclose(a[k], b[k])


def test_mean_ci_ignores_nan():
    m, ci, n = mean_ci([1.0, 2.0, float("nan"), 3.0])
    assert m == 2.0 and n == 3 and ci > 0
