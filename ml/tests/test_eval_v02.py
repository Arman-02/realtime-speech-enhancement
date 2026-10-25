"""Evaluation suite v0.2: decomposition, statistics, MOS, results merging (PLAN.md §3.3, §4.6, §4.8, §5.1.6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ans.data import SyntheticDataset
from ans.decompose import crn_decompose, decomposition_metrics, wiener_decompose
from ans.models import ClassicalEnhancer, build_model
from ans.stats import holm, mos, wilcoxon


def _pair(seed=0):
    it = SyntheticDataset(2, train=False, seed=seed)[0]
    return it["noisy"].numpy(), it["clean"].numpy()


@pytest.mark.parametrize("k", [0, 2])
@pytest.mark.parametrize("atten", [None, 12.0])
def test_crn_decomposition_consistent(k, atten):
    torch.manual_seed(0)
    model = build_model("ans_crn", hidden=64, lookahead=k).eval()
    noisy, clean = _pair()
    y, s_hat, n_hat = crn_decompose(model, noisy, clean, atten)
    with torch.no_grad():
        ref = model.enhance(torch.from_numpy(noisy)[None], atten_lim_db=atten)[0].numpy()
    np.testing.assert_allclose(y, ref, atol=1e-4)
    np.testing.assert_allclose(s_hat + n_hat, y, atol=1e-4)  # iSTFT(M⊙S) + iSTFT(M⊙N) = ŷ
    m = decomposition_metrics(clean, noisy - clean, s_hat, n_hat)
    assert np.isfinite(m["speech_distortion_sisdr"]) and np.isfinite(m["noise_reduction_db"])


def test_wiener_decomposition_consistent():
    noisy, clean = _pair(1)
    enh = ClassicalEnhancer()
    y, s_hat, n_hat = wiener_decompose(enh, noisy, clean)
    np.testing.assert_allclose(y, enh.enhance_np(noisy), atol=1e-4)
    np.testing.assert_allclose(s_hat + n_hat, y, atol=1e-4)
    m = decomposition_metrics(clean, noisy - clean, s_hat, n_hat)
    assert m["noise_reduction_db"] > 0  # Wiener gains <= 1 remove noise


def test_atten_limit_caps_noise_reduction():
    """With a limit of L dB, |M| >= 10^(-L/20): noise energy can drop by ~L dB at most (plus OLA effects)."""
    torch.manual_seed(0)
    model = build_model("ans_crn", hidden=64).eval()
    with torch.no_grad():
        model.decoder[-1].conv.bias.fill_(0.0)
        model.decoder[-1].conv.weight.mul_(1e-3)  # near-zero mask: removes almost everything
    noisy, clean = _pair()
    free = decomposition_metrics(clean, noisy - clean, *crn_decompose(model, noisy, clean)[1:])
    lim = decomposition_metrics(clean, noisy - clean, *crn_decompose(model, noisy, clean, 6.0)[1:])
    assert free["noise_reduction_db"] > 20
    assert lim["noise_reduction_db"] < 10


def test_wilcoxon_matches_scipy(rng):
    from scipy.stats import wilcoxon as sw

    a = rng.normal(2.5, 0.3, 80)
    b = a - rng.normal(0.05, 0.1, 80)
    r = wilcoxon(a, b)
    ref = sw(a, b)
    assert r["p_wilcoxon"] == pytest.approx(ref.pvalue) and r["statistic"] == pytest.approx(ref.statistic)
    assert r["n"] == 80 and r["median_diff"] == pytest.approx(np.median(a - b))
    a2 = a.copy()
    a2[3] = np.nan
    assert wilcoxon(a2, b)["n"] == 79
    assert wilcoxon(a, a)["p_wilcoxon"] == 1.0


def test_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = holm(p)
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)=0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([0.01, float("nan")])[1]) and holm([0.01, float("nan")])[0] == pytest.approx(0.01)


def _resp(rater, clean_scores, noisy_scores, crn=4):
    ratings = [{"trial": "practice", "condition": "clean", "score": 1, "position": 0, "ms": 10}]
    for i, (c, n) in enumerate(zip(clean_scores, noisy_scores, strict=True)):
        t = f"t{i:02d}"
        ratings += [
            {"trial": t, "condition": "clean", "score": c, "position": 0, "ms": 1000},
            {"trial": t, "condition": "noisy", "score": n, "position": 1, "ms": 1500},
            {"trial": t, "condition": "crn", "score": crn, "position": 2, "ms": 2000},
        ]
    return {"version": 1, "rater_id": rater, "started_at": "x", "device_note": "", "ratings": ratings}


def test_mos_screening_and_aggregation():
    good = _resp("a", [5, 5, 4, 5], [1, 2, 1, 2])
    good2 = _resp("b", [4, 5, 5, 5], [2, 1, 1, 1], crn=3)
    bad = _resp("c", [1, 1, 2, 5], [5, 5, 4, 1])  # clean < noisy on 3/4 trials
    table = mos([good, good2, bad])
    assert table["raters"] == 2 and [e["rater_id"] for e in table["excluded"]] == ["c"]
    assert table["conditions"]["crn"]["mean"] == pytest.approx(3.5) and table["conditions"]["crn"]["n"] == 8
    assert table["conditions"]["clean"]["n"] == 8  # practice ratings ignored


def test_dnsmos_polyfit_matches_reference_constants():
    from ans.dnsmos import POLY

    # values from microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (non-personalised)
    assert POLY[False]["ovrl"](3.0) == pytest.approx(-0.06766283 * 9 + 1.11546468 * 3 + 0.04602535)
    assert POLY[True]["bak"].order == 3


def test_merge_upgrades_v01_rows():
    from ans.evaluate import merge_results, upgrade_v01

    v01 = {
        "dataset": "VB-D",
        "rows": [{"model": "ANS-CRN (ours)", "pesq": 2.76}, {"model": "Noisy (unprocessed)", "pesq": 1.97}],
        "per_noise_type": {"bus": {"ANS-CRN (ours)": {"pesq": 3.0, "n": 10}}},
        "per_snr": {},
    }
    up = upgrade_v01(v01)
    assert {(r["model"], r["set"]) for r in up["rows"]} == {
        ("CRN-256 (benchmark)", "vbd"),
        ("Noisy (unprocessed)", "vbd"),
    }
    assert up["per_noise_type"] == {"vbd": {"bus": {"CRN-256 (benchmark)": {"pesq": 3.0, "n": 10}}}}
    new = {"rows": [{"model": "Noisy (unprocessed)", "set": "vbd", "pesq": 2.0}], "per_noise_type": {}, "per_snr": {}}
    merged = merge_results(v01, new)
    assert sorted((r["model"], r["pesq"]) for r in merged["rows"]) == [
        ("CRN-256 (benchmark)", 2.76),
        ("Noisy (unprocessed)", 2.0),
    ]
    assert upgrade_v01(up)["per_noise_type"] == up["per_noise_type"]  # idempotent on v0.2
