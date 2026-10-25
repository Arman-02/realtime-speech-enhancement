"""Listening-test selection and failure explanations (PLAN.md §3.6, §5.3)."""

from __future__ import annotations

import json

import pytest

from ans.data import manifest_path
from ans.render import SAMPLES_JSON, explain, pick_trials


def test_explanations_follow_measurements():
    pump = explain(
        "silence",
        {
            "pause_s": 5,
            "pause_attenuation_db": 8,
            "pause_noise_db_enhanced": -50,
            "pause_fluctuation_db": 9.0,
            "pause_fluctuation_db_noisy": 4.0,
        },
        "",
    )
    calm = explain(
        "silence",
        {
            "pause_s": 5,
            "pause_attenuation_db": 8,
            "pause_noise_db_enhanced": -50,
            "pause_fluctuation_db": 4.0,
            "pause_fluctuation_db_noisy": 4.0,
        },
        "",
    )
    assert "pumping" in pump and "no audible pumping" in calm
    assert "largely suppressed" in explain("music", {"music_energy_kept_db": -18, "gain_fluctuation_db": 9}, "")
    assert "only partly" in explain("music", {"music_energy_kept_db": -5, "gain_fluctuation_db": 9}, "")
    talk = explain(
        "talker",
        {
            "noise_reduction_db": 2,
            "speech_distortion_sisdr": 4,
            "si_sdr_noisy": 0,
            "si_sdr_enhanced": -3,
            "pesq_enhanced": 1.1,
        },
        "",
    )
    assert "target is damaged" in talk
    assert explain("other", {}, "fallback note") == "fallback note"


@pytest.mark.skipif(not manifest_path("test").exists(), reason="VB-D test set not prepared")
def test_listening_trials_stratified_and_exclude_demo():
    trials, practice = pick_trials(20, seed=0)
    assert len(trials) == 20 and len({t.id for t in trials}) == 20
    cells = {(t.noise_type, t.snr_db) for t in trials}
    assert len(cells) == 20  # one per (noise type, SNR) cell of the 5 x 4 design
    demo = {s["id"] for s in json.loads(SAMPLES_JSON.read_text())} if SAMPLES_JSON.exists() else set()
    assert not demo & {t.id for t in trials} and practice.id not in {t.id for t in trials}
    assert [t.id for t in pick_trials(20, seed=0)[0]] == [t.id for t in trials]  # deterministic
