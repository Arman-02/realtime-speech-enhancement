"""Augmentation (PLAN.md §4.1, §4.8) and the corpus-separation rule (§3.5)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from ans.augment import (
    AugmentConfig,
    apply_response,
    check_corpus_separation,
    find_data_manifest,
    mic_response,
    prepare_rir,
    read_audio_manifest,
    reverberate,
)
from ans.data import MixConfig, VBDDataset, read_manifest
from tests.test_data import _fake_vbd


def _synthetic_rir(rng, pre=40, sr=16000, t60=0.4):
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / t60) * 0.3
    rir = np.concatenate([np.zeros(pre), [0.0, 1.0], tail]).astype(np.float32)
    return rir


def _write_list(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_reverb_target_is_early_reflections(rng):
    clean = rng.standard_normal(8000).astype(np.float32)
    rir = _synthetic_rir(rng)
    wet, target = reverberate(clean, rir, early_ms=50, pre_ms=1)
    assert wet.shape == target.shape == clean.shape
    full, early = prepare_rir(rir, 50, 1)
    peak = int(np.argmax(np.abs(rir)))
    start = peak - 16
    assert len(early) == 16 + 800 and early[16] == pytest.approx(1.0)
    np.testing.assert_allclose(early, rir[start : start + 816] / rir[peak], atol=1e-6)
    np.testing.assert_allclose(target, np.convolve(clean, early)[: len(clean)], atol=1e-4)
    np.testing.assert_allclose(wet, np.convolve(clean, full)[: len(clean)], atol=1e-3)
    # target keeps the direct path aligned with the dry signal (up to the 1 ms pre-peak margin)
    xc = np.correlate(target, clean, "full")
    assert int(np.argmax(xc)) - (len(clean) - 1) == 16
    # a pure delta RIR leaves the signal untouched
    w2, t2 = reverberate(clean, np.r_[np.zeros(10), 1.0].astype(np.float32), pre_ms=0)
    np.testing.assert_allclose(w2, clean, atol=1e-5)
    np.testing.assert_allclose(t2, clean, atol=1e-5)


def test_mic_response_zero_phase(rng):
    x = rng.standard_normal(16000).astype(np.float32)
    np.testing.assert_allclose(apply_response(x, lambda n: mic_response(n)), x, atol=1e-5)
    y = apply_response(x, lambda n: mic_response(n, hpf_hz=150, tilt_db=4, lpf_hz=5000))
    xc = np.correlate(y, x, "full")
    assert int(np.argmax(xc)) == len(x) - 1  # no delay
    h = mic_response(1024, hpf_hz=150, tilt_db=0, lpf_hz=5000)
    f = np.fft.rfftfreq(1024, 1 / 16000)
    assert h[0] == pytest.approx(0, abs=1e-6) and h[np.searchsorted(f, 1000)] > 0.95 and h[-1] < 0.05


@pytest.fixture
def aug_setup(tmp_path, rng):
    vbd = _fake_vbd(tmp_path, n=4, rng=rng)
    (tmp_path / "noise").mkdir()
    (tmp_path / "rir").mkdir()
    noise_rows, rir_rows = [], []
    for i in range(3):
        sf.write(tmp_path / "noise" / f"n{i}.wav", rng.standard_normal(30000).astype(np.float32) * 0.1, 16000)
        noise_rows.append({"id": f"musan_n{i}", "path": str(tmp_path / "noise" / f"n{i}.wav"), "corpus": "musan"})
        sf.write(tmp_path / "rir" / f"r{i}.wav", _synthetic_rir(rng), 16000)
        rir_rows.append({"id": f"sim_r{i}", "path": str(tmp_path / "rir" / f"r{i}.wav"), "source": "sim"})
    noise_csv = _write_list(tmp_path / "noise_train.csv", noise_rows)
    rir_csv = _write_list(tmp_path / "rir_train.csv", rir_rows)
    return read_manifest(vbd, root=tmp_path), str(noise_csv), str(rir_csv)


def test_augmented_items_aligned(aug_setup):
    items, noise_csv, rir_csv = aug_setup
    cfg = AugmentConfig.from_dict(
        {
            "extra_noise": {"prob": 1.0, "manifest": noise_csv},
            "reverb": {"prob": 1.0, "manifest": rir_csv},
            "mic": {"prob": 1.0, "clip_prob": 1.0, "lpf_prob": 1.0},
            "level": {"prob": 1.0},
        }
    )
    ds = VBDDataset(items, train=True, mix=MixConfig(segment=12000), augment=cfg, seed=0)
    assert len(ds.augmentor.noise) == 3 and len(ds.augmentor.rirs) == 3
    for i in range(len(ds)):
        it = ds[i]
        assert it["noisy"].shape == it["clean"].shape == (min(12000, items[i].num_samples),)
        assert torch.isfinite(it["noisy"]).all() and it["noisy"].abs().max() <= 1.0
        rms = 20 * np.log10(it["noisy"].pow(2).mean().sqrt().item() + 1e-12)
        assert -46.0 <= rms <= -2.9


def test_reverb_only_target_matches_early_rir(aug_setup, tmp_path):
    items, _, rir_csv = aug_setup
    one = read_audio_manifest(rir_csv)[0]
    single = _write_list(tmp_path / "one_rir.csv", [{"id": one.id, "path": str(one.path), "corpus": "sim"}])
    cfg = AugmentConfig.from_dict({"reverb": {"prob": 1.0, "manifest": str(single)}})
    mix = MixConfig(segment=40000, remix_prob=0.0, gain_prob=0.0)  # full utterances, no remix/gain
    ds = VBDDataset(items, train=True, mix=mix, augment=cfg, seed=0)
    rir, _ = sf.read(one.path, dtype="float32")
    for i in range(len(items)):
        it = ds[i]
        clean, _ = sf.read(items[i].clean_path, dtype="float32")
        noisy, _ = sf.read(items[i].noisy_path, dtype="float32")
        wet, target = reverberate(clean, rir)
        np.testing.assert_allclose(it["clean"].numpy(), target, atol=1e-5)  # target = direct + 50 ms early
        np.testing.assert_allclose(it["noisy"].numpy(), wet + (noisy - clean), atol=1e-5)


def test_all_off_is_default_path(aug_setup):
    items, *_ = aug_setup
    mix = MixConfig(segment=8000)
    a = VBDDataset(items, train=True, mix=mix, seed=3)
    b = VBDDataset(items, train=True, mix=mix, seed=3, augment={"mic": {"prob": 0.0}, "level": {"prob": 0}})
    assert b.augmentor is None
    for i in range(len(items)):
        assert torch.equal(a[i]["noisy"], b[i]["noisy"]) and torch.equal(a[i]["clean"], b[i]["clean"])


def test_corpus_separation_logic(tmp_path):
    tr = _write_list(tmp_path / "a.csv", [{"id": "x1", "path": "a/x1.wav", "corpus": "musan"}])
    te = _write_list(tmp_path / "b.csv", [{"id": "y1", "path": "b/y1.wav", "corpus": "esc50"}])
    assert check_corpus_separation(read_audio_manifest(tr), read_audio_manifest(te)) == []
    bad = _write_list(tmp_path / "c.csv", [{"id": "x1", "path": "a/x1.wav", "corpus": "MUSAN"}])
    assert len(check_corpus_separation(read_audio_manifest(tr), read_audio_manifest(bad))) == 3


@pytest.mark.parametrize(("train_name", "test_name"), [("noise_train", "noise_ood"), ("rir_train", "rir_test")])
def test_corpus_separation_real_manifests(train_name, test_name):
    tr, te = find_data_manifest(train_name), find_data_manifest(test_name)
    if tr is None or te is None:
        pytest.skip(f"{train_name}/{test_name} manifests not built yet")
    problems = check_corpus_separation(read_audio_manifest(tr), read_audio_manifest(te))
    if train_name == "rir_train":  # held-out RIRs may come from the same simulator; ids/files must differ
        problems = [p for p in problems if not p.startswith("shared source corpora")]
    assert problems == []
