from __future__ import annotations

import csv

import numpy as np
import soundfile as sf
import torch

from ans.data import MixConfig, SyntheticDataset, VBDDataset, collate, read_manifest


def _fake_vbd(tmp_path, n=4, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = []
    for i in range(n):
        length = 16000 + 4000 * i
        clean = (0.1 * np.sin(np.arange(length) * 0.05 * (i + 1))).astype(np.float32)
        noisy = clean + (0.02 * rng.standard_normal(length)).astype(np.float32)
        for kind, sig in (("clean", clean), ("noisy", noisy)):
            (tmp_path / kind).mkdir(exist_ok=True)
            sf.write(tmp_path / kind / f"u{i}.wav", sig, 16000, subtype="FLOAT")
        rows.append(
            {
                "id": f"u{i}",
                "speaker": "p1",
                "clean_path": f"clean/u{i}.wav",
                "noisy_path": f"noisy/u{i}.wav",
                "num_samples": length,
                "noise_type": "cafe" if i % 2 else "bus",
                "snr_db": 2.5 * (i + 1),
            }
        )
    path = tmp_path / "train.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_manifest_and_train_crops(tmp_path):
    items = read_manifest(_fake_vbd(tmp_path), root=tmp_path)
    assert items[1].noise_type == "cafe" and items[0].snr_db == 2.5
    ds = VBDDataset(items, train=True, mix=MixConfig(segment=20000, remix_prob=1.0, gain_prob=1.0))
    for i in range(len(ds)):
        it = ds[i]
        assert it["noisy"].shape == it["clean"].shape
        assert it["noisy"].shape[0] == min(20000, items[i].num_samples)
        assert it["noisy"].abs().max() <= 1.0


def test_no_remix_preserves_pair(tmp_path):
    items = read_manifest(_fake_vbd(tmp_path), root=tmp_path)
    ds = VBDDataset(items, train=True, mix=MixConfig(segment=8000, remix_prob=0.0, gain_prob=0.0))
    it = ds[3]
    resid = (it["noisy"] - it["clean"]).std().item()
    assert 0.015 < resid < 0.025


def test_eval_mode_full_utterances(tmp_path):
    items = read_manifest(_fake_vbd(tmp_path), root=tmp_path)
    ds = VBDDataset(items, train=False)
    assert [ds[i]["noisy"].shape[0] for i in range(len(ds))] == [it.num_samples for it in items]
    batch = collate([ds[0], ds[2]])
    assert batch["noisy"].shape == (2, items[2].num_samples)
    assert batch["mask"][0].sum() == items[0].num_samples
    assert batch["noise_type"] == ["bus", "bus"]


def test_synthetic_deterministic_eval():
    a, b = SyntheticDataset(4, train=False, seed=7), SyntheticDataset(4, train=False, seed=7)
    assert torch.equal(a[2]["noisy"], b[2]["noisy"])
    tr = SyntheticDataset(4, train=True, segment=8000)
    assert tr[0]["noisy"].shape == (8000,)
