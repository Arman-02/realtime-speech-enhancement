from __future__ import annotations

import json

import torch

from ans.data import SyntheticDataset, collate
from ans.evaluate import evaluate
from ans.losses import EnhancementLoss
from ans.models import build_model, load_checkpoint
from ans.train import load_config, train, train_step


def test_overfit_single_batch():
    torch.manual_seed(0)
    ds = SyntheticDataset(4, train=False, seed=0)
    batch = collate([ds[i] for i in range(4)])
    model = build_model("ans_crn")
    loss_fn = EnhancementLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = [train_step(model, batch, loss_fn, opt, torch.device("cpu"))[0] for _ in range(50)]
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


def test_train_smoke_resume_and_evaluate(tmp_path):
    cfg = load_config(
        None,
        [
            "device=cpu",
            "batch_size=4",
            "data.num_workers=0",
            "data.synthetic_items=8",
            "data.segment=8000",
            "val.subset=4",
            "val.workers=1",
            "log_every=1",
            "schedule.epochs=2",
            "run_name=smoke",
        ],
    )
    out = train(cfg, synthetic=True, ckpt_root=tmp_path / "ck", log_root=tmp_path / "runs")
    assert out["steps"] == 4
    ck = tmp_path / "ck" / "smoke"
    assert (ck / "best.pt").exists() and (ck / "last.pt").exists()
    cfg["schedule"]["epochs"] = 3
    out2 = train(cfg, synthetic=True, resume=str(ck / "last.pt"), ckpt_root=tmp_path / "ck", log_root=tmp_path / "runs")
    assert out2["steps"] == 6
    assert load_checkpoint(ck / "best.pt").state_shape == [2, 1, 256]

    res = evaluate(
        ["noisy", "classical", f"crn:{ck / 'best.pt'}"],
        synthetic=True,
        limit=3,
        device="cpu",
        workers=1,
        out_json=tmp_path / "results.json",
        decompose=True,
        atten_sweep=[6.0, None],
        dnsmos=False,
    )
    assert [r["model"] for r in res["rows"]] == [
        "Noisy (unprocessed)",
        "Spectral subtraction (Wiener)",
        "CRN-256 (3 ep)",  # registry display name from the checkpoint config
    ]
    assert {r["set"] for r in res["rows"]} == {"synthetic"}
    crn_row = res["rows"][2]
    assert crn_row["family"] == "ours" and crn_row["hidden"] == 256 and crn_row["distilled"] is False
    assert crn_row["speech_distortion_sisdr"] is not None and crn_row["noise_reduction_db"] is not None
    assert res["rows"][0]["speech_distortion_sisdr"] is None  # not mask-based
    assert [e["atten_lim_db"] for e in res["atten_sweep"]] == [6.0, None]
    assert res["pareto"] == [] or all("set" in e for e in res["pareto"])
    meta = json.loads((ck / "meta.json").read_text())
    assert meta["completed"] is True and meta["git"]["source_hash"] and meta["pip_freeze"]["sha256"]
    for r in res["rows"]:
        assert {
            "pesq",
            "stoi",
            "estoi",
            "si_sdr",
            "params",
            "macs_per_frame",
            "ms_per_frame_cpu",
            "rtf",
            "causal",
        } <= set(r)
    assert res["rows"][2]["params"] > 1e6
    assert res["per_noise_type"] and (tmp_path / "per_utterance.csv").exists()
