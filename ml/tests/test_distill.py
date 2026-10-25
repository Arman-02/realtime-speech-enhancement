"""Knowledge distillation (PLAN.md §4.4, §4.8, §4.9)."""

from __future__ import annotations

import copy
import json

import torch

from ans.data import SyntheticDataset, collate
from ans.distill import DistillConfig, Distiller
from ans.losses import EnhancementLoss
from ans.models import build_model, load_checkpoint
from ans.train import load_config, train


def _batch():
    ds = SyntheticDataset(2, train=False, seed=0)
    return collate([ds[0], ds[1]])


def test_kd_loss_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    student = build_model("ans_crn", hidden=64)
    teacher = copy.deepcopy(student)
    b = _batch()
    loss_fn = EnhancementLoss()
    d = Distiller(teacher, student, DistillConfig(teacher_ckpt="x", beta=1.0, gamma=1.0))
    with torch.no_grad():
        d.proj.weight.copy_(torch.eye(64))
        d.proj.bias.zero_()
    student.eval()  # identical numerics (no dropout anyway)
    loss, terms = d(student, b["noisy"], b["clean"], b["mask"], loss_fn)
    target, _ = loss_fn(student.enhance(b["noisy"]), b["clean"], b["mask"])
    assert abs(terms["kd_spec"]) < 1e-8 and abs(terms["kd_feat"]) < 1e-8
    torch.testing.assert_close(loss, target, atol=1e-7, rtol=1e-6)


def test_kd_loss_positive_and_teacher_frozen():
    torch.manual_seed(0)
    student = build_model("ans_crn", hidden=64)
    teacher = build_model("crn_teacher", hidden=32)
    d = Distiller(teacher, student, DistillConfig(teacher_ckpt="x", beta=0.5, gamma=0.1))
    assert d.proj.in_features == 64 and d.proj.out_features == 64  # bi-GRU 32 -> 64 features
    b = _batch()
    loss, terms = d(student, b["noisy"], b["clean"], b["mask"], EnhancementLoss())
    loss.backward()
    assert terms["kd_spec"] > 0 and terms["kd_feat"] > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert d.proj.weight.grad is not None and student.fc_in[0].weight.grad is not None
    assert "teacher fp32" in d.calibrate(b["noisy"])  # CPU -> fp32


def test_teacher_then_kd_student_smoke(tmp_path):
    common = [
        "device=cpu",
        "batch_size=4",
        "data.num_workers=0",
        "data.synthetic_items=8",
        "data.segment=8000",
        "val.subset=4",
        "val.workers=1",
        "schedule.epochs=1",
    ]
    tcfg = load_config(None, [*common, "run_name=teacher", "model.name=crn_teacher", "model.kwargs.hidden=32"])
    train(tcfg, synthetic=True, ckpt_root=tmp_path / "ck", log_root=tmp_path / "runs")
    tck = tmp_path / "ck" / "teacher" / "best.pt"
    scfg = load_config(
        None,
        [*common, "run_name=kd", "model.kwargs.hidden=32", f"distill.teacher_ckpt={tck}", "distill.gamma=0.1"],
    )
    out = train(scfg, synthetic=True, ckpt_root=tmp_path / "ck", log_root=tmp_path / "runs")
    assert out["steps"] == 2
    ck = torch.load(tmp_path / "ck" / "kd" / "last.pt", weights_only=False)
    assert ck["distilled"] is True and ck["distill_proj"] is not None and ck["lookahead"] == 0
    assert load_checkpoint(tmp_path / "ck" / "kd" / "best.pt").cfg.hidden == 32
    meta = json.loads((tmp_path / "ck" / "kd" / "meta.json").read_text())
    assert meta["completed"] is True and meta["config"]["distill"]["beta"] == 1.0
