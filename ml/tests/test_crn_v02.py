"""Contract v0.2 model features: attenuation limit, return_mask, teacher, lookahead (PLAN.md §3.2, §4.8, §4.11)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ans.dsp import N_BINS
from ans.models import build_model, count_params
from ans.models.crn import apply_mask, atten_floor, delay_frames
from ans.stream import StreamingEnhancer, enhance_streaming, stream_model, torch_frame_fn


def _mag(m: torch.Tensor) -> torch.Tensor:
    return m.pow(2).sum(-1).sqrt()


@pytest.mark.parametrize("lim", [0.0, 6.0, 12.0, 24.0])
def test_atten_floor_respected_and_phase_kept(crn, lim):
    raw = torch.randn(2, 2, 7, N_BINS) * torch.logspace(-6, 1, N_BINS)  # many near-zero masks
    floor = atten_floor(lim)
    free, lim_m = crn.mask(raw), crn.mask(raw, floor)
    assert (_mag(lim_m) >= float(floor) - 1e-6).all()
    assert (_mag(lim_m) <= crn.cfg.mask_max + 1e-5).all()
    big = _mag(free) > 1e-6
    ang_free = torch.atan2(free[..., 1], free[..., 0])
    ang_lim = torch.atan2(lim_m[..., 1], lim_m[..., 0])
    assert (ang_free - ang_lim)[big].abs().max() < 1e-4
    untouched = _mag(free) >= float(floor) + 1e-6
    assert torch.allclose(free[untouched], lim_m[untouched], atol=1e-6)


def test_forward_atten_and_return_mask(crn):
    spec = torch.randn(1, 30, N_BINS, 2)
    with torch.no_grad():
        out, st = crn(spec)
        out12, st12, m12 = crn(spec, atten_lim_db=12.0, return_mask=True)
        out100, _ = crn(spec, atten_lim_db=torch.tensor([100.0]))
    assert (_mag(m12) >= 10 ** (-12 / 20) - 1e-6).all()
    torch.testing.assert_close(apply_mask(spec, m12), out12)
    torch.testing.assert_close(st, st12)  # the clamp is after the network: state unaffected
    assert (out100 - out).abs().max() < 1e-4  # lim = 100 matches the unlimited model
    assert (out12 - out).abs().max() > 1e-3


def test_atten_waveform_enhance(crn, rng):
    x = torch.from_numpy((0.1 * rng.standard_normal(4000)).astype(np.float32))[None]
    with torch.no_grad():
        a, b = crn.enhance(x), crn.enhance(x, atten_lim_db=100.0)
        c = crn.enhance(x, atten_lim_db=6.0)
    assert (a - b).abs().max() < 1e-4 and c.shape == a.shape
    # with a 6 dB limit at most 6 dB of energy can be removed (up to OLA/window effects)
    assert c.pow(2).sum() > 0.2 * x.pow(2).sum()


def test_teacher_topology():
    torch.manual_seed(0)
    teacher = build_model("crn_teacher").eval()
    student = build_model("ans_crn")
    assert teacher.cfg.hidden == 512 and teacher.cfg.bidirectional and not teacher.causal
    assert teacher.state_shape == [4, 1, 512] and teacher.feature_dim == 1024
    assert count_params(teacher) > 5 * count_params(student)
    # same encoder/decoder topology: identical shapes outside the recurrent part
    ts, ss = teacher.state_dict(), student.state_dict()
    for k in ss:
        if k.startswith(("encoder", "decoder", "in_norm")):
            assert ts[k].shape == ss[k].shape, k
    spec = torch.randn(1, 20, N_BINS, 2)
    spec2 = spec.clone()
    spec2[:, 15:] = torch.randn(1, 5, N_BINS, 2)
    with torch.no_grad():
        a, _ = teacher(spec)
        b, _ = teacher(spec2)
    assert (a[:, :10] - b[:, :10]).abs().max() > 1e-6  # non-causal
    assert teacher.enhance(torch.randn(2, 3000) * 0.1).shape == (2, 3000)


def test_student_state_dict_keys_unchanged(crn):
    keys = list(crn.state_dict())
    assert keys[:3] == ["in_norm.weight", "in_norm.bias", "encoder.0.conv.weight"]
    assert "gru.weight_ih_l0" in keys and not any("reverse" in k for k in keys)
    assert crn.state_dict()["fc_out.0.weight"].shape == (64 * 11, 256)


# ---------------------------------------------------------------------------------------------
# Lookahead (§4.11)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2])
def test_lookahead_offline_definition_and_causality(k):
    torch.manual_seed(0)
    model = build_model("ans_crn", hidden=64, lookahead=k).eval()
    spec = torch.randn(1, 40, N_BINS, 2)
    with torch.no_grad():
        out, _, m = model(spec, return_mask=True)
        assert torch.equal(out, apply_mask(delay_frames(spec, k), m))  # frame t = M_t ⊙ X_{t-k}
        assert out[:, :k].abs().max() == 0
        spec2 = spec.clone()
        spec2[:, 25:] = torch.randn(1, 15, N_BINS, 2)
        out2, _ = model(spec2)
    # emitting frame t-k needs input frames <= t only
    assert (out[:, :25] - out2[:, :25]).abs().max() < 1e-6


@pytest.mark.parametrize("k", [0, 1, 2])
def test_lookahead_streaming_equals_offline_shifted(k, rng):
    torch.manual_seed(0)
    model = build_model("ans_crn", hidden=64, lookahead=k).eval()
    spec = torch.randn(1, 30, N_BINS, 2)
    fn = torch_frame_fn(model)
    state = np.zeros(model.state_shape, np.float32)
    outs = []
    for t in range(spec.shape[1]):
        o, state = fn(spec[:, t : t + 1].numpy(), state)
        outs.append(o)
    with torch.no_grad():
        offline, _ = model(spec)
    np.testing.assert_allclose(np.concatenate(outs, 1), offline.numpy(), atol=1e-5)

    # waveform level: aligned streaming == aligned offline enhance; raw stream is delayed by (1+k) hops
    x = (0.1 * rng.standard_normal(4800)).astype(np.float32)
    with torch.no_grad():
        ref = model.enhance(torch.from_numpy(x)[None])[0].numpy()
    np.testing.assert_allclose(stream_model(x, model), ref, atol=1e-4)
    raw = StreamingEnhancer(torch_frame_fn(model), model.state_shape).push(x)
    d = 160 * (1 + k)
    np.testing.assert_allclose(raw[d:], ref[: len(raw) - d], atol=1e-4)
    al = enhance_streaming(x, torch_frame_fn(model), model.state_shape, align=True, lookahead=k)
    np.testing.assert_allclose(al, ref, atol=1e-4)


def test_lookahead_enhance_alignment():
    """A lookahead model that passes the input through (mask = 1) must return the input, aligned."""
    torch.manual_seed(0)
    model = build_model("ans_crn", hidden=32, lookahead=2).eval()
    with torch.no_grad():
        for p in model.decoder[-1].parameters():
            p.zero_()
        model.decoder[-1].conv.bias[0] = 20.0  # mask -> mask_max * tanh(20) ≈ 1.2 (real)
    x = torch.randn(1, 3200) * 0.1
    with torch.no_grad():
        y = model.enhance(x)
    torch.testing.assert_close(y, 1.2 * x, atol=1e-4, rtol=1e-4)
