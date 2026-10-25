from __future__ import annotations

import numpy as np
import pytest
import torch

from ans.benchmark import crn_macs_per_frame, macs_per_frame
from ans.data import SyntheticDataset
from ans.dsp import N_BINS, STFT
from ans.metrics import si_sdr
from ans.models import ClassicalEnhancer, build_model, count_params
from ans.stream import StreamingEnhancer, enhance_streaming, torch_frame_fn


def test_crn_size_and_macs(crn):
    assert 1.0e6 <= count_params(crn) <= 1.5e6
    assert 0.5e6 <= crn_macs_per_frame(crn) <= 2.5e6
    assert crn.state_shape == [2, 1, 256]


@pytest.mark.parametrize("t", [1, 7, 50])
def test_crn_shapes(crn, t):
    spec = torch.randn(2, t, N_BINS, 2)
    out, state = crn(spec)
    assert out.shape == spec.shape and state.shape == (2, 2, 256)


def test_crn_streaming_equals_offline(crn):
    torch.manual_seed(1)
    spec = torch.randn(1, 60, N_BINS, 2) * 2
    with torch.no_grad():
        full, st_full = crn(spec)
        state, outs = None, []
        for t in range(spec.shape[1]):
            o, state = crn(spec[:, t : t + 1], state)
            outs.append(o)
    assert (torch.cat(outs, 1) - full).abs().max() < 1e-4
    assert (state - st_full).abs().max() < 1e-4


def test_stream_loop_matches_offline_enhance(crn, rng):
    x = (0.1 * rng.standard_normal(5000)).astype(np.float32)
    with torch.no_grad():
        offline = crn.enhance(torch.from_numpy(x)[None])[0].numpy()
    streamed = enhance_streaming(x, torch_frame_fn(crn), crn.state_shape, align=True)
    np.testing.assert_allclose(streamed, offline, atol=1e-4)
    live = StreamingEnhancer(torch_frame_fn(crn), crn.state_shape)
    chunks = [live.push(c) for c in np.array_split(x, 37)]
    out = np.concatenate(chunks)
    assert len(out) == (len(x) // 160) * 160
    np.testing.assert_allclose(out[160:], offline[: len(out) - 160], atol=1e-4)


def test_crn_is_causal(crn):
    spec = torch.randn(1, 40, N_BINS, 2)
    spec2 = spec.clone()
    spec2[:, 25:] = torch.randn(1, 15, N_BINS, 2)
    with torch.no_grad():
        a, _ = crn(spec)
        b, _ = crn(spec2)
    assert (a[:, :25] - b[:, :25]).abs().max() < 1e-6


def test_crn_mask_is_bounded(crn):
    m = crn.mask(torch.randn(2, 2, 5, N_BINS) * 100)
    assert (m.pow(2).sum(-1).sqrt() <= crn.cfg.mask_max + 1e-5).all()


def test_conv_tasnet_shapes_and_causality():
    torch.manual_seed(0)
    model = build_model("conv_tasnet").eval()
    assert 0.5e6 < count_params(model) < 3e6
    x = torch.randn(2, 8003) * 0.1
    with torch.no_grad():
        y = model(x)
        x2 = x.clone()
        x2[:, 5000:] = torch.randn(2, 3003)
        y2 = model(x2)
    assert y.shape == x.shape
    # causal up to the 16-sample encoder window
    assert (y[:, :4980] - y2[:, :4980]).abs().max() < 1e-5
    assert macs_per_frame(model) > 0


def test_classical_improves_si_sdr_on_stationary_noise():
    item = SyntheticDataset(4, train=False, seed=3)[0]
    rng = np.random.default_rng(0)
    clean = item["clean"].numpy()
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2))  # 0 dB white noise
    noisy = clean + noise
    enh = ClassicalEnhancer().enhance_np(noisy)
    assert enh.shape == noisy.shape and np.isfinite(enh).all()
    assert si_sdr(clean, enh) > si_sdr(clean, noisy) + 1.0


def test_stft_module_moves_with_model(crn):
    assert isinstance(crn.stft, STFT)
    assert not any(k.startswith("stft") for k in crn.state_dict())
