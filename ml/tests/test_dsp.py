from __future__ import annotations

import numpy as np
import torch

from ans.dsp import HOP, N_FFT, STFT, analysis_frame, sqrt_hann, stft_np
from ans.stream import enhance_streaming, identity_frame_fn


def test_window_is_periodic_sqrt_hann_and_cola():
    w = sqrt_hann(N_FFT)
    assert w[0] == 0.0 and np.isclose(w[N_FFT // 2], 1.0)
    w2 = w**2
    np.testing.assert_allclose(w2[:HOP] + w2[HOP:], 1.0, atol=1e-12)


def test_round_trip_exact(rng):
    x = torch.from_numpy(rng.standard_normal((3, 16_037)).astype(np.float32))
    stft = STFT()
    y = stft.inverse(stft(x), length=x.shape[-1])
    assert y.shape == x.shape
    assert (y - x).abs().max() < 1e-5


def test_round_trip_center_false_interior(rng):
    x = torch.from_numpy(rng.standard_normal(8000).astype(np.float32))
    stft = STFT()
    y = stft.inverse(stft(x, center=False), center=False)
    assert (y[HOP : len(y) - HOP] - x[HOP : len(y) - HOP]).abs().max() < 1e-5


def test_torch_matches_numpy_reference(rng):
    x = rng.standard_normal(4000).astype(np.float32)
    ref = stft_np(x)
    got = STFT()(torch.from_numpy(x), center=False).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-4)
    np.testing.assert_allclose(analysis_frame(x[:N_FFT])[..., 0], np.fft.rfft(sqrt_hann() * x[:N_FFT]).real, atol=1e-4)


def test_center_framing_equals_streaming_framing(rng):
    x = rng.standard_normal(3200).astype(np.float32)
    padded = np.concatenate([np.zeros(HOP, np.float32), x])
    ref = stft_np(padded)
    got = STFT()(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got[: len(ref)], ref, atol=1e-4)


def test_identity_streaming_reconstructs_with_one_hop_delay(rng):
    x = rng.standard_normal(4800).astype(np.float32)
    y = enhance_streaming(x, identity_frame_fn, [1])
    assert len(y) == len(x)
    np.testing.assert_allclose(y[HOP:], x[:-HOP], atol=1e-5)
    aligned = enhance_streaming(x, identity_frame_fn, [1], align=True)
    np.testing.assert_allclose(aligned, x, atol=1e-5)
