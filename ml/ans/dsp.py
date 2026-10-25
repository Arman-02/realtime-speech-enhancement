"""Reference STFT / iSTFT (PLAN.md §3.1).

Spectra are real tensors of shape ``(..., T, F, 2)`` with the last dim = (real, imag).
The DFT is implemented as a matmul against precomputed cos/sin bases, so the hot path
never touches complex dtypes (MPS- and ONNX-friendly).

Framing conventions
-------------------
* ``frame t = window * x[t*hop : t*hop + n_fft]`` (``center=False``).
* The *streaming* framing prepends ``hop`` zeros (the zero-initialised input history), which is
  identical to ``center=True`` with constant zero padding. :func:`stft` with ``center=True`` and
  :func:`istft` with ``length=N`` therefore reproduce the streaming engine exactly, aligned so that
  output sample ``n`` corresponds to input sample ``n``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SAMPLE_RATE = 16_000
N_FFT = 320
HOP = 160
N_BINS = N_FFT // 2 + 1


@dataclass(frozen=True)
class STFTConfig:
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop: int = HOP

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1


def sqrt_hann(n: int = N_FFT, dtype: np.dtype = np.float64) -> np.ndarray:
    """Periodic sqrt-Hann window: ``sqrt(0.5 - 0.5 cos(2 pi i / n))``."""
    i = np.arange(n, dtype=np.float64)
    return np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)).astype(dtype)


def hann(n: int, dtype: np.dtype = np.float64) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)).astype(dtype)


def _dft_bases(n_fft: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward (n_fft, F) and inverse (F, n_fft) real DFT bases in float64."""
    n_bins = n_fft // 2 + 1
    n = np.arange(n_fft, dtype=np.float64)[:, None]
    k = np.arange(n_bins, dtype=np.float64)[None, :]
    ang = 2.0 * np.pi * n * k / n_fft
    fwd_re, fwd_im = np.cos(ang), -np.sin(ang)
    # irfft: x[n] = 1/N * sum_k w_k (Re X_k cos - Im X_k sin), w = 1 for DC/Nyquist, else 2.
    w = np.full(n_bins, 2.0)
    w[0] = 1.0
    if n_fft % 2 == 0:
        w[-1] = 1.0
    inv_re = (w[:, None] * np.cos(ang.T)) / n_fft
    inv_im = (-w[:, None] * np.sin(ang.T)) / n_fft
    if n_fft % 2 == 0:
        inv_im[-1] = 0.0
    inv_im[0] = 0.0
    return fwd_re, fwd_im, inv_re, inv_im


def overlap_add(frames: Tensor, hop: int) -> Tensor:
    """Overlap-add ``(..., T, n_fft)`` frames with ``n_fft % hop == 0`` -> ``(..., (T-1)*hop + n_fft)``."""
    *lead, t, n_fft = frames.shape
    if n_fft % hop:
        raise ValueError("overlap_add requires n_fft to be a multiple of hop")
    r = n_fft // hop
    chunks = frames.reshape(*lead, t, r, hop)
    out = frames.new_zeros(*lead, t + r - 1, hop)
    for i in range(r):
        out = out + F.pad(chunks[..., i, :], (0, 0, i, r - 1 - i))
    return out.reshape(*lead, (t + r - 1) * hop)


class STFT(nn.Module):
    """Batch STFT/iSTFT with real-valued ``(..., T, F, 2)`` spectra.

    ``window="sqrt_hann"`` is the §3.1 analysis/synthesis window; ``"hann"`` is used for the
    multi-resolution loss (analysis only).
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP, window: str = "sqrt_hann") -> None:
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        win = sqrt_hann(n_fft) if window == "sqrt_hann" else hann(n_fft)
        fwd_re, fwd_im, inv_re, inv_im = _dft_bases(n_fft)
        self.register_buffer("window", torch.from_numpy(win).float(), persistent=False)
        # Window folded into the bases: analysis (n_fft, 2F), synthesis (2F, n_fft).
        fwd = np.concatenate([fwd_re, fwd_im], axis=1) * win[:, None]
        inv = np.concatenate([inv_re, inv_im], axis=0) * win[None, :]
        self.register_buffer("fwd", torch.from_numpy(fwd).float(), persistent=False)
        self.register_buffer("inv", torch.from_numpy(inv).float(), persistent=False)

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    def frames_to_spec(self, frames: Tensor) -> Tensor:
        """Un-windowed frames ``(..., T, n_fft)`` -> spectrum ``(..., T, F, 2)``."""
        y = frames @ self.fwd
        return torch.stack(y.split(self.n_bins, dim=-1), dim=-1)

    def spec_to_frames(self, spec: Tensor) -> Tensor:
        """Spectrum ``(..., T, F, 2)`` -> windowed time frames ``(..., T, n_fft)`` (ready for OLA)."""
        flat = torch.cat([spec[..., 0], spec[..., 1]], dim=-1)
        return flat @ self.inv

    def forward(self, x: Tensor, center: bool = True) -> Tensor:
        """Waveform ``(..., N)`` -> ``(..., T, F, 2)``.

        ``center=True`` pads ``hop`` zeros in front (the streaming history) and enough zeros at the
        end so that every input sample is covered by two windows; ``T = ceil(N / hop) + 1``.
        """
        if center:
            n = x.shape[-1]
            t = math.ceil(n / self.hop) + 1
            total = (t - 1) * self.hop + self.n_fft
            x = F.pad(x, (self.hop, total - n - self.hop))
        frames = x.unfold(-1, self.n_fft, self.hop)
        return self.frames_to_spec(frames)

    def inverse(self, spec: Tensor, length: int | None = None, center: bool = True) -> Tensor:
        """Spectrum ``(..., T, F, 2)`` -> waveform. With ``center=True`` the leading ``hop`` samples
        are dropped so output aligns with the input of :meth:`forward`."""
        y = overlap_add(self.spec_to_frames(spec), self.hop)
        if center:
            y = y[..., self.hop :]
        if length is not None:
            y = y[..., :length]
        return y


def stft(x: Tensor, center: bool = True) -> Tensor:
    return _default_stft(x.device)(x, center=center)


def istft(spec: Tensor, length: int | None = None, center: bool = True) -> Tensor:
    return _default_stft(spec.device).inverse(spec, length=length, center=center)


_CACHE: dict[str, STFT] = {}


def _default_stft(device: torch.device) -> STFT:
    key = str(device)
    if key not in _CACHE:
        _CACHE[key] = STFT().to(device)
    return _CACHE[key]


# --------------------------------------------------------------------------------------------
# NumPy frame-level reference (what the JS engine does, in float64 -> float32).
# --------------------------------------------------------------------------------------------

WINDOW_F64 = sqrt_hann(N_FFT)


def analysis_frame(frame: np.ndarray) -> np.ndarray:
    """Raw 320-sample frame -> ``(161, 2)`` float32 spectrum (§3.1)."""
    spec = np.fft.rfft(WINDOW_F64 * frame.astype(np.float64), n=N_FFT)
    return np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)


def synthesis_frame(spec: np.ndarray) -> np.ndarray:
    """``(161, 2)`` spectrum -> windowed 320-sample frame (to be overlap-added)."""
    z = spec[..., 0].astype(np.float64) + 1j * spec[..., 1].astype(np.float64)
    return (WINDOW_F64 * np.fft.irfft(z, n=N_FFT)).astype(np.float32)


def stft_np(x: np.ndarray) -> np.ndarray:
    """``center=False`` framing of a 1-D signal -> ``(T, 161, 2)``."""
    t = 1 + (len(x) - N_FFT) // HOP
    return np.stack([analysis_frame(x[i * HOP : i * HOP + N_FFT]) for i in range(t)])
