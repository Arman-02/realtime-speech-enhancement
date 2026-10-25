"""Classical baseline: causal Wiener filter with minimum-statistics noise tracking (Martin, 2001, simplified)
and decision-directed a-priori SNR (Ephraim & Malah). No training; frame-by-frame like the neural model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ans.dsp import N_BINS, STFT


@dataclass
class WienerConfig:
    psd_smooth: float = 0.8  # recursive periodogram smoothing
    sub_windows: int = 8  # U
    sub_window_frames: int = 12  # V  -> min-search window D = U*V frames (~1 s)
    bias: float = 1.8  # min-statistics bias compensation
    dd_alpha: float = 0.98  # decision-directed smoothing
    gain_floor: float = 0.1  # -20 dB
    over_subtraction: float = 1.0


class MinStatsWiener:
    """Stateful per-frame enhancer operating on ``(161, 2)`` spectra."""

    def __init__(self, cfg: WienerConfig | None = None) -> None:
        self.cfg = cfg or WienerConfig()
        self.reset()

    def reset(self) -> None:
        self.psd: np.ndarray | None = None
        self.cur_min = np.full(N_BINS, np.inf)
        self.minima: deque[np.ndarray] = deque(maxlen=self.cfg.sub_windows)
        self.count = 0
        self.prev_clean_pow = np.zeros(N_BINS)

    def noise_psd(self) -> np.ndarray:
        stored = np.min(np.stack(self.minima), axis=0) if self.minima else np.inf
        return self.cfg.bias * np.minimum(stored, self.cur_min)

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        return self.process_frame_gain(spec)[0]

    def process_frame_gain(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(161, 2)`` spectrum -> (enhanced spectrum, real gain ``(161,)``)."""
        c = self.cfg
        x = spec[..., 0].astype(np.float64) + 1j * spec[..., 1].astype(np.float64)
        p = np.abs(x) ** 2
        self.psd = p if self.psd is None else c.psd_smooth * self.psd + (1 - c.psd_smooth) * p
        self.cur_min = np.minimum(self.cur_min, self.psd)
        self.count += 1
        if self.count % c.sub_window_frames == 0:
            self.minima.append(self.cur_min.copy())
            self.cur_min = self.psd.copy()
        noise = np.maximum(c.over_subtraction * self.noise_psd(), 1e-10)
        gamma = p / noise
        xi = c.dd_alpha * self.prev_clean_pow / noise + (1 - c.dd_alpha) * np.maximum(gamma - 1, 0)
        gain = np.maximum(xi / (1 + xi), c.gain_floor)
        y = gain * x
        self.prev_clean_pow = np.abs(y) ** 2
        return np.stack([y.real, y.imag], axis=-1).astype(np.float32), gain


class ClassicalEnhancer:
    """``enhance(wav[N] | wav[B,N]) -> same shape`` using the §3.1 framing (same as the CRN)."""

    params = 0
    causal = True

    def __init__(self, cfg: WienerConfig | None = None) -> None:
        self.cfg = cfg or WienerConfig()
        self.stft = STFT()

    def enhance_np(self, wav: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(wav, np.float32))
        spec = self.stft(x).numpy()
        f = MinStatsWiener(self.cfg)
        out = np.stack([f.process_frame(s) for s in spec])
        return self.stft.inverse(torch.from_numpy(out), length=len(wav)).numpy()

    def gains_np(self, wav: np.ndarray) -> np.ndarray:
        """Per-frame real Wiener gains ``[T, F]`` (the mask, for the §4.6 decomposition)."""
        spec = self.stft(torch.from_numpy(np.asarray(wav, np.float32))).numpy()
        f = MinStatsWiener(self.cfg)
        return np.stack([f.process_frame_gain(s)[1] for s in spec]).astype(np.float32)

    def enhance(self, wav: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        is_tensor = isinstance(wav, torch.Tensor)
        arr = wav.detach().cpu().numpy() if is_tensor else np.asarray(wav)
        out = self.enhance_np(arr) if arr.ndim == 1 else np.stack([self.enhance_np(a) for a in arr])
        return torch.from_numpy(out).to(wav.device) if is_tensor else out
