"""Speech distortion vs. noise reduction (PLAN.md §4.6) for mask-based models (CRN family, Wiener).

``Y = M ⊙ X = M ⊙ S + M ⊙ N`` with ``N = noisy - clean`` known, so the predicted mask is applied to the
clean and noise STFTs separately:

* speech distortion = SI-SDR(iSTFT(M ⊙ S), s) in dB (higher = less damage to the voice);
* noise reduction   = 10·log10(‖n‖² / ‖iSTFT(M ⊙ N)‖²) in dB (higher = more noise removed).

By linearity ``iSTFT(M⊙S) + iSTFT(M⊙N) = ŷ`` (tested to 1e-4). Lookahead-k models apply the mask of frame
``t`` to frame ``t - k``, exactly as in :meth:`ans.models.crn.CRN.forward`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ans.dsp import HOP
from ans.metrics import si_sdr
from ans.models.classical import ClassicalEnhancer
from ans.models.crn import apply_mask, delay_frames


@torch.no_grad()
def crn_decompose(
    model, noisy: np.ndarray, clean: np.ndarray, atten_lim_db: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ŷ, ŝ = iSTFT(M⊙S), n̂ = iSTFT(M⊙N))``, all aligned with the input and ``len(noisy)`` long."""
    k = int(getattr(model.cfg, "lookahead", 0) or 0)
    n = len(noisy)
    dev = next(model.parameters()).device
    x = torch.from_numpy(np.stack([noisy, clean, noisy - clean]).astype(np.float32)).to(dev)
    x = F.pad(x, (0, k * HOP)) if k else x
    spec = model.stft(x)  # [3, T, F, 2]
    m, _, _ = model.masks(spec[:1], None, atten_lim_db)
    out = apply_mask(delay_frames(spec, k), m.expand(3, -1, -1, -1))
    y = model.stft.inverse(out, length=n + k * HOP)[..., k * HOP :].float().cpu().numpy()
    return y[0], y[1], y[2]


def wiener_decompose(enh: ClassicalEnhancer, noisy: np.ndarray, clean: np.ndarray):
    gains = torch.from_numpy(enh.gains_np(noisy))[..., None]  # [T, F, 1] real mask
    x = torch.from_numpy(np.stack([noisy, clean, noisy - clean]).astype(np.float32))
    y = enh.stft.inverse(enh.stft(x) * gains, length=len(noisy)).numpy()
    return y[0], y[1], y[2]


def decomposition_metrics(clean: np.ndarray, noise: np.ndarray, s_hat: np.ndarray, n_hat: np.ndarray) -> dict:
    e_n = float(np.sum(np.square(noise, dtype=np.float64)))
    e_nh = float(np.sum(np.square(n_hat, dtype=np.float64)))
    nr = 10 * math.log10((e_n + 1e-12) / (e_nh + 1e-12)) if e_n > 1e-10 else math.nan
    return {"speech_distortion_sisdr": si_sdr(clean, s_hat), "noise_reduction_db": nr}
