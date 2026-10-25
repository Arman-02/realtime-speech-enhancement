"""Training losses (PLAN.md §4.4). All operate on waveforms ``[B, N]`` with an optional validity mask."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ans.dsp import STFT

EPS = 1e-8


def compressed_spectral_loss(est: Tensor, ref: Tensor, c: float = 0.3, alpha: float = 0.3) -> Tensor:
    """``alpha * MSE(|Y|^c e^{j phi_Y}, |S|^c e^{j phi_S}) + (1 - alpha) * MSE(|Y|^c, |S|^c)`` on ``[...,F,2]``."""

    def compress(x: Tensor) -> tuple[Tensor, Tensor]:
        mag2 = x[..., 0] ** 2 + x[..., 1] ** 2 + EPS
        mag_c = mag2 ** (c / 2)
        return mag_c, x * (mag_c / mag2.sqrt()).unsqueeze(-1)

    ym, yc = compress(est)
    sm, sc = compress(ref)
    complex_term = (yc - sc).pow(2).sum(-1).mean()
    mag_term = (ym - sm).pow(2).mean()
    return alpha * complex_term + (1 - alpha) * mag_term


def si_sdr(est: Tensor, ref: Tensor, eps: float = EPS) -> Tensor:
    """Per-item SI-SDR in dB for ``[B, N]`` (zero-mean)."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


class MultiResSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 averaged over several resolutions (Hann windows)."""

    def __init__(self, ffts: tuple[int, ...] = (256, 512, 1024)) -> None:
        super().__init__()
        self.stfts = nn.ModuleList(STFT(n, n // 4, window="hann") for n in ffts)

    def forward(self, est: Tensor, ref: Tensor) -> Tensor:
        total = est.new_zeros(())
        for stft in self.stfts:
            ye, yr = stft(est), stft(ref)
            me = (ye.pow(2).sum(-1) + EPS).sqrt()
            mr = (yr.pow(2).sum(-1) + EPS).sqrt()
            sc = torch.linalg.vector_norm(mr - me) / (torch.linalg.vector_norm(mr) + EPS)
            mag = (mr.log() - me.log()).abs().mean()
            total = total + sc + mag
        return total / len(self.stfts)


@dataclass
class LossConfig:
    spec_weight: float = 1.0
    compress: float = 0.3
    alpha: float = 0.3
    mrstft_weight: float = 0.0
    mrstft_ffts: list[int] = field(default_factory=lambda: [256, 512, 1024])
    sisdr_weight: float = 0.0


class EnhancementLoss(nn.Module):
    """Weighted sum of the configured terms. ``forward(est, ref, mask) -> (loss, {name: value})``."""

    def __init__(self, cfg: LossConfig | None = None, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig(**kwargs)
        self.stft = STFT()
        self.mrstft = MultiResSTFTLoss(tuple(self.cfg.mrstft_ffts)) if self.cfg.mrstft_weight > 0 else None

    def forward(self, est: Tensor, ref: Tensor, mask: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
        if mask is not None:
            est, ref = est * mask, ref * mask
        cfg = self.cfg
        terms: dict[str, Tensor] = {}
        if cfg.spec_weight > 0:
            terms["spec"] = cfg.spec_weight * compressed_spectral_loss(
                self.stft(est), self.stft(ref), cfg.compress, cfg.alpha
            )
        if self.mrstft is not None:
            terms["mrstft"] = cfg.mrstft_weight * self.mrstft(est, ref)
        if cfg.sisdr_weight > 0:
            terms["sisdr"] = -cfg.sisdr_weight * si_sdr(est, ref).mean()
        total = sum(terms.values(), est.new_zeros(()))
        return total, {k: float(v.detach()) for k, v in terms.items()}
