"""Causal Conv-TasNet (Luo & Mesgarani, 2019) with cumulative layer norm — small config (PLAN.md §4.3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ConvTasNetConfig:
    N: int = 256  # encoder basis
    L: int = 16  # window length (samples)
    B: int = 128  # bottleneck channels
    H: int = 256  # conv block channels
    P: int = 3  # depthwise kernel
    X: int = 6  # blocks per repeat
    R: int = 2  # repeats


class CumulativeLayerNorm(nn.Module):
    """cLN: normalise ``[B, C, T]`` with statistics over channels and all frames up to ``t``."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        c, t = x.shape[1], x.shape[2]
        cum_sum = x.sum(1, keepdim=True).cumsum(-1)
        cum_pow = x.pow(2).sum(1, keepdim=True).cumsum(-1)
        count = c * torch.arange(1, t + 1, device=x.device, dtype=x.dtype).view(1, 1, t)
        mean = cum_sum / count
        var = (cum_pow / count - mean.pow(2)).clamp_min(0.0)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class TemporalBlock(nn.Module):
    def __init__(self, b: int, h: int, p: int, dilation: int) -> None:
        super().__init__()
        self.pad = (p - 1) * dilation
        self.in_conv = nn.Conv1d(b, h, 1)
        self.act1, self.norm1 = nn.PReLU(), CumulativeLayerNorm(h)
        self.dconv = nn.Conv1d(h, h, p, dilation=dilation, groups=h)
        self.act2, self.norm2 = nn.PReLU(), CumulativeLayerNorm(h)
        self.res = nn.Conv1d(h, b, 1)
        self.skip = nn.Conv1d(h, b, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y = self.norm1(self.act1(self.in_conv(x)))
        y = self.norm2(self.act2(self.dconv(F.pad(y, (self.pad, 0)))))
        return x + self.res(y), self.skip(y)


class ConvTasNet(nn.Module):
    """Single-output (speech) causal Conv-TasNet; ``forward(wav[B,N]) -> wav[B,N]``."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        cfg = ConvTasNetConfig(**kwargs)
        self.cfg = cfg
        self.stride = cfg.L // 2
        self.encoder = nn.Conv1d(1, cfg.N, cfg.L, stride=self.stride, bias=False)
        self.in_norm = CumulativeLayerNorm(cfg.N)
        self.bottleneck = nn.Conv1d(cfg.N, cfg.B, 1)
        self.blocks = nn.ModuleList(
            TemporalBlock(cfg.B, cfg.H, cfg.P, 2**x) for _ in range(cfg.R) for x in range(cfg.X)
        )
        self.mask_head = nn.Sequential(nn.PReLU(), nn.Conv1d(cfg.B, cfg.N, 1))
        self.decoder = nn.ConvTranspose1d(cfg.N, 1, cfg.L, stride=self.stride, bias=False)

    def forward(self, wav: Tensor) -> Tensor:
        n = wav.shape[-1]
        pad = (-(n - self.cfg.L)) % self.stride if n >= self.cfg.L else self.cfg.L - n
        x = F.pad(wav.unsqueeze(1), (0, pad + self.stride))
        w = F.relu(self.encoder(x))
        y = self.bottleneck(self.in_norm(w))
        skip = torch.zeros_like(y)
        for block in self.blocks:
            y, s = block(y)
            skip = skip + s
        mask = torch.sigmoid(self.mask_head(skip))
        return self.decoder(w * mask).squeeze(1)[..., :n]

    def enhance(self, wav: Tensor) -> Tensor:
        return self(wav)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
