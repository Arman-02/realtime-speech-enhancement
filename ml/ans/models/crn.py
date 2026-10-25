"""``ans_crn``: causal convolutional-recurrent network with a bounded complex ratio mask (PLAN.md §4.2),
and ``crn_teacher``: the same topology with a bidirectional GRU (non-causal, §4.9).

Convolutions run along frequency only, so all temporal context lives in the GRU and the streaming
state is just the GRU hidden state ``[layers, B, hidden]``.

Extras (all off by default, so the default path is unchanged):

* ``atten_lim_db`` (§3.2): the mask magnitude is floored at ``10^(-lim/20)``, keeping its phase.
* ``lookahead`` k (§4.11): the mask computed at frame ``t`` (from frames ``<= t``) is applied to input
  frame ``t - k``. The GRU stays causal; the waveform output is aligned again by :meth:`CRN.enhance`
  and :func:`ans.stream.enhance_streaming`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ans.dsp import HOP, N_BINS, STFT


@dataclass
class CRNConfig:
    n_bins: int = N_BINS
    channels: list[int] = field(default_factory=lambda: [16, 32, 48, 64])
    kernel: int = 3
    hidden: int = 256
    gru_layers: int = 2
    mask_max: float = 1.2
    compress: float = 0.3
    bidirectional: bool = False
    lookahead: int = 0


def _freq_out(n: int, levels: int) -> list[int]:
    sizes = [n]
    for _ in range(levels):
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


def atten_floor(atten_lim_db: float | Tensor) -> Tensor:
    """Linear mask-magnitude floor ``10^(-lim/20)`` (ONNX-friendly: Mul + Exp)."""
    lim = atten_lim_db if isinstance(atten_lim_db, Tensor) else torch.tensor([float(atten_lim_db)])
    return torch.exp(lim * (-math.log(10.0) / 20.0))


def apply_mask(spec: Tensor, m: Tensor) -> Tensor:
    """Complex multiply ``M ⊙ X`` on ``[..., F, 2]`` tensors."""
    xr, xi = spec[..., 0], spec[..., 1]
    return torch.stack([m[..., 0] * xr - m[..., 1] * xi, m[..., 0] * xi + m[..., 1] * xr], dim=-1)


def delay_frames(spec: Tensor, k: int) -> Tensor:
    """``[B, T, F, 2]`` delayed by ``k`` frames (zeros in front, same length). ``k = 0`` is the identity."""
    if k == 0:
        return spec
    return F.pad(spec, (0, 0, 0, 0, k, 0))[:, : spec.shape[1]]


class FrameNorm(nn.Module):
    """LayerNorm over (C, F) independently for every frame of a ``[B, C, T, F]`` tensor."""

    def __init__(self, channels: int, freqs: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm([channels, freqs])

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, fout: int, k: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (1, k), stride=(1, 2), padding=(0, k // 2))
        self.norm = FrameNorm(cout, fout)
        self.act = nn.PReLU(cout)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, fout: int, k: int, last: bool = False) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(2 * cin, cout, (1, k), stride=(1, 2), padding=(0, k // 2))
        self.post = nn.Identity() if last else nn.Sequential(FrameNorm(cout, fout), nn.PReLU(cout))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.post(self.conv(torch.cat([x, skip], dim=1)))


class CRN(nn.Module):
    """``forward(spec[B,T,F,2], state | None, atten_lim_db=None, return_mask=False)
    -> (spec_out[B,T,F,2], state[L*D,B,H] [, mask[B,T,F,2]])``."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        cfg = CRNConfig(**kwargs)
        if cfg.lookahead < 0:
            raise ValueError("lookahead must be >= 0")
        self.cfg = cfg
        ch, k = cfg.channels, cfg.kernel
        freqs = _freq_out(cfg.n_bins, len(ch))  # 161, 81, 41, 21, 11
        self.freqs = freqs
        self.in_norm = nn.LayerNorm(cfg.n_bins)
        cins = [3, *ch[:-1]]
        self.encoder = nn.ModuleList(
            EncoderBlock(ci, co, fo, k) for ci, co, fo in zip(cins, ch, freqs[1:], strict=True)
        )
        flat = ch[-1] * freqs[-1]
        dirs = 2 if cfg.bidirectional else 1
        self.fc_in = nn.Sequential(nn.Linear(flat, cfg.hidden), nn.PReLU())
        self.gru = nn.GRU(
            cfg.hidden, cfg.hidden, num_layers=cfg.gru_layers, batch_first=True, bidirectional=cfg.bidirectional
        )
        self.fc_out = nn.Sequential(nn.Linear(dirs * cfg.hidden, flat), nn.PReLU())
        couts = [*cins[1:][::-1], 2]  # 48, 32, 16, 2
        self.decoder = nn.ModuleList(
            DecoderBlock(ci, co, fo, k, last=i == len(ch) - 1)
            for i, (ci, co, fo) in enumerate(zip(ch[::-1], couts, freqs[-2::-1], strict=True))
        )
        self.stft = STFT()

    # -- metadata ---------------------------------------------------------------------------------
    @property
    def causal(self) -> bool:
        return not self.cfg.bidirectional

    @property
    def lookahead(self) -> int:
        return self.cfg.lookahead

    @property
    def feature_dim(self) -> int:
        """Width of the GRU output used for feature distillation."""
        return self.cfg.hidden * (2 if self.cfg.bidirectional else 1)

    @property
    def state_shape(self) -> list[int]:
        return [self.cfg.gru_layers * (2 if self.cfg.bidirectional else 1), 1, self.cfg.hidden]

    def init_state(self, batch: int = 1, device: torch.device | str = "cpu") -> Tensor:
        layers = self.cfg.gru_layers * (2 if self.cfg.bidirectional else 1)
        return torch.zeros(layers, batch, self.cfg.hidden, device=device)

    # -- network ----------------------------------------------------------------------------------
    def features(self, spec: Tensor) -> Tensor:
        """``[B,T,F,2]`` -> ``[B,3,T,F]``: |X|^c and the compressed (re, im) = X/|X|^(1-c)."""
        re, im = spec[..., 0], spec[..., 1]
        mag2 = re * re + im * im + 1e-12
        c = self.cfg.compress
        mag_c = mag2 ** (c / 2)
        scale = mag2 ** ((c - 1) / 2)
        feat = torch.stack([mag_c, re * scale, im * scale], dim=1)
        return self.in_norm(feat)

    def mask(self, m: Tensor, floor: Tensor | None = None) -> Tensor:
        """Raw ``[B,2,T,F]`` -> bounded complex mask ``[B,T,F,2]`` with |M| = mask_max * tanh(|m|).

        With ``floor`` the magnitude is raised to at least ``floor`` while the phase is kept.
        """
        mr, mi = m[:, 0], m[:, 1]
        mag = torch.sqrt(mr * mr + mi * mi + 1e-8)
        g = self.cfg.mask_max * torch.tanh(mag) / mag
        if floor is None:
            return torch.stack([mr * g, mi * g], dim=-1)
        out_r, out_i = mr * g, mi * g
        abs_m = torch.sqrt(out_r * out_r + out_i * out_i + 1e-20)
        s = torch.maximum(abs_m, floor.to(abs_m.dtype)) / abs_m
        return torch.stack([out_r * s, out_i * s], dim=-1)

    def masks(
        self, spec: Tensor, state: Tensor | None = None, atten_lim_db: float | Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Network only: ``spec -> (mask[B,T,F,2], state, gru_features[B,T,D*H])``.

        The mask at frame ``t`` depends on frames ``<= t`` (causal GRU) and is meant for input frame
        ``t - lookahead``.
        """
        b, t = spec.shape[0], spec.shape[1]
        x = self.features(spec)
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        c, f = x.shape[1], x.shape[3]
        z = self.fc_in(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        if state is None:
            state = self.init_state(b, spec.device)
        z, state = self.gru(z, state)
        x = self.fc_out(z).reshape(b, t, c, f).permute(0, 2, 1, 3)
        for dec, skip in zip(self.decoder, reversed(skips), strict=True):
            x = dec(x, skip)
        floor = None if atten_lim_db is None else atten_floor(atten_lim_db).to(spec.device)
        return self.mask(x.float(), floor), state, z

    def forward(
        self,
        spec: Tensor,
        state: Tensor | None = None,
        atten_lim_db: float | Tensor | None = None,
        return_mask: bool = False,
    ):
        m, state, _ = self.masks(spec, state, atten_lim_db)
        out = apply_mask(delay_frames(spec, self.cfg.lookahead), m)
        return (out, state, m) if return_mask else (out, state)

    def enhance(self, wav: Tensor, atten_lim_db: float | Tensor | None = None) -> Tensor:
        """Waveform ``[B,N]`` -> enhanced ``[B,N]`` (offline, identical to streaming, aligned with ``wav``)."""
        k = self.cfg.lookahead
        if k == 0 and atten_lim_db is None:
            out, _ = self(self.stft(wav))
            return self.stft.inverse(out, length=wav.shape[-1])
        n = wav.shape[-1]
        x = F.pad(wav, (0, k * HOP)) if k else wav
        out, _ = self(self.stft(x), atten_lim_db=atten_lim_db)
        return self.stft.inverse(out, length=n + k * HOP)[..., k * HOP :]

    def config_dict(self) -> dict:
        return asdict(self.cfg)


class CRNTeacher(CRN):
    """``crn_teacher`` (§4.9): bidirectional GRU, hidden 512, non-causal. Never exported."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("hidden", 512)
        kwargs.setdefault("bidirectional", True)
        if kwargs.get("lookahead", 0):
            raise ValueError("the non-causal teacher has no lookahead parameter")
        super().__init__(**kwargs)
