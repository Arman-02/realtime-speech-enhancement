"""Knowledge distillation from the non-causal ``crn_teacher`` into causal students (PLAN.md §4.4, §4.9).

``L_student = L_target(Y_s, S) + beta * L_spec(Y_s, sg(Y_t)) [+ gamma * MSE(h_s W, sg(h_t))]``

* ``L_spec`` is the same compressed complex spectral loss as the target term (c, alpha from ``loss``).
* The teacher runs online under ``torch.no_grad()`` (augmented crops are unique, so outputs can't be
  cached). On MPS it runs under fp16 autocast if a check on the first batch shows the result matches
  fp32 (relative error < ``fp16_tol``), otherwise in fp32. STFT, features and the mask stay fp32.
* Feature distillation (``gamma > 0``) maps the student's GRU output through a learned linear
  projection onto the teacher's (bidirectional) GRU output. The projection is trained with the student
  but is not part of the student's state_dict.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ans.dsp import STFT
from ans.losses import EnhancementLoss, compressed_spectral_loss
from ans.models.crn import apply_mask


@dataclass
class DistillConfig:
    teacher_ckpt: str | None = None
    beta: float = 1.0
    gamma: float = 0.0
    teacher_dtype: str = "auto"  # auto | fp32 | fp16
    fp16_tol: float = 1e-2

    @classmethod
    def from_dict(cls, d: dict | None) -> DistillConfig | None:
        if not d or not d.get("teacher_ckpt"):
            return None
        return cls(**d)


def resolve_ckpt(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else root / p


class Distiller(nn.Module):
    """Holds the frozen teacher and the (optional) feature projection."""

    def __init__(self, teacher: nn.Module, student: nn.Module, cfg: DistillConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(student.feature_dim, teacher.feature_dim) if cfg.gamma > 0 else None
        if cfg.gamma > 0 and getattr(student.cfg, "lookahead", 0):
            raise ValueError("feature distillation with a lookahead student is not supported")
        self.stft = STFT()
        self.use_fp16 = cfg.teacher_dtype == "fp16"
        self._calibrated = cfg.teacher_dtype != "auto"

    def train(self, mode: bool = True) -> Distiller:
        super().train(mode)
        self.teacher.eval()  # the teacher is always in eval mode
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.proj.parameters()) if self.proj is not None else []

    def _autocast(self, device: torch.device, fp16: bool):
        if fp16 and device.type in ("mps", "cuda"):
            return torch.autocast(device.type, dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def teacher_forward(self, noisy: Tensor, fp16: bool | None = None) -> tuple[Tensor, Tensor]:
        """Teacher waveform estimate ``[B,N]`` and GRU features ``[B,T,D]`` (both fp32, detached)."""
        fp16 = self.use_fp16 if fp16 is None else fp16
        spec = self.stft(noisy)
        with self._autocast(noisy.device, fp16):
            m, _, z = self.teacher.masks(spec)
        out = apply_mask(spec, m.float())
        return self.stft.inverse(out, length=noisy.shape[-1]), z.float()

    @torch.no_grad()
    def calibrate(self, noisy: Tensor) -> str:
        """Pick fp16 vs fp32 for the teacher on the first batch (``teacher_dtype: auto``)."""
        self._calibrated = True
        if noisy.device.type != "mps":
            self.use_fp16 = False
            return f"teacher fp32 (device {noisy.device.type})"
        ref, _ = self.teacher_forward(noisy, fp16=False)
        try:
            y16, _ = self.teacher_forward(noisy, fp16=True)
            rel = float((y16 - ref).norm() / (ref.norm() + 1e-12))
        except Exception as e:  # autocast unsupported on this torch/MPS build
            self.use_fp16 = False
            return f"teacher fp32 (fp16 autocast failed: {type(e).__name__}: {e})"
        self.use_fp16 = bool(torch.isfinite(torch.tensor(rel))) and rel < self.cfg.fp16_tol
        return f"teacher {'fp16' if self.use_fp16 else 'fp32'} (fp16 vs fp32 relative error {rel:.2e})"

    def student_forward(self, student: nn.Module, noisy: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.proj is None:
            return student.enhance(noisy), None
        spec = student.stft(noisy)
        m, _, z = student.masks(spec)
        out = apply_mask(spec, m)
        return student.stft.inverse(out, length=noisy.shape[-1]), z

    def forward(
        self, student: nn.Module, noisy: Tensor, clean: Tensor, mask: Tensor, loss_fn: EnhancementLoss
    ) -> tuple[Tensor, dict[str, float]]:
        if not self._calibrated:
            print("  distill: " + self.calibrate(noisy), flush=True)
        y_t, z_t = self.teacher_forward(noisy)
        est, z_s = self.student_forward(student, noisy)
        est = est.float()
        loss, terms = loss_fn(est, clean, mask)
        lc = loss_fn.cfg
        kd = compressed_spectral_loss(loss_fn.stft(est * mask), loss_fn.stft(y_t * mask), lc.compress, lc.alpha)
        loss = loss + self.cfg.beta * kd
        terms["kd_spec"] = float(self.cfg.beta * kd.detach())
        if self.proj is not None and z_s is not None:
            feat = torch.mean((self.proj(z_s) - z_t) ** 2)
            loss = loss + self.cfg.gamma * feat
            terms["kd_feat"] = float(self.cfg.gamma * feat.detach())
        return loss, terms
