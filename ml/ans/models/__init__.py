"""Model registry. Every trainable model exposes ``enhance(wav[B,N]) -> wav[B,N]``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ans.models.classical import ClassicalEnhancer
from ans.models.conv_tasnet import ConvTasNet
from ans.models.crn import CRN, CRNTeacher

MODELS: dict[str, type[nn.Module]] = {
    "ans_crn": CRN,
    "crn_teacher": CRNTeacher,
    "conv_tasnet": ConvTasNet,
}

ALIASES = {"crn": "ans_crn", "teacher": "crn_teacher", "tasnet": "conv_tasnet"}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    name = ALIASES.get(name, name)
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu", ema: bool = True) -> nn.Module:
    """Rebuild a model from a ``train.py`` checkpoint (EMA weights by default)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return model_from_checkpoint(ckpt, device, ema)


def model_from_checkpoint(ckpt: dict, device: torch.device | str = "cpu", ema: bool = True) -> nn.Module:
    model = build_model(ckpt["model_name"], **ckpt.get("model_kwargs", {}))
    key = "ema_state_dict" if ema and ckpt.get("ema_state_dict") is not None else "state_dict"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval()


__all__ = [
    "CRN",
    "MODELS",
    "CRNTeacher",
    "ClassicalEnhancer",
    "ConvTasNet",
    "build_model",
    "count_params",
    "load_checkpoint",
    "model_from_checkpoint",
]
