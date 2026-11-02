"""Run registry helpers (PLAN.md §6.2): display names and run attributes derived from a run's config.

Display names: ``CRN-256 (benchmark)`` for R0, ``CRN-128``, ``CRN-128 (KD)``, ``CRN-128 (KD, seed 1)``,
``CRN-128 (KD β=0.5)``, ``CRN-256 (k=1)``, ``CRN-256 (+MUSAN, 50 ep)``, ``Teacher (bi-GRU 512)`` …
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

BENCHMARK_RUN = "crn_small"
DEMO_RUN = "demo_crn256"
AUG_LABELS = {"extra_noise": "+MUSAN", "reverb": "+reverb", "mic": "+mic", "level": "+level"}


def _augment_tags(cfg: dict) -> list[str]:
    aug = (cfg.get("data") or {}).get("augment") or {}
    return [AUG_LABELS[k] for k in AUG_LABELS if (aug.get(k) or {}).get("prob", 0) > 0]


def run_info(cfg: dict) -> dict[str, Any]:
    """Attributes used by results.json rows / the Pareto table."""
    model = cfg.get("model") or {}
    kw = model.get("kwargs") or {}
    name = model.get("name", "ans_crn")
    teacher = name in ("crn_teacher", "teacher")
    distill = cfg.get("distill") or {}
    distilled = bool(distill.get("teacher_ckpt"))
    return {
        "run": cfg.get("run_name"),
        "arch": name,
        "hidden": kw.get("hidden", 512 if teacher else 256),
        "distilled": distilled,
        "beta": distill.get("beta", 1.0) if distilled else None,
        "seed": cfg.get("seed", 0),
        "lookahead": int(kw.get("lookahead", 0) or 0),
        "augment": [t.lstrip("+") for t in _augment_tags(cfg)],
        "epochs": (cfg.get("schedule") or {}).get("epochs", 100),
        "causal": not (teacher or kw.get("bidirectional", False)),
        "teacher": teacher,
    }


def display_name(cfg: dict) -> str:
    if cfg.get("display_name"):
        return str(cfg["display_name"])
    info = run_info(cfg)
    if info["teacher"]:
        aug = _augment_tags(cfg)
        return f"Teacher (bi-GRU {info['hidden']}" + (f", {', '.join(aug)})" if aug else ")")
    if info["arch"] not in ("ans_crn", "crn"):
        return str(cfg.get("run_name") or info["arch"])
    if info["run"] == BENCHMARK_RUN:
        return f"CRN-{info['hidden']} (benchmark)"
    tags = []
    if info["distilled"]:
        tags.append("KD" if info["beta"] == 1.0 else f"KD β={info['beta']:g}")
    if info["lookahead"]:
        tags.append(f"k={info['lookahead']}")
    tags += _augment_tags(cfg)
    if info["epochs"] != 100:
        tags.append(f"{info['epochs']} ep")
    if info["seed"]:
        tags.append(f"seed {info['seed']}")
    return f"CRN-{info['hidden']}" + (f" ({', '.join(tags)})" if tags else "")


def ckpt_config(path: str | Path) -> dict:
    """The training config stored in a checkpoint (falls back to ``config.yaml`` next to it)."""
    import torch
    import yaml

    path = Path(path)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck.get("config"), dict):
            return ck["config"]
    except Exception:
        pass
    cy = path.parent / "config.yaml"
    return yaml.safe_load(cy.read_text()) if cy.exists() else {}
