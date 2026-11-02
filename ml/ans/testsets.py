"""Evaluation sets (PLAN.md §3.3, §3.5, §4.10): discovery, metadata and loading.

* ``vbd``     ``data/vbd16k/manifests/test.csv`` (VB-D schema)
* ``ood_esc`` ``data/**/manifests/ood_esc.csv`` — VB-D clean test speech + ESC-50 noise
* ``talker``  ``data/**/manifests/ood_talker.csv`` — VB-D test speech + LibriSpeech test-clean talker
* ``real``    ``data/**/manifests/real.csv`` — real recordings, non-intrusive (DNSMOS only)

Intrusive manifests use the VB-D columns ``id, clean_path, noisy_path[, noise_type, snr_db]`` (aliases
``clean``/``noisy``, ``category``/``noise_class``, ``snr``/``sir_db`` are accepted). ``real`` needs ``id, path``.
Paths are relative to the repo root (or the manifest's data area).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from ans.augment import PATH_KEYS, find_data_manifest
from ans.data import REPO_ROOT, manifest_path

SET_ORDER = ("vbd", "ood_esc", "talker", "reverb", "real")
SET_META = {
    "vbd": {"label": "VoiceBank+DEMAND test", "intrusive": True, "noise_seen_in_training": False},
    "ood_esc": {"label": "VB-D speech + ESC-50 noise", "intrusive": True, "noise_seen_in_training": False},
    "talker": {"label": "Competing talker", "intrusive": True, "noise_seen_in_training": False},
    "reverb": {"label": "Reverberant speech (held-out rooms)", "intrusive": True, "noise_seen_in_training": False},
    "real": {"label": "Real recordings", "intrusive": False, "noise_seen_in_training": False},
}
MANIFEST_NAMES = {
    "ood_esc": ("ood_esc",),
    "talker": ("ood_talker", "talker"),
    "reverb": ("ood_reverb",),
    "real": ("real",),
}


@dataclass(frozen=True)
class EvalItem:
    id: str
    noisy_path: Path
    clean_path: Path | None
    noise_type: str
    snr_db: float


def set_manifest(name: str) -> Path | None:
    if name == "vbd":
        p = manifest_path("test")
        return p if p.exists() else None
    for cand in MANIFEST_NAMES.get(name, (name,)):
        if p := find_data_manifest(cand):
            return p
    return None


def available_sets(names: tuple[str, ...] | list[str] = SET_ORDER) -> list[str]:
    return [n for n in names if set_manifest(n) is not None]


def _path(raw: str, manifest: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (REPO_ROOT, manifest.parent.parent, manifest.parent, REPO_ROOT / "data"):
        if (base / p).exists():
            return base / p
    return REPO_ROOT / p


def _float(v: str | None) -> float:
    try:
        return float(v) if v not in (None, "") else math.nan
    except ValueError:
        return math.nan


def read_set(name: str, limit: int | None = None) -> list[EvalItem]:
    path = set_manifest(name)
    if path is None:
        raise FileNotFoundError(f"test set {name!r}: manifest not found")
    items = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            noisy = row.get("noisy_path") or row.get("noisy") or next((row[k] for k in PATH_KEYS if row.get(k)), "")
            clean = row.get("clean_path") or row.get("clean") or ""
            items.append(
                EvalItem(
                    id=row.get("id") or Path(noisy).stem,
                    noisy_path=_path(noisy, path),
                    clean_path=_path(clean, path) if clean and SET_META.get(name, {}).get("intrusive", True) else None,
                    noise_type=row.get("noise_type") or row.get("category") or row.get("noise_class") or "unknown",
                    snr_db=_float(row.get("snr_db") or row.get("snr") or row.get("sir_db") or row.get("sir")),
                )
            )
    if limit is not None and limit < len(items):
        import numpy as np

        idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
        items = [items[i] for i in idx]
    return items


def set_metadata(name: str, n: int) -> dict:
    m = dict(SET_META.get(name, {"label": name, "intrusive": True, "noise_seen_in_training": False}))
    m["label"] = f"{m['label']} ({n})"
    m["n"] = n
    return m
