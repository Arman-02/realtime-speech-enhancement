"""Run provenance (PLAN.md §4.5, §6.2, §8): ``checkpoints/<run>/meta.json``.

Records the resolved config, git hash (``"uncommitted"`` + a hash of the tracked source files while the
repo has no commits), seed, data-manifest hashes, a pip-freeze digest and ``"completed": true`` at the end.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
SOURCE_GLOBS = (
    "ml/ans/**/*.py",
    "ml/configs/*.yaml",
    "ml/configs/*.txt",
    "ml/scripts/*.sh",
    "ml/pyproject.toml",
    "ml/uv.lock",
    "scripts/*.py",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def source_hash(root: Path = REPO_ROOT) -> str:
    """SHA-256 over (relative path, content) of the tracked source files, in sorted order."""
    h = hashlib.sha256()
    files = sorted({p for g in SOURCE_GLOBS for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts})
    for p in files:
        h.update(str(p.relative_to(root)).encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def git_info(root: Path = REPO_ROOT) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": "uncommitted", "dirty": None, "source_hash": source_hash(root)}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if commit.returncode == 0 and commit.stdout.strip():
            info["commit"] = commit.stdout.strip()
            st = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            info["dirty"] = bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def git_hash(root: Path = REPO_ROOT) -> str:
    """Short provenance string for results/manifests: commit or ``uncommitted+<source hash[:12]>``."""
    g = git_info(root)
    return g["commit"] if g["commit"] != "uncommitted" else f"uncommitted+{g['source_hash'][:12]}"


def pip_freeze() -> dict[str, Any]:
    from importlib import metadata

    pkgs = sorted(
        {f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]}
    )
    return {"sha256": hashlib.sha256("\n".join(pkgs).encode()).hexdigest(), "n_packages": len(pkgs)}


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def manifest_hashes(paths: dict[str, str | Path | None]) -> dict[str, dict[str, Any]]:
    out = {}
    for k, p in paths.items():
        if p is None:
            continue
        p = Path(p)
        out[k] = {
            "path": str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p),
            "sha256": file_sha256(p),
        }
    return out


def read_meta(run_dir: str | Path) -> dict[str, Any] | None:
    p = Path(run_dir) / "meta.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_meta(run_dir: str | Path, meta: dict[str, Any]) -> Path:
    p = Path(run_dir) / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    tmp.replace(p)
    return p


def start_meta(
    run_dir: Path, cfg: dict, manifests: dict[str, str | Path | None], extra: dict | None = None
) -> dict[str, Any]:
    """Create (or, on resume, update) meta.json with ``completed: false``."""
    old = read_meta(run_dir) or {}
    meta = {
        "run_name": cfg.get("run_name"),
        "config": cfg,
        "seed": cfg.get("seed"),
        "git": git_info(),
        "manifests": manifest_hashes(manifests),
        "pip_freeze": pip_freeze(),
        "environment": environment(),
        "started_at": old.get("started_at", now()),
        "resumed_at": [*old.get("resumed_at", []), now()] if old else [],
        "completed": False,
        **(extra or {}),
    }
    write_meta(run_dir, meta)
    return meta


def finish_meta(run_dir: Path, meta: dict[str, Any], **fields: Any) -> dict[str, Any]:
    meta = {**meta, **fields, "finished_at": now(), "completed": True}
    write_meta(run_dir, meta)
    return meta
