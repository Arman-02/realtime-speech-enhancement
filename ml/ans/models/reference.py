"""Off-the-shelf reference systems: RNNoise and DeepFilterNet3 (PLAN.md §4.3, §4.10).

Both run natively at 48 kHz; we resample 16 -> 48 -> 16 kHz (soxr VHQ) and compensate their fixed delay.
They are framed honestly as *trained on far more data, at full band*.

They run in **isolated environments** (``uv run --no-project --with …``) so the main lockfile and torch
version stay untouched:

* RNNoise: ``pyrnnoise`` (bundles the BSD-licensed C library; wheels for macOS arm64 / Python 3.12).
* DeepFilterNet3: ``deepfilternet`` on Python 3.11 with ``torch==2.1.2`` / ``torchaudio==2.1.2`` — the
  package's ``df.io`` imports ``torchaudio.backend.common``, which newer torchaudio removed, and the
  unpinned resolution does not even install torchaudio. Model weights download to the user cache.

Outputs are cached as float32 wavs under ``ml/exports/ref_cache/<system>/<set>/<id>.wav``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ML_ROOT = Path(__file__).resolve().parents[2]
CACHE = ML_ROOT / "exports" / "ref_cache"
WORKER = Path(__file__).with_name("ref_worker.py")

ENVS: dict[str, list[str]] = {
    "rnnoise": ["--python", "3.12", "--with", "pyrnnoise==0.4.5", "--with", "soundfile", "--with", "soxr"],
    "deepfilternet": [
        "--python",
        "3.11",
        "--with",
        "deepfilternet==0.5.6",
        "--with",
        "torch==2.1.2",
        "--with",
        "torchaudio==2.1.2",
        "--with",
        "numpy<2",
        "--with",
        "soundfile",
        "--with",
        "soxr",
    ],
}
DISPLAY = {"rnnoise": "RNNoise (reference)", "deepfilternet": "DeepFilterNet3 (reference)"}
INFO = {
    # Published sizes only (left out when no reliable figure is known, e.g. RNNoise MACs).
    # DeepFilterNet3: 2.13 M params, 0.344 GMAC/s (Schröter et al., Interspeech 2023) -> 3.44 M MACs per 10 ms.
    "rnnoise": {"params": None, "macs_per_frame": None, "native_sr": 48000, "license": "BSD-3-Clause"},
    "deepfilternet": {
        "params": 2_130_000,
        "macs_per_frame": 3_440_000,
        "native_sr": 48000,
        "license": "MIT/Apache-2.0",
    },
}


def cache_path(system: str, set_name: str, uid: str) -> Path:
    return CACHE / system / set_name / f"{uid}.wav"


def ensure_outputs(system: str, set_name: str, items: list[tuple[str, Path]], force: bool = False) -> dict:
    """Run ``system`` on ``(id, noisy_path)`` items not yet cached; returns timing info (may be cached)."""
    out_dir = CACHE / system / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(uid, p) for uid, p in items if force or not cache_path(system, set_name, uid).exists()]
    timing_file = out_dir / f"timing_{system}.json"
    if todo:
        jobs = out_dir / "jobs.tsv"
        jobs.write_text(
            "".join(f"{uid}\t{Path(p).resolve()}\t{cache_path(system, set_name, uid).resolve()}\n" for uid, p in todo)
        )
        cmd = ["uv", "run", "--no-project", "--quiet", *ENVS[system], "python", "-I", str(WORKER), system, str(jobs)]
        print(f"[reference] {system}/{set_name}: processing {len(todo)} clips in an isolated env", flush=True)
        subprocess.run(cmd, check=True, cwd=out_dir)
    return json.loads(timing_file.read_text()) if timing_file.exists() else {}


def load_output(system: str, set_name: str, uid: str, n: int) -> np.ndarray:
    y, _ = sf.read(cache_path(system, set_name, uid), dtype="float32")
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
