"""Training augmentation (PLAN.md §4.1). Every augmentation is off by default (probability 0).

* ``extra_noise``: with probability p the noise comes from the training-only noise corpus (MUSAN,
  ``noise_train`` manifest) instead of VoiceBank+DEMAND.
* ``reverb``: with probability p the clean speech is convolved with a training RIR (``rir_train``).
  The **target** is the direct path plus the first ``early_ms`` (50 ms) of reflections.
* ``mic``: input-only coloration (random high-pass 80–200 Hz, ±6 dB tilt, optional 4–8 kHz low-pass,
  occasional soft clipping). All filters are zero-phase, so input and target stay sample-aligned.
* ``level``: extended gain range ``[-45, -3]`` dBFS (replaces the default gain stage when it fires).

Manifests are found by name (``noise_train``, ``rir_train`` …) under ``data/**/manifests/`` (§3.5) or
given as explicit paths. Corpus separation (§3.5): training augmentation only ever reads ``*_train``
manifests; :func:`check_corpus_separation` verifies train/test manifests are disjoint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ans.dsp import SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
PATH_KEYS = ("path", "wav", "file", "filepath", "noise_path", "rir_path", "audio_path")
CORPUS_KEYS = ("corpus", "source", "dataset", "origin")


# ------------------------------------------------------------------------------------------------
# Manifests
# ------------------------------------------------------------------------------------------------


def find_data_manifest(name: str, data_root: str | Path = DATA_ROOT) -> Path | None:
    """``noise_train`` -> ``data/<area>/manifests/noise_train.csv`` (or ``data/manifests/…``); a path
    (anything ending in ``.csv``) is returned as is if it exists."""
    if name.endswith(".csv"):
        p = Path(name)
        p = p if p.is_absolute() else REPO_ROOT / p
        return p if p.exists() else None
    root = Path(data_root)
    for cand in (root / "manifests" / f"{name}.csv", *sorted(root.glob(f"*/manifests/{name}.csv"))):
        if cand.exists():
            return cand
    hits = sorted(root.glob(f"**/manifests/{name}.csv"))
    return hits[0] if hits else None


@dataclass(frozen=True)
class AudioEntry:
    id: str
    path: Path
    corpus: str
    num_samples: int | None = None


def _resolve(raw: str, manifest: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (REPO_ROOT, manifest.parent.parent, manifest.parent, DATA_ROOT):
        if (base / p).exists():
            return base / p
    return REPO_ROOT / p


def read_audio_manifest(path: str | Path) -> list[AudioEntry]:
    """Generic ``id,path[,corpus][,num_samples]…`` manifest (column names are matched loosely)."""
    path = Path(path)
    out = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            raw = next((row[k] for k in PATH_KEYS if row.get(k)), None)
            if raw is None:
                raise KeyError(f"{path}: no path column (one of {PATH_KEYS})")
            corpus = next((row[k] for k in CORPUS_KEYS if row.get(k)), "") or path.stem
            n = row.get("num_samples") or row.get("frames") or ""
            out.append(
                AudioEntry(
                    id=row.get("id") or Path(raw).stem or str(i),
                    path=_resolve(raw, path),
                    corpus=corpus,
                    num_samples=int(float(n)) if n else None,
                )
            )
    return out


def check_corpus_separation(train: list[AudioEntry], test: list[AudioEntry]) -> list[str]:
    """Problems found (empty list = separated): shared ids, shared files or shared corpora."""
    problems = []
    if ids := {e.id for e in train} & {e.id for e in test}:
        problems.append(f"{len(ids)} shared ids, e.g. {sorted(ids)[:3]}")
    if files := {e.path.resolve() for e in train} & {e.path.resolve() for e in test}:
        problems.append(f"{len(files)} shared files, e.g. {sorted(map(str, files))[:3]}")
    if corpora := {e.corpus.lower() for e in train} & {e.corpus.lower() for e in test}:
        problems.append(f"shared source corpora: {sorted(corpora)}")
    return problems


# ------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------


@dataclass
class ExtraNoiseConfig:
    prob: float = 0.0
    manifest: str = "noise_train"


@dataclass
class ReverbConfig:
    prob: float = 0.0
    manifest: str = "rir_train"
    early_ms: float = 50.0
    pre_ms: float = 1.0  # kept before the direct-path peak


@dataclass
class MicConfig:
    prob: float = 0.0
    hpf_hz: tuple[float, float] = (80.0, 200.0)
    tilt_db: tuple[float, float] = (-6.0, 6.0)
    lpf_prob: float = 0.3
    lpf_hz: tuple[float, float] = (4000.0, 8000.0)
    clip_prob: float = 0.1
    clip_drive: tuple[float, float] = (1.5, 6.0)


@dataclass
class LevelConfig:
    prob: float = 0.0
    dbfs: tuple[float, float] = (-45.0, -3.0)


@dataclass
class AugmentConfig:
    extra_noise: ExtraNoiseConfig = field(default_factory=ExtraNoiseConfig)
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    mic: MicConfig = field(default_factory=MicConfig)
    level: LevelConfig = field(default_factory=LevelConfig)

    @property
    def enabled(self) -> bool:
        return any(getattr(self, k).prob > 0 for k in ("extra_noise", "reverb", "mic", "level"))

    @classmethod
    def from_dict(cls, d: dict | None) -> AugmentConfig:
        d = d or {}

        def mk(klass, sub):
            sub = dict(sub or {})
            for k, v in sub.items():
                if isinstance(v, list):
                    sub[k] = tuple(v)
            return klass(**sub)

        return cls(
            extra_noise=mk(ExtraNoiseConfig, d.get("extra_noise")),
            reverb=mk(ReverbConfig, d.get("reverb")),
            mic=mk(MicConfig, d.get("mic")),
            level=mk(LevelConfig, d.get("level")),
        )

    def manifests(self) -> dict[str, str]:
        """Manifest names/paths needed by the enabled augmentations."""
        out = {}
        if self.extra_noise.prob > 0:
            out["extra_noise"] = self.extra_noise.manifest
        if self.reverb.prob > 0:
            out["reverb"] = self.reverb.manifest
        return out


# ------------------------------------------------------------------------------------------------
# Signal operations
# ------------------------------------------------------------------------------------------------


def _fftconv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution truncated to ``len(x)``."""
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[: len(x)]
    return y.astype(np.float32)


def prepare_rir(rir: np.ndarray, early_ms: float = 50.0, pre_ms: float = 1.0, sr: int = SAMPLE_RATE):
    """Trim the pre-delay (keep ``pre_ms`` before the direct-path peak), normalise the peak to 1, and
    return ``(full, early)`` where ``early`` is the first ``pre_ms + early_ms`` of ``full``."""
    rir = np.asarray(rir, np.float64)
    peak = int(np.argmax(np.abs(rir)))
    start = max(0, peak - round(pre_ms * sr / 1000))
    full = rir[start:] / (np.abs(rir[peak]) + 1e-12)
    early = full[: (peak - start) + round(early_ms * sr / 1000)]
    return full.astype(np.float32), early.astype(np.float32)


def reverberate(clean: np.ndarray, rir: np.ndarray, early_ms: float = 50.0, pre_ms: float = 1.0):
    """``clean -> (reverberant speech, early-reflection target)``, both ``len(clean)`` and aligned."""
    full, early = prepare_rir(rir, early_ms, pre_ms)
    return _fftconv(clean, full), _fftconv(clean, early)


def mic_response(
    n_fft: int,
    hpf_hz: float | None = None,
    tilt_db: float = 0.0,
    lpf_hz: float | None = None,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Zero-phase magnitude response on the ``rfft`` grid: Butterworth-shaped HPF (order 2), log-frequency
    tilt (``tilt_db`` between 100 Hz and 8 kHz, 0 dB at the geometric centre), Butterworth LPF (order 8)."""
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    h = np.ones_like(f)
    if hpf_hz:
        h *= 1 / np.sqrt(1 + (hpf_hz / np.maximum(f, 1e-3)) ** 4)
    if tilt_db:
        u = np.clip(np.log(np.maximum(f, 100) / 100) / np.log(80), 0, 1)
        h *= 10 ** (tilt_db * (u - 0.5) / 20)
    if lpf_hz:
        h *= 1 / np.sqrt(1 + (f / lpf_hz) ** 16)
    return h


def apply_response(x: np.ndarray, h_fn) -> np.ndarray:
    nfft = 1 << (len(x) - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * h_fn(nfft), nfft)[: len(x)]
    return y.astype(np.float32)


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    """``tanh`` saturation; ``drive`` > 1 means the peak is driven ``drive`` times into the knee."""
    peak = float(np.abs(x).max()) + 1e-9
    d = drive / peak
    return (np.tanh(d * x) / d).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# Augmentor
# ------------------------------------------------------------------------------------------------


def _read_mono(path: Path, start: int = 0, stop: int | None = None) -> np.ndarray:
    x, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        x = soxr.resample(x, sr, SAMPLE_RATE).astype(np.float32)
    return x


class Augmentor:
    """Holds the noise / RIR pools; all randomness comes from the caller's ``rng``."""

    def __init__(self, cfg: AugmentConfig) -> None:
        self.cfg = cfg
        self.noise: list[AudioEntry] = []
        self.rirs: list[AudioEntry] = []
        for key, name in cfg.manifests().items():
            path = find_data_manifest(name)
            if path is None:
                raise FileNotFoundError(f"augmentation {key!r} needs manifest {name!r} (data/**/manifests/{name}.csv)")
            entries = read_audio_manifest(path)
            if not entries:
                raise ValueError(f"{path} is empty")
            if key == "extra_noise":
                self.noise = entries
            else:
                self.rirs = entries

    def noise_crop(self, n: int, rng: np.random.Generator) -> np.ndarray:
        e = self.noise[int(rng.integers(len(self.noise)))]
        total = e.num_samples or sf.info(str(e.path)).frames
        if total > n:
            start = int(rng.integers(0, total - n + 1))
            x = _read_mono(e.path, start, start + n)
        else:
            x = _read_mono(e.path)
        if len(x) < n:
            x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
        x = x[:n]
        return x if np.any(x) else x + 1e-6 * rng.standard_normal(n).astype(np.float32)

    def random_rir(self, rng: np.random.Generator) -> np.ndarray:
        return _read_mono(self.rirs[int(rng.integers(len(self.rirs)))].path)

    def mic(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg.mic
        hpf = float(rng.uniform(*c.hpf_hz))
        tilt = float(rng.uniform(*c.tilt_db))
        lpf = float(rng.uniform(*c.lpf_hz)) if rng.random() < c.lpf_prob else None
        y = apply_response(x, lambda nfft: mic_response(nfft, hpf, tilt, lpf))
        if rng.random() < c.clip_prob:
            y = soft_clip(y, float(rng.uniform(*c.clip_drive)))
        return y
