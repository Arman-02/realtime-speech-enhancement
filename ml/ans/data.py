"""Datasets: VoiceBank+DEMAND manifests with dynamic mixing (PLAN.md §3.5, §4.1) and a synthetic set."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from ans.dsp import SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = REPO_ROOT / "data" / "vbd16k" / "manifests"


@dataclass(frozen=True)
class Utterance:
    id: str
    speaker: str
    clean_path: Path
    noisy_path: Path
    num_samples: int
    noise_type: str
    snr_db: float


def read_manifest(path: str | Path, root: str | Path = REPO_ROOT) -> list[Utterance]:
    """Read ``id,speaker,clean_path,noisy_path,num_samples,noise_type,snr_db`` (paths relative to ``root``)."""
    root = Path(root)
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            snr = row.get("snr_db", "")
            out.append(
                Utterance(
                    id=row["id"],
                    speaker=row.get("speaker", ""),
                    clean_path=root / row["clean_path"],
                    noisy_path=root / row["noisy_path"],
                    num_samples=int(row["num_samples"]),
                    noise_type=row.get("noise_type") or "unknown",
                    snr_db=float(snr) if snr not in ("", None) else float("nan"),
                )
            )
    return out


def manifest_path(split: str, manifests_dir: str | Path | None = None) -> Path:
    base = Path(manifests_dir) if manifests_dir else DEFAULT_MANIFESTS
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / f"{split}.csv"


@dataclass
class MixConfig:
    segment: int = 2 * SAMPLE_RATE
    remix_prob: float = 0.5
    snr_range: tuple[float, float] = (-5.0, 20.0)
    gain_prob: float = 0.5
    gain_dbfs: tuple[float, float] = (-25.0, -5.0)
    extra_noise_dir: str | None = None
    extra_noise_prob: float = 0.5


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    scale = rms(clean) / (rms(noise) * 10 ** (snr_db / 20))
    return clean + scale * noise


def apply_gain(noisy: np.ndarray, clean: np.ndarray, dbfs: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale both signals so the mixture has RMS ``dbfs``; back off to avoid clipping."""
    g = 10 ** (dbfs / 20) / rms(noisy)
    peak = max(np.abs(noisy).max(), np.abs(clean).max()) * g
    if peak > 0.99:
        g *= 0.99 / peak
    return noisy * g, clean * g


def _fit_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1))
        return x[start : start + n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def _read(path: Path, start: int = 0, stop: int | None = None) -> np.ndarray:
    x, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return x if x.ndim == 1 else x.mean(axis=1)


class _RNGMixin:
    """Lazily-created per-process RNG (different in every DataLoader worker, advances across epochs)."""

    _rng: np.random.Generator | None = None
    _pid: int | None = None
    seed: int = 0

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None or self._pid != os.getpid():
            info = torch.utils.data.get_worker_info()
            base = info.seed if info is not None else self.seed
            self._rng = np.random.default_rng(base % 2**32)
            self._pid = os.getpid()
        return self._rng


class VBDDataset(_RNGMixin, Dataset):
    """VoiceBank+DEMAND pairs. ``train=True``: random crops + dynamic mixing; else full utterances."""

    def __init__(
        self,
        manifest: str | Path | list[Utterance],
        train: bool = True,
        mix: MixConfig | None = None,
        max_items: int | None = None,
        seed: int = 0,
        augment=None,
    ) -> None:
        """``augment``: an :class:`ans.augment.AugmentConfig` (or dict); ``None``/all-off keeps the default
        VB-D pipeline bit-for-bit."""
        items = manifest if isinstance(manifest, list) else read_manifest(manifest)
        if max_items is not None and max_items < len(items):
            idx = np.linspace(0, len(items) - 1, max_items).round().astype(int)
            items = [items[i] for i in idx]
        self.items = items
        self.train = train
        self.mix = mix or MixConfig()
        self.seed = seed
        self.extra_noise: list[Path] = []
        if self.mix.extra_noise_dir:
            self.extra_noise = sorted(Path(self.mix.extra_noise_dir).rglob("*.wav"))
        self.augmentor = None
        if augment is not None and train:
            from ans.augment import AugmentConfig, Augmentor

            cfg = augment if isinstance(augment, AugmentConfig) else AugmentConfig.from_dict(augment)
            if cfg.enabled:
                self.augmentor = Augmentor(cfg)

    def __len__(self) -> int:
        return len(self.items)

    def _crop(self, utt: Utterance, n: int) -> tuple[np.ndarray, np.ndarray]:
        if utt.num_samples > n:
            start = int(self.rng.integers(0, utt.num_samples - n + 1))
            return _read(utt.clean_path, start, start + n), _read(utt.noisy_path, start, start + n)
        return _read(utt.clean_path), _read(utt.noisy_path)

    def _other_noise(self, idx: int, n: int) -> np.ndarray:
        if self.extra_noise and self.rng.random() < self.mix.extra_noise_prob:
            path = self.extra_noise[int(self.rng.integers(len(self.extra_noise)))]
            x, sr = sf.read(str(path), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != SAMPLE_RATE:
                import soxr

                x = soxr.resample(x, sr, SAMPLE_RATE).astype(np.float32)
            return _fit_length(x, n, self.rng)
        j = int(self.rng.integers(len(self.items) - 1))
        j = j + 1 if j >= idx else j
        clean, noisy = self._crop(self.items[j], n)
        return _fit_length(noisy - clean, n, self.rng)

    def __getitem__(self, idx: int) -> dict:
        utt = self.items[idx]
        if not self.train:
            clean, noisy = _read(utt.clean_path), _read(utt.noisy_path)
            n = min(len(clean), len(noisy))
            return _item(noisy[:n], clean[:n], utt.id, utt.noise_type, utt.snr_db)
        if self.augmentor is not None:
            return self._getitem_augmented(idx, utt)
        seg = self.mix.segment
        clean, noisy = self._crop(utt, seg)
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        if len(self.items) > 1 and self.rng.random() < self.mix.remix_prob:
            noise = self._other_noise(idx, n)
            noisy = mix_at_snr(clean, noise, float(self.rng.uniform(*self.mix.snr_range)))
        if self.rng.random() < self.mix.gain_prob:
            noisy, clean = apply_gain(noisy, clean, float(self.rng.uniform(*self.mix.gain_dbfs)))
        return _item(noisy, clean, utt.id, utt.noise_type, utt.snr_db)

    def _getitem_augmented(self, idx: int, utt: Utterance) -> dict:
        """§4.1 pipeline: crop -> [reverb: target = early reflections] -> noise (MUSAN | remix | original)
        -> [mic coloration, input only] -> gain ([level] range or the default one)."""
        from ans.augment import reverberate

        a, rng, mix = self.augmentor, self.rng, self.mix
        c = a.cfg
        clean, noisy = self._crop(utt, mix.segment)
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        speech = target = clean
        if c.reverb.prob > 0 and rng.random() < c.reverb.prob:
            speech, target = reverberate(clean, a.random_rir(rng), c.reverb.early_ms, c.reverb.pre_ms)
        if c.extra_noise.prob > 0 and rng.random() < c.extra_noise.prob:
            noisy = mix_at_snr(speech, a.noise_crop(n, rng), float(rng.uniform(*mix.snr_range)))
        elif len(self.items) > 1 and rng.random() < mix.remix_prob:
            noisy = mix_at_snr(speech, self._other_noise(idx, n), float(rng.uniform(*mix.snr_range)))
        else:
            noisy = speech + (noisy - clean)
        if c.mic.prob > 0 and rng.random() < c.mic.prob:
            noisy = a.mic(noisy, rng)
        if c.level.prob > 0 and rng.random() < c.level.prob:
            noisy, target = apply_gain(noisy, target, float(rng.uniform(*c.level.dbfs)))
        elif rng.random() < mix.gain_prob:
            noisy, target = apply_gain(noisy, target, float(rng.uniform(*mix.gain_dbfs)))
        return _item(noisy, target, utt.id, utt.noise_type, utt.snr_db)


def _item(noisy: np.ndarray, clean: np.ndarray, uid: str, noise_type: str, snr_db: float) -> dict:
    return {
        "noisy": torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32)),
        "clean": torch.from_numpy(np.ascontiguousarray(clean, dtype=np.float32)),
        "id": uid,
        "noise_type": noise_type,
        "snr_db": snr_db,
    }


# --------------------------------------------------------------------------------------------
# Synthetic data (tests, smoke runs)
# --------------------------------------------------------------------------------------------

SYNTH_NOISES = ("white", "pink", "hum", "babble")


def synth_speech(n: int, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Harmonic 'voice' with a gliding f0 and a syllabic on/off envelope."""
    t = np.arange(n) / sr
    f0 = rng.uniform(90, 240) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t + rng.uniform(0, 6)))
    f0 *= np.exp(np.linspace(0, rng.uniform(-0.3, 0.3), n))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros(n)
    for k in range(1, int(rng.integers(6, 16))):
        if k * f0.max() >= sr / 2 - 200:
            break
        sig += rng.uniform(0.3, 1.0) / k * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    rate = rng.uniform(2.5, 6.0)
    env = np.clip(np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi)), 0, None) ** 0.7
    return (sig * env).astype(np.float32)


def synth_noise(n: int, kind: str, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        spec = np.fft.rfft(rng.standard_normal(n))
        spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
        x = np.fft.irfft(spec, n=n)
    elif kind == "hum":
        t = np.arange(n) / sr
        f = rng.choice([50.0, 60.0])
        x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8)) + 0.1 * rng.standard_normal(n)
    else:  # babble: a few overlapping synthetic talkers
        x = sum(synth_speech(n, rng, sr) for _ in range(4))
    return (x / (np.std(x) + 1e-8)).astype(np.float32)


class SyntheticDataset(_RNGMixin, Dataset):
    """Synthetic noisy/clean pairs. Eval mode is deterministic per index with variable lengths."""

    def __init__(
        self,
        n_items: int = 1024,
        train: bool = True,
        segment: int = 2 * SAMPLE_RATE,
        snr_range: tuple[float, float] = (-5.0, 20.0),
        seed: int = 0,
    ) -> None:
        self.n_items, self.train, self.segment = n_items, train, segment
        self.snr_range, self.seed = snr_range, seed

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> dict:
        rng = self.rng if self.train else np.random.default_rng((self.seed, idx))
        n = self.segment if self.train else int(rng.integers(SAMPLE_RATE * 3 // 2, 3 * SAMPLE_RATE))
        clean = synth_speech(n, rng)
        kind = SYNTH_NOISES[int(rng.integers(len(SYNTH_NOISES)))]
        snr = float(rng.uniform(*self.snr_range))
        noisy = mix_at_snr(clean, synth_noise(n, kind, rng), snr)
        noisy, clean = apply_gain(noisy, clean, float(rng.uniform(-25, -10)))
        return _item(noisy, clean, f"synth_{idx:05d}", kind, round(snr, 1))


def collate(batch: list[dict]) -> dict:
    """Zero-pad to the longest item; returns ``noisy, clean [B,N]``, ``lengths [B]``, ``mask [B,N]``."""
    lengths = torch.tensor([b["noisy"].shape[-1] for b in batch])
    n = int(lengths.max())
    noisy = torch.zeros(len(batch), n)
    clean = torch.zeros(len(batch), n)
    for i, b in enumerate(batch):
        noisy[i, : lengths[i]] = b["noisy"]
        clean[i, : lengths[i]] = b["clean"]
    mask = (torch.arange(n)[None] < lengths[:, None]).float()
    return {
        "noisy": noisy,
        "clean": clean,
        "lengths": lengths,
        "mask": mask,
        "id": [b["id"] for b in batch],
        "noise_type": [b["noise_type"] for b in batch],
        "snr_db": [b["snr_db"] for b in batch],
    }


def default_workers() -> int:
    return max(0, min(8, (os.cpu_count() or 2) - 2))


@dataclass
class LoaderConfig:
    batch_size: int = 32
    num_workers: int | None = None
    shuffle: bool = True
    drop_last: bool = True


def make_loader(ds: Dataset, cfg: LoaderConfig) -> DataLoader:
    workers = default_workers() if cfg.num_workers is None else cfg.num_workers
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=collate,
        pin_memory=False,
    )
