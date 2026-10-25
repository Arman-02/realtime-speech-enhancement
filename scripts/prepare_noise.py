# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "soundfile>=0.12",
#     "soxr>=0.3.7",
#     "tqdm>=4.66",
#     "requests>=2.31",
#     "pyroomacoustics>=0.7",
# ]
# ///
"""Download and prepare the extra noise / talker / RIR corpora at 16 kHz (PLAN.md §3.5, §4.1, §4.10).

Corpora and their role (corpus separation rule, PLAN.md §3.5):
    MUSAN noise (openslr.org/17)               TRAIN only     data/noise/musan16k/        manifests/noise_train.csv
    MUSAN music (~20 x 10 s excerpts)          failure gallery only (never training)  data/failures_src/music/
    ESC-50, ~20 classes (github ESC-50)        TEST only      data/noise/esc50_16k/       manifests/noise_ood.csv
    LibriSpeech test-clean (openslr.org/12)    TEST only      data/talkers/librispeech_test_clean16k/  manifests/talkers_test.csv
    Simulated shoebox RIRs (pyroomacoustics)   TRAIN + TEST   data/rir/sim/               manifests/rir_{train,test}.csv
        (train and test drawn from disjoint seed streams; no parameter tuple shared)
    Measured RIRs, OpenSLR 28 real_rirs_isotropic_noises   TEST only   data/rir/real/   manifests/rir_test.csv

All manifests live in <out>/manifests/ and store paths relative to the repo root.

Usage:
    uv run scripts/prepare_noise.py --out data
    uv run scripts/prepare_noise.py --out data --skip-download         # use archives already present
    uv run scripts/prepare_noise.py --out data --only rir_sim check    # rebuild a subset of steps
    uv run scripts/prepare_noise.py --out data --only check            # corpus-separation self-check only

Idempotent and resumable: downloads resume via HTTP Range, extraction skips files already
present, conversion skips outputs that exist, and manifests are rewritten atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import stat
import sys
import tarfile
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import soxr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_SR = 16000
MAX_WORKERS = 4  # a training run shares this machine; never use more than 4 processes

ESC_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"  # ESC-50 master, 2023-10-13

# name -> download metadata. `md5` = checksum published by the provider (hard error on mismatch).
# `sha256` = pinned by us on first download (no checksum is published for these); mismatch is an
# error unless --allow-unpinned is given.
SOURCES = {
    "librispeech": {
        "file": "test-clean.tar.gz",
        "urls": [
            "https://openslr.elda.org/resources/12/test-clean.tar.gz",
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
        ],
        "size": 346663984,
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",  # https://www.openslr.org/resources/12/md5sum.txt
    },
    "esc50": {
        "file": f"ESC-50-{ESC_COMMIT}.zip",
        "urls": [f"https://github.com/karolpiczak/ESC-50/archive/{ESC_COMMIT}.zip"],
        "size": None,
        "sha256": "661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f",
    },
    "rirs_noises": {
        "file": "rirs_noises.zip",
        "urls": [
            "https://openslr.elda.org/resources/28/rirs_noises.zip",
            "https://www.openslr.org/resources/28/rirs_noises.zip",
            "https://openslr.trmal.net/resources/28/rirs_noises.zip",
        ],
        "size": 1311166223,
        "sha256": "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7",
    },
    "musan": {
        "file": "musan.tar.gz",
        "urls": [
            "https://openslr.elda.org/resources/17/musan.tar.gz",
            "https://openslr.trmal.net/resources/17/musan.tar.gz",
            "https://www.openslr.org/resources/17/musan.tar.gz",
            "https://openslr.magicdatatech.com/resources/17/musan.tar.gz",
        ],
        "size": 11086114085,
        "sha256": "86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1",
    },
}

LICENCES = {
    "musan": "MUSAN (Snyder, Chen & Povey 2015), https://www.openslr.org/17/ - corpus CC BY 4.0; noise files "
             "from freesound.org / soundbible.com under the per-file terms in musan/noise/*/LICENSE",
    "esc50": "ESC-50 (Piczak 2015), https://github.com/karolpiczak/ESC-50 - CC BY-NC 3.0 (non-commercial); "
             "clips from freesound.org, per-clip attribution in the ESC-50 LICENSE file",
    "librispeech": "LibriSpeech (Panayotov et al. 2015), https://www.openslr.org/12/ - CC BY 4.0",
    "rirs_noises": "OpenSLR 28 RIRS_NOISES (Ko et al. 2017), https://www.openslr.org/28/ - Apache 2.0 package; "
                   "real RIRs come from RWCP (research and development use only), REVERB 2014 (research use) and "
                   "Aachen AIR (licence not stated) - test only, never redistributed",
    "sim": "Simulated with pyroomacoustics (MIT) by this script - no third-party audio",
}

# ESC-50 classes for the OOD noise set (PLAN.md §4.10.1). ESC-50 has no "traffic" class; "airplane"
# (broadband transport rumble) is the closest stand-in. "dishes/can opening" -> "can_opening".
ESC_CLASSES = [
    "rain", "wind", "airplane", "car_horn", "engine", "train", "siren", "keyboard_typing", "mouse_click",
    "washing_machine", "vacuum_cleaner", "footsteps", "door_wood_knock", "clock_alarm", "crying_baby",
    "laughing", "coughing", "pouring_water", "can_opening", "helicopter",
]

RIR_SEEDS = {"train": 20261023_1, "test": 20261023_2}  # disjoint SeedSequence entropy per split
N_MUSIC = 20
MUSIC_SECONDS = 10.0
UA = {"User-Agent": "ans-prepare-noise/1.0"}

NOISE_COLUMNS = ["id", "path", "duration", "source", "licence_note"]
RIR_COLUMNS = ["id", "path", "split", "source", "t60", "t60_target", "direct_idx", "early_end_idx", "drr_db",
               "num_samples", "room", "room_dims", "src_pos", "mic_pos", "distance_m", "seed", "licence_note"]


def rel(p: Path) -> str:
    return os.path.relpath(p, REPO_ROOT)


# ----------------------------------------------------------------------------- download


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, meta: dict, allow_unpinned: bool) -> str:
    """Verify checksum; returns a short description of what was verified."""
    if meta.get("md5"):
        got = _hash_file(path, "md5")
        if got != meta["md5"]:
            raise RuntimeError(f"MD5 mismatch for {path.name}: got {got}, expected {meta['md5']}")
        return f"md5 {got} (published)"
    got = _hash_file(path, "sha256")
    if meta.get("sha256"):
        if got != meta["sha256"]:
            msg = f"SHA-256 mismatch for {path.name}: got {got}, pinned {meta['sha256']}"
            if not allow_unpinned:
                raise RuntimeError(msg + " (pass --allow-unpinned to accept)")
            print(f"[warn] {msg}", file=sys.stderr)
        return f"sha256 {got} (pinned)"
    print(f"[warn] {path.name}: no published checksum and none pinned; sha256 = {got}", file=sys.stderr)
    return f"sha256 {got} (unpinned)"


def download(name: str, dl_dir: Path, allow_unpinned: bool) -> Path:
    meta = SOURCES[name]
    dest = dl_dir / meta["file"]
    ok = dl_dir / f"{meta['file']}.ok"
    if dest.exists() and ok.exists():
        return dest
    part = dl_dir / f"{meta['file']}.part"
    size = meta["size"]
    if not dest.exists():
        for attempt in range(1, 7):
            url = meta["urls"][(attempt - 1) % len(meta["urls"])]
            have = part.stat().st_size if part.exists() else 0
            if size is not None and have == size:
                break
            if size is not None and have > size:
                part.unlink()
                have = 0
            headers = dict(UA)
            if have and size is not None:
                headers["Range"] = f"bytes={have}-"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                    if r.status_code == 416 and size is not None and have == size:
                        break
                    if r.status_code == 200 and have:  # server ignored Range (or size unknown): restart
                        have = 0
                        part.unlink(missing_ok=True)
                    elif r.status_code not in (200, 206):
                        r.raise_for_status()
                    if "text/html" in r.headers.get("Content-Type", ""):
                        raise RuntimeError(f"got HTML instead of an archive from {url}")
                    with open(part, "ab") as f, tqdm(total=size, initial=have, unit="B", unit_scale=True,
                                                     unit_divisor=1024, desc=meta["file"], dynamic_ncols=True) as bar:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            bar.update(len(chunk))
            except (requests.RequestException, OSError) as e:
                print(f"[warn] {meta['file']}: attempt {attempt} ({url}) failed: {e}", file=sys.stderr)
                continue
            if size is None or part.stat().st_size == size:
                break
        else:
            raise RuntimeError(f"failed to download {meta['file']}")
        if size is not None and part.stat().st_size != size:
            raise RuntimeError(f"{meta['file']}: size {part.stat().st_size} != expected {size}")
        part.rename(dest)
    elif size is not None and dest.stat().st_size != size:
        raise RuntimeError(f"{dest}: size {dest.stat().st_size} != expected {size}; delete it and re-run")
    print(f"verifying {meta['file']} ...")
    try:
        what = _verify(dest, meta, allow_unpinned)
    except RuntimeError:
        if meta.get("md5"):
            dest.rename(part.with_suffix(".bad"))
        raise
    ok.write_text(what + "\n")
    print(f"  {meta['file']}: {what}")
    return dest


# ----------------------------------------------------------------------------- safe extraction


def _check_member_name(name: str, root: Path) -> Path:
    parts = Path(name).parts
    if not parts or os.path.isabs(name) or ".." in parts:
        raise RuntimeError(f"unsafe archive path: {name!r}")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RuntimeError(f"unsafe archive path: {name!r}")
    return target


def _write_stream(src, target: Path, size: int) -> None:
    if target.exists() and target.stat().st_size == size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    tmp.rename(target)


def extract_zip(zip_path: Path, dest: Path, keep=lambda name: True) -> int:
    """Extract regular files only (no links, no absolute/parent paths, no macOS junk)."""
    marker = dest / f".complete_{zip_path.name}"
    if marker.exists():
        return int(marker.read_text() or 0)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        for info in tqdm(infos, desc=f"extract {zip_path.name}", dynamic_ncols=True):
            name = info.filename
            if "__MACOSX" in Path(name).parts or Path(name).name.startswith("._") or not keep(name):
                continue
            mode = info.external_attr >> 16
            if mode and not stat.S_ISREG(mode):
                continue
            target = _check_member_name(name, dest)
            with zf.open(info) as src:
                _write_stream(src, target, info.file_size)
            n += 1
    marker.write_text(str(n))
    return n


def extract_tar(tar_path: Path, dest: Path) -> int:
    marker = dest / f".complete_{tar_path.name}"
    if marker.exists():
        return int(marker.read_text() or 0)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, mode="r|gz") as tf:
        for m in tqdm(tf, desc=f"extract {tar_path.name}", unit=" files", dynamic_ncols=True):
            if not m.isreg():
                continue
            target = _check_member_name(m.name, dest)
            _write_stream(tf.extractfile(m), target, m.size)
            n += 1
    marker.write_text(str(n))
    return n


# ----------------------------------------------------------------------------- audio helpers


def _load_mono(path: str | io.BytesIO) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    return x if sr == TARGET_SR else soxr.resample(x, sr, TARGET_SR, quality="VHQ")


def _to_pcm16(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


def _write_atomic(path: Path, y: np.ndarray, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.wav")
    sf.write(tmp, y, TARGET_SR, subtype=subtype)
    tmp.rename(path)


def _convert_job(job: tuple[str, str, bool]) -> tuple[str, int]:
    """Resample one file to 16 kHz mono PCM16. Optionally trims digital-silence padding."""
    src, dst, trim = job
    dst_p = Path(dst)
    if dst_p.exists():
        return dst, sf.info(dst).frames
    x, sr = _load_mono(src)
    if trim:  # ESC-50 clips are zero-padded to 5 s; drop the padding
        nz = np.flatnonzero(np.abs(x) > 1e-4)
        x = x[nz[0]: nz[-1] + 1] if len(nz) else x[:0]
    y = _to_pcm16(_resample(x, sr))
    _write_atomic(dst_p, y)
    return dst, len(y)


def _run_pool(fn, jobs, workers: int, desc: str):
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        return list(tqdm(ex.map(fn, jobs, chunksize=8), total=len(jobs), desc=desc, dynamic_ncols=True))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- MUSAN


def _is_music_candidate(name: str) -> bool:
    # deterministic ~1/12 subset of music files, decided from the name alone (stream order unknown)
    return zlib.crc32(Path(name).name.encode()) % 12 == 0


def _music_excerpt(data: bytes) -> np.ndarray:
    x, sr = _load_mono(io.BytesIO(data))
    x = _resample(x, sr)
    n = int(MUSIC_SECONDS * TARGET_SR)
    if len(x) > n:
        start = min(30 * TARGET_SR, (len(x) - n) // 2)
        x = x[start: start + n]
    return x


def extract_musan(tar_path: Path, raw: Path) -> dict[str, np.ndarray]:
    """One streaming pass over musan.tar.gz: extract musan/noise/** + top-level docs + music
    ANNOTATIONS to <raw>; return 10 s excerpts of a deterministic subset of music files.
    Speech and the bulk of the music are never written to disk."""
    marker = raw / ".complete_musan_noise"
    music: dict[str, np.ndarray] = {}
    n = 0
    raw.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r|gz") as tf:
        bar = tqdm(tf, desc="stream musan.tar.gz", unit=" members", dynamic_ncols=True)
        for m in bar:
            if not m.isreg():
                continue
            parts = Path(m.name).parts
            if len(parts) < 2 or parts[0] != "musan":
                continue
            sub = parts[1]
            if sub == "speech":
                continue
            if sub == "noise" or len(parts) == 2 or (sub == "music" and parts[-1] in ("ANNOTATIONS", "LICENSE")):
                target = _check_member_name(m.name, raw)
                _write_stream(tf.extractfile(m), target, m.size)
                n += 1
            elif sub == "music" and m.name.endswith(".wav") and _is_music_candidate(m.name):
                _check_member_name(m.name, raw)
                music[Path(m.name).stem] = _music_excerpt(tf.extractfile(m).read())
            bar.set_postfix(noise_files=n, music=len(music))
    marker.write_text(str(n))
    return music


def _read_annotations(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.split()
        if t:
            out[t[0]] = t[1:]
    return out


def write_music_subset(out: Path, raw: Path, music: dict[str, np.ndarray]) -> list[dict]:
    """Pick ~N_MUSIC excerpts (about half with vocals) for the failure gallery."""
    dst = out / "failures_src" / "music"
    vocals: dict[str, bool] = {}
    for ann in (raw / "musan" / "music").glob("*/ANNOTATIONS"):
        for k, toks in _read_annotations(ann).items():
            vocals[k] = len(toks) > 1 and toks[1].upper() == "Y"  # "<id> <genre(s)> <vocals Y/N> <artist> ..."
    rng = np.random.default_rng(17)
    ids = sorted(music)
    voc = [i for i in ids if vocals.get(i)]
    ins = [i for i in ids if not vocals.get(i)]
    rng.shuffle(voc)
    rng.shuffle(ins)
    n_voc = min(len(voc), N_MUSIC // 2)
    chosen = sorted(voc[:n_voc] + ins[: N_MUSIC - n_voc])
    rows = []
    for k in chosen:
        x = music[k]
        rms = np.sqrt(np.mean(x**2)) + 1e-12
        x = x * min(10 ** (-20 / 20) / rms, 0.9 / (np.max(np.abs(x)) + 1e-12))  # -20 dBFS RMS, peak <= 0.9
        p = dst / f"musan_{k}.wav"
        if not p.exists():
            _write_atomic(p, _to_pcm16(x))
        rows.append({"id": f"musan_{k}", "path": rel(p), "duration": f"{len(x) / TARGET_SR:.3f}",
                     "vocals": "Y" if vocals.get(k) else "N", "source": f"musan/music/{k.split('-')[1]}",
                     "licence_note": "MUSAN music, CC BY 4.0 corpus; failure gallery only, NOT for training"})
    write_csv(dst / "music.csv", rows, ["id", "path", "duration", "vocals", "source", "licence_note"])
    return rows


def _musan_noise_meta(noise_root: Path) -> dict[str, dict]:
    """Per-file background/foreground tag (ANNOTATIONS) and licence (LICENSE) for MUSAN noise."""
    meta: dict[str, dict] = defaultdict(dict)
    for ann in noise_root.glob("*/ANNOTATIONS"):
        tag = ""
        for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("background"):
                tag = "background"
            elif line.lower().startswith("foreground"):
                tag = "foreground"
            elif line.startswith("noise-"):
                meta[line.split()[0]]["fg_bg"] = tag
    lic = noise_root / "sound-bible" / "LICENSE"
    if lic.exists():
        for block in re.split(r"^=+\s*$", lic.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE):
            lines = [x.strip() for x in block.strip().splitlines() if x.strip()]
            if not lines or not lines[0].startswith("noise-"):
                continue
            lic_s = next((x.split(":", 1)[1].strip() for x in lines if x.startswith("License:")), "unknown")
            by = next((x.removeprefix("Recorded by").strip() for x in lines if x.startswith("Recorded by")), "")
            url = next((x for x in lines if x.startswith("http")), "")
            meta[lines[0]]["licence"] = f"soundbible.com, {lic_s}" + (f", by {by}" if by else "") + (f", {url}" if url else "")
    return meta


def prepare_musan(out: Path, dl: Path, workers: int, allow_unpinned: bool) -> None:
    tar_path = download("musan", dl, allow_unpinned)
    raw = out / "raw" / "musan"
    music_dir = out / "failures_src" / "music"
    have_music = len(list(music_dir.glob("musan_*.wav"))) >= N_MUSIC
    if not (raw / ".complete_musan_noise").exists() or not have_music:
        music = extract_musan(tar_path, raw)
        write_music_subset(out, raw, music)
    noise_root = raw / "musan" / "noise"
    srcs = sorted(p for p in noise_root.rglob("*.wav") if not p.name.startswith("._"))
    meta = _musan_noise_meta(noise_root)
    dst = out / "noise" / "musan16k"
    jobs = [(str(p), str(dst / f"{p.stem}.wav"), False) for p in srcs]
    res = dict(_run_pool(_convert_job, jobs, workers, "musan noise -> 16k"))
    rows = []
    for p in srcs:
        d = dst / f"{p.stem}.wav"
        sub = p.parent.name
        m = meta.get(p.stem, {})
        lic = m.get("licence") or ("freesound.org, public domain (per MUSAN free-sound/LICENSE)" if sub == "free-sound"
                                   else f"see musan/noise/{sub}/LICENSE")
        rows.append({"id": p.stem, "path": rel(d), "duration": f"{res[str(d)] / TARGET_SR:.3f}",
                     "source": f"musan/noise/{sub}", "licence_note": f"MUSAN (CC BY 4.0 corpus); {lic}; train only",
                     "fg_bg": m.get("fg_bg") or ("foreground" if sub == "free-sound" else "")})
    write_csv(out / "manifests" / "noise_train.csv", rows, NOISE_COLUMNS + ["fg_bg"])


# ----------------------------------------------------------------------------- ESC-50


OVERLAP_SR = 2000
OVERLAP_SCREEN = 0.3  # 2 kHz NCC above which a candidate is re-checked at 16 kHz
OVERLAP_THRESHOLD = 0.5  # 16 kHz full-band NCC at the matched offset => same recording


def _low_rate(path: str) -> np.ndarray:
    x, sr = _load_mono(path)
    return soxr.resample(x, sr, OVERLAP_SR, quality="HQ").astype(np.float32)


def _ncc16(job: tuple[str, str, int]) -> float:
    """Full-band NCC between an ESC clip and a MUSAN file around a 2 kHz match offset (+-5 ms)."""
    esc_path, musan_path, off2k = job
    t, _ = _load_mono(esc_path)
    m, _ = _load_mono(musan_path)
    t = t - t.mean()
    L, c0, w = len(t), off2k * (TARGET_SR // OVERLAP_SR), int(0.005 * TARGET_SR)
    a = max(0, c0 - w)
    seg = m[a: c0 + w + L]
    if len(seg) < L or not np.any(t):
        return 0.0
    c = np.correlate(seg, t, mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(seg**2)])
    local = np.maximum(np.sqrt(cs[L:] - cs[:-L]), 3e-4 * np.sqrt(L))
    return float(np.max(np.abs(c) / (np.linalg.norm(t) * local)))


def musan_esc_overlap(musan_paths: list[str], esc: dict[str, str], cache: Path, workers: int) -> dict[str, tuple[float, str]]:
    """Content-based leakage guard. MUSAN noise and ESC-50 both come from freesound.org and MUSAN
    lists no source ids, so search every ESC clip inside all MUSAN noise with a normalized
    cross-correlation at 2 kHz. Returns {esc_id: (max NCC, best MUSAN id)}; cached in `cache`."""
    import scipy.fft as sfft

    done: dict[str, dict] = {}
    if cache.exists():
        done = {r["esc_id"]: r for r in read_csv(cache) if "ncc_16k" in r}
    todo = [k for k in sorted(esc) if k not in done]
    if not todo:
        return {k: (float(r["ncc_16k"] or r["ncc_2k"]), r["musan_id"]) for k, r in done.items()}
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        segs = list(tqdm(ex.map(_low_rate, musan_paths, chunksize=8), total=len(musan_paths),
                         desc="MUSAN @2 kHz", dynamic_ncols=True))
        tmpl = dict(zip(todo, ex.map(_low_rate, [esc[k] for k in todo], chunksize=8)))
    gap = 6 * OVERLAP_SR  # > longest ESC clip, so no match can straddle two MUSAN files
    starts, parts, pos = [], [], 0
    for x in segs:
        starts.append(pos)
        parts += [x, np.zeros(gap, np.float32)]
        pos += len(x) + gap
    m = np.concatenate(parts)
    starts = np.array(starts)
    nfft = sfft.next_fast_len(len(m) + gap, real=True)
    M = sfft.rfft(m, nfft, workers=workers)
    cs = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
    for k in tqdm(todo, desc="ESC vs MUSAN xcorr", dynamic_ncols=True):
        t = tmpl[k].astype(np.float64)
        t = t - t.mean()
        L = len(t)
        tn = np.linalg.norm(t)
        if L < OVERLAP_SR // 4 or tn == 0:
            done[k] = {"esc_id": k, "ncc_2k": "0", "musan_id": "", "offset_s": "", "ncc_16k": ""}
            continue
        c = sfft.irfft(M * sfft.rfft(t[::-1].astype(np.float32), nfft, workers=workers), nfft, workers=workers)
        c = c[L - 1: L - 1 + len(m) - L + 1]
        local = np.sqrt(np.maximum(cs[L:] - cs[:-L], 0.0))
        local = np.maximum(local, 3e-4 * np.sqrt(L))  # ignore digital-silence windows
        ncc = np.abs(c) / (tn * local)
        j = int(np.argmax(ncc))
        owner = int(np.searchsorted(starts, j, side="right") - 1)
        done[k] = {"esc_id": k, "ncc_2k": f"{ncc[j]:.4f}", "musan_id": Path(musan_paths[owner]).stem,
                   "offset_s": f"{(j - starts[owner]) / OVERLAP_SR:.4f}", "ncc_16k": "", "_owner": owner,
                   "_off": int(j - starts[owner])}
    cand = [k for k in todo if float(done[k]["ncc_2k"]) >= OVERLAP_SCREEN]
    jobs = [(esc[k], musan_paths[done[k]["_owner"]], done[k]["_off"]) for k in cand]
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        for k, v in zip(cand, tqdm(ex.map(_ncc16, jobs), total=len(jobs), desc="16 kHz re-check", dynamic_ncols=True)):
            done[k]["ncc_16k"] = f"{v:.4f}"
    write_csv(cache, [done[k] for k in sorted(done)], ["esc_id", "ncc_2k", "musan_id", "offset_s", "ncc_16k"])
    return {k: (float(r["ncc_16k"] or r["ncc_2k"]), r["musan_id"]) for k, r in done.items()}


def prepare_esc50(out: Path, dl: Path, workers: int, allow_unpinned: bool) -> None:
    zp = download("esc50", dl, allow_unpinned)
    raw = out / "raw" / "esc50"
    extract_zip(zp, raw)
    root = raw / f"ESC-50-{ESC_COMMIT}"
    meta = read_csv(root / "meta" / "esc50.csv")
    cats = {r["category"] for r in meta}
    missing = [c for c in ESC_CLASSES if c not in cats]
    if missing:
        raise SystemExit(f"ESC-50 classes not found: {missing}")
    sel = [r for r in meta if r["category"] in ESC_CLASSES]
    dst = out / "noise" / "esc50_16k"
    jobs = [(str(root / "audio" / r["filename"]), str(dst / f"esc_{Path(r['filename']).stem}.wav"), True) for r in sel]
    res = dict(_run_pool(_convert_job, jobs, workers, "esc50 -> 16k"))

    # Leakage guard against the training noise corpus (MUSAN must be prepared first).
    nt = out / "manifests" / "noise_train.csv"
    if not nt.exists():
        raise SystemExit("noise_train.csv missing: run the 'musan' step before 'esc50' (overlap guard)")
    musan_paths = [str(REPO_ROOT / r["path"]) for r in read_csv(nt)]
    overlap = musan_esc_overlap(musan_paths, {Path(d).stem: d for _, d, _ in jobs},
                                out / "manifests" / "esc50_musan_overlap.csv", workers)
    rows, dropped = [], []
    for r, (_, d, _) in zip(sel, jobs):
        n = res[d]
        ncc, mid = overlap.get(Path(d).stem, (0.0, ""))
        if ncc >= OVERLAP_THRESHOLD:
            dropped.append((Path(d).stem, mid, ncc))
            Path(d).unlink(missing_ok=True)
            continue
        if n < TARGET_SR // 4:
            print(f"[warn] ESC clip {r['filename']} shorter than 0.25 s after trimming; skipped", file=sys.stderr)
            continue
        rows.append({"id": Path(d).stem, "path": rel(Path(d)), "duration": f"{n / TARGET_SR:.3f}",
                     "source": "esc50", "licence_note": "ESC-50, CC BY-NC 3.0; test only",
                     "category": r["category"], "fold": r["fold"], "freesound_id": r["src_file"]})
    print(f"ESC-50 vs MUSAN content overlap: {len(dropped)} clip(s) with 16 kHz NCC >= {OVERLAP_THRESHOLD} dropped "
          f"(see manifests/esc50_musan_overlap.csv): " + ", ".join(f"{a}~{b} ({c:.2f})" for a, b, c in dropped))
    write_csv(out / "manifests" / "noise_ood.csv", rows, NOISE_COLUMNS + ["category", "fold", "freesound_id"])


# ----------------------------------------------------------------------------- LibriSpeech


def prepare_librispeech(out: Path, dl: Path, workers: int, allow_unpinned: bool) -> None:
    tp = download("librispeech", dl, allow_unpinned)
    raw = out / "raw" / "librispeech"
    extract_tar(tp, raw)
    root = raw / "LibriSpeech"
    gender = {}
    for line in (root / "SPEAKERS.TXT").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(";"):
            continue
        t = [s.strip() for s in line.split("|")]
        if len(t) >= 3 and t[2] == "test-clean":
            gender[t[0]] = t[1]
    srcs = sorted((root / "test-clean").rglob("*.flac"))
    dst = out / "talkers" / "librispeech_test_clean16k"
    jobs = [(str(p), str(dst / f"{p.stem}.wav"), False) for p in srcs]
    res = dict(_run_pool(_convert_job, jobs, workers, "librispeech -> 16k"))
    rows = []
    for p, (_, d, _) in zip(srcs, jobs):
        spk, chap, _ = p.stem.split("-")
        rows.append({"id": p.stem, "speaker": spk, "gender": gender.get(spk, ""), "chapter": chap, "path": rel(Path(d)),
                     "duration": f"{res[d] / TARGET_SR:.3f}", "source": "librispeech/test-clean",
                     "licence_note": "LibriSpeech, CC BY 4.0; test only"})
    write_csv(out / "manifests" / "talkers_test.csv", rows,
              ["id", "speaker", "gender", "chapter", "path", "duration", "source", "licence_note"])


# ----------------------------------------------------------------------------- RIRs


def _edc_trim(h: np.ndarray, floor_db: float = -70.0) -> np.ndarray:
    edc = np.cumsum((h**2)[::-1])[::-1]
    keep = np.flatnonzero(edc > edc[0] * 10 ** (floor_db / 10))
    return h[: keep[-1] + 1] if len(keep) else h


def _t60_schroeder(h: np.ndarray, fs: int = TARGET_SR, start_idx: int = 0) -> float:
    """T60 from the Schroeder decay (T30: -5 to -35 dB, linear fit, extrapolated to 60 dB);
    falls back to T20 when the dynamic range is too small."""
    e = h[start_idx:].astype(np.float64) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
    for lo, hi in ((-5.0, -35.0), (-5.0, -25.0)):
        idx = np.flatnonzero((edc_db <= lo) & (edc_db >= hi))
        if len(idx) > 10 and edc_db[-1] < hi:
            t = idx / fs
            slope = np.polyfit(t, edc_db[idx], 1)[0]
            if slope < 0:
                return float(-60.0 / slope)
    return float("nan")


def _drr_db(h: np.ndarray, d: int, fs: int = TARGET_SR) -> float:
    w = int(0.0025 * fs)
    a, b = max(0, d - w), d + w + 1
    direct = np.sum(h[a:b] ** 2)
    rest = np.sum(h**2) - direct
    return float(10 * np.log10(direct / max(rest, 1e-30)))


def _sim_rir(job: tuple[str, int, str]) -> dict:
    import pyroomacoustics as pra

    split, idx, out_path = job
    rng = np.random.default_rng([RIR_SEEDS[split], idx])
    c = pra.constants.get("c")
    half = pra.constants.get("frac_delay_length") // 2
    for _attempt in range(200):
        t60 = rng.uniform(0.1, 0.8)
        L = rng.uniform(3.0, 10.0)
        W = rng.uniform(2.5, min(L, 8.0))
        H = rng.uniform(2.4, 4.0)
        dims = np.array([L, W, H])
        mic = np.array([rng.uniform(0.5, L - 0.5), rng.uniform(0.5, W - 0.5), rng.uniform(0.8, 1.8)])
        dist = rng.uniform(0.3, 4.0)
        az = rng.uniform(0, 2 * np.pi)
        src = mic + np.array([dist * np.cos(az), dist * np.sin(az), rng.uniform(-0.4, 0.6)])
        if not (np.all(src[:2] >= 0.5) and np.all(src[:2] <= dims[:2] - 0.5) and 1.0 <= src[2] <= 2.0):
            continue
        try:
            e_abs, max_order = pra.inverse_sabine(t60, dims)
        except ValueError:  # T60 not achievable in this room
            continue
        room = pra.ShoeBox(dims, fs=TARGET_SR, materials=pra.Material(e_abs), max_order=min(max_order, 80),
                           air_absorption=False)
        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float64)
        d_true = float(np.linalg.norm(src - mic))
        direct_idx = round(d_true / c * TARGET_SR) + half
        h = _edc_trim(h)
        t60_meas = _t60_schroeder(h, start_idx=direct_idx)
        # rejection: Sabine and ISM disagree for elongated rooms; keep only rooms whose
        # measured T60 is in range and close to the drawn target
        if not (0.08 <= t60_meas <= 0.9) or abs(t60_meas - t60) > 0.25 * t60 + 0.05:
            continue
        h = h / np.max(np.abs(h))
        if not Path(out_path).exists():
            _write_atomic(Path(out_path), h.astype(np.float32), subtype="FLOAT")
        return {
            "id": Path(out_path).stem, "path": rel(Path(out_path)), "split": split, "source": "sim",
            "t60": f"{t60_meas:.3f}", "t60_target": f"{t60:.3f}", "direct_idx": direct_idx,
            "early_end_idx": direct_idx + int(0.05 * TARGET_SR), "drr_db": f"{_drr_db(h, direct_idx):.2f}",
            "num_samples": len(h), "room": f"sim_{split}_{idx:05d}", "room_dims": " ".join(f"{v:.3f}" for v in dims),
            "src_pos": " ".join(f"{v:.3f}" for v in src), "mic_pos": " ".join(f"{v:.3f}" for v in mic),
            "distance_m": f"{d_true:.3f}", "seed": f"{RIR_SEEDS[split]}:{idx}",
            "licence_note": LICENCES["sim"],
        }
    raise RuntimeError(f"RIR {split}/{idx}: no valid room after 200 draws")


def prepare_rir_sim(out: Path, n_train: int, n_test: int, workers: int) -> dict[str, list[dict]]:
    dst = out / "rir" / "sim"
    jobs = [("train", i, str(dst / f"sim_tr_{i:05d}.wav")) for i in range(n_train)]
    jobs += [("test", i, str(dst / f"sim_te_{i:05d}.wav")) for i in range(n_test)]
    rows = _run_pool(_sim_rir, jobs, workers, "simulate RIRs")
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "test")}


def _noise_floor_cut(h: np.ndarray, d: int, fs: int = TARGET_SR) -> np.ndarray:
    """Cut a measured RIR where its 10 ms energy envelope reaches the measurement noise floor
    (+3 dB). Without this the Schroeder integral of the noise tail inflates T60."""
    w = int(0.01 * fs)
    env = 10 * np.log10(np.convolve(h**2, np.ones(w) / w, mode="same") + 1e-30)
    floor = np.median(env[-max(len(env) // 10, w):])
    after = np.flatnonzero(env[d:] < floor + 3.0)
    return h[: d + after[0]] if len(after) and after[0] > int(0.02 * fs) else h


def _real_rir_job(job: tuple[str, str, str, str, int]) -> dict | None:
    _rid, room, src, out_path, channel = job
    x, sr = sf.read(src, dtype="float64", always_2d=True)
    h = _resample(x[:, min(channel, x.shape[1] - 1)], sr)
    # Direct path = first local maximum reaching 50 % of the global peak (in low-DRR rooms a
    # reflection can be the global maximum). Measured RIRs have a variable pre-delay; keep 2.5 ms
    # before the direct path.
    a = np.abs(h)
    onset = int(np.flatnonzero(a >= 0.5 * a.max())[0])
    peak = onset + int(np.argmax(a[onset: onset + int(0.001 * TARGET_SR) + 1]))
    start = max(0, peak - int(0.0025 * TARGET_SR))
    h = h[start:]
    d = peak - start
    h = _edc_trim(_noise_floor_cut(h, d))
    h = h / np.max(np.abs(h))
    t60 = _t60_schroeder(h, start_idx=d)
    if not (np.isfinite(t60) and 0.05 <= t60 <= 3.0):
        return None
    if not Path(out_path).exists():
        _write_atomic(Path(out_path), h.astype(np.float32), subtype="FLOAT")
    db = Path(src).stem.split("_")[0]  # RVB2014 | RWCP | air
    return {"id": Path(out_path).stem, "path": rel(Path(out_path)), "split": "test", "source": f"openslr28/{db}",
            "t60": f"{t60:.3f}", "t60_target": "", "direct_idx": d, "early_end_idx": d + int(0.05 * TARGET_SR),
            "drr_db": f"{_drr_db(h, d):.2f}", "num_samples": len(h), "room": room, "room_dims": "", "src_pos": "", "mic_pos": "",
            "distance_m": "", "seed": "", "licence_note": "OpenSLR 28 (Apache 2.0 package); " + {
                "RWCP": "RWCP: research and development use only",
                "air": "AIR: licence not stated in the OpenSLR 20 mirror",
                "RVB2014": "REVERB 2014 challenge RIRs: research use",
            }.get(db, "") + "; test only, never redistributed"}


def _room_group(room: str) -> str:
    """Collapse rir_list room ids into physical rooms (e.g. RWCP angle files, AIR positions)."""
    if re.fullmatch(r"RWCP_[mp]\d+[lr]\.wav", room) or room == "RWCP_000l.wav" or room == "RWCP_000r.wav":
        return "RWCP_angles"
    m = re.fullmatch(r"air_binaural_([a-z]+(?:_[a-z]+)*)(?:_\d+)*", room)
    return f"air_binaural_{m.group(1)}" if m else room


def prepare_rir_real(out: Path, dl: Path, workers: int, allow_unpinned: bool, per_room: int = 2) -> list[dict]:
    """A small, room-stratified subset of the OpenSLR 28 measured RIRs (<= per_room per physical room)."""
    zp = download("rirs_noises", dl, allow_unpinned)
    raw = out / "raw" / "rirs_noises"
    extract_zip(zp, raw, keep=lambda n: "/real_rirs_isotropic_noises/" in n or n.count("/") <= 1)
    rir_list = raw / "RIRS_NOISES" / "real_rirs_isotropic_noises" / "rir_list"
    groups: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    # rir_list lines: "--rir-id <id> --room-id <room> RIRS_NOISES/real_rirs_isotropic_noises/<file>.wav"
    for line in rir_list.read_text().splitlines():
        m = re.search(r"--rir-id\s+(\S+)\s+--room-id\s+(\S+)\s+\S*(real_rirs_isotropic_noises/\S+\.wav)", line)
        if m and (raw / "RIRS_NOISES" / m.group(3)).exists():
            groups[_room_group(m.group(2))].append((m.group(1), m.group(2), raw / "RIRS_NOISES" / m.group(3)))
    rng = np.random.default_rng(28)
    jobs = []
    for g in sorted(groups):
        items = sorted(groups[g])
        for k in sorted(rng.permutation(len(items))[:per_room]):
            rid, room, f = items[k]
            jobs.append((rid, room, str(f), str(out / "rir" / "real" / f"real_{f.stem}.wav"), 0))
    rows = [r for r in _run_pool(_real_rir_job, jobs, workers, "real RIRs") if r]
    print(f"real RIRs: {len(rows)} of {len(jobs)} selected ({len(groups)} rooms, <= {per_room} per room)")
    return rows


# ----------------------------------------------------------------------------- self-check


def _audio_hash(path: str) -> tuple[str, str]:
    x, _ = sf.read(path, dtype="int16" if sf.info(path).subtype == "PCM_16" else "float32")
    return path, hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()


def separation_check(out: Path, workers: int) -> bool:
    m = out / "manifests"
    need = {"noise_train": "train", "rir_train": "train", "noise_ood": "test", "talkers_test": "test", "rir_test": "test"}
    rows = {k: read_csv(m / f"{k}.csv") for k in need if (m / f"{k}.csv").exists()}
    missing = sorted(set(need) - set(rows))
    print("\n=== Corpus separation self-check ===")
    ok = True
    if missing:
        print(f"[FAIL] missing manifests: {missing}")
        ok = False
    train = [(k, r) for k, rs in rows.items() if need[k] == "train" for r in rs]
    test = [(k, r) for k, rs in rows.items() if need[k] == "test" for r in rs]

    def corpus(r):
        return r["source"].split("/")[0]

    tr_corp = {corpus(r) for _, r in train}
    te_corp = {corpus(r) for _, r in test}
    shared = tr_corp & te_corp
    print(f"train corpora: {sorted(tr_corp)}   test corpora: {sorted(te_corp)}")
    if shared - {"sim"}:
        ok = False
        print(f"[FAIL] corpora shared between train and test: {sorted(shared - {'sim'})}")
    else:
        print(f"shared corpora: {sorted(shared)} (allowed: simulated RIRs, split by seed)")
    for label, key in (("ids", "id"), ("paths", "path")):
        inter = {r[key] for _, r in train} & {r[key] for _, r in test}
        if inter:
            ok = False
            print(f"[FAIL] {len(inter)} {label} in both train and test, e.g. {sorted(inter)[:3]}")
        else:
            print(f"no shared {label}")
    # simulated RIRs: disjoint seed streams and parameter tuples
    sim_tr = [r for _, r in train if r["source"] == "sim"]
    sim_te = [r for _, r in test if r["source"] == "sim"]
    seeds_tr = {r["seed"].split(":")[0] for r in sim_tr}
    seeds_te = {r["seed"].split(":")[0] for r in sim_te}
    params = lambda r: (r["room_dims"], r["src_pos"], r["mic_pos"])
    pshared = {params(r) for r in sim_tr} & {params(r) for r in sim_te}
    dshared = {r["room_dims"] for r in sim_tr} & {r["room_dims"] for r in sim_te}
    if seeds_tr & seeds_te or pshared or dshared:
        ok = False
        print(f"[FAIL] simulated RIR seeds/params overlap: seeds {seeds_tr & seeds_te}, {len(pshared)} tuples, {len(dshared)} rooms")
    else:
        print(f"simulated RIRs: seed streams train {sorted(seeds_tr)} vs test {sorted(seeds_te)}; no shared room or geometry")
    # content overlap between MUSAN (train) and ESC-50 (test), computed by prepare_esc50
    ov = m / "esc50_musan_overlap.csv"
    if ov.exists():
        ovr = {r["esc_id"]: float(r["ncc_16k"] or r["ncc_2k"]) for r in read_csv(ov)}
        used = [ovr.get(r["id"], float("nan")) for r in rows.get("noise_ood", [])]
        if not used or any(not (u < OVERLAP_THRESHOLD) for u in used):
            ok = False
            print("[FAIL] ESC-50 test clip(s) overlap MUSAN content or were never checked")
        else:
            print(f"ESC-50 test clips vs MUSAN noise: max normalized xcorr {max(used):.3f} (< {OVERLAP_THRESHOLD}, 16 kHz re-check), "
                  f"{len(used)} clips checked")
    else:
        ok = False
        print("[FAIL] esc50_musan_overlap.csv missing (run the esc50 step)")
    # identical audio content across the split
    paths_tr = [str(REPO_ROOT / r["path"]) for _, r in train]
    paths_te = [str(REPO_ROOT / r["path"]) for _, r in test]
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        h = dict(tqdm(ex.map(_audio_hash, paths_tr + paths_te, chunksize=32), total=len(paths_tr) + len(paths_te),
                      desc="hash audio", dynamic_ncols=True))
    dup = {h[p] for p in paths_tr} & {h[p] for p in paths_te}
    if dup:
        ok = False
        print(f"[FAIL] {len(dup)} identical audio payloads in train and test")
    else:
        print(f"no identical audio across train ({len(paths_tr)}) and test ({len(paths_te)}) files")
    # MUSAN music (failure gallery) must not be in the training noise manifest
    music = {p.stem for p in (out / "failures_src" / "music").glob("*.wav")}
    if music & {r["id"] for _, r in train}:
        ok = False
        print("[FAIL] failure-gallery music appears in a training manifest")
    print("Corpus separation: OK" if ok else "Corpus separation: FAILED")
    return ok


# ----------------------------------------------------------------------------- summary


def _fmt_check(path: str) -> tuple[str, int, int, str]:
    i = sf.info(path)
    return path, i.samplerate, i.channels, i.subtype


def summary(out: Path, workers: int) -> bool:
    m = out / "manifests"
    ok = True
    print("\n=== Summary ===")
    all_paths = []
    for name in ("noise_train", "noise_ood", "talkers_test", "rir_train", "rir_test"):
        p = m / f"{name}.csv"
        if not p.exists():
            print(f"{name:13s} (missing)")
            continue
        rows = read_csv(p)
        all_paths += [str(REPO_ROOT / r["path"]) for r in rows]
        if name.startswith("rir"):
            t60 = np.array([float(r["t60"]) for r in rows])
            by = Counter(r["source"].split("/")[0] for r in rows)
            print(f"{name:13s} {len(rows):5d} RIRs  {dict(by)}  T60 {t60.min():.2f}-{t60.max():.2f} s "
                  f"(median {np.median(t60):.2f}; >=0.7 s: {(t60 >= 0.7).sum()})")
        else:
            hrs = sum(float(r["duration"]) for r in rows) / 3600
            extra = ""
            if name == "noise_ood":
                c = Counter(r["category"] for r in rows)
                extra = f"  {len(c)} classes: " + ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))
            elif name == "talkers_test":
                extra = f"  {len({r['speaker'] for r in rows})} speakers ({dict(Counter(r['gender'] for r in rows))} utts)"
            else:
                extra = "  " + str(dict(Counter(r["source"] for r in rows)))
            print(f"{name:13s} {len(rows):5d} files {hrs:6.2f} h{extra}")
    music = sorted((out / "failures_src" / "music").glob("musan_*.wav"))
    all_paths += [str(p) for p in music]
    print(f"failures_src/music  {len(music)} excerpts (failure gallery only)")
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        bad = [r for r in ex.map(_fmt_check, all_paths, chunksize=64)
               if r[1] != TARGET_SR or r[2] != 1 or r[3] not in ("PCM_16", "FLOAT")]
    if bad:
        ok = False
        print(f"[FAIL] {len(bad)} files not 16 kHz mono: {bad[:3]}")
    else:
        print(f"all {len(all_paths)} files are 16 kHz mono (noise/speech PCM16, RIRs float32)")
    print("\nLicences:")
    for k, v in LICENCES.items():
        print(f"  {k:12s} {v}")
    return ok


# ----------------------------------------------------------------------------- main

STEPS = ["musan", "esc50", "librispeech", "rir_sim", "rir_real", "check"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"process pool size (capped at {MAX_WORKERS})")
    ap.add_argument("--only", nargs="+", choices=STEPS, help="run only these steps")
    ap.add_argument("--skip-download", action="store_true", help="fail instead of downloading missing archives")
    ap.add_argument("--allow-unpinned", action="store_true", help="accept archives whose pinned SHA-256 changed")
    ap.add_argument("--n-rir-train", type=int, default=3000)
    ap.add_argument("--n-rir-test", type=int, default=300)
    args = ap.parse_args()
    out = args.out.resolve()
    workers = max(1, min(args.workers, MAX_WORKERS))
    steps = args.only or STEPS
    dl = out / "raw" / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        for name, meta in SOURCES.items():
            if not (dl / meta["file"]).exists() and any(s.startswith(name[:4]) for s in steps):
                print(f"[warn] {meta['file']} not present", file=sys.stderr)
    mdir = out / "manifests"

    if "musan" in steps:
        prepare_musan(out, dl, workers, args.allow_unpinned)
    if "esc50" in steps:  # after MUSAN so the freesound-id overlap guard can run
        prepare_esc50(out, dl, workers, args.allow_unpinned)
    if "librispeech" in steps:
        prepare_librispeech(out, dl, workers, args.allow_unpinned)
    if "rir_sim" in steps or "rir_real" in steps:
        old_test = read_csv(mdir / "rir_test.csv") if (mdir / "rir_test.csv").exists() else []
        if "rir_sim" in steps:
            sim = prepare_rir_sim(out, args.n_rir_train, args.n_rir_test, workers)
            write_csv(mdir / "rir_train.csv", sim["train"], RIR_COLUMNS)
            sim_test = sim["test"]
        else:
            sim_test = [r for r in old_test if r["source"] == "sim"]
        if "rir_real" in steps:
            real = prepare_rir_real(out, dl, workers, args.allow_unpinned)
        else:
            real = [r for r in old_test if r["source"] != "sim"]
        write_csv(mdir / "rir_test.csv", sim_test + real, RIR_COLUMNS)

    ok = summary(out, workers)
    if "check" in steps:
        ok = separation_check(out, workers) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
