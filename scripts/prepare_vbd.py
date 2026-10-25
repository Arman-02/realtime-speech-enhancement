# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26",
#     "soundfile>=0.12",
#     "soxr>=0.3.7",
#     "tqdm>=4.66",
#     "requests>=2.31",
# ]
# ///
"""Download and prepare VoiceBank+DEMAND (28-speaker version) at 16 kHz.

Source: Valentini-Botinhao, C. (2017). Noisy speech database for training speech
enhancement algorithms and TTS models. University of Edinburgh, DataShare.
https://doi.org/10.7488/ds/2117  (handle 10283/2791, CC BY 4.0)

Usage:
    uv run scripts/prepare_vbd.py --out data
    uv run scripts/prepare_vbd.py --out data --skip-download   # use existing archives
    uv run scripts/prepare_vbd.py --out data --samples-only    # only re-pick demo clips

Produces (see PLAN.md section 3.5):
    <out>/raw/downloads/*.zip                       original archives (MD5-verified)
    <out>/raw/vbd/<archive>/...                     extracted 48 kHz originals
    <out>/vbd16k/{train,test}/{clean,noisy}/<id>.wav  16 kHz mono PCM16
    <out>/vbd16k/manifests/{train,val,test}.csv
and (PLAN.md section 3.4) web/public/samples/{<id>_noisy.wav,<id>_clean.wav,samples.json,ATTRIBUTION.txt}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import soxr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "web" / "public" / "samples"

TARGET_SR = 16000
VAL_SPEAKERS = {"p226", "p287"}
EXPECTED_TRAINVAL = 11572
EXPECTED_TEST = 824

DATASHARE = "https://datashare.ed.ac.uk"
HANDLE = "10283/2791"
DATASET_URL = f"{DATASHARE}/handle/{HANDLE}"
DOI_URL = "https://doi.org/10.7488/ds/2117"

# Files we need from the item. Fallback bitstream UUIDs / sizes / MD5s (as listed by
# the DSpace REST API in Oct 2026) are used only if the API lookup fails.
FILES = {
    "clean_trainset_28spk_wav.zip": ("245452b6-6235-44b6-a6f9-e7eb19797769", 2486057279, "d2d5a45ec32f8fcbf201bde0447e20ba"),
    "noisy_trainset_28spk_wav.zip": ("ecb5a102-bb00-46d3-8af5-40c79823b837", 2830205201, "1fca9e8bafb8cd069f6653c6d92f9e9c"),
    "clean_testset_wav.zip": ("dec213d3-bf57-4777-9663-c24bdce92d5e", 154332064, "34eb1c0ba7ef667e9b966866c542fc16"),
    "noisy_testset_wav.zip": ("13c1bfbf-14a6-41db-9b41-8f7310f01ad5", 170578854, "fb1b86caa31e8ba5b506c0c64da9aab5"),
    # log_trainset_28spk.txt and log_testset.txt are distributed inside logfiles.zip
    "logfiles.zip": ("11185dc8-9cf1-405b-b858-35bd6a04aedd", 115932, "0d16f5b6afe12d64238d1e4a0e6ff6e8"),
}

# split -> (clean archive, noisy archive, log file name)
SPLITS = {
    "train": ("clean_trainset_28spk_wav.zip", "noisy_trainset_28spk_wav.zip", "log_trainset_28spk.txt"),
    "test": ("clean_testset_wav.zip", "noisy_testset_wav.zip", "log_testset.txt"),
}

# Log-file noise codes -> readable names. Train noises: 8 DEMAND recordings + 2
# artificial (babble, speech-shaped noise). Test noises: 5 different DEMAND recordings.
# Note: test "cafe" is DEMAND SCAFE (outdoor street cafe), not the train "cafeteria" (PCAFETER).
NOISE_NAMES = {
    # test
    "bus": "bus",
    "cafe": "cafe",
    "living": "living room",
    "office": "office",
    "psquare": "public square",
    # train
    "babble": "babble",
    "cafeteria": "cafeteria",
    "car": "car",
    "kitchen": "kitchen",
    "meeting": "meeting room",
    "metro": "metro",
    "restaurant": "restaurant",
    "ssn": "speech-shaped noise",
    "station": "station",
    "traffic": "traffic",
}

MANIFEST_COLUMNS = ["id", "speaker", "clean_path", "noisy_path", "num_samples", "noise_type", "snr_db"]

UA = {"User-Agent": "ans-prepare-vbd/1.0 (+https://datashare.ed.ac.uk/handle/10283/2791)"}


# ----------------------------------------------------------------------------- download


def resolve_bitstreams(session: requests.Session) -> dict[str, dict]:
    """Map file name -> {url, size, md5} using the DSpace 7 REST API, with fallback."""
    out = {name: {"uuid": u, "size": s, "md5": m} for name, (u, s, m) in FILES.items()}
    try:
        r = session.get(f"{DATASHARE}/server/api/pid/find", params={"id": f"hdl:{HANDLE}"}, timeout=30)
        r.raise_for_status()
        item_uuid = r.json()["uuid"]
        r = session.get(
            f"{DATASHARE}/server/api/core/items/{item_uuid}/bundles",
            params={"embed": "bitstreams", "embed.size": "bitstreams=100"},
            timeout=30,
        )
        r.raise_for_status()
        found = {}
        for bundle in r.json()["_embedded"]["bundles"]:
            if bundle.get("name") != "ORIGINAL":
                continue
            for bs in bundle["_embedded"]["bitstreams"]["_embedded"]["bitstreams"]:
                found[bs["name"]] = {
                    "uuid": bs["uuid"],
                    "size": int(bs["sizeBytes"]),
                    "md5": (bs.get("checkSum") or {}).get("value"),
                }
        for name in FILES:
            if name in found:
                out[name] = found[name]
            else:
                print(f"[warn] {name} not listed by the API; using built-in bitstream id", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] DataShare API lookup failed ({e}); using built-in bitstream ids", file=sys.stderr)
    for meta in out.values():
        meta["url"] = f"{DATASHARE}/server/api/core/bitstreams/{meta['uuid']}/content"
    return out


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(session: requests.Session, name: str, meta: dict, dest_dir: Path, position: int) -> Path:
    dest = dest_dir / name
    ok_marker = dest_dir / f"{name}.md5ok"
    if dest.exists() and ok_marker.exists() and dest.stat().st_size == meta["size"]:
        return dest
    if dest.exists() and dest.stat().st_size == meta["size"]:
        if not meta.get("md5") or md5_file(dest) == meta["md5"]:
            ok_marker.write_text(meta.get("md5") or "")
            return dest
        dest.unlink()

    part = dest_dir / f"{name}.part"
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        if have > meta["size"]:
            part.unlink()
            have = 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with session.get(meta["url"], headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if have and r.status_code == 200:  # server ignored Range: restart
                    have = 0
                    part.unlink(missing_ok=True)
                elif r.status_code not in (200, 206):
                    r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    raise RuntimeError(f"got HTML instead of the archive from {meta['url']}")
                with open(part, "ab") as f, tqdm(
                    total=meta["size"], initial=have, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=name, position=position, leave=True, dynamic_ncols=True,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        except (requests.RequestException, OSError) as e:
            print(f"[warn] {name}: attempt {attempt} failed: {e}", file=sys.stderr)
            continue
        if part.stat().st_size == meta["size"]:
            break
    else:
        raise RuntimeError(f"failed to download {name} after 5 attempts")

    if meta.get("md5"):
        got = md5_file(part)
        if got != meta["md5"]:
            part.unlink()
            raise RuntimeError(f"MD5 mismatch for {name}: got {got}, expected {meta['md5']} (partial file removed)")
    part.rename(dest)
    ok_marker.write_text(meta.get("md5") or "")
    return dest


def download_all(out: Path) -> None:
    dl_dir = out / "raw" / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(UA)
    metas = resolve_bitstreams(session)
    print("Downloading from DataShare:")
    for name, meta in metas.items():
        print(f"  {name:32s} {meta['size'] / 1e6:9.1f} MB  {meta['url']}")
    # Small files first, big ones in parallel.
    order = sorted(metas, key=lambda n: metas[n]["size"])
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(download_one, session, n, metas[n], dl_dir, i): n for i, n in enumerate(order)}
        for fut in as_completed(futs):
            fut.result()
    print("All archives present and verified.")


# ----------------------------------------------------------------------------- extract


def _safe_extract(zip_path: str, dest_dir: str) -> tuple[str, int]:
    """Extract regular files only, refusing absolute/parent paths, links and macOS junk."""
    zip_path, dest = Path(zip_path), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            parts = Path(name).parts
            if info.is_dir() or not parts or "__MACOSX" in parts or Path(name).name.startswith("._"):
                continue
            mode = info.external_attr >> 16
            if mode and not (stat.S_ISREG(mode) or mode & 0o170000 == 0):
                continue  # symlinks / devices
            target = (dest / name).resolve()
            if not target.is_relative_to(root) or os.path.isabs(name) or ".." in parts:
                raise RuntimeError(f"unsafe path in {zip_path.name}: {name!r}")
            if target.exists() and target.stat().st_size == info.file_size:
                n += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            tmp.rename(target)
            n += 1
    return zip_path.name, n


def extract_all(out: Path, workers: int) -> None:
    dl_dir = out / "raw" / "downloads"
    raw = out / "raw" / "vbd"
    jobs = []
    for name in FILES:
        stem = name.removesuffix(".zip")
        dest = raw / stem
        if (dest / ".complete").exists():
            continue
        zp = dl_dir / name
        if not zp.exists():
            raise SystemExit(f"missing archive {zp}; run without --skip-download")
        jobs.append((str(zp), str(dest)))
    if not jobs:
        print("Archives already extracted.")
        return
    print(f"Extracting {len(jobs)} archive(s) to {raw} ...")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        futs = {ex.submit(_safe_extract, zp, d): d for zp, d in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
            name, n = fut.result()
            (Path(futs[fut]) / ".complete").write_text(str(n))
            tqdm.write(f"  {name}: {n} files")


def wav_index(dir_: Path) -> dict[str, Path]:
    idx = {}
    for p in dir_.rglob("*.wav"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        idx[p.stem] = p
    return idx


def read_log(path: Path) -> dict[str, tuple[str, float]]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        uid, code, snr = parts
        try:
            out[uid] = (code, float(snr))
        except ValueError:
            continue
    return out


def find_log(out: Path, log_name: str) -> Path:
    hits = [p for p in (out / "raw" / "vbd").rglob(log_name) if "__MACOSX" not in p.parts]
    if not hits:
        raise SystemExit(f"log file {log_name} not found under {out / 'raw' / 'vbd'}")
    return hits[0]


# ----------------------------------------------------------------------------- resample


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def _to_pcm16(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


def _write_atomic(path: Path, y16: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.wav")
    sf.write(tmp, y16, TARGET_SR, subtype="PCM_16")
    tmp.rename(path)


def _process_pair(job: tuple[str, str, str, str, str]) -> tuple[str, int, int, str]:
    """Resample one clean/noisy pair. Returns (id, n_clean, n_noisy, note)."""
    uid, src_c, src_n, dst_c, dst_n = job
    dst_c, dst_n = Path(dst_c), Path(dst_n)
    if dst_c.exists() and dst_n.exists():
        return uid, sf.info(dst_c).frames, sf.info(dst_n).frames, "skip"
    xc, src_c_sr = _load_mono(src_c)
    xn, src_n_sr = _load_mono(src_n)
    note = "ok"
    if len(xc) != len(xn) or src_c_sr != src_n_sr:
        note = f"src_len_mismatch:{len(xc)}@{src_c_sr}/{len(xn)}@{src_n_sr}"
        if src_c_sr == src_n_sr:  # keep the pair sample-aligned
            n = min(len(xc), len(xn))
            xc, xn = xc[:n], xn[:n]
    outs = []
    for x, sr in ((xc, src_c_sr), (xn, src_n_sr)):
        y = x if sr == TARGET_SR else soxr.resample(x, sr, TARGET_SR, quality="VHQ")
        outs.append(_to_pcm16(y))
    _write_atomic(dst_c, outs[0])
    _write_atomic(dst_n, outs[1])
    return uid, len(outs[0]), len(outs[1]), note


def build_split(out: Path, split: str, workers: int) -> list[dict]:
    clean_zip, noisy_zip, log_name = SPLITS[split]
    raw = out / "raw" / "vbd"
    clean_idx = wav_index(raw / clean_zip.removesuffix(".zip"))
    noisy_idx = wav_index(raw / noisy_zip.removesuffix(".zip"))
    log = read_log(find_log(out, log_name))

    ids = sorted(set(clean_idx) & set(noisy_idx))
    for label, missing in (
        ("clean without noisy", set(clean_idx) - set(noisy_idx)),
        ("noisy without clean", set(noisy_idx) - set(clean_idx)),
        ("pairs without log entry", set(ids) - set(log)),
        ("log entries without audio", set(log) - set(ids)),
    ):
        if missing:
            print(f"[warn] {split}: {len(missing)} {label}: {sorted(missing)[:5]}", file=sys.stderr)

    dst = out / "vbd16k" / split
    (dst / "clean").mkdir(parents=True, exist_ok=True)
    (dst / "noisy").mkdir(parents=True, exist_ok=True)
    jobs = [
        (uid, str(clean_idx[uid]), str(noisy_idx[uid]), str(dst / "clean" / f"{uid}.wav"), str(dst / "noisy" / f"{uid}.wav"))
        for uid in ids
    ]
    results = {}
    notes = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for uid, nc, nn, note in tqdm(
            ex.map(_process_pair, jobs, chunksize=16), total=len(jobs), desc=f"resample {split}", dynamic_ncols=True
        ):
            results[uid] = (nc, nn)
            if note not in ("ok", "skip"):
                notes[note.split(":")[0]].append(uid)
    for k, v in notes.items():
        print(f"[warn] {split}: {len(v)} pairs with {k} (trimmed to common length): {v[:5]}", file=sys.stderr)

    rows = []
    for uid in ids:
        code, snr = log.get(uid, ("unknown", float("nan")))
        rows.append({
            "id": uid,
            "speaker": uid.split("_")[0],
            "clean_path": os.path.relpath(dst / "clean" / f"{uid}.wav", REPO_ROOT),
            "noisy_path": os.path.relpath(dst / "noisy" / f"{uid}.wav", REPO_ROOT),
            "num_samples": results[uid][0],
            "noise_type": NOISE_NAMES.get(code, code),
            "snr_db": f"{snr:g}",
            "_noisy_samples": results[uid][1],
        })
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- checks


def _check_file(path: str) -> tuple[str, int, int, str]:
    info = sf.info(path)
    return path, info.samplerate, info.channels, info.subtype


def sanity_checks(manifests: dict[str, list[dict]], workers: int) -> bool:
    ok = True
    n_trainval = len(manifests["train"]) + len(manifests["val"])
    n_test = len(manifests["test"])
    print("\n=== Sanity checks ===")
    print(f"train+val pairs: {n_trainval} (expected {EXPECTED_TRAINVAL})  test pairs: {n_test} (expected {EXPECTED_TEST})")
    if n_trainval != EXPECTED_TRAINVAL or n_test != EXPECTED_TEST:
        ok = False
        print("[FAIL] unexpected counts")
    val_spk = {r["speaker"] for r in manifests["val"]}
    if val_spk != VAL_SPEAKERS or {r["speaker"] for r in manifests["train"]} & VAL_SPEAKERS:
        ok = False
        print(f"[FAIL] val speakers {sorted(val_spk)} / leakage into train")

    paths = [REPO_ROOT / r[k] for rows in manifests.values() for r in rows for k in ("clean_path", "noisy_path")]
    bad_fmt = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, sr, ch, sub in tqdm(ex.map(_check_file, map(str, paths), chunksize=64), total=len(paths), desc="verify"):
            if sr != TARGET_SR or ch != 1 or sub != "PCM_16":
                bad_fmt.append((p, sr, ch, sub))
    if bad_fmt:
        ok = False
        print(f"[FAIL] {len(bad_fmt)} files not 16 kHz mono PCM16, e.g. {bad_fmt[:3]}")
    else:
        print(f"all {len(paths)} wavs are {TARGET_SR} Hz mono PCM16")

    mism = [r["id"] for rows in manifests.values() for r in rows if "_noisy_samples" in r and r["_noisy_samples"] != r["num_samples"]]
    if mism:
        ok = False
        print(f"[FAIL] {len(mism)} clean/noisy length mismatches: {mism[:5]}")
    else:
        print("clean/noisy lengths equal for every pair")

    unknown = [r["id"] for rows in manifests.values() for r in rows if r["noise_type"] == "unknown"]
    if unknown:
        ok = False
        print(f"[FAIL] {len(unknown)} utterances without noise label")

    print("\n=== Summary ===")
    for split, rows in manifests.items():
        secs = sum(int(r["num_samples"]) for r in rows) / TARGET_SR
        spk = sorted({r["speaker"] for r in rows})
        print(f"\n[{split}] {len(rows)} pairs, {secs / 3600:.2f} h, {len(spk)} speakers"
              + (f" ({', '.join(spk)})" if len(spk) <= 4 else ""))
        by_noise = defaultdict(list)
        for r in rows:
            by_noise[r["noise_type"]].append(float(r["snr_db"]))
        snrs = sorted({float(r["snr_db"]) for r in rows})
        print(f"  SNRs (dB): {', '.join(f'{s:g}' for s in snrs)}")
        for noise in sorted(by_noise):
            v = by_noise[noise]
            dist = " ".join(f"{s:g}:{v.count(s)}" for s in snrs)
            print(f"  {noise:22s} {len(v):5d}   [{dist}]")
    print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
    return ok


# ----------------------------------------------------------------------------- samples

# (noise_type, snr_db) slots for the demo clips: 5 noise types, SNR 2.5-12.5 dB.
SAMPLE_SLOTS = [
    ("cafe", 2.5),
    ("public square", 2.5),
    ("bus", 2.5),
    ("living room", 7.5),
    ("office", 7.5),
    ("cafe", 12.5),
]
NOISE_TITLES = {
    "cafe": "Street cafe",
    "public square": "Public square",
    "living room": "Living room",
    "office": "Office",
    "bus": "Bus",
}

ATTRIBUTION = f"""Demo audio clips in this directory are taken from the VoiceBank+DEMAND test set:

  Valentini-Botinhao, Cassia. (2017). Noisy speech database for training speech
  enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh.
  School of Informatics. Centre for Speech Technology Research (CSTR).
  {DOI_URL}
  {DATASET_URL}

Licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/

The clean speech is from the VCTK / Voice Bank corpus (Veaux, Yamagishi & King, 2013);
the noise is from the DEMAND database (Thiemann, Ito & Vincent, 2013).

Modifications: resampled from 48 kHz to 16 kHz mono (soxr, VHQ) and stored as 16-bit PCM WAV.
"""


def _clip_stats(path: Path) -> tuple[float, float, float]:
    """Return (duration_s, active_ratio, peak) for a clean clip."""
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    frame = sr // 50
    n = len(x) // frame
    if n == 0:
        return len(x) / sr, 0.0, 0.0
    e = 10 * np.log10(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    active = float(np.mean(e > e.max() - 30.0))
    return len(x) / sr, active, float(np.max(np.abs(x)))


def make_samples(out: Path) -> None:
    test_csv = out / "vbd16k" / "manifests" / "test.csv"
    if not test_csv.exists():
        raise SystemExit(f"{test_csv} not found; run the full preparation first")
    rows = read_manifest(test_csv)
    by_slot = defaultdict(list)
    for r in rows:
        by_slot[(r["noise_type"], float(r["snr_db"]))].append(r)

    chosen = []
    used = set()
    speakers = ["p232", "p257"]
    for i, (noise, snr) in enumerate(SAMPLE_SLOTS):
        cands = [r for r in by_slot.get((noise, snr), []) if r["id"] not in used]
        if not cands:
            print(f"[warn] no test clip for {noise} @ {snr} dB", file=sys.stderr)
            continue
        want_spk = speakers[i % 2]
        scored = []
        for r in cands:
            dur = int(r["num_samples"]) / TARGET_SR
            if not 3.0 <= dur <= 6.0:
                continue
            _, active, peak = _clip_stats(REPO_ROOT / r["clean_path"])
            npk = float(np.max(np.abs(sf.read(REPO_ROOT / r["noisy_path"], dtype="float64")[0])))
            if peak >= 0.99 or npk >= 0.99:
                continue  # avoid clipped clips
            score = active - 0.05 * abs(dur - 4.5) + (0.1 if r["speaker"] == want_spk else 0.0)
            scored.append((score, r["id"], r))
        if not scored:
            print(f"[warn] no 3-6 s clip for {noise} @ {snr} dB", file=sys.stderr)
            continue
        scored.sort(key=lambda t: (-t[0], t[1]))
        r = scored[0][2]
        used.add(r["id"])
        chosen.append(r)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    keep = set()
    entries = []
    for r in chosen:
        uid = r["id"]
        for kind in ("noisy", "clean"):
            dst = SAMPLES_DIR / f"{uid}_{kind}.wav"
            shutil.copyfile(REPO_ROOT / r[f"{kind}_path"], dst)
            keep.add(dst.name)
        snr = float(r["snr_db"])
        entries.append({
            "id": uid,
            "title": f"{NOISE_TITLES.get(r['noise_type'], r['noise_type'].capitalize())}, {snr:g} dB",
            "noise_type": r["noise_type"],
            "snr_db": snr,
            "noisy": f"samples/{uid}_noisy.wav",
            "clean": f"samples/{uid}_clean.wav",
        })
    # Remove demo clips left over from a previous selection.
    for p in SAMPLES_DIR.glob("*.wav"):
        if (p.name.endswith("_noisy.wav") or p.name.endswith("_clean.wav")) and p.name not in keep:
            p.unlink()
    (SAMPLES_DIR / "samples.json").write_text(json.dumps(entries, indent=2) + "\n")
    (SAMPLES_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
    print(f"\n=== Demo samples -> {os.path.relpath(SAMPLES_DIR, REPO_ROOT)} ===")
    for e, r in zip(entries, chosen):
        print(f"  {e['id']}  {int(r['num_samples']) / TARGET_SR:4.1f} s  {e['title']}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data", help="output data directory (default: <repo>/data)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="worker processes")
    ap.add_argument("--skip-download", action="store_true", help="use archives already in <out>/raw/downloads")
    ap.add_argument("--samples-only", action="store_true", help="only (re)build web/public/samples from the test manifest")
    args = ap.parse_args()
    out = args.out.resolve()

    if args.samples_only:
        make_samples(out)
        return

    if not args.skip_download:
        download_all(out)
    extract_all(out, args.workers)

    split_rows = {s: build_split(out, s, args.workers) for s in SPLITS}
    manifests = {
        "train": [r for r in split_rows["train"] if r["speaker"] not in VAL_SPEAKERS],
        "val": [r for r in split_rows["train"] if r["speaker"] in VAL_SPEAKERS],
        "test": split_rows["test"],
    }
    mdir = out / "vbd16k" / "manifests"
    for name, rows in manifests.items():
        write_manifest(mdir / f"{name}.csv", rows)
    print(f"Wrote manifests to {mdir}")

    ok = sanity_checks(manifests, args.workers)
    make_samples(out)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
