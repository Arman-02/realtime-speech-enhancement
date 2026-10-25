# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "soundfile>=0.12",
#     "soxr>=0.3.7",
#     "tqdm>=4.66",
#     "av>=12",
# ]
# ///
"""Build the fixed-seed out-of-domain test sets, failure-gallery sources and real-recording set
(PLAN.md §4.10, §5.3). Run scripts/prepare_vbd.py and scripts/prepare_noise.py first.

Outputs (paths in manifests are relative to the repo root):
    data/ood/esc/{clean,noisy}/<id>.wav       manifests/ood_esc.csv     VB-D test speech + ESC-50 noise (500)
    data/ood/talker/{clean,noisy}/<id>.wav    manifests/ood_talker.csv  VB-D test speech + LibriSpeech talker (300)
    data/failures_src/<category>/*.wav        data/failures_src/manifest.csv
    data/real/<id>.wav                        manifests/real.csv        from recordings/manifest.csv (if any)

SNR / SIR definition. Speech level = ITU-T P.56 active speech level (method B: 30 ms envelope,
200 ms hangover, 15.9 dB margin), the same convention used to build VoiceBank+DEMAND.
    ood_esc:  SNR = 10 log10( ASL(clean)^2 / mean(noise^2) )   noise power over the whole
              (cropped or looped) noise segment, i.e. the noise as it is heard in the clip.
    talker:   SIR = 10 log10( ASL(clean)^2 / ASL(interferer)^2 )   both P.56 active levels.
Mixtures are scaled down together when needed so nothing clips (the ratio is unchanged).

Usage:
    uv run scripts/build_ood_sets.py --out data --seed 0
    uv run scripts/build_ood_sets.py --out data --seed 0 --only real
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import fftconvolve, lfilter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SR = 16000
MAX_WORKERS = 4

ESC_SNRS = [-5, 0, 5, 10, 15]
TALKER_SIRS = [0, 5, 10]
N_ESC = 500
N_TALKER = 300
MANIFEST_COLUMNS = ["id", "speaker", "clean_path", "noisy_path", "num_samples", "noise_type", "snr_db"]
MAX_FAILURE_S = 10.0


def rel(p: Path) -> str:
    return os.path.relpath(p, REPO_ROOT)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)


def load(path: str | Path) -> np.ndarray:
    x, sr = sf.read(REPO_ROOT / path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    return x if sr == SR else soxr.resample(x, sr, SR, quality="VHQ")


def to_pcm16(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


def write_wav(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.wav")
    sf.write(tmp, to_pcm16(y), SR, subtype="PCM_16")
    tmp.rename(path)


# ----------------------------------------------------------------------------- levels


def active_level(x: np.ndarray, fs: int = SR) -> float:
    """ITU-T P.56 method B active speech level, returned as power (mean square over active samples)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sq = float(np.sum(x**2))
    if n == 0 or sq <= 0:
        return 0.0
    g = np.exp(-1.0 / (fs * 0.03))
    hang = int(np.ceil(0.2 * fs))
    p = lfilter([1 - g], [1, -g], np.abs(x))
    q = lfilter([1 - g], [1, -g], p)
    c = 2.0 ** np.arange(-15, 1)  # thresholds for a [-1, 1] full-scale signal
    idx = np.arange(n)
    lo = np.maximum(0, idx - hang)
    a = np.empty(len(c))
    for j, cj in enumerate(c):
        cs = np.concatenate([[0], np.cumsum(q >= cj)])
        a[j] = np.count_nonzero(cs[idx + 1] - cs[lo])  # active if above threshold within the hangover
    with np.errstate(divide="ignore"):
        A = 10 * np.log10(sq / a)  # level (dB) given each threshold's activity
    C = 20 * np.log10(c)
    diff = A - C
    margin = 15.9
    for j in range(len(c)):
        if a[j] == 0:
            break
        if diff[j] <= margin:
            if j == 0:
                return float(10 ** (A[0] / 10))
            f = (diff[j - 1] - margin) / (diff[j - 1] - diff[j])
            return float(10 ** ((A[j - 1] + f * (A[j] - A[j - 1])) / 10))
    return sq / n  # degenerate (no crossing): fall back to plain mean square


def mean_sq(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop if long enough, else loop from a random start with 20 ms raised-cosine crossfades."""
    if len(noise) >= n:
        s = int(rng.integers(0, len(noise) - n + 1))
        return noise[s: s + n].copy()
    xf = int(min(0.02 * SR, len(noise) // 4))
    s = int(rng.integers(0, len(noise)))
    buf = np.concatenate([noise[s:], noise[:s]])
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf)) if xf else np.zeros(0)
    while len(buf) < n:
        if xf:
            buf[-xf:] = buf[-xf:] * (1 - fade) + noise[:xf] * fade
            buf = np.concatenate([buf, noise[xf:]])
        else:
            buf = np.concatenate([buf, noise])
    return buf[:n]


def noise_segment(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """fit_length, retrying offsets so that sparse noises (clicks, knocks) are not near-silent in the clip."""
    ref = mean_sq(noise)
    best = None
    for _ in range(8):
        seg = fit_length(noise, n, rng)
        if best is None or mean_sq(seg) > mean_sq(best):
            best = seg
        if mean_sq(seg) >= 0.25 * ref:
            return seg
    return best


def mix_at(clean: np.ndarray, interf: np.ndarray, ratio_db: float, level=mean_sq) -> tuple[np.ndarray, np.ndarray]:
    """Return (k*clean, k*clean + g*interf) with ASL(k*clean)^2 / level(g*interf) == ratio_db and peak <= 0.99.

    P.56 is not exactly scale-invariant on short clips, so both levels are evaluated at the final
    output scale (iterating gain and anti-clipping scale) - the ratio then holds for the written files."""
    k, g = 1.0, 1.0
    for _ in range(20):
        c = k * clean
        target = active_level(c) / 10 ** (ratio_db / 10)
        for _ in range(10):
            pi = level(g * interf)
            g *= np.sqrt(target / pi)
            if abs(10 * np.log10(target / pi)) < 0.002:
                break
        y = c + g * interf
        peak = max(np.max(np.abs(y)), np.max(np.abs(c)), 1e-9)
        if peak <= 0.99 + 1e-9:
            return c, y
        k *= 0.99 / peak
        g *= 0.99 / peak
    raise RuntimeError("mix_at did not converge")


# ----------------------------------------------------------------------------- OOD sets


def _clear_stale(d: Path, keep: set[str]) -> None:
    for p in d.glob("*.wav"):
        if p.stem not in keep:
            p.unlink()


def build_ood_esc(out: Path, seed: int) -> list[dict]:
    vbd = read_csv(out / "vbd16k" / "manifests" / "test.csv")
    noise = read_csv(out / "manifests" / "noise_ood.csv")
    by_cat = defaultdict(list)
    for r in noise:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    per_cell, rem = divmod(N_ESC, len(cats) * len(ESC_SNRS))
    if rem:
        raise SystemExit(f"{N_ESC} clips do not split evenly over {len(cats)} categories x {len(ESC_SNRS)} SNRs")
    rng = np.random.default_rng([seed, 101])
    speech = [vbd[i] for i in rng.choice(len(vbd), size=N_ESC, replace=False)]
    cells = []
    for cat in cats:
        pool = [by_cat[cat][i] for i in rng.permutation(len(by_cat[cat]))]
        k = 0
        for snr in ESC_SNRS:
            for _ in range(per_cell):
                cells.append((cat, snr, pool[k % len(pool)]))
                k += 1
    order = rng.permutation(len(cells))
    cells = [cells[i] for i in order]
    base = out / "ood" / "esc"
    rows = []
    for i, (u, (cat, snr, nz)) in enumerate(tqdm(list(zip(speech, cells)), desc="ood_esc", dynamic_ncols=True)):
        uid = f"esc_{i:03d}"
        r = np.random.default_rng([seed, 102, i])
        clean = load(u["clean_path"])
        seg = noise_segment(load(nz["path"]), len(clean), r)
        c, y = mix_at(clean, seg, snr)
        write_wav(base / "clean" / f"{uid}.wav", c)
        write_wav(base / "noisy" / f"{uid}.wav", y)
        rows.append({"id": uid, "speaker": u["speaker"], "clean_path": rel(base / "clean" / f"{uid}.wav"),
                     "noisy_path": rel(base / "noisy" / f"{uid}.wav"), "num_samples": len(clean),
                     "noise_type": cat.replace("_", " "), "snr_db": snr, "clean_src": u["id"], "noise_src": nz["id"]})
    keep = {r["id"] for r in rows}
    _clear_stale(base / "clean", keep)
    _clear_stale(base / "noisy", keep)
    write_csv(out / "manifests" / "ood_esc.csv", rows, MANIFEST_COLUMNS + ["clean_src", "noise_src"])
    return rows


def talker_signal(utts: list[dict], n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Concatenate one speaker's utterances (random start, in order) and take a random n-sample window."""
    k = int(rng.integers(0, len(utts)))
    parts, used, total = [], [], 0
    while total < n + SR:
        u = utts[k % len(utts)]
        x = load(u["path"])
        parts.append(x)
        used.append(u["id"])
        total += len(x)
        k += 1
        if len(used) > 4 * len(utts):
            break
    sig = np.concatenate(parts)
    s = int(rng.integers(0, max(1, len(sig) - n + 1)))
    return fit_length(sig[s:], n, rng), used


def build_ood_talker(out: Path, seed: int) -> list[dict]:
    vbd = read_csv(out / "vbd16k" / "manifests" / "test.csv")
    talkers = read_csv(out / "manifests" / "talkers_test.csv")
    by_spk = defaultdict(list)
    for t in talkers:
        by_spk[t["speaker"]].append(t)
    for v in by_spk.values():
        v.sort(key=lambda t: t["id"])
    spks = sorted(by_spk)
    rng = np.random.default_rng([seed, 201])
    speech = [vbd[i] for i in rng.choice(len(vbd), size=N_TALKER, replace=False)]
    per = N_TALKER // len(TALKER_SIRS)
    sirs = [s for s in TALKER_SIRS for _ in range(per)]
    interf = list(np.resize(rng.permutation(spks), N_TALKER))
    order = rng.permutation(N_TALKER)
    sirs = [sirs[i] for i in order]
    base = out / "ood" / "talker"
    rows = []
    for i, (u, sir, spk) in enumerate(tqdm(list(zip(speech, sirs, interf)), desc="talker", dynamic_ncols=True)):
        uid = f"talker_{i:03d}"
        r = np.random.default_rng([seed, 202, i])
        clean = load(u["clean_path"])
        it, used = talker_signal(by_spk[spk], len(clean), r)
        c, y = mix_at(clean, it, sir, level=active_level)
        write_wav(base / "clean" / f"{uid}.wav", c)
        write_wav(base / "noisy" / f"{uid}.wav", y)
        rows.append({"id": uid, "speaker": u["speaker"], "clean_path": rel(base / "clean" / f"{uid}.wav"),
                     "noisy_path": rel(base / "noisy" / f"{uid}.wav"), "num_samples": len(clean),
                     "noise_type": "competing talker", "snr_db": sir, "clean_src": u["id"],
                     "interferer_speaker": spk, "interferer_gender": by_spk[spk][0]["gender"],
                     "interferer_src": " ".join(used)})
    keep = {r["id"] for r in rows}
    _clear_stale(base / "clean", keep)
    _clear_stale(base / "noisy", keep)
    write_csv(out / "manifests" / "ood_talker.csv", rows,
              MANIFEST_COLUMNS + ["clean_src", "interferer_speaker", "interferer_gender", "interferer_src"])
    return rows


# ----------------------------------------------------------------------------- failure sources


def build_failures(out: Path, seed: int) -> list[dict]:
    """2-3 source clips per failure category (PLAN.md §5.3). Only redistributable audio is used
    (VB-D CC BY 4.0, MUSAN CC BY 4.0, LibriSpeech CC BY 4.0, simulated RIRs): these clips end up on
    the public website, so ESC-50 (CC BY-NC) and the measured RIRs are deliberately not used here."""
    fdir = out / "failures_src"
    vbd = read_csv(out / "vbd16k" / "manifests" / "test.csv")
    rng = np.random.default_rng([seed, 301])
    used: set[str] = set()

    def pick(lo: float, hi: float, noise_type: str | None = None, snr: float | None = None) -> dict:
        c = [r for r in vbd if lo <= int(r["num_samples"]) / SR <= hi and r["id"] not in used
             and (noise_type is None or r["noise_type"] == noise_type) and (snr is None or float(r["snr_db"]) == snr)]
        r = c[int(rng.integers(0, len(c)))]
        used.add(r["id"])
        return r

    rows: list[dict] = []

    def emit(fid: str, cat: str, noisy: np.ndarray, clean: np.ndarray | None, note: str) -> None:
        d = fdir / cat
        npath = d / f"{fid}_noisy.wav"
        write_wav(npath, noisy)
        cpath = ""
        if clean is not None:
            write_wav(d / f"{fid}_clean.wav", clean)
            cpath = rel(d / f"{fid}_clean.wav")
        rows.append({"id": fid, "category": cat, "noisy_path": rel(npath), "clean_path": cpath, "note": note})

    def scale_pair(c: np.ndarray, y: np.ndarray, peak: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        k = peak / max(np.max(np.abs(y)), np.max(np.abs(c)), 1e-9)
        return c * k, y * k

    def vbd_noise(r: dict) -> np.ndarray:
        return load(r["noisy_path"]) - load(r["clean_path"])

    # --- music: MUSAN excerpts (written by prepare_noise.py) + speech over background music
    music = read_csv(fdir / "music" / "music.csv")
    voc = [m for m in music if m["vocals"] == "Y"]
    ins = [m for m in music if m["vocals"] == "N"]
    if voc:
        m = voc[0]
        rows.append({"id": "music_vocals", "category": "music", "noisy_path": m["path"], "clean_path": "",
                     "note": f"MUSAN {m['id'][6:]} (singing), 10 s excerpt; no clean reference - the model should "
                             "pass or suppress it consistently, singing tends to be partly kept as speech"})
    if ins:
        m = ins[0]
        rows.append({"id": "music_instrumental", "category": "music", "noisy_path": m["path"], "clean_path": "",
                     "note": f"MUSAN {m['id'][6:]} (instrumental), 10 s excerpt; tonal music is not in the "
                             "training noise, expect partial suppression and warbling"})
        r = pick(3.0, 8.0)
        clean = load(r["clean_path"])
        bg = noise_segment(load(m["path"]), len(clean), rng)
        c, y = mix_at(clean, bg, 5.0)
        emit("music_speech_bg", "music", y, c,
             f"VB-D {r['id']} over MUSAN {m['id'][6:]} (instrumental) at 5 dB SNR")

    # --- silence: speech, a long low-noise pause, speech (<= 10 s total)
    for k, (ntype, snr_db) in enumerate([("living room", 25.0), ("office", 20.0)]):
        a, b = pick(1.8, 2.4), pick(1.8, 2.4)
        sa, sb = load(a["clean_path"]), load(b["clean_path"])
        n = int(MAX_FAILURE_S * SR)
        gap = n - len(sa) - len(sb)
        clean = np.concatenate([sa, np.zeros(gap), sb])
        srcs = [r for r in vbd if r["noise_type"] == ntype]
        pieces, tot = [], 0
        for j in rng.permutation(len(srcs)):
            pieces.append(vbd_noise(srcs[j]))
            tot += len(pieces[-1])
            if tot > n:
                break
        bg = fit_length(np.concatenate(pieces), n, rng)
        ps = active_level(np.concatenate([sa, sb]))
        bg *= np.sqrt(ps / (mean_sq(bg) * 10 ** (snr_db / 10)))
        c, y = scale_pair(clean, clean + bg)
        emit(f"silence_{k + 1}", "silence", y, c,
             f"VB-D {a['id']} ({len(sa) / SR:.1f} s), {gap / SR:.1f} s pause, {b['id']} ({len(sb) / SR:.1f} s); "
             f"VB-D '{ntype}' noise at {snr_db:g} dB SNR throughout - watch for noise pumping in the pause")

    # --- low SNR: VB-D test speech with its own (unseen-type) noise rescaled to -5 / -10 dB
    for ntype, snr_db in [("bus", -5.0), ("cafe", -10.0), ("public square", -10.0)]:
        r = pick(3.0, 8.0, noise_type=ntype)
        clean = load(r["clean_path"])
        nz = vbd_noise(r)
        c, y = mix_at(clean, nz, snr_db)
        emit(f"low_snr_{ntype.replace(' ', '_')}_m{int(-snr_db)}db", "low_snr", y, c,
             f"VB-D {r['id']} with its own '{ntype}' noise rescaled from {r['snr_db']} dB to {snr_db:g} dB SNR")

    # --- competing talker at 0 dB SIR (LibriSpeech test-clean)
    talkers = read_csv(out / "manifests" / "talkers_test.csv")
    by_spk = defaultdict(list)
    for t in talkers:
        by_spk[t["speaker"]].append(t)
    gender_of = {"p232": "M", "p257": "F"}
    for k, same in enumerate([False, True]):
        r = pick(3.0, 8.0)
        g = gender_of.get(r["speaker"], "M")
        cands = sorted(s for s, v in by_spk.items() if (v[0]["gender"] == g) == same)
        spk = cands[int(rng.integers(0, len(cands)))]
        clean = load(r["clean_path"])
        it, _src = talker_signal(sorted(by_spk[spk], key=lambda t: t["id"]), len(clean), rng)
        c, y = mix_at(clean, it, 0.0, level=active_level)
        emit(f"talker_{k + 1}", "talker", y, c,
             f"VB-D {r['id']} + LibriSpeech speaker {spk} ({'same' if same else 'other'} gender) at 0 dB SIR - "
             "the model was never trained to remove speech, so the interferer is expected to stay")

    # --- heavy reverb: simulated test RIRs with T60 >= 0.7 s (redistributable)
    rirs = [x for x in read_csv(out / "manifests" / "rir_test.csv") if x["source"] == "sim" and float(x["t60"]) >= 0.7]
    rirs.sort(key=lambda x: (-float(x["t60"]), x["id"]))
    for k, h_row in enumerate(rirs[:2]):
        r = pick(3.0, 7.0)
        clean = load(r["clean_path"])
        h = load(h_row["path"])
        d = int(h_row["direct_idx"])
        n = min(len(clean) + int(0.6 * SR), int(MAX_FAILURE_S * SR))
        wet = fftconvolve(clean, h)[:n]
        dry = np.zeros(n)
        dry[d: d + len(clean)] = clean[: n - d] * h[d]  # direct path only, time-aligned
        c, y = scale_pair(dry, wet)
        emit(f"reverb_{k + 1}", "reverb", y, c,
             f"VB-D {r['id']} convolved with {h_row['id']} (simulated, T60 {float(h_row['t60']):.2f} s, "
             f"DRR {float(h_row['drr_db']):.1f} dB); clean = time-aligned direct path. "
             "Heavy reverb is outside the benchmark model's training data (reverb augmentation is demo-only)")

    # --- clipping: VB-D test noisy input amplified and hard-clipped at full scale
    for k, frac in enumerate([0.03, 0.15]):
        r = pick(3.0, 8.0)
        x = load(r["noisy_path"])
        t = np.quantile(np.abs(x), 1 - frac)
        y = np.clip(x / t, -1.0, 1.0) * 0.99
        gain_db = 20 * np.log10(0.99 / t)
        emit(f"clipping_{k + 1}", "clipping", y, load(r["clean_path"]),
             f"VB-D {r['id']} noisy ({r['noise_type']}, {r['snr_db']} dB) +{gain_db:.1f} dB then hard-clipped "
             f"({100 * np.mean(np.abs(y) >= 0.989):.0f}% of samples clipped); clean is the unclipped reference "
             "at its original level")

    keep = {REPO_ROOT / r[k] for r in rows for k in ("noisy_path", "clean_path") if r[k]}
    for p in fdir.glob("*/*.wav"):
        if p not in keep and (p.stem.endswith("_noisy") or p.stem.endswith("_clean")):
            p.unlink()  # stale generated clip from an earlier build (MUSAN excerpts are never touched)
    write_csv(fdir / "manifest.csv", rows, ["id", "category", "noisy_path", "clean_path", "note"])
    return rows


# ----------------------------------------------------------------------------- real recordings


def decode_any(path: Path) -> tuple[np.ndarray, int]:
    try:
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        return x.mean(axis=1), sr
    except Exception:  # noqa: BLE001 - m4a/aac/mp3 etc.: decode with PyAV (bundled FFmpeg)
        import av

        with av.open(str(path)) as cont:
            st = cont.streams.audio[0]
            rs = av.AudioResampler(format="flt", layout="mono", rate=st.rate)
            chunks = []
            for frame in cont.decode(st):
                chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
            chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
            return np.concatenate(chunks).astype(np.float64), st.rate


def build_real(out: Path) -> list[dict]:
    rec = REPO_ROOT / "recordings"
    man = rec / "manifest.csv"
    rows_in = [r for r in read_csv(man) if (r.get("id") or "").strip()] if man.exists() else []
    if not rows_in:
        print(f"\n[real] {rel(man)} lists no recordings yet - skipping (see recordings/README.md)")
        return []
    rows = []
    for r in rows_in:
        rid = r["id"].strip()
        src = rec / r["file"].strip()
        if not src.exists():
            print(f"[warn] real: {rel(src)} not found; skipped", file=sys.stderr)
            continue
        x, sr = decode_any(src)
        y = x if sr == SR else soxr.resample(x, sr, SR, quality="VHQ")
        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak > 1.0:
            y = y / peak * 0.99
        dur = len(y) / SR
        if not 8.0 <= dur <= 15.0:
            print(f"[warn] real: {rid} is {dur:.1f} s (recommended 8-15 s)", file=sys.stderr)
        dst = out / "real" / f"{rid}.wav"
        write_wav(dst, y)
        rows.append({"id": rid, "speaker": "author", "clean_path": "", "noisy_path": rel(dst), "num_samples": len(y),
                     "noise_type": r.get("place", "").strip(), "snr_db": "", "device": r.get("device", "").strip(),
                     "place": r.get("place", "").strip(), "notes": r.get("notes", "").strip(), "file": r["file"].strip()})
    write_csv(out / "manifests" / "real.csv", rows, MANIFEST_COLUMNS + ["device", "place", "notes", "file"])
    print(f"\n[real] {len(rows)} recordings -> {rel(out / 'real')}, manifests/real.csv")
    return rows


# ----------------------------------------------------------------------------- verification


def _measure(job: tuple[str, str, str]) -> tuple[str, int, int, str, int, int, float]:
    cpath, npath, kind = job
    ic, inn = sf.info(REPO_ROOT / cpath), sf.info(REPO_ROOT / npath)
    c = sf.read(REPO_ROOT / cpath, dtype="float64")[0]
    y = sf.read(REPO_ROOT / npath, dtype="float64")[0]
    if len(c) != len(y):
        return npath, ic.samplerate, inn.samplerate, inn.subtype, len(c), len(y), float("nan")
    resid = y - c
    p_int = active_level(resid) if kind == "talker" else mean_sq(resid)
    return npath, ic.samplerate, inn.samplerate, inn.subtype, len(c), len(y), 10 * np.log10(active_level(c) / p_int)


def verify(out: Path, esc: list[dict], talker: list[dict], fails: list[dict], workers: int) -> bool:
    ok = True
    print("\n=== Verification ===")
    for name, rows, kind, n_exp, levels in (("ood_esc", esc, "esc", N_ESC, ESC_SNRS),
                                           ("ood_talker", talker, "talker", N_TALKER, TALKER_SIRS)):
        if not rows:
            continue
        jobs = [(r["clean_path"], r["noisy_path"], kind) for r in rows]
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
            res = list(ex.map(_measure, jobs, chunksize=16))
        err = np.array([m[6] - float(r["snr_db"]) for m, r in zip(res, rows)])
        fmt_bad = [m for m in res if m[1] != SR or m[2] != SR or m[3] != "PCM_16" or m[4] != m[5]]
        print(f"[{name}] {len(rows)} clips (expected {n_exp}); "
              f"measured-target {'SIR' if kind == 'talker' else 'SNR'} error: max |err| {np.nanmax(np.abs(err)):.3f} dB, "
              f"mean {np.nanmean(err):+.3f} dB")
        if len(rows) != n_exp or fmt_bad or not np.all(np.abs(err) <= 0.5):
            ok = False
            print(f"  [FAIL] count/format/length problems: {len(fmt_bad)}; |err| > 0.5 dB: {int(np.sum(~(np.abs(err) <= 0.5)))}")
        cnt = Counter(float(r["snr_db"]) for r in rows)
        print("  per level: " + ", ".join(f"{v:g} dB: {cnt[float(v)]}" for v in levels))
        if kind == "esc":
            cats = Counter(r["noise_type"] for r in rows)
            print(f"  {len(cats)} categories x {sorted(set(cats.values()))} clips: " + ", ".join(sorted(cats)))
            cell = Counter((r["noise_type"], r["snr_db"]) for r in rows)
            if len(set(cell.values())) != 1 or len(cell) != len(cats) * len(levels):
                ok = False
                print("  [FAIL] not balanced over category x SNR")
            noise_ood = {r["id"] for r in read_csv(out / "manifests" / "noise_ood.csv")}
            noise_train = {r["id"] for r in read_csv(out / "manifests" / "noise_train.csv")}
            used = {r["noise_src"] for r in rows}
            if not used <= noise_ood or used & noise_train:
                ok = False
                print("  [FAIL] noise not from the test-only noise corpus")
            else:
                print(f"  {len(used)} distinct ESC-50 noise clips, all from noise_ood.csv, none in noise_train.csv")
        else:
            spk = Counter(r["interferer_speaker"] for r in rows)
            print(f"  {len(spk)} interferer speakers ({min(spk.values())}-{max(spk.values())} clips each), "
                  f"genders {dict(Counter(r['interferer_gender'] for r in rows))}")
        print(f"  target speakers {dict(Counter(r['speaker'] for r in rows))}, "
              f"{len({r['clean_src'] for r in rows})} distinct VB-D test utterances")
    if fails:
        bad = []
        for r in fails:
            for k in ("noisy_path", "clean_path"):
                if r[k]:
                    i = sf.info(REPO_ROOT / r[k])
                    if i.samplerate != SR or i.channels != 1 or i.subtype != "PCM_16" or i.duration > MAX_FAILURE_S + 1e-6:
                        bad.append((r[k], i.samplerate, i.duration))
        cats = Counter(r["category"] for r in fails)
        print(f"[failures_src] {len(fails)} clips: {dict(cats)}")
        if bad or set(cats) != {"music", "silence", "low_snr", "talker", "reverb", "clipping"} or min(cats.values()) < 2:
            ok = False
            print(f"  [FAIL] format/duration/category problems: {bad[:3]}")
        else:
            print("  all <= 10 s, 16 kHz mono PCM16; every category has >= 2 clips")
    print("All checks passed." if ok else "SOME CHECKS FAILED.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--only", nargs="+", choices=["esc", "talker", "failures", "real"])
    args = ap.parse_args()
    out = args.out.resolve()
    steps = args.only or ["esc", "talker", "failures", "real"]
    workers = max(1, min(args.workers, MAX_WORKERS))
    esc = build_ood_esc(out, args.seed) if "esc" in steps else []
    talker = build_ood_talker(out, args.seed) if "talker" in steps else []
    fails = build_failures(out, args.seed) if "failures" in steps else []
    if "real" in steps:
        build_real(out)
    if not verify(out, esc, talker, fails, workers):
        sys.exit(1)


if __name__ == "__main__":
    main()
