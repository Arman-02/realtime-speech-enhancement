"""Render the listening-test clips and the failure-case gallery (PLAN.md §3.6, §5.1.6, §5.3).

uv run python -m ans.render --listening --failures --out ../web/public

--listening  20 VB-D test trials + 1 practice trial, stratified over noise type x SNR (one per cell of the
             5 x 4 VB-D test design), excluding the six demo clips. Conditions: noisy, wiener, crn (benchmark),
             crn_kd (if its checkpoint exists), clean (hidden reference). Every file is RMS-normalised to the
             same level (-26 dBFS) so loudness does not bias the ratings. -> listening/index.json + wavs
--failures   Each clip in data/failures_src/manifest.csv enhanced by the benchmark model; the explanation is
             generated from measurements of what the model actually did (residual noise in pauses, music
             energy kept, interferer kept, reverb tail removed, …). -> failures/index.json + wavs

All clips are 16 kHz mono PCM16, at most 10 s.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ans.data import ML_ROOT, REPO_ROOT, read_manifest
from ans.decompose import crn_decompose, decomposition_metrics
from ans.dsp import SAMPLE_RATE
from ans.metrics import compute_metrics
from ans.models import ClassicalEnhancer, load_checkpoint
from ans.registry import ckpt_config, display_name

MAX_S = 10.0
TARGET_DBFS = -26.0
FAILURES_SRC = REPO_ROOT / "data" / "failures_src" / "manifest.csv"
SAMPLES_JSON = REPO_ROOT / "web" / "public" / "samples" / "samples.json"
DEFAULT_CRN = ML_ROOT / "checkpoints" / "crn_small" / "best.pt"
DEFAULT_KD = ML_ROOT / "checkpoints" / "crn256_kd" / "best.pt"
TITLES = {
    "music": "Music and singing",
    "silence": "Long pause with residual noise",
    "low_snr": "Very low SNR",
    "talker": "Competing talker",
    "reverb": "Heavy reverberation",
    "clipping": "Clipped input",
}


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, np.float32)[: int(MAX_S * SAMPLE_RATE)]
    sf.write(str(path), np.clip(x, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def db(x: float) -> float:
    return 10 * math.log10(max(x, 1e-20))


def rms_db(x: np.ndarray) -> float:
    return db(float(np.mean(np.square(x, dtype=np.float64))))


def normalise(x: np.ndarray, target: float = TARGET_DBFS) -> np.ndarray:
    g = 10 ** ((target - rms_db(x)) / 20)
    peak = float(np.abs(x).max()) * g
    if peak > 0.99:
        g *= 0.99 / peak
    return (x * g).astype(np.float32)


def read(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        x = soxr.resample(x, sr, SAMPLE_RATE).astype(np.float32)
    return x


@torch.no_grad()
def enhance(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, ClassicalEnhancer):
        return model.enhance_np(x)
    return model.enhance(torch.from_numpy(x)[None])[0].numpy()


# ------------------------------------------------------------------------------------------------
# Listening test
# ------------------------------------------------------------------------------------------------


def pick_trials(n_trials: int = 20, seed: int = 0) -> tuple[list, object]:
    """One utterance per (noise type, SNR) cell of the VB-D test set (round-robin if n != cells), excluding
    the demo clips; plus a practice utterance from a different speaker/cell. Deterministic for ``seed``."""
    from ans.data import manifest_path

    items = read_manifest(manifest_path("test"))
    demo: set[str] = set()
    with contextlib.suppress(OSError, json.JSONDecodeError):
        demo = {s["id"] for s in json.loads(SAMPLES_JSON.read_text())}
    rng = np.random.default_rng(seed)
    pool = [u for u in items if u.id not in demo and 2.0 * SAMPLE_RATE <= u.num_samples <= MAX_S * SAMPLE_RATE]
    cells: dict[tuple[str, float], list] = {}
    for u in pool:
        cells.setdefault((u.noise_type, u.snr_db), []).append(u)
    keys = sorted(cells)
    chosen = []
    while len(chosen) < n_trials:
        for k in keys:
            if len(chosen) >= n_trials:
                break
            cands = [u for u in cells[k] if u not in chosen]
            if cands:
                chosen.append(cands[int(rng.integers(len(cands)))])
    rest = [u for u in pool if u not in chosen]
    practice = rest[int(rng.integers(len(rest)))]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order], practice


def render_listening(out: Path, crn_ckpt: Path, kd_ckpt: Path | None, n_trials: int = 20, seed: int = 0) -> dict:
    lis = out / "listening"
    lis.mkdir(parents=True, exist_ok=True)
    models: dict[str, object] = {"wiener": ClassicalEnhancer(), "crn": load_checkpoint(crn_ckpt)}
    names = {"noisy": "Noisy (unprocessed)", "wiener": "Spectral subtraction (Wiener)", "clean": "Clean reference"}
    names["crn"] = display_name(ckpt_config(crn_ckpt))
    if kd_ckpt is not None and kd_ckpt.exists():
        models["crn_kd"] = load_checkpoint(kd_ckpt)
        names["crn_kd"] = display_name(ckpt_config(kd_ckpt))
    conditions = ["noisy", *models, "clean"]
    trials, practice = pick_trials(n_trials, seed)

    def render(tag: str, u) -> dict:
        noisy, clean = read(u.noisy_path), read(u.clean_path)
        n = min(len(noisy), len(clean))
        sigs = {"noisy": noisy[:n], "clean": clean[:n]}
        for k, m in models.items():
            sigs[k] = enhance(m, noisy[:n])
        files = {}
        for k in conditions:
            rel = f"listening/{tag}_{k}.wav"
            write_wav(out / rel, normalise(sigs[k]))
            files[k] = rel
        return files

    index = {
        "version": 1,
        "conditions": conditions,
        "hidden_reference": "clean",
        "low_anchor": "noisy",
        "models": names,
        "normalisation": f"every file RMS-normalised to {TARGET_DBFS:g} dBFS (peak-limited to -0.1 dBFS)",
        "practice": {"id": "practice", "utterance": practice.id, "files": render("practice", practice)},
        "trials": [],
    }
    for i, u in enumerate(trials, 1):
        tid = f"t{i:02d}"
        index["trials"].append(
            {
                "id": tid,
                "utterance": u.id,
                "set": "vbd",
                "noise_type": u.noise_type,
                "snr_db": u.snr_db,
                "files": render(tid, u),
            }
        )
    (lis / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"listening: {len(trials)} trials + practice, conditions {conditions} -> {lis}")
    return index


# ------------------------------------------------------------------------------------------------
# Failure gallery
# ------------------------------------------------------------------------------------------------


def frame_db(x: np.ndarray, hop: int = 320) -> np.ndarray:
    n = len(x) // hop
    f = x[: n * hop].reshape(n, hop).astype(np.float64)
    return 10 * np.log10(np.mean(f * f, axis=1) + 1e-12)


def measure(cat: str, noisy: np.ndarray, enh: np.ndarray, clean: np.ndarray | None, model) -> dict:
    m: dict = {"level_change_db": round(rms_db(enh) - rms_db(noisy), 1)}
    if clean is not None:
        before = compute_metrics(clean, noisy, which=("pesq", "stoi", "si_sdr"))
        after = compute_metrics(clean, enh, which=("pesq", "stoi", "si_sdr"))
        for k in ("pesq", "stoi", "si_sdr"):
            m[f"{k}_noisy"], m[f"{k}_enhanced"] = round(before[k], 3), round(after[k], 3)
        if hasattr(model, "masks"):
            _, s_hat, n_hat = crn_decompose(model, noisy, clean)
            d = decomposition_metrics(clean, noisy - clean, s_hat, n_hat)
            m["noise_reduction_db"] = round(d["noise_reduction_db"], 1)
            m["speech_distortion_sisdr"] = round(d["speech_distortion_sisdr"], 1)
    if cat == "silence" and clean is not None:
        cdb = frame_db(clean)
        pause = cdb < cdb.max() - 50  # frames where the clean speech is (nearly) silent
        if pause.sum() >= 10:
            nd, ed = frame_db(noisy)[pause], frame_db(enh)[pause]
            m["pause_s"] = round(pause.sum() * 0.02, 1)
            m["pause_noise_db_noisy"] = round(float(np.mean(nd)), 1)
            m["pause_noise_db_enhanced"] = round(float(np.mean(ed)), 1)
            m["pause_attenuation_db"] = round(float(np.mean(nd) - np.mean(ed)), 1)
            m["pause_fluctuation_db"] = round(float(np.std(ed)), 1)  # pumping
            m["pause_fluctuation_db_noisy"] = round(float(np.std(nd)), 1)
    if cat == "music":
        nd, ed = frame_db(noisy), frame_db(enh)
        active = nd > nd.max() - 40
        gain = ed[active] - nd[active]
        m["music_energy_kept_db"] = round(float(np.mean(gain)), 1)
        m["gain_fluctuation_db"] = round(float(np.std(gain)), 1)  # warbling
    if cat == "clipping":
        m["clipped_fraction_input"] = round(float(np.mean(np.abs(noisy) >= 0.999 * np.abs(noisy).max())), 3)
    return m


def explain(cat: str, m: dict, note: str, ref_nr: float | None = None) -> str:
    """One or two sentences, every claim conditional on the measurements in ``m``."""
    g = m.get

    def f(k: str, nd: int = 1) -> str:
        v = g(k)
        return "—" if v is None else f"{v:.{nd}f}"

    vbd = f" (VB-D test average: {ref_nr:.1f} dB)" if ref_nr is not None else ""
    if cat == "silence" and g("pause_attenuation_db") is not None:
        pump = g("pause_fluctuation_db") > g("pause_fluctuation_db_noisy") + 2
        return (
            f"In the {f('pause_s')} s pause the background is turned down by only {f('pause_attenuation_db')} dB "
            f"(to {f('pause_noise_db_enhanced')} dBFS): the model leaves a noise floor instead of gating pauses to "
            "silence. "
            + (
                f"Its level fluctuates more than in the input ({f('pause_fluctuation_db')} vs "
                f"{f('pause_fluctuation_db_noisy')} dB frame to frame), heard as pumping."
                if pump
                else f"The residual keeps the input's texture ({f('pause_fluctuation_db')} vs "
                f"{f('pause_fluctuation_db_noisy')} dB frame-to-frame spread), so there is no audible pumping."
            )
        )
    if cat == "music" and g("noise_reduction_db") is None:
        kept = g("music_energy_kept_db")
        how = "only partly suppressed" if kept > -10 else "largely suppressed"
        return (
            f"Music never appears in training. Here it is {how} ({f('music_energy_kept_db')} dB on average), but the "
            f"gain jumps by {f('gain_fluctuation_db')} dB from frame to frame as sung or tonal passages are "
            "mistaken for voiced speech, so what remains warbles in level."
        )
    if cat == "music":
        return (
            f"Music under speech is reduced by {f('noise_reduction_db')} dB{vbd}; its harmonics look like voiced "
            f"speech to the model, so it partly survives (PESQ {f('pesq_noisy', 2)} → {f('pesq_enhanced', 2)})."
        )
    if cat == "low_snr":
        sd = g("speech_distortion_sisdr")
        damage = "the speech itself is badly damaged" if sd < 10 else "the speech is moderately distorted"
        return (
            f"The model removes {f('noise_reduction_db')} dB of noise and SI-SDR rises from {f('si_sdr_noisy')} to "
            f"{f('si_sdr_enhanced')} dB, but {damage} (speech-only SI-SDR {f('speech_distortion_sisdr')} dB) and "
            f"quality stays low (PESQ {f('pesq_noisy', 2)} → {f('pesq_enhanced', 2)}, STOI {f('stoi_noisy', 2)} → "
            f"{f('stoi_enhanced', 2)}): at this SNR the mask cannot separate speech from noise reliably."
        )
    if cat == "talker":
        sd = g("speech_distortion_sisdr")
        extra = (
            f" Worse, the target is damaged too (speech-only SI-SDR {f('speech_distortion_sisdr')} dB; SI-SDR "
            f"{f('si_sdr_noisy')} → {f('si_sdr_enhanced')} dB)."
            if sd < 10
            else f" PESQ stays at {f('pesq_enhanced', 2)}."
        )
        return (
            f"The interfering voice is attenuated by just {f('noise_reduction_db')} dB: the model is trained to keep "
            "all speech and has no notion of which talker is the target." + extra
        )
    if cat == "reverb":
        stoi_drop = (g("stoi_noisy") or 0) - (g("stoi_enhanced") or 0)
        extra = (
            f" Intelligibility even drops (STOI {f('stoi_noisy', 2)} → {f('stoi_enhanced', 2)})."
            if stoi_drop > 0.02
            else ""
        )
        return (
            f"The reverberant tail is reduced by only {f('noise_reduction_db')} dB relative to the direct path: "
            "the benchmark model was trained without reverb and treats smeared speech as speech." + extra
        )
    if cat == "clipping":
        return (
            f"Clipping distortion ({(g('clipped_fraction_input') or 0) * 100:.0f}% of samples at full scale) is "
            f"correlated with the speech, so it is kept: the error against the unclipped reference (noise plus "
            f"distortion) drops by only {f('noise_reduction_db')} dB and PESQ goes {f('pesq_noisy', 2)} → "
            f"{f('pesq_enhanced', 2)}, although the background noise itself is removed."
        )
    return note


def _benchmark_nr() -> float | None:
    try:
        res = json.loads((ML_ROOT / "exports" / "results.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for r in res.get("rows", []):
        if r.get("run") == "crn_small" and r.get("set") == "vbd" and r.get("noise_reduction_db") is not None:
            return float(r["noise_reduction_db"])
    return None


def render_failures(out: Path, crn_ckpt: Path, src: Path = FAILURES_SRC) -> list[dict]:
    fdir = out / "failures"
    fdir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(crn_ckpt)
    name = display_name(ckpt_config(crn_ckpt))
    entries = []
    ref_nr = _benchmark_nr()
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cat, fid = row["category"], row["id"]
        noisy = read(REPO_ROOT / row["noisy_path"])[: int(MAX_S * SAMPLE_RATE)]
        clean = None
        if row.get("clean_path"):
            clean = read(REPO_ROOT / row["clean_path"])
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]
        enh = enhance(model, noisy)
        m = measure(cat, noisy, enh, clean, model)
        try:
            from ans.dnsmos import dnsmos

            m["dnsmos_ovrl_noisy"] = round(dnsmos(noisy)["dnsmos_ovrl"], 2)
            m["dnsmos_ovrl_enhanced"] = round(dnsmos(enh)["dnsmos_ovrl"], 2)
        except Exception:
            pass
        files = {"noisy": f"failures/{fid}_noisy.wav", "enhanced": f"failures/{fid}_enhanced.wav"}
        write_wav(out / files["noisy"], noisy)
        write_wav(out / files["enhanced"], enh)
        clean_rel = None
        if clean is not None:
            clean_rel = f"failures/{fid}_clean.wav"
            write_wav(out / clean_rel, clean)
        entries.append(
            {
                "id": fid,
                "title": f"{TITLES.get(cat, cat)} — {fid.replace('_', ' ')}",
                "category": cat,
                "explanation": explain(cat, m, row.get("note", ""), ref_nr),
                "noisy": files["noisy"],
                "enhanced": files["enhanced"],
                "clean": clean_rel,
                "model": name,
                "source_note": row.get("note", ""),
                "metrics": m,
            }
        )
        print(f"  {fid:<28s} {entries[-1]['explanation'][:110]}…")
    (fdir / "index.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print(f"failures: {len(entries)} clips -> {fdir}")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listening", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "web" / "public"))
    ap.add_argument("--crn", default=str(DEFAULT_CRN), help="benchmark checkpoint (condition 'crn')")
    ap.add_argument("--crn-kd", default=str(DEFAULT_KD), help="KD checkpoint (condition 'crn_kd'; skipped if missing)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.listening or args.failures):
        ap.error("pass --listening and/or --failures")
    torch.set_num_threads(4)
    out = Path(args.out)
    if args.listening:
        render_listening(out, Path(args.crn), Path(args.crn_kd) if args.crn_kd else None, args.trials, args.seed)
    if args.failures:
        render_failures(out, Path(args.crn))


if __name__ == "__main__":
    main()
