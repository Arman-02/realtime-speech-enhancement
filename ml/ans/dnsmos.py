"""DNSMOS P.835 (SIG, BAK, OVRL) — non-intrusive quality (PLAN.md §4.6).

Mirrors Microsoft's ``DNSMOS/dnsmos_local.py`` (DNS-Challenge): 16 kHz input, 9.01 s windows hopped by
1 s (short clips are tiled to 9.01 s), raw outputs mapped through the published P.835 polynomials and
averaged over windows. Only the P.835 model (``sig_bak_ovr.onnx``) is used; the P.808 model is not needed.

Model files are **downloaded at runtime** into ``ml/.cache/dnsmos/`` (git-ignored) and never committed.
Terms: DNS-Challenge code is MIT (``LICENSE-CODE``); the repository's models/data are CC BY 4.0
(``LICENSE``) — attribute "DNSMOS P.835, Reddy et al., ICASSP 2022" when publishing scores.

uv run python -m ans.dnsmos file1.wav [file2.wav ...]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from ans.dsp import SAMPLE_RATE

ML_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("ANS_DNSMOS_CACHE", ML_ROOT / ".cache" / "dnsmos"))
BASE_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS"
MODELS = {
    False: ("DNSMOS/sig_bak_ovr.onnx", "sig_bak_ovr.onnx"),
    True: ("pDNSMOS/sig_bak_ovr.onnx", "p_sig_bak_ovr.onnx"),
}
INPUT_LENGTH = 9.01
CITATION = "DNSMOS P.835 (Reddy, Gopal & Cutler, ICASSP 2022), models from microsoft/DNS-Challenge (CC BY 4.0)"

# Polynomial calibration from dnsmos_local.py (get_polyfit_val).
POLY = {
    False: {
        "ovrl": np.poly1d([-0.06766283, 1.11546468, 0.04602535]),
        "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
        "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
    },
    True: {
        "ovrl": np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046]),
        "sig": np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726]),
        "bak": np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132]),
    },
}


def model_path(personalized: bool = False, download: bool = True) -> Path:
    remote, local = MODELS[personalized]
    path = CACHE / local
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{remote}", tmp)
        tmp.replace(path)
    return path


def model_sha256(personalized: bool = False) -> str:
    return hashlib.sha256(model_path(personalized).read_bytes()).hexdigest()


class DNSMOS:
    """``DNSMOS()(wav16k) -> {"dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"}``."""

    def __init__(self, personalized: bool = False, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path(personalized)), opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.poly = POLY[personalized]

    def __call__(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
        audio = np.asarray(audio, np.float64)
        if sr != SAMPLE_RATE:
            import soxr

            audio = soxr.resample(audio, sr, SAMPLE_RATE)
        if len(audio) == 0 or not np.any(audio):
            return {"dnsmos_sig": float("nan"), "dnsmos_bak": float("nan"), "dnsmos_ovrl": float("nan")}
        fs = SAMPLE_RATE
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        sig, bak, ovr = [], [], []
        for idx in range(num_hops):
            seg = audio[int(idx * fs) : int((idx + INPUT_LENGTH) * fs)]
            if len(seg) < len_samples:
                continue
            raw = self.sess.run(None, {self.input: seg.astype(np.float32)[np.newaxis, :]})[0][0]
            sig.append(self.poly["sig"](raw[0]))
            bak.append(self.poly["bak"](raw[1]))
            ovr.append(self.poly["ovrl"](raw[2]))
        return {
            "dnsmos_sig": float(np.mean(sig)),
            "dnsmos_bak": float(np.mean(bak)),
            "dnsmos_ovrl": float(np.mean(ovr)),
        }


_SINGLETON: dict[bool, DNSMOS] = {}


def dnsmos(audio: np.ndarray, sr: int = SAMPLE_RATE, personalized: bool = False) -> dict[str, float]:
    """Process-local cached scorer (for multiprocessing workers)."""
    if personalized not in _SINGLETON:
        _SINGLETON[personalized] = DNSMOS(personalized)
    return _SINGLETON[personalized](audio, sr)


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-p", "--personalized", action="store_true", help="pDNSMOS (penalises interfering talkers)")
    args = ap.parse_args()
    scorer = DNSMOS(args.personalized)
    print(f"{'file':<50s}  SIG    BAK    OVRL")
    for f in args.files:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        r = scorer(x.mean(axis=1), sr)
        print(f"{Path(f).name:<50s} {r['dnsmos_sig']:.3f}  {r['dnsmos_bak']:.3f}  {r['dnsmos_ovrl']:.3f}")


if __name__ == "__main__":
    main()
