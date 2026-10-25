"""Enhance audio files (any sample rate / channel count -> 16 kHz mono).

uv run python -m ans.enhance noisy.wav -o enhanced.wav --ckpt checkpoints/crn_small/best.pt
uv run python -m ans.enhance a.wav b.flac -o out_dir/ --classical
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

from ans.dsp import SAMPLE_RATE
from ans.models import ClassicalEnhancer, load_checkpoint
from ans.stream import stream_model


def load_audio(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        x = soxr.resample(x, sr, SAMPLE_RATE, quality="VHQ")
    return x.astype(np.float32)


@torch.no_grad()
def enhance_array(x: np.ndarray, model, stream: bool = False, atten_lim_db: float | None = None) -> np.ndarray:
    """Aligned enhancement (lookahead-aware via ``model.enhance`` / :func:`ans.stream.stream_model`)."""
    if isinstance(model, ClassicalEnhancer):
        return model.enhance_np(x)
    if stream:
        return stream_model(x, model, atten_lim_db)
    if atten_lim_db is None:
        return model.enhance(torch.from_numpy(x)[None])[0].numpy()
    return model.enhance(torch.from_numpy(x)[None], atten_lim_db=atten_lim_db)[0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True, help="output file (single input) or directory")
    ap.add_argument("--ckpt", help="train.py checkpoint")
    ap.add_argument("--classical", action="store_true", help="use the Wiener/min-statistics baseline")
    ap.add_argument("--stream", action="store_true", help="use the frame-by-frame streaming loop")
    ap.add_argument("--atten-lim-db", type=float, help="maximum attenuation in dB (default: unlimited)")
    args = ap.parse_args()
    if not (args.ckpt or args.classical):
        ap.error("pass --ckpt or --classical")
    model = ClassicalEnhancer() if args.classical else load_checkpoint(args.ckpt, "cpu")

    out = Path(args.out)
    multi = len(args.inputs) > 1 or out.is_dir() or args.out.endswith("/")
    if multi:
        out.mkdir(parents=True, exist_ok=True)
    for inp in args.inputs:
        y = enhance_array(load_audio(inp), model, args.stream, args.atten_lim_db)
        dst = out / f"{Path(inp).stem}_enhanced.wav" if multi else out
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), np.clip(y, -1, 1), SAMPLE_RATE, subtype="PCM_16")
        print(f"{inp} -> {dst}")


if __name__ == "__main__":
    main()
