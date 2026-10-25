"""Standalone worker for the reference systems (run in an isolated environment by ``reference.py``).

Imports nothing from ``ans``. Reads a TSV of ``id<TAB>input.wav<TAB>output.wav`` (16 kHz in/out), runs the
system at 48 kHz (soxr VHQ resampling 16 -> 48 -> 16 kHz), compensates the system's fixed delay so the
output is sample-aligned with the input, and writes float32 wavs plus ``timing.json``.

  python ref_worker.py rnnoise jobs.tsv
  python ref_worker.py deepfilternet jobs.tsv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR, SR_REF = 16000, 48000


def rnnoise_fn():
    import ctypes

    from pyrnnoise import rnnoise as r

    frame = r.FRAME_SIZE  # 480 samples = 10 ms at 48 kHz

    def run(x48: np.ndarray) -> np.ndarray:
        state = r.lib.rnnoise_create(None)
        try:
            n = len(x48)
            # 20 ms analysis window, 10 ms hop: output is delayed by two frames (960 samples, measured).
            delay = 2 * frame
            pad = np.concatenate([x48, np.zeros(delay + (-n) % frame, np.float32)]) * 32768.0
            out = np.zeros_like(pad, dtype=np.float32)
            buf_in = np.zeros(frame, np.float32)
            buf_out = np.zeros(frame, np.float32)
            p_in = buf_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            p_out = buf_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            for i in range(0, len(pad), frame):
                buf_in[:] = pad[i : i + frame]
                r.lib.rnnoise_process_frame(state, p_out, p_in)
                out[i : i + frame] = buf_out
            return (out[delay : delay + n] / 32768.0).astype(np.float32)
        finally:
            r.lib.rnnoise_destroy(state)

    return run, {"system": "RNNoise", "package": "pyrnnoise", "frame_ms": 10}


def deepfilternet_fn():
    import torch
    from df.enhance import enhance, init_df

    torch.set_num_threads(1)
    model, df_state, _ = init_df()  # DeepFilterNet3, weights downloaded to the user cache at runtime

    def run(x48: np.ndarray) -> np.ndarray:
        y = enhance(model, df_state, torch.from_numpy(x48)[None], pad=True)  # pad=True compensates delay
        return y[0].numpy().astype(np.float32)

    return run, {"system": "DeepFilterNet3", "package": "deepfilternet", "frame_ms": 10}


def main() -> None:
    system, jobs_path = sys.argv[1], Path(sys.argv[2])
    run, info = {"rnnoise": rnnoise_fn, "deepfilternet": deepfilternet_fn}[system]()
    jobs = [line.rstrip("\n").split("\t") for line in jobs_path.read_text().splitlines() if line.strip()]
    proc_s = audio_s = 0.0
    for k, (_uid, src, dst) in enumerate(jobs):
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = soxr.resample(x, sr, SR, quality="VHQ").astype(np.float32)
        t0 = time.perf_counter()
        x48 = soxr.resample(x, SR, SR_REF, quality="VHQ").astype(np.float32)
        y48 = run(x48)
        y = soxr.resample(y48, SR_REF, SR, quality="VHQ").astype(np.float32)[: len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        proc_s += time.perf_counter() - t0
        audio_s += len(x) / SR
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, y, SR, subtype="FLOAT")
        if (k + 1) % 100 == 0:
            print(f"{system}: {k + 1}/{len(jobs)}", flush=True)
    timing = {**info, "clips": len(jobs), "audio_s": audio_s, "proc_s": proc_s, "rtf": proc_s / max(audio_s, 1e-9)}
    (jobs_path.parent / f"timing_{system}.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
