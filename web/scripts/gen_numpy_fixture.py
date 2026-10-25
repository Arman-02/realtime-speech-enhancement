"""Regenerate src/test/fixtures/numpy_fft.json (numpy rfft/irfft reference for the JS DSP tests).

    python3 scripts/gen_numpy_fixture.py      # any Python with numpy, e.g. ml/.venv/bin/python
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(1234)
n = 320
x = rng.standard_normal(n)
X = np.fft.rfft(x)
Z = rng.standard_normal(161) + 1j * rng.standard_normal(161)  # non-Hermitian DC/Nyquist imag on purpose
z = np.fft.irfft(Z, n=n)
win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))
sig = rng.standard_normal(1600) * 0.3
frames = [np.fft.rfft(win * sig[t * 160 : t * 160 + 320]) for t in range(1 + (1600 - 320) // 160)]

out = {
    "x": x.tolist(),
    "X_re": X.real.tolist(),
    "X_im": X.imag.tolist(),
    "Z_re": Z.real.tolist(),
    "Z_im": Z.imag.tolist(),
    "z": z.tolist(),
    "sig": sig.tolist(),
    "frames_re": [f.real.tolist() for f in frames],
    "frames_im": [f.imag.tolist() for f in frames],
}
p = Path(__file__).resolve().parent.parent / "src/test/fixtures/numpy_fft.json"
p.write_text(json.dumps(out))
print("wrote", p)
