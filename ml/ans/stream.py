"""Frame-by-frame reference streaming enhancer — mirrors the JS engine (PLAN.md §3.1, §5.2).

State per stream: a 160-sample input history (zeros at start) and a 160-sample OLA tail (zeros at
start). Every 160 input samples produce exactly 160 output samples; output sample ``n`` is the
enhanced version of input sample ``n - hop`` (one hop of buffering delay, 20 ms algorithmic latency
in total including the window).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ans.dsp import HOP, N_FFT, analysis_frame, synthesis_frame

FrameFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""``(spec[1,1,161,2], state) -> (spec_out[1,1,161,2], state)`` — the ONNX graph signature."""


@dataclass
class StreamingSTFT:
    """Analysis buffer + OLA tail for one stream."""

    history: np.ndarray = field(default_factory=lambda: np.zeros(N_FFT - HOP, np.float32))
    tail: np.ndarray = field(default_factory=lambda: np.zeros(N_FFT - HOP, np.float32))

    def analyze(self, hop_samples: np.ndarray) -> np.ndarray:
        frame = np.concatenate([self.history, hop_samples.astype(np.float32)])
        self.history = frame[HOP:]
        return analysis_frame(frame)

    def synthesize(self, spec: np.ndarray) -> np.ndarray:
        frame = synthesis_frame(spec)
        out = self.tail + frame[:HOP]
        self.tail = frame[HOP:].copy()
        return out


class StreamingEnhancer:
    """Push arbitrary-size chunks; returns as many output samples as full hops were completed."""

    def __init__(self, frame_fn: FrameFn, state_shape: list[int] | tuple[int, ...]) -> None:
        self.frame_fn = frame_fn
        self.state_shape = tuple(state_shape)
        self.reset()

    def reset(self) -> None:
        self.stft = StreamingSTFT()
        self.state = np.zeros(self.state_shape, np.float32)
        self.pending = np.zeros(0, np.float32)
        if hasattr(self.frame_fn, "reset"):
            self.frame_fn.reset()

    def process_hop(self, hop_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One hop -> (output samples, input spectrum, model output spectrum)."""
        spec = self.stft.analyze(hop_samples)
        spec_out, self.state = self.frame_fn(spec[None, None], self.state)
        spec_out = np.asarray(spec_out, np.float32)[0, 0]
        return self.stft.synthesize(spec_out), spec, spec_out

    def push(self, samples: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.pending, samples.astype(np.float32)])
        n = len(buf) // HOP
        outs = [self.process_hop(buf[i * HOP : (i + 1) * HOP])[0] for i in range(n)]
        self.pending = buf[n * HOP :]
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def lookahead_of(model) -> int:
    """Lookahead k (frames) of a model; 0 for anything that doesn't declare one (§4.11)."""
    cfg = getattr(model, "cfg", None)
    return int(getattr(cfg, "lookahead", 0) or 0)


class TorchFrameFn:
    """Wrap a streaming torch model as a FrameFn, optionally with an attenuation limit.

    Lookahead-k models emit the mask for frame ``t`` after seeing frame ``t`` but apply it to the input
    frame ``t - k``, so this keeps a k-frame input buffer (cleared by :meth:`reset`).
    """

    def __init__(self, model: torch.nn.Module, atten_lim_db: float | None = None) -> None:
        self.model = model
        self.atten = None if atten_lim_db is None else torch.tensor([float(atten_lim_db)])
        self.k = lookahead_of(model)
        self.reset()

    def reset(self) -> None:
        self.buf: deque[torch.Tensor] = deque(torch.zeros(1, 1, N_FFT // 2 + 1, 2) for _ in range(self.k))

    @torch.no_grad()
    def __call__(self, spec: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, st = torch.from_numpy(spec), torch.from_numpy(state)
        if self.k == 0:
            out, st = self.model(x, st) if self.atten is None else self.model(x, st, atten_lim_db=self.atten)
            return out.numpy(), st.numpy()
        from ans.models.crn import apply_mask

        m, st, _ = self.model.masks(x, st, self.atten)
        self.buf.append(x)
        return apply_mask(self.buf.popleft(), m).numpy(), st.numpy()


def torch_frame_fn(model: torch.nn.Module, atten_lim_db: float | None = None) -> FrameFn:
    """Wrap a streaming torch model (``forward(spec, state) -> (spec, state)``) as a FrameFn."""
    return TorchFrameFn(model, atten_lim_db)


def ort_frame_fn(session, atten_lim_db: float | None = None) -> FrameFn:
    """Wrap an ``onnxruntime.InferenceSession`` for the §3.2 graph as a FrameFn.

    ``atten_lim_db`` is fed only when the graph declares it (v0.2); ``None`` -> 100 dB (unlimited).
    """
    has_atten = any(i.name == "atten_lim_db" for i in session.get_inputs())
    lim = np.array([100.0 if atten_lim_db is None else atten_lim_db], np.float32)

    def fn(spec: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feeds = {"spec": spec, "state_in": state}
        if has_atten:
            feeds["atten_lim_db"] = lim
        out, st = session.run(["spec_out", "state_out"], feeds)
        return out, st

    return fn


def identity_frame_fn(spec: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return spec, state


def enhance_streaming(
    x: np.ndarray, frame_fn: FrameFn, state_shape, align: bool = False, lookahead: int = 0
) -> np.ndarray:
    """Run a whole signal through the streaming loop.

    ``align=False`` returns exactly what a live stream would emit (``len(x)`` samples rounded down to
    whole hops, delayed by one hop, plus ``lookahead`` more hops for a lookahead-k model).
    ``align=True`` flushes ``1 + lookahead`` extra hops of zeros and removes the delay, so the output
    aligns with ``x`` and matches the model's offline ``enhance``.
    """
    enh = StreamingEnhancer(frame_fn, state_shape)
    if not align:
        return enh.push(x)
    n = len(x)
    delay = HOP * (1 + lookahead)
    pad = (-n) % HOP + delay
    y = enh.push(np.concatenate([x.astype(np.float32), np.zeros(pad, np.float32)]))
    return y[delay : delay + n]


def stream_model(x: np.ndarray, model: torch.nn.Module, atten_lim_db: float | None = None) -> np.ndarray:
    """Aligned frame-by-frame enhancement of a whole signal with a torch model (lookahead-aware)."""
    return enhance_streaming(
        x, torch_frame_fn(model, atten_lim_db), model.state_shape, align=True, lookahead=lookahead_of(model)
    )
