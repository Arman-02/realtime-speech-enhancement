/**
 * DSP contract (PLAN.md §3.1). Must match `ml/ans/dsp.py` + `ml/ans/stream.py` exactly.
 *
 *   sample rate 16 kHz · n_fft 320 · hop 160 · window sqrt(hann(320, periodic)) at analysis AND synthesis
 *   frame t = window * x[t*hop : t*hop + 320] -> rfft (unnormalised)       -> 161 bins
 *   synthesis = window * irfft(Y_t, n=320) (1/N), overlap-added at t*hop     (periodic Hann sums to 1)
 *
 * Spectra are interleaved `Float32Array(161 * 2)` = [re0, im0, re1, im1, ...], which is exactly the
 * row-major memory layout of the ONNX `[1, 1, 161, 2]` tensors.
 *
 * 320 = 4·4·4·5 is not a power of two, so this file carries a small mixed-radix (kissfft-style)
 * complex FFT computed in float64. It reproduces numpy's rfft/irfft (norm="backward") to ~1e-13.
 */

export const SAMPLE_RATE = 16_000;
export const N_FFT = 320;
export const HOP = 160;
export const N_BINS = N_FFT / 2 + 1;
export const SPEC_LEN = N_BINS * 2;

/** Periodic sqrt-Hann window, computed in float64 like numpy: sqrt(0.5 - 0.5 cos(2πi/n)). */
export function sqrtHannPeriodic(n: number = N_FFT): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = Math.sqrt(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
  return w;
}

export const WINDOW = sqrtHannPeriodic(N_FFT);

// ---------------------------------------------------------------------------------------------
// Mixed-radix complex FFT (decimation in time, recursive, generic butterflies).
// ---------------------------------------------------------------------------------------------

function factorize(n: number): number[] {
  const out: number[] = [];
  let p = 4;
  const floorSqrt = Math.floor(Math.sqrt(n));
  let rem = n;
  do {
    while (rem % p !== 0) {
      if (p === 4) p = 2;
      else if (p === 2) p = 3;
      else p += 2;
      if (p > floorSqrt) p = rem;
    }
    rem /= p;
    out.push(p, rem);
  } while (rem > 1);
  return out;
}

export class ComplexFFT {
  readonly n: number;
  private readonly factors: number[];
  private readonly twRe: Float64Array;
  private readonly twIm: Float64Array;
  private readonly scRe: Float64Array;
  private readonly scIm: Float64Array;
  private readonly tmpRe: Float64Array;
  private readonly tmpIm: Float64Array;

  constructor(n: number) {
    if (!Number.isInteger(n) || n < 1) throw new Error(`invalid FFT size ${n}`);
    this.n = n;
    this.factors = factorize(n);
    this.twRe = new Float64Array(n);
    this.twIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = (-2 * Math.PI * i) / n;
      this.twRe[i] = Math.cos(a);
      this.twIm[i] = Math.sin(a);
    }
    let maxP = 1;
    for (let i = 0; i < this.factors.length; i += 2) maxP = Math.max(maxP, this.factors[i]);
    this.scRe = new Float64Array(maxP);
    this.scIm = new Float64Array(maxP);
    this.tmpRe = new Float64Array(n);
    this.tmpIm = new Float64Array(n);
  }

  /** Unnormalised forward DFT: X[k] = Σ x[n] e^{-2πikn/N}. Output arrays must not alias inputs. */
  forward(inRe: ArrayLike<number>, inIm: ArrayLike<number>, outRe: Float64Array, outIm: Float64Array): void {
    this.work(outRe, outIm, 0, inRe, inIm, 0, 1, 0);
  }

  /** Unnormalised inverse DFT (no 1/N): x[n] = Σ X[k] e^{+2πikn/N}. */
  inverse(inRe: ArrayLike<number>, inIm: ArrayLike<number>, outRe: Float64Array, outIm: Float64Array): void {
    // ifft(X) = conj(fft(conj(X)))
    const cRe = this.tmpRe;
    const cIm = this.tmpIm;
    for (let i = 0; i < this.n; i++) {
      cRe[i] = inRe[i];
      cIm[i] = -inIm[i];
    }
    this.work(outRe, outIm, 0, cRe, cIm, 0, 1, 0);
    for (let i = 0; i < this.n; i++) outIm[i] = -outIm[i];
  }

  private work(
    outRe: Float64Array,
    outIm: Float64Array,
    outOff: number,
    inRe: ArrayLike<number>,
    inIm: ArrayLike<number>,
    inOff: number,
    fstride: number,
    fi: number,
  ): void {
    const p = this.factors[fi];
    const m = this.factors[fi + 1];
    if (m === 1) {
      for (let k = 0; k < p; k++) {
        const src = inOff + k * fstride;
        outRe[outOff + k] = inRe[src];
        outIm[outOff + k] = inIm[src];
      }
    } else {
      for (let k = 0; k < p; k++) {
        this.work(outRe, outIm, outOff + k * m, inRe, inIm, inOff + k * fstride, fstride * p, fi + 2);
      }
    }
    this.butterfly(outRe, outIm, outOff, fstride, m, p);
  }

  private butterfly(re: Float64Array, im: Float64Array, off: number, fstride: number, m: number, p: number): void {
    const n = this.n;
    const twRe = this.twRe;
    const twIm = this.twIm;
    const scRe = this.scRe;
    const scIm = this.scIm;
    for (let u = 0; u < m; u++) {
      for (let q1 = 0, k = u; q1 < p; q1++, k += m) {
        scRe[q1] = re[off + k];
        scIm[q1] = im[off + k];
      }
      for (let q1 = 0, k = u; q1 < p; q1++, k += m) {
        let accRe = scRe[0];
        let accIm = scIm[0];
        let tw = 0;
        for (let q = 1; q < p; q++) {
          tw += fstride * k;
          if (tw >= n) tw -= n;
          const a = scRe[q];
          const b = scIm[q];
          const c = twRe[tw];
          const d = twIm[tw];
          accRe += a * c - b * d;
          accIm += a * d + b * c;
        }
        re[off + k] = accRe;
        im[off + k] = accIm;
      }
    }
  }
}

/** numpy-compatible rfft / irfft (norm="backward") for an even length n. */
export class RealFFT {
  readonly n: number;
  readonly bins: number;
  private readonly fft: ComplexFFT;
  private readonly zeros: Float64Array;
  private readonly bufRe: Float64Array;
  private readonly bufIm: Float64Array;
  private readonly outRe: Float64Array;
  private readonly outIm: Float64Array;

  constructor(n: number) {
    if (n % 2 !== 0) throw new Error('RealFFT requires an even length');
    this.n = n;
    this.bins = n / 2 + 1;
    this.fft = new ComplexFFT(n);
    this.zeros = new Float64Array(n);
    this.bufRe = new Float64Array(n);
    this.bufIm = new Float64Array(n);
    this.outRe = new Float64Array(n);
    this.outIm = new Float64Array(n);
  }

  /** x (length n) -> re/im (length n/2+1). */
  forward(x: ArrayLike<number>, re: Float64Array, im: Float64Array): void {
    this.fft.forward(x, this.zeros, this.outRe, this.outIm);
    for (let k = 0; k < this.bins; k++) {
      re[k] = this.outRe[k];
      im[k] = this.outIm[k];
    }
  }

  /**
   * re/im (length n/2+1) -> x (length n), scaled by 1/n. Like numpy (pocketfft c2r), the imaginary
   * parts of the DC and Nyquist bins are ignored (the spectrum is treated as Hermitian).
   */
  inverse(re: ArrayLike<number>, im: ArrayLike<number>, x: Float64Array): void {
    const n = this.n;
    const h = n / 2;
    const bRe = this.bufRe;
    const bIm = this.bufIm;
    bRe[0] = re[0];
    bIm[0] = 0;
    bRe[h] = re[h];
    bIm[h] = 0;
    for (let k = 1; k < h; k++) {
      bRe[k] = re[k];
      bIm[k] = im[k];
      bRe[n - k] = re[k];
      bIm[n - k] = -im[k];
    }
    this.fft.inverse(bRe, bIm, this.outRe, this.outIm);
    const s = 1 / n;
    for (let i = 0; i < n; i++) x[i] = this.outRe[i] * s;
  }
}

// ---------------------------------------------------------------------------------------------
// Frame-level STFT / iSTFT (§3.1)
// ---------------------------------------------------------------------------------------------

const rfft = new RealFFT(N_FFT);
const fRe = new Float64Array(N_BINS);
const fIm = new Float64Array(N_BINS);
const fBuf = new Float64Array(N_FFT);
const fOut = new Float64Array(N_FFT);

/**
 * Analysis of one raw (un-windowed) 320-sample frame -> interleaved spectrum (161 × [re, im]).
 * Mirrors `analysis_frame` in ml/ans/dsp.py (float64 maths, float32 result).
 */
export function analysisFrame(frame: ArrayLike<number>, out: Float32Array = new Float32Array(SPEC_LEN)): Float32Array {
  for (let i = 0; i < N_FFT; i++) fBuf[i] = WINDOW[i] * frame[i];
  rfft.forward(fBuf, fRe, fIm);
  for (let k = 0; k < N_BINS; k++) {
    out[2 * k] = fRe[k];
    out[2 * k + 1] = fIm[k];
  }
  return out;
}

/** Synthesis of one spectrum -> windowed 320-sample frame, ready for overlap-add (`synthesis_frame`). */
export function synthesisFrame(spec: ArrayLike<number>, out: Float32Array = new Float32Array(N_FFT)): Float32Array {
  for (let k = 0; k < N_BINS; k++) {
    fRe[k] = spec[2 * k];
    fIm[k] = spec[2 * k + 1];
  }
  rfft.inverse(fRe, fIm, fOut);
  for (let i = 0; i < N_FFT; i++) out[i] = WINDOW[i] * fOut[i];
  return out;
}

/**
 * Offline STFT. `center=false`: frame t = x[t*hop : t*hop+320] (T = 1 + floor((N-320)/hop)).
 * `center=true`: prepends `hop` zeros (the streaming history) and zero-pads the end so that every
 * sample is covered by two windows (T = ceil(N/hop) + 1) — identical to the Python `center=True`.
 */
export function stft(x: Float32Array, center = false): Float32Array[] {
  let src = x;
  if (center) {
    const t = Math.ceil(x.length / HOP) + 1;
    const total = (t - 1) * HOP + N_FFT;
    src = new Float32Array(total);
    src.set(x, HOP);
  }
  const frames: Float32Array[] = [];
  if (src.length < N_FFT) return frames;
  const count = 1 + Math.floor((src.length - N_FFT) / HOP);
  for (let t = 0; t < count; t++) frames.push(analysisFrame(src.subarray(t * HOP, t * HOP + N_FFT)));
  return frames;
}

/** Offline iSTFT by overlap-add. With `center=true` the leading hop is dropped (inverse of `stft(x, true)`). */
export function istft(frames: ArrayLike<number>[], length?: number, center = false): Float32Array {
  const total = frames.length === 0 ? 0 : (frames.length - 1) * HOP + N_FFT;
  const y = new Float32Array(total);
  const buf = new Float32Array(N_FFT);
  for (let t = 0; t < frames.length; t++) {
    synthesisFrame(frames[t], buf);
    const off = t * HOP;
    for (let i = 0; i < N_FFT; i++) y[off + i] += buf[i];
  }
  let out = center ? y.subarray(HOP) : y;
  if (length !== undefined) out = out.subarray(0, Math.min(length, out.length));
  return out.slice();
}

/**
 * Streaming analysis buffer + OLA tail (§3.1 "Streaming state"). Both start at zeros. Each 160-sample
 * hop produces one spectrum; each synthesized spectrum produces 160 output samples. Output sample n
 * is the reconstruction of input sample n - 160 (one hop of buffering, 20 ms algorithmic latency).
 */
export class StreamingSTFT {
  private readonly history = new Float32Array(N_FFT - HOP);
  private readonly tail = new Float32Array(N_FFT - HOP);
  private readonly frame = new Float32Array(N_FFT);
  private readonly synth = new Float32Array(N_FFT);

  reset(): void {
    this.history.fill(0);
    this.tail.fill(0);
  }

  /** One hop (160 samples) -> spectrum of [history, hop]. */
  analyze(hop: ArrayLike<number>, out?: Float32Array): Float32Array {
    const f = this.frame;
    f.set(this.history, 0);
    for (let i = 0; i < HOP; i++) f[N_FFT - HOP + i] = hop[i];
    this.history.set(f.subarray(HOP));
    return analysisFrame(f, out);
  }

  /** Spectrum -> 160 output samples (tail + frame[:160]); keeps frame[160:] as the new tail. */
  synthesize(spec: ArrayLike<number>, out: Float32Array = new Float32Array(HOP)): Float32Array {
    const s = synthesisFrame(spec, this.synth);
    for (let i = 0; i < HOP; i++) out[i] = this.tail[i] + s[i];
    this.tail.set(s.subarray(HOP));
    return out;
  }
}

// ---------------------------------------------------------------------------------------------
// Helpers for visualisation
// ---------------------------------------------------------------------------------------------

/** 10·log10(|X|² + eps) per bin, written into `out` (length 161). */
export function specToDb(spec: ArrayLike<number>, out: Float32Array, offset = 0): void {
  for (let k = 0; k < N_BINS; k++) {
    const re = spec[2 * k];
    const im = spec[2 * k + 1];
    out[offset + k] = 10 * Math.log10(re * re + im * im + 1e-10);
  }
}

/** Log-magnitude spectrogram (center framing) as a flat Float32Array [frames × 161]. */
export function logSpectrogram(x: Float32Array): { db: Float32Array; frames: number; max: number } {
  const frames = Math.ceil(x.length / HOP) + 1;
  const padded = new Float32Array((frames - 1) * HOP + N_FFT);
  padded.set(x, HOP);
  const db = new Float32Array(frames * N_BINS);
  const spec = new Float32Array(SPEC_LEN);
  let max = -Infinity;
  for (let t = 0; t < frames; t++) {
    analysisFrame(padded.subarray(t * HOP, t * HOP + N_FFT), spec);
    specToDb(spec, db, t * N_BINS);
  }
  for (let i = 0; i < db.length; i++) if (db[i] > max) max = db[i];
  return { db, frames, max };
}
