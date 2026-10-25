import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  analysisFrame,
  ComplexFFT,
  HOP,
  istft,
  N_BINS,
  N_FFT,
  RealFFT,
  sqrtHannPeriodic,
  stft,
  StreamingSTFT,
} from '../engine/dsp';
import { BypassModel, type FrameModel, type FrameResult } from '../engine/model';
import { Enhancer, enhanceOffline } from '../engine/enhancer';
import { resampleOffline, StreamResampler } from '../engine/resample';
import { runSelfChecks } from '../engine/diagnostics';
import { siSdr } from '../lib/audio';

interface NumpyFixture {
  x: number[];
  X_re: number[];
  X_im: number[];
  Z_re: number[];
  Z_im: number[];
  z: number[];
  sig: number[];
  frames_re: number[][];
  frames_im: number[][];
}
const np = JSON.parse(readFileSync(new URL('./fixtures/numpy_fft.json', import.meta.url), 'utf8')) as NumpyFixture;

function rand(n: number, seed = 1): Float32Array {
  let s = seed;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 2 ** 32) * 2 - 1;
  }
  return out;
}

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function naiveDft(re: ArrayLike<number>, im: ArrayLike<number>, sign = -1): [Float64Array, Float64Array] {
  const n = re.length;
  const oRe = new Float64Array(n);
  const oIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let a = 0;
    let b = 0;
    for (let t = 0; t < n; t++) {
      const ang = (sign * 2 * Math.PI * k * t) / n;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      a += re[t] * c - im[t] * s;
      b += re[t] * s + im[t] * c;
    }
    oRe[k] = a;
    oIm[k] = b;
  }
  return [oRe, oIm];
}

describe('FFT', () => {
  it.each([320, 160, 64, 45, 7, 1])('complex FFT(%i) matches a naive DFT', (n) => {
    const re = rand(n, 3);
    const im = rand(n, 4);
    const fft = new ComplexFFT(n);
    const oRe = new Float64Array(n);
    const oIm = new Float64Array(n);
    fft.forward(re, im, oRe, oIm);
    const [eRe, eIm] = naiveDft(re, im);
    expect(maxAbs(oRe, eRe)).toBeLessThan(1e-9);
    expect(maxAbs(oIm, eIm)).toBeLessThan(1e-9);
  });

  it('rfft(320) matches numpy.fft.rfft', () => {
    const fft = new RealFFT(N_FFT);
    const re = new Float64Array(N_BINS);
    const im = new Float64Array(N_BINS);
    fft.forward(np.x, re, im);
    expect(maxAbs(re, np.X_re)).toBeLessThan(1e-11);
    expect(maxAbs(im, np.X_im)).toBeLessThan(1e-11);
  });

  it('irfft(320) matches numpy.fft.irfft (ignores DC/Nyquist imag)', () => {
    const fft = new RealFFT(N_FFT);
    const z = new Float64Array(N_FFT);
    fft.inverse(np.Z_re, np.Z_im, z);
    expect(maxAbs(z, np.z)).toBeLessThan(1e-12);
  });

  it('rfft -> irfft round trip', () => {
    const fft = new RealFFT(N_FFT);
    const x = rand(N_FFT, 9);
    const re = new Float64Array(N_BINS);
    const im = new Float64Array(N_BINS);
    const y = new Float64Array(N_FFT);
    fft.forward(x, re, im);
    fft.inverse(re, im, y);
    expect(maxAbs(y, x)).toBeLessThan(1e-12);
  });
});

describe('STFT (§3.1)', () => {
  it('window is periodic sqrt-Hann and satisfies COLA (w² sums to 1 at 50 % overlap)', () => {
    const w = sqrtHannPeriodic(N_FFT);
    expect(w[0]).toBe(0);
    expect(w[160]).toBeCloseTo(1, 12);
    for (let i = 0; i < HOP; i++) expect(w[i] ** 2 + w[i + HOP] ** 2).toBeCloseTo(1, 12);
  });

  it('analysis frames match numpy (window * x[t*hop : t*hop+320] -> rfft)', () => {
    const sig = Float32Array.from(np.sig);
    np.frames_re.forEach((fr, t) => {
      const ours = analysisFrame(sig.subarray(t * HOP, t * HOP + N_FFT));
      for (let k = 0; k < N_BINS; k++) {
        // the signal is float32 in JS, float64 in numpy -> allow float32 rounding
        expect(Math.abs(ours[2 * k] - fr[k])).toBeLessThan(1e-4);
        expect(Math.abs(ours[2 * k + 1] - np.frames_im[t][k])).toBeLessThan(1e-4);
      }
    });
  });

  it('STFT -> iSTFT reconstructs the interior exactly (center=false)', () => {
    const x = rand(16000 + 123, 5);
    const frames = stft(x, false);
    expect(frames.length).toBe(1 + Math.floor((x.length - N_FFT) / HOP));
    const y = istft(frames);
    expect(maxAbs(y.subarray(HOP, y.length - HOP), x.subarray(HOP, y.length - HOP))).toBeLessThan(1e-6);
  });

  it('STFT -> iSTFT reconstructs everything with center=true', () => {
    const x = rand(4000 + 7, 6);
    const frames = stft(x, true);
    expect(frames.length).toBe(Math.ceil(x.length / HOP) + 1);
    expect(maxAbs(istft(frames, x.length, true), x)).toBeLessThan(1e-6);
  });
});

describe('streaming == offline', () => {
  it('StreamingSTFT frames equal offline center=true frames, and OLA output equals offline iSTFT', () => {
    const x = rand(HOP * 50, 8);
    const offline = stft(x, true);
    const s = new StreamingSTFT();
    const out = new Float32Array(x.length);
    for (let t = 0; t < 50; t++) {
      const spec = s.analyze(x.subarray(t * HOP, (t + 1) * HOP));
      expect(maxAbs(spec, offline[t])).toBe(0);
      out.set(s.synthesize(spec), t * HOP);
    }
    // streaming output n reconstructs input n - HOP
    expect(maxAbs(out.subarray(HOP), x.subarray(0, x.length - HOP))).toBeLessThan(1e-6);
    expect(maxAbs(out.subarray(0, HOP), new Float32Array(HOP))).toBeLessThan(1e-6);
  });

  it('Enhancer.push is independent of chunk size and enhanceOffline aligns with the input', async () => {
    // A stateful toy model: spectral gain that depends on the running state.
    const toy: FrameModel = {
      kind: 'onnx',
      stateShape: [2],
      stateSize: 2,
      supportsAttenLimit: false,
      attenLimitDb: null,
      setAttenLimit: () => undefined,
      initialState: () => new Float32Array(2),
      dispose: () => Promise.resolve(),
      processFrame(spec, state): Promise<FrameResult> {
        const g = 0.5 + 0.4 * Math.tanh(state[0]);
        const out = spec.map((v) => v * g);
        let e = 0;
        for (let i = 0; i < spec.length; i++) e += spec[i] * spec[i];
        return Promise.resolve({ specOut: out, stateOut: Float32Array.from([0.9 * state[0] + 0.001 * Math.sqrt(e), state[1] + 1]) });
      },
    };
    const x = rand(HOP * 40 + 77, 12);
    const a = new Enhancer(toy);
    const ya = await a.push(x);
    const b = new Enhancer(toy);
    const parts: Float32Array[] = [];
    for (let i = 0; i < x.length; i += 37) parts.push(await b.push(x.subarray(i, Math.min(x.length, i + 37))));
    const yb = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let off = 0;
    for (const p of parts) {
      yb.set(p, off);
      off += p.length;
    }
    expect(ya.length).toBe(Math.floor(x.length / HOP) * HOP);
    expect(maxAbs(ya, yb)).toBe(0);

    const { audio } = await enhanceOffline(toy, x);
    expect(audio.length).toBe(x.length);
    expect(maxAbs(audio.subarray(0, ya.length - HOP), ya.subarray(HOP))).toBeLessThan(1e-7);
  });

  it('bypass enhanceOffline is the identity', async () => {
    const x = rand(12345, 2);
    const { audio } = await enhanceOffline(new BypassModel([2, 1, 256]), x);
    expect(maxAbs(audio, x)).toBeLessThan(1e-6);
  });

  it('diagnostics self-checks all pass', async () => {
    const res = await runSelfChecks();
    for (const r of res) expect(r.status, `${r.name}: ${r.detail ?? r.maxAbsErr}`).toBe('pass');
  });
});

describe('resampler', () => {
  function tone(rate: number, f: number, n: number): Float32Array {
    return Float32Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / rate));
  }

  it.each([
    [48000, 16000],
    [44100, 16000],
    [16000, 48000],
    [16000, 44100],
  ])('%i -> %i preserves an in-band tone (SI-SDR > 40 dB)', (inR, outR) => {
    const x = tone(inR, 1000, inR);
    const y = resampleOffline(x, inR, outR);
    expect(y.length).toBe(outR);
    const ref = tone(outR, 1000, outR);
    // ignore filter edges
    const e = Math.round(outR * 0.01);
    expect(siSdr(y.subarray(e, -e), ref.subarray(e, -e))).toBeGreaterThan(40);
  });

  it('48k -> 16k rejects an out-of-band tone (> 50 dB)', () => {
    const x = tone(48000, 12000, 48000);
    const y = resampleOffline(x, 48000, 16000);
    let p = 0;
    for (let i = 200; i < y.length - 200; i++) p = Math.max(p, Math.abs(y[i]));
    expect(20 * Math.log10(p)).toBeLessThan(-50);
  });

  it('streaming chunks equal one-shot processing', () => {
    const x = tone(48000, 440, 4800);
    const a = new StreamResampler(48000, 16000).process(x);
    const r = new StreamResampler(48000, 16000);
    const parts: number[] = [];
    for (let i = 0; i < x.length; i += 128) parts.push(...r.process(x.subarray(i, i + 128)));
    expect(parts.length).toBe(a.length);
    expect(maxAbs(parts, a)).toBeLessThan(1e-7);
  });
});

describe('SI-SDR', () => {
  it('is scale invariant and large for identical signals', () => {
    const x = rand(8000, 1);
    expect(siSdr(x, x)).toBeGreaterThan(60);
    expect(siSdr(x.map((v) => v * 0.3), x)).toBeGreaterThan(60);
    const noisy = x.map((v, i) => v + 0.1 * Math.sin(i));
    expect(siSdr(noisy, x)).toBeLessThan(siSdr(x, x));
  });
});
