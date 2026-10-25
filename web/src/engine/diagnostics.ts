/**
 * Python/JS parity checks. `runSelfChecks` needs nothing external; `runVectorChecks` replays
 * `public/models/test_vectors.json` (written by ml/ans/export_onnx.py) through the JS DSP + model.
 * Shared by the #/diagnostics page (runs in the worker against onnxruntime-web) and the Vitest suite.
 */
import { analysisFrame, ComplexFFT, HOP, istft, N_BINS, N_FFT, RealFFT, SPEC_LEN, stft } from './dsp';
import { ATTEN_UNLIMITED_DB, BypassModel, type FrameModel } from './model';
import { Enhancer, enhanceOffline } from './enhancer';

export interface TestVectors {
  description?: string;
  conventions?: Record<string, string>;
  sample_rate: number;
  n_fft: number;
  hop: number;
  state_shape: number[];
  tolerances: { stft: number; model: number; waveform: number };
  input_wav: number[];
  stft_frames: number[][][];
  model_out_frames: number[][][];
  output_wav: number[];
  /** v0.2: attenuation limit used for the main case (absent = unlimited, 100 dB). */
  atten_lim_db?: number;
  [extra: string]: unknown;
}

/** One extra case run at a given attenuation limit (v0.2). Missing arrays fall back to the main case. */
export interface AttenCase {
  atten_lim_db: number;
  input_wav?: number[];
  stft_frames?: number[][][];
  model_out_frames?: number[][][];
  output_wav?: number[];
  tolerances?: Partial<TestVectors['tolerances']>;
}

function asAttenCase(v: unknown): AttenCase | null {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const db = typeof o.atten_lim_db === 'number' ? o.atten_lim_db : null;
  if (db === null || !Number.isFinite(db)) return null;
  if (!Array.isArray(o.model_out_frames) && !Array.isArray(o.output_wav)) return null;
  return o as unknown as AttenCase;
}

/**
 * Find the extra attenuation-limit case(s). §3.2 only says "one extra case run at atten_lim_db = 12", so
 * accept any top-level value that is such a case, or an array of them (e.g. `cases`, `atten_cases`,
 * `atten_lim_db_12`). The main case itself is never returned.
 */
export function findAttenCases(tv: TestVectors): AttenCase[] {
  const out: AttenCase[] = [];
  for (const [k, v] of Object.entries(tv)) {
    if (k === 'stft_frames' || k === 'model_out_frames' || k === 'output_wav' || k === 'input_wav') continue;
    const one = asAttenCase(v);
    if (one) out.push(one);
    else if (Array.isArray(v)) for (const item of v) {
      const c = asAttenCase(item);
      if (c) out.push(c);
    }
  }
  return out;
}

export type CheckStatus = 'pass' | 'fail' | 'skip';

export interface CheckResult {
  id: string;
  group: 'self' | 'vectors';
  name: string;
  status: CheckStatus;
  maxAbsErr?: number;
  tol?: number;
  detail?: string;
  ms?: number;
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (!(d <= m)) m = d; // also propagates NaN as NaN
  }
  return m;
}

function verdict(err: number, tol: number): CheckStatus {
  return Number.isFinite(err) && err <= tol ? 'pass' : 'fail';
}

function seededNoise(n: number, seed = 1): Float32Array {
  // Mulberry32 -> uniform [-1, 1)
  let s = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

function flattenFrame(frame: number[][]): Float32Array {
  const f = new Float32Array(SPEC_LEN);
  for (let k = 0; k < N_BINS; k++) {
    f[2 * k] = frame[k][0];
    f[2 * k + 1] = frame[k][1];
  }
  return f;
}

async function timed(fn: () => Promise<Omit<CheckResult, 'ms'>> | Omit<CheckResult, 'ms'>): Promise<CheckResult> {
  const t0 = now();
  try {
    const r = await fn();
    return { ...r, ms: now() - t0 };
  } catch (e) {
    return { id: 'error', group: 'self', name: 'Unexpected error', status: 'fail', detail: String(e), ms: now() - t0 };
  }
}

/** DSP checks that need no external files. */
export async function runSelfChecks(): Promise<CheckResult[]> {
  const results: CheckResult[] = [];

  results.push(
    await timed(() => {
      const x = seededNoise(N_FFT, 7);
      const fft = new RealFFT(N_FFT);
      const re = new Float64Array(N_BINS);
      const im = new Float64Array(N_BINS);
      fft.forward(x, re, im);
      let err = 0;
      for (let k = 0; k < N_BINS; k++) {
        let r = 0;
        let i = 0;
        for (let n = 0; n < N_FFT; n++) {
          const a = (-2 * Math.PI * k * n) / N_FFT;
          r += x[n] * Math.cos(a);
          i += x[n] * Math.sin(a);
        }
        err = Math.max(err, Math.abs(r - re[k]), Math.abs(i - im[k]));
      }
      return { id: 'rfft-vs-dft', group: 'self', name: 'rfft(320) vs naive DFT', status: verdict(err, 1e-9), maxAbsErr: err, tol: 1e-9 };
    }),
  );

  results.push(
    await timed(() => {
      const fft = new ComplexFFT(N_FFT);
      const re = seededNoise(N_FFT, 3);
      const im = seededNoise(N_FFT, 4);
      const oRe = new Float64Array(N_FFT);
      const oIm = new Float64Array(N_FFT);
      const bRe = new Float64Array(N_FFT);
      const bIm = new Float64Array(N_FFT);
      fft.forward(re, im, oRe, oIm);
      fft.inverse(oRe, oIm, bRe, bIm);
      let err = 0;
      for (let i = 0; i < N_FFT; i++) err = Math.max(err, Math.abs(bRe[i] / N_FFT - re[i]), Math.abs(bIm[i] / N_FFT - im[i]));
      return { id: 'fft-roundtrip', group: 'self', name: 'Complex FFT forward/inverse round trip', status: verdict(err, 1e-12), maxAbsErr: err, tol: 1e-12 };
    }),
  );

  results.push(
    await timed(() => {
      const x = seededNoise(16000, 11);
      const y = istft(stft(x, false), x.length);
      // Interior samples (covered by two windows) must reconstruct exactly.
      const err = maxAbsDiff(x.subarray(HOP, y.length - HOP), y.subarray(HOP, y.length - HOP));
      return { id: 'stft-roundtrip', group: 'self', name: 'STFT → iSTFT round trip (interior)', status: verdict(err, 1e-5), maxAbsErr: err, tol: 1e-5 };
    }),
  );

  results.push(
    await timed(async () => {
      const x = seededNoise(16000 + 37, 5);
      const offline = istft(stft(x, true), x.length, true);
      const { audio } = await enhanceOffline(new BypassModel(), x);
      const err = Math.max(maxAbsDiff(offline, audio), maxAbsDiff(x, audio));
      return { id: 'stream-eq-offline', group: 'self', name: 'Streaming == offline (bypass, aligned)', status: verdict(err, 1e-5), maxAbsErr: err, tol: 1e-5 };
    }),
  );

  return results;
}

/**
 * Replays the Python test vectors. `model` should be the ONNX model; model checks are skipped for bypass.
 * The main case runs at `tv.atten_lim_db` (default: unlimited); each extra attenuation case runs at its
 * own limit.
 */
export async function runVectorChecks(tv: TestVectors, model: FrameModel | null): Promise<CheckResult[]> {
  const results: CheckResult[] = [];
  const tol = tv.tolerances ?? { stft: 1e-4, model: 1e-3, waveform: 1e-3 };
  const x = Float32Array.from(tv.input_wav);
  const mainAtten = typeof tv.atten_lim_db === 'number' ? tv.atten_lim_db : ATTEN_UNLIMITED_DB;
  const attenCases = findAttenCases(tv);

  results.push(
    await timed(() => {
      const ok = tv.sample_rate === 16000 && tv.n_fft === N_FFT && tv.hop === HOP;
      return {
        id: 'params',
        group: 'vectors',
        name: 'DSP parameters match (16 kHz, 320/160)',
        status: ok ? 'pass' : 'fail',
        detail: `sample_rate=${tv.sample_rate} n_fft=${tv.n_fft} hop=${tv.hop}`,
      };
    }),
  );

  results.push(
    await timed(() => {
      let err = 0;
      for (let t = 0; t < tv.stft_frames.length; t++) {
        const ours = analysisFrame(x.subarray(t * HOP, t * HOP + N_FFT));
        err = Math.max(err, maxAbsDiff(ours, flattenFrame(tv.stft_frames[t])));
      }
      return {
        id: 'stft-frames',
        group: 'vectors',
        name: `STFT frames (${tv.stft_frames.length}) vs Python`,
        status: verdict(err, tol.stft),
        maxAbsErr: err,
        tol: tol.stft,
      };
    }),
  );

  if (!model || model.kind !== 'onnx') {
    const why = 'No ONNX model loaded (bypass mode)';
    results.push({ id: 'model-frames', group: 'vectors', name: 'Model outputs vs Python ORT', status: 'skip', detail: why });
    results.push({ id: 'output-wav', group: 'vectors', name: 'Full streaming waveform vs Python', status: 'skip', detail: why });
    for (const c of attenCases) {
      results.push({ id: `atten-${c.atten_lim_db}`, group: 'vectors', name: `atten_lim_db = ${c.atten_lim_db} dB case vs Python`, status: 'skip', detail: why });
    }
    return results;
  }

  results.push(
    await timed(() => {
      const same = model.stateShape.length === tv.state_shape.length && model.stateShape.every((v, i) => v === tv.state_shape[i]);
      return {
        id: 'state-shape',
        group: 'vectors',
        name: 'State shape (manifest vs test vectors)',
        status: same ? 'pass' : 'fail',
        detail: `[${model.stateShape.join(', ')}] vs [${tv.state_shape.join(', ')}]`,
      };
    }),
  );

  results.push(
    await timed(async () => {
      let state = model.initialState();
      let err = 0;
      for (let t = 0; t < tv.model_out_frames.length; t++) {
        const { specOut, stateOut } = await model.processFrame(flattenFrame(tv.stft_frames[t]), state, mainAtten);
        state = stateOut;
        err = Math.max(err, maxAbsDiff(specOut, flattenFrame(tv.model_out_frames[t])));
      }
      return {
        id: 'model-frames',
        group: 'vectors',
        name: `Model outputs (${tv.model_out_frames.length} frames, stateful) vs Python ORT`,
        status: verdict(err, tol.model),
        maxAbsErr: err,
        tol: tol.model,
      };
    }),
  );

  results.push(
    await timed(async () => {
      const enh = new Enhancer(model);
      enh.attenLimDb = mainAtten;
      const y = await enh.push(x);
      const ref = tv.output_wav;
      const err = y.length === ref.length ? maxAbsDiff(y, ref) : Infinity;
      return {
        id: 'output-wav',
        group: 'vectors',
        name: `Full streaming waveform (${ref.length} samples) vs Python`,
        status: verdict(err, tol.waveform),
        maxAbsErr: err,
        tol: tol.waveform,
        detail: y.length === ref.length ? undefined : `length ${y.length} vs ${ref.length}`,
      };
    }),
  );

  for (const c of attenCases) results.push(...(await runAttenCase(tv, c, model)));

  return results;
}

async function runAttenCase(tv: TestVectors, c: AttenCase, model: FrameModel): Promise<CheckResult[]> {
  const db = c.atten_lim_db;
  const tol = { ...(tv.tolerances ?? { stft: 1e-4, model: 1e-3, waveform: 1e-3 }), ...(c.tolerances ?? {}) };
  const out: CheckResult[] = [];
  if (!model.supportsAttenLimit) {
    out.push({
      id: `atten-${db}-input`,
      group: 'vectors',
      name: `atten_lim_db = ${db} dB case`,
      status: 'fail',
      detail: 'test_vectors.json has an attenuation case but the loaded ONNX graph has no atten_lim_db input',
    });
    return out;
  }
  if (Array.isArray(c.model_out_frames)) {
    const frames = c.stft_frames ?? tv.stft_frames;
    out.push(
      await timed(async () => {
        let state = model.initialState();
        let err = 0;
        for (let t = 0; t < c.model_out_frames!.length; t++) {
          const { specOut, stateOut } = await model.processFrame(flattenFrame(frames[t]), state, db);
          state = stateOut;
          err = Math.max(err, maxAbsDiff(specOut, flattenFrame(c.model_out_frames![t])));
        }
        return {
          id: `atten-${db}-model-frames`,
          group: 'vectors',
          name: `Model outputs at atten_lim_db = ${db} dB (${c.model_out_frames!.length} frames) vs Python ORT`,
          status: verdict(err, tol.model),
          maxAbsErr: err,
          tol: tol.model,
        };
      }),
    );
  }
  if (Array.isArray(c.output_wav)) {
    const xin = Float32Array.from(c.input_wav ?? tv.input_wav);
    out.push(
      await timed(async () => {
        const enh = new Enhancer(model);
        enh.attenLimDb = db;
        const y = await enh.push(xin);
        const ref = c.output_wav!;
        const err = y.length === ref.length ? maxAbsDiff(y, ref) : Infinity;
        return {
          id: `atten-${db}-output-wav`,
          group: 'vectors',
          name: `Streaming waveform at atten_lim_db = ${db} dB vs Python`,
          status: verdict(err, tol.waveform),
          maxAbsErr: err,
          tol: tol.waveform,
          detail: y.length === ref.length ? undefined : `length ${y.length} vs ${ref.length}`,
        };
      }),
    );
  }
  return out;
}
