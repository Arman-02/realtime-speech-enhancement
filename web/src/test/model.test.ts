/** Contract v0.2 in the engine: optional atten_lim_db input, lookahead alignment, atten test-vector cases. */
import { describe, expect, it } from 'vitest';
import { HOP, SPEC_LEN } from '../engine/dsp';
import { Enhancer, enhanceOffline } from '../engine/enhancer';
import { findAttenCases, type TestVectors } from '../engine/diagnostics';
import { algorithmicLatencyMs, attenSpec, manifestDeclaresAtten, OnnxModel, type FrameModel, type FrameResult, type OrtLike } from '../engine/model';

interface FakeTensor {
  data: Float32Array;
  dims: readonly number[];
}

/** Minimal ORT stand-in: spec_out = spec * gain, where gain = 10^(-atten/20) when the input exists. */
function fakeOrt(inputNames: string[]) {
  const calls: Record<string, FakeTensor>[] = [];
  const session = {
    inputNames,
    outputNames: ['spec_out', 'state_out'],
    run(feeds: Record<string, FakeTensor>) {
      calls.push(feeds);
      for (const n of inputNames) if (!(n in feeds)) return Promise.reject(new Error(`missing input ${n}`));
      for (const n of Object.keys(feeds)) if (!inputNames.includes(n)) return Promise.reject(new Error(`unknown input ${n}`));
      const g = feeds.atten_lim_db ? Math.pow(10, -feeds.atten_lim_db.data[0] / 20) : 1;
      return Promise.resolve({
        spec_out: { data: feeds.spec.data.map((v) => v * g), dims: feeds.spec.dims },
        state_out: { data: feeds.state_in.data.map((v) => v + 1), dims: feeds.state_in.dims },
      });
    },
    release: () => Promise.resolve(),
  };
  const ort = {
    InferenceSession: { create: () => Promise.resolve(session) },
    Tensor: class {
      constructor(
        readonly type: string,
        readonly data: Float32Array,
        readonly dims: readonly number[],
      ) {}
    },
  } as unknown as OrtLike;
  return { ort, calls };
}

const V01 = { state_shape: [2, 1, 4], inputs: { spec: [1, 1, 161, 2], state_in: [2, 1, 4] } };
const V02 = { ...V01, inputs: { ...V01.inputs, atten_lim_db: [1] }, atten_lim_db: { default: 100, min: 0, max: 100 } };

describe('OnnxModel and atten_lim_db', () => {
  it('v0.1 manifest: never feeds atten_lim_db and reports no support', async () => {
    const { ort, calls } = fakeOrt(['spec', 'state_in']);
    const m = await OnnxModel.create(ort, new Uint8Array(1), V01);
    expect(manifestDeclaresAtten(V01)).toBe(false);
    expect(m.supportsAttenLimit).toBe(false);
    expect(m.attenLimitDb).toBeNull();
    m.setAttenLimit(12);
    const spec = new Float32Array(SPEC_LEN).fill(1);
    const r = await m.processFrame(spec, m.initialState(), 6);
    expect(Object.keys(calls[0]).sort()).toEqual(['spec', 'state_in']);
    expect(r.specOut[0]).toBe(1);
  });

  it('v0.2 manifest: feeds a [1] tensor, default from the manifest, setAttenLimit applies on the next frame', async () => {
    const { ort, calls } = fakeOrt(['spec', 'state_in', 'atten_lim_db']);
    const m = await OnnxModel.create(ort, new Uint8Array(1), { ...V02, atten_lim_db: { default: 24, min: 0, max: 100 } });
    expect(m.supportsAttenLimit).toBe(true);
    const spec = new Float32Array(SPEC_LEN).fill(1);
    let st = m.initialState();
    st = (await m.processFrame(spec, st)).stateOut;
    expect(calls[0].atten_lim_db.dims).toEqual([1]);
    expect(calls[0].atten_lim_db.data[0]).toBe(24);
    m.setAttenLimit(12);
    const r = await m.processFrame(spec, st);
    expect(calls[1].atten_lim_db.data[0]).toBe(12);
    expect(r.specOut[0]).toBeCloseTo(Math.pow(10, -12 / 20), 6);
    expect(r.stateOut[0]).toBe(2); // state threads through unchanged by the limit
    // Per-call override, clamping and "infinity".
    await m.processFrame(spec, st, 6);
    await m.processFrame(spec, st, 500);
    m.setAttenLimit(Infinity);
    await m.processFrame(spec, st);
    m.setAttenLimit(-5);
    await m.processFrame(spec, st);
    expect(calls.slice(2).map((c) => c.atten_lim_db.data[0])).toEqual([6, 100, 100, 0]);
  });

  it('feeds the input when the graph has it even if the manifest forgot to declare it', async () => {
    const { ort } = fakeOrt(['spec', 'state_in', 'atten_lim_db']);
    const m = await OnnxModel.create(ort, new Uint8Array(1), V01);
    await expect(m.processFrame(new Float32Array(SPEC_LEN), m.initialState())).resolves.toBeTruthy();
    expect(m.attenLimitDb).toBe(100);
  });

  it('Enhancer.attenLimDb reaches the model on the next frame without resetting state', async () => {
    const { ort, calls } = fakeOrt(['spec', 'state_in', 'atten_lim_db']);
    const m = await OnnxModel.create(ort, new Uint8Array(1), V02);
    const enh = new Enhancer(m);
    await enh.push(new Float32Array(HOP * 2));
    enh.attenLimDb = 18;
    await enh.push(new Float32Array(HOP));
    expect(calls.map((c) => c.atten_lim_db.data[0])).toEqual([100, 100, 18]);
    expect(calls[2].state_in.data[0]).toBe(2);
  });
});

describe('manifest helpers', () => {
  it('attenSpec fills contract defaults and clamps the default', () => {
    expect(attenSpec(null)).toEqual({ default: 100, min: 0, max: 100 });
    expect(attenSpec({ atten_lim_db: { default: 150 } })).toEqual({ default: 100, min: 0, max: 100 });
    expect(attenSpec({ atten_lim_db: { default: 30, min: 0, max: 60 } })).toEqual({ default: 30, min: 0, max: 60 });
  });
  it('algorithmic latency is 20 ms plus 10 ms per lookahead frame', () => {
    expect(algorithmicLatencyMs(null)).toBe(20);
    expect(algorithmicLatencyMs({ lookahead_frames: 2, n_fft: 320, hop: 160, sample_rate: 16000 })).toBe(40);
  });
});

describe('lookahead alignment', () => {
  it('enhanceOffline removes the extra k-frame delay of a lookahead model', async () => {
    // A k = 2 "model" that emits the frame it saw two calls ago (zeros first): a pure delay of 2 hops.
    const k = 2;
    const delayModel: FrameModel = {
      kind: 'onnx',
      stateShape: [k * SPEC_LEN],
      stateSize: k * SPEC_LEN,
      supportsAttenLimit: false,
      attenLimitDb: null,
      setAttenLimit: () => undefined,
      initialState: () => new Float32Array(k * SPEC_LEN),
      dispose: () => Promise.resolve(),
      processFrame(spec, state): Promise<FrameResult> {
        const out = state.slice(0, SPEC_LEN);
        const next = new Float32Array(k * SPEC_LEN);
        next.set(state.subarray(SPEC_LEN));
        next.set(spec, (k - 1) * SPEC_LEN);
        return Promise.resolve({ specOut: out, stateOut: next });
      },
    };
    const x = Float32Array.from({ length: HOP * 30 + 53 }, (_, i) => Math.sin(i * 0.05) * 0.5);
    const { audio } = await enhanceOffline(delayModel, x, { lookaheadFrames: k });
    expect(audio.length).toBe(x.length);
    let err = 0;
    for (let i = 0; i < x.length; i++) err = Math.max(err, Math.abs(audio[i] - x[i]));
    expect(err).toBeLessThan(1e-5);
  });
});

describe('findAttenCases', () => {
  const base = { sample_rate: 16000, n_fft: 320, hop: 160, state_shape: [1], tolerances: { stft: 1, model: 1, waveform: 1 }, input_wav: [], stft_frames: [], model_out_frames: [], output_wav: [] };
  it('finds a single object, an array, and ignores unrelated keys', () => {
    const one = { ...base, atten_case: { atten_lim_db: 12, model_out_frames: [], output_wav: [] } } as unknown as TestVectors;
    expect(findAttenCases(one).map((c) => c.atten_lim_db)).toEqual([12]);
    const many = { ...base, cases: [{ atten_lim_db: 12, output_wav: [] }, { atten_lim_db: 6, model_out_frames: [] }, { foo: 1 }] } as unknown as TestVectors;
    expect(findAttenCases(many).map((c) => c.atten_lim_db)).toEqual([12, 6]);
    expect(findAttenCases({ ...base, atten_lim_db: 100 } as unknown as TestVectors)).toEqual([]);
  });
});
