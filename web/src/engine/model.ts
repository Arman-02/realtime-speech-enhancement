/**
 * Frame models (PLAN.md §3.2). One call processes one STFT frame:
 *   spec [1,1,161,2] + state_in [state_shape] (+ atten_lim_db [1], contract v0.2)
 *     -> spec_out [1,1,161,2] + state_out [state_shape]
 * The runtime is generic over `manifest.state_shape`, so any model that packs its recurrent state into
 * one tensor works without code changes. `atten_lim_db` is fed only when the model declares it, so v0.1
 * exports keep working (the attenuation-limit control is then disabled).
 */
import { N_BINS, SPEC_LEN } from './dsp';

export interface AttenLimitSpec {
  default: number;
  min: number;
  max: number;
}

export interface Manifest {
  name: string;
  version: string;
  /** v0.2: "demo" | "benchmark" */
  variant?: string;
  trained: boolean;
  sample_rate: number;
  n_fft: number;
  hop: number;
  window: string;
  onnx: string;
  onnx_int8?: string;
  inputs?: Record<string, number[]>;
  outputs?: Record<string, number[]>;
  state_shape: number[];
  /** v0.2: frames of lookahead; total algorithmic latency = 20 + 10·k ms. Absent = 0. */
  lookahead_frames?: number;
  /** v0.2: range and default of the `atten_lim_db` input (present when the graph has that input). */
  atten_lim_db?: Partial<AttenLimitSpec> | null;
  /** v0.2: deployment budget (weights, activations, MMAC/s, ms/frame on various targets). */
  budget?: Record<string, unknown> | null;
  training?: Record<string, unknown> | null;
  params?: number | null;
  macs_per_frame?: number | null;
  metrics?: Record<string, number | null> | null;
  [extra: string]: unknown;
}

export interface FrameResult {
  specOut: Float32Array;
  stateOut: Float32Array;
}

export interface FrameModel {
  readonly kind: 'onnx' | 'bypass';
  readonly stateShape: readonly number[];
  readonly stateSize: number;
  /** True when the graph takes the `atten_lim_db` input (contract v0.2). */
  readonly supportsAttenLimit: boolean;
  /**
   * Process one frame. `attenLimDb` overrides the model-wide limit set with `setAttenLimit` for this
   * call only; it is ignored by models without the input.
   */
  processFrame(spec: Float32Array, state: Float32Array, attenLimDb?: number | null): Promise<FrameResult>;
  /** Model-wide default attenuation limit (dB); applies from the next frame. No-op without the input. */
  setAttenLimit(db: number): void;
  readonly attenLimitDb: number | null;
  initialState(): Float32Array;
  dispose(): Promise<void>;
}

export function shapeSize(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export const ATTEN_INPUT = 'atten_lim_db';
/** Contract §3.2: 100 dB is "effectively unlimited". */
export const ATTEN_UNLIMITED_DB = 100;

/** Whether the manifest declares the v0.2 `atten_lim_db` input. */
export function manifestDeclaresAtten(m: Pick<Manifest, 'inputs'> | null | undefined): boolean {
  return !!m?.inputs && Object.prototype.hasOwnProperty.call(m.inputs, ATTEN_INPUT);
}

/** Range/default for the attenuation limit, with contract defaults filled in. */
export function attenSpec(m: Pick<Manifest, 'atten_lim_db'> | null | undefined): AttenLimitSpec {
  const a = m?.atten_lim_db ?? {};
  const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
  const min = num(a.min, 0);
  const max = Math.max(min + 1, num(a.max, ATTEN_UNLIMITED_DB));
  const def = Math.min(max, Math.max(min, num(a.default, max)));
  return { default: def, min, max };
}

/** Lookahead frames declared by the manifest (0 when absent or invalid). */
export function lookaheadFrames(m: Pick<Manifest, 'lookahead_frames'> | null | undefined): number {
  const k = m?.lookahead_frames;
  return typeof k === 'number' && Number.isInteger(k) && k > 0 ? k : 0;
}

/** Total algorithmic latency in ms: one 20 ms window plus 10 ms per lookahead frame (§4.11). */
export function algorithmicLatencyMs(m: Pick<Manifest, 'lookahead_frames' | 'n_fft' | 'hop' | 'sample_rate'> | null | undefined): number {
  const sr = m?.sample_rate || 16000;
  const win = ((m?.n_fft || 320) / sr) * 1000;
  const hop = ((m?.hop || 160) / sr) * 1000;
  return win + hop * lookaheadFrames(m);
}

/** Identity model (spec_out = spec). Used automatically when no ONNX model is available. */
export class BypassModel implements FrameModel {
  readonly kind = 'bypass' as const;
  readonly stateShape: readonly number[];
  readonly stateSize: number;
  readonly supportsAttenLimit = false;
  readonly attenLimitDb = null;

  constructor(stateShape: readonly number[] = [1]) {
    this.stateShape = stateShape;
    this.stateSize = shapeSize(stateShape);
  }

  processFrame(spec: Float32Array, state: Float32Array): Promise<FrameResult> {
    return Promise.resolve({ specOut: spec, stateOut: state });
  }

  setAttenLimit(): void {
    /* no attenuation input */
  }

  initialState(): Float32Array {
    return new Float32Array(this.stateSize);
  }

  dispose(): Promise<void> {
    return Promise.resolve();
  }
}

// Minimal structural types shared by onnxruntime-web and onnxruntime-node (both re-export onnxruntime-common).
interface OrtTensorLike {
  readonly data: unknown;
  readonly dims: readonly number[];
}
interface OrtSessionLike {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
  release(): Promise<void>;
}
export interface OrtLike {
  InferenceSession: {
    create(data: Uint8Array, options?: Record<string, unknown>): Promise<unknown>;
  };
  Tensor: new (type: 'float32', data: Float32Array, dims: readonly number[]) => OrtTensorLike;
}

const SPEC_DIMS = [1, 1, N_BINS, 2] as const;

export class OnnxModel implements FrameModel {
  readonly kind = 'onnx' as const;
  readonly stateShape: readonly number[];
  readonly stateSize: number;
  readonly supportsAttenLimit: boolean;
  private readonly names: { spec: string; state: string; specOut: string; stateOut: string };
  private atten: number;
  private readonly attenRange: AttenLimitSpec;

  private constructor(
    private readonly ort: OrtLike,
    private readonly session: OrtSessionLike,
    manifest: Pick<Manifest, 'state_shape' | 'inputs' | 'atten_lim_db'>,
  ) {
    this.stateShape = manifest.state_shape;
    this.stateSize = shapeSize(manifest.state_shape);
    const declared = manifestDeclaresAtten(manifest);
    const inGraph = session.inputNames.includes(ATTEN_INPUT);
    if (declared && !inGraph) console.warn(`[model] manifest declares ${ATTEN_INPUT} but the ONNX graph has no such input; ignoring it`);
    if (!declared && inGraph) console.warn(`[model] the ONNX graph takes ${ATTEN_INPUT} but the manifest does not declare it; feeding the default`);
    // The graph is the source of truth: feeding an undeclared input would fail, and omitting a required one too.
    this.supportsAttenLimit = inGraph;
    this.attenRange = attenSpec(manifest);
    this.atten = this.attenRange.default;
    const others = session.inputNames.filter((n) => n !== ATTEN_INPUT);
    const pick = (list: readonly string[], want: string, fallback: number) =>
      list.includes(want) ? want : list[fallback];
    this.names = {
      spec: pick(others, 'spec', 0),
      state: pick(others, 'state_in', 1),
      specOut: pick(session.outputNames, 'spec_out', 0),
      stateOut: pick(session.outputNames, 'state_out', 1),
    };
  }

  get attenLimitDb(): number | null {
    return this.supportsAttenLimit ? this.atten : null;
  }

  setAttenLimit(db: number): void {
    if (!Number.isFinite(db)) db = this.attenRange.max;
    this.atten = Math.min(this.attenRange.max, Math.max(this.attenRange.min, db));
  }

  static async create(
    ort: OrtLike,
    bytes: Uint8Array,
    manifest: Pick<Manifest, 'state_shape'> & Partial<Pick<Manifest, 'inputs' | 'atten_lim_db'>>,
    options: Record<string, unknown> = {},
  ): Promise<OnnxModel> {
    if (!Array.isArray(manifest.state_shape) || manifest.state_shape.length === 0) {
      throw new Error('manifest.state_shape is missing or empty');
    }
    const session = (await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      ...options,
    })) as OrtSessionLike;
    if (session.inputNames.length < 2 || session.outputNames.length < 2) {
      throw new Error(`unexpected ONNX signature: inputs=${session.inputNames.join(',')} outputs=${session.outputNames.join(',')}`);
    }
    return new OnnxModel(ort, session, manifest);
  }

  async processFrame(spec: Float32Array, state: Float32Array, attenLimDb?: number | null): Promise<FrameResult> {
    if (spec.length !== SPEC_LEN) throw new Error(`spec has ${spec.length} values, expected ${SPEC_LEN}`);
    const feeds: Record<string, OrtTensorLike> = {
      [this.names.spec]: new this.ort.Tensor('float32', spec, SPEC_DIMS),
      [this.names.state]: new this.ort.Tensor('float32', state, this.stateShape),
    };
    if (this.supportsAttenLimit) {
      const db = attenLimDb ?? this.atten;
      const v = Number.isFinite(db) ? Math.min(this.attenRange.max, Math.max(this.attenRange.min, db)) : this.attenRange.max;
      feeds[ATTEN_INPUT] = new this.ort.Tensor('float32', Float32Array.of(v), [1]);
    }
    const out = await this.session.run(feeds);
    return {
      specOut: out[this.names.specOut].data as Float32Array,
      stateOut: out[this.names.stateOut].data as Float32Array,
    };
  }

  initialState(): Float32Array {
    return new Float32Array(this.stateSize);
  }

  dispose(): Promise<void> {
    return this.session.release();
  }
}
