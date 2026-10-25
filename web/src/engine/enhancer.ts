/**
 * Streaming enhancer — the single code path for both offline (Studio) and live (mic) processing.
 * Mirrors `StreamingEnhancer` in ml/ans/stream.py.
 */
import { HOP, StreamingSTFT } from './dsp';
import type { FrameModel } from './model';

export interface EnhancerStats {
  frames: number;
  /** Wall time spent in model.processFrame (ms). */
  modelMs: number;
  /** Wall time spent per frame including STFT/iSTFT (ms). */
  totalMs: number;
}

export type FrameTap = (inSpec: Float32Array, outSpec: Float32Array) => void;

/** Yield to the event loop without the 4 ms nested-setTimeout clamp. */
export function yieldToEventLoop(): Promise<void> {
  if (typeof MessageChannel === 'undefined') return new Promise((r) => setTimeout(r, 0));
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => {
      ch.port1.close();
      resolve();
    };
    ch.port2.postMessage(null);
  });
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export class Enhancer {
  /** When true, the model still runs (keeping its state warm) but the noisy spectrum is resynthesised. */
  bypass = false;
  /**
   * Attenuation limit (dB) fed to models with the v0.2 `atten_lim_db` input; null = the model's own
   * setting. Changing it takes effect on the next frame (no state reset).
   */
  attenLimDb: number | null = null;
  onFrame: FrameTap | null = null;
  readonly stats: EnhancerStats = { frames: 0, modelMs: 0, totalMs: 0 };

  private stft = new StreamingSTFT();
  private state: Float32Array;
  private pending = new Float32Array(HOP * 64);
  private pendingLen = 0;
  /** Number of output samples emitted so far (stream index of the next output sample). */
  emitted = 0;

  constructor(private readonly model: FrameModel) {
    this.state = model.initialState();
  }

  reset(): void {
    this.stft.reset();
    this.state = this.model.initialState();
    this.pendingLen = 0;
    this.emitted = 0;
    this.stats.frames = 0;
    this.stats.modelMs = 0;
    this.stats.totalMs = 0;
  }

  private chain: Promise<unknown> = Promise.resolve();

  /**
   * Push any number of samples; resolves to 160·k output samples (k = number of hops completed).
   * Calls are serialised, so it is safe to call push() again before the previous call resolves.
   */
  push(samples: Float32Array): Promise<Float32Array> {
    const copy = samples.slice();
    const p = this.chain.then(() => this.pushNow(copy));
    this.chain = p.catch(() => undefined);
    return p;
  }

  private async pushNow(samples: Float32Array): Promise<Float32Array> {
    this.append(samples);
    const hops = Math.floor(this.pendingLen / HOP);
    const out = new Float32Array(hops * HOP);
    for (let h = 0; h < hops; h++) {
      await this.processHop(this.pending.subarray(h * HOP, (h + 1) * HOP), out.subarray(h * HOP, (h + 1) * HOP));
    }
    const used = hops * HOP;
    this.pending.copyWithin(0, used, this.pendingLen);
    this.pendingLen -= used;
    this.emitted += out.length;
    return out;
  }

  private append(samples: Float32Array): void {
    const need = this.pendingLen + samples.length;
    if (need > this.pending.length) {
      const grown = new Float32Array(Math.max(need, this.pending.length * 2));
      grown.set(this.pending.subarray(0, this.pendingLen));
      this.pending = grown;
    }
    this.pending.set(samples, this.pendingLen);
    this.pendingLen = need;
  }

  private async processHop(hop: Float32Array, out: Float32Array): Promise<void> {
    const t0 = now();
    const spec = this.stft.analyze(hop);
    const t1 = now();
    const { specOut, stateOut } = await this.model.processFrame(spec, this.state, this.attenLimDb);
    const t2 = now();
    this.state = stateOut;
    const y = this.bypass ? spec : specOut;
    this.stft.synthesize(y, out);
    this.onFrame?.(spec, y);
    const t3 = now();
    this.stats.frames += 1;
    this.stats.modelMs += t2 - t1;
    this.stats.totalMs += t3 - t0;
  }
}

export interface OfflineResult {
  audio: Float32Array;
  stats: EnhancerStats;
  /** Wall-clock processing time (ms) and real-time factor (processing time / audio duration). */
  elapsedMs: number;
  rtf: number;
}

export interface OfflineOptions {
  /** Called with (framesDone, framesTotal). */
  onProgress?: (done: number, total: number) => void;
  /** Return true to abort. */
  shouldCancel?: () => boolean;
  /** Frames per progress callback / yield. */
  chunkFrames?: number;
  /** Attenuation limit in dB for models with the `atten_lim_db` input (null/undefined = model default). */
  attenLimDb?: number | null;
  /** Extra output delay of the model in frames (`manifest.lookahead_frames`), removed when aligning. */
  lookaheadFrames?: number;
}

/**
 * Enhance a whole 16 kHz clip through the streaming path. The input is zero-padded to whole hops plus
 * flush hops, and the streaming delay (one hop, plus one hop per lookahead frame) is removed so that the
 * output aligns with the input sample-for-sample (same as `enhance_streaming(align=True)` in Python).
 */
export async function enhanceOffline(
  model: FrameModel,
  x: Float32Array,
  onProgressOrOpts?: OfflineOptions['onProgress'] | OfflineOptions,
): Promise<OfflineResult> {
  const opts: OfflineOptions = typeof onProgressOrOpts === 'function' ? { onProgress: onProgressOrOpts } : (onProgressOrOpts ?? {});
  const chunkFrames = opts.chunkFrames ?? 100;
  const n = x.length;
  const delay = HOP * (1 + Math.max(0, Math.floor(opts.lookaheadFrames ?? 0)));
  const pad = ((HOP - (n % HOP)) % HOP) + delay;
  const padded = new Float32Array(n + pad);
  padded.set(x);
  const totalFrames = padded.length / HOP;
  const enh = new Enhancer(model);
  enh.attenLimDb = opts.attenLimDb ?? null;
  const out = new Float32Array(padded.length);
  const t0 = now();
  let done = 0;
  opts.onProgress?.(0, totalFrames);
  while (done < totalFrames) {
    if (opts.shouldCancel?.()) throw new DOMException('Enhancement cancelled', 'AbortError');
    const k = Math.min(chunkFrames, totalFrames - done);
    const y = await enh.push(padded.subarray(done * HOP, (done + k) * HOP));
    out.set(y, done * HOP);
    done += k;
    opts.onProgress?.(done, totalFrames);
    // Yield so cancel / other messages can be handled between chunks.
    await yieldToEventLoop();
  }
  const elapsedMs = now() - t0;
  return {
    audio: out.slice(delay, delay + n),
    stats: { ...enh.stats },
    elapsedMs,
    rtf: n > 0 ? elapsedMs / 1000 / (n / 16000) : 0,
  };
}
