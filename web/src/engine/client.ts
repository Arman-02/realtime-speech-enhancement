/** Main-thread handle to the engine worker (singleton, created lazily). */
import type { DiagnosticsReport, FromWorker, LiveStats, ModelStatus, ToWorker } from './protocol';
import type { EnhancerStats } from './enhancer';
import type { Manifest } from './model';
import { assetUrl } from '../lib/fetchers';

export interface EnhanceResult {
  audio: Float32Array;
  stats: EnhancerStats;
  elapsedMs: number;
  rtf: number;
}

export interface BenchmarkResult {
  frames: number;
  msPerFrame: number;
  modelMsPerFrame: number;
  p50: number;
  p95: number;
}

type Listener<T> = (v: T) => void;

interface Pending {
  resolve: (v: never) => void;
  reject: (e: Error) => void;
  onProgress?: (done: number, total: number) => void;
}

class EngineClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private statusListeners = new Set<Listener<ModelStatus>>();
  private vizListeners = new Set<Listener<{ frames: number; inDb: Float32Array; outDb: Float32Array }>>();
  private liveStatsListeners = new Set<Listener<LiveStats>>();
  status: ModelStatus = { kind: 'loading' };
  manifest: Manifest | null = null;
  private readyPromise: Promise<ModelStatus> | null = null;
  private resolveReady: ((s: ModelStatus) => void) | null = null;

  /** Spawns the worker and starts loading the model (idempotent). */
  ensure(): Promise<ModelStatus> {
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = new Promise((resolve) => {
      this.resolveReady = resolve;
    });
    try {
      this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'ans-engine' });
    } catch (e) {
      this.setStatus({ kind: 'bypass', reason: 'error', message: `Web Workers unavailable: ${String(e)}` }, null);
      return this.readyPromise;
    }
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => this.onMessage(ev.data);
    this.worker.onerror = (ev) => {
      console.error('[engine] worker error', ev.message);
      if (this.status.kind === 'loading') this.setStatus({ kind: 'bypass', reason: 'error', message: ev.message || 'worker failed to start' }, null);
      for (const p of this.pending.values()) p.reject(new Error(ev.message || 'engine worker crashed'));
      this.pending.clear();
    };
    this.send({ type: 'init', baseUrl: assetUrl('') });
    return this.readyPromise;
  }

  private setStatus(s: ModelStatus, manifest: Manifest | null): void {
    this.status = s;
    this.manifest = manifest;
    this.statusListeners.forEach((l) => l(s));
    if (s.kind !== 'loading') this.resolveReady?.(s);
  }

  private send(msg: ToWorker, transfer: Transferable[] = []): void {
    this.worker?.postMessage(msg, transfer);
  }

  private onMessage(msg: FromWorker): void {
    switch (msg.type) {
      case 'status':
        this.setStatus(msg.status, msg.manifest);
        return;
      case 'progress':
        this.pending.get(msg.id)?.onProgress?.(msg.done, msg.total);
        return;
      case 'enhanced': {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        (p?.resolve as Listener<EnhanceResult> | undefined)?.({ audio: msg.audio, stats: msg.stats, elapsedMs: msg.elapsedMs, rtf: msg.rtf });
        return;
      }
      case 'diagnostics': {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        (p?.resolve as Listener<DiagnosticsReport> | undefined)?.(msg.report);
        return;
      }
      case 'benchmark': {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        const { frames, msPerFrame, modelMsPerFrame, p50, p95 } = msg;
        (p?.resolve as Listener<BenchmarkResult> | undefined)?.({ frames, msPerFrame, modelMsPerFrame, p50, p95 });
        return;
      }
      case 'error': {
        if (msg.id !== undefined) {
          const p = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          const err = new Error(msg.message);
          if (msg.cancelled) err.name = 'AbortError';
          p?.reject(err);
        } else {
          console.error('[engine]', msg.message);
        }
        return;
      }
      case 'live-viz':
        this.vizListeners.forEach((l) => l(msg));
        return;
      case 'live-stats':
        this.liveStatsListeners.forEach((l) => l(msg.stats));
        return;
    }
  }

  private request<T>(build: (id: number) => { msg: ToWorker; transfer?: Transferable[] }, onProgress?: Pending['onProgress']): { id: number; promise: Promise<T> } {
    const id = this.nextId++;
    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: never) => void, reject, onProgress });
    });
    const { msg, transfer } = build(id);
    void this.ensure().then(() => this.send(msg, transfer));
    return { id, promise };
  }

  /**
   * Enhance a 16 kHz clip. The input array is copied (not transferred). `attenLimDb` is the attenuation
   * limit for models with the v0.2 input (ignored otherwise; null = manifest default).
   */
  enhance(
    audio: Float32Array,
    onProgress?: (done: number, total: number) => void,
    attenLimDb: number | null = null,
  ): { promise: Promise<EnhanceResult>; cancel: () => void } {
    const copy = audio.slice();
    const { id, promise } = this.request<EnhanceResult>((id) => ({ msg: { type: 'enhance', id, audio: copy, attenLimDb }, transfer: [copy.buffer] }), onProgress);
    return { promise, cancel: () => this.send({ type: 'cancel', id }) };
  }

  diagnostics(): Promise<DiagnosticsReport> {
    return this.request<DiagnosticsReport>((id) => ({ msg: { type: 'diagnostics', id } })).promise;
  }

  benchmark(frames = 500): Promise<BenchmarkResult> {
    return this.request<BenchmarkResult>((id) => ({ msg: { type: 'benchmark', id, frames } })).promise;
  }

  async startLive(capture: MessagePort, playback: MessagePort, bypass: boolean, attenLimDb: number | null = null): Promise<void> {
    await this.ensure();
    this.send({ type: 'live-start', capture, playback, bypass, attenLimDb }, [capture, playback]);
  }

  /** Change the live attenuation limit; applies from the next frame, no restart. */
  setLiveAttenLimit(attenLimDb: number | null): void {
    this.send({ type: 'live-atten', attenLimDb });
  }

  stopLive(): void {
    this.send({ type: 'live-stop' });
  }

  setLiveBypass(bypass: boolean): void {
    this.send({ type: 'live-bypass', bypass });
  }

  onStatus(l: Listener<ModelStatus>): () => void {
    this.statusListeners.add(l);
    return () => this.statusListeners.delete(l);
  }

  onLiveViz(l: Listener<{ frames: number; inDb: Float32Array; outDb: Float32Array }>): () => void {
    this.vizListeners.add(l);
    return () => this.vizListeners.delete(l);
  }

  onLiveStats(l: Listener<LiveStats>): () => void {
    this.liveStatsListeners.add(l);
    return () => this.liveStatsListeners.delete(l);
  }
}

export const engine = new EngineClient();
