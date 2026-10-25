/** Messages between the main thread (EngineClient) and the engine worker. */
import type { CheckResult } from './diagnostics';
import type { EnhancerStats } from './enhancer';
import type { Manifest } from './model';

export type ModelStatus =
  | { kind: 'loading' }
  | {
      kind: 'onnx';
      name: string;
      version: string;
      trained: boolean;
      backend: string;
      loadMs: number;
      modelBytes: number;
      /** The graph takes the v0.2 `atten_lim_db` input. */
      attenSupported: boolean;
    }
  | { kind: 'bypass'; reason: 'missing' | 'error'; message: string };

export type ToWorker =
  | { type: 'init'; baseUrl: string }
  | { type: 'enhance'; id: number; audio: Float32Array; attenLimDb?: number | null }
  | { type: 'cancel'; id: number }
  | { type: 'diagnostics'; id: number }
  | { type: 'benchmark'; id: number; frames: number }
  | { type: 'live-start'; capture: MessagePort; playback: MessagePort; bypass: boolean; attenLimDb?: number | null }
  | { type: 'live-stop' }
  | { type: 'live-bypass'; bypass: boolean }
  | { type: 'live-atten'; attenLimDb: number | null };

export interface DiagnosticsReport {
  self: CheckResult[];
  vectors: CheckResult[] | null;
  vectorsInfo: string | null;
  model: ModelStatus;
}

export interface LiveStats {
  frames: number;
  msPerFrame: number;
  p95MsPerFrame: number;
  backlog: number;
  dropped: number;
}

export type FromWorker =
  | { type: 'status'; status: ModelStatus; manifest: Manifest | null }
  | { type: 'progress'; id: number; done: number; total: number }
  | { type: 'enhanced'; id: number; audio: Float32Array; stats: EnhancerStats; elapsedMs: number; rtf: number }
  | { type: 'error'; id?: number; message: string; cancelled?: boolean }
  | { type: 'diagnostics'; id: number; report: DiagnosticsReport }
  | { type: 'benchmark'; id: number; frames: number; msPerFrame: number; modelMsPerFrame: number; p50: number; p95: number }
  | { type: 'live-viz'; frames: number; inDb: Float32Array; outDb: Float32Array }
  | { type: 'live-stats'; stats: LiveStats };

/** Capture worklet -> worker. `endIndex` = 16 kHz sample count after this chunk; `endTime` = context time of its last sample. */
export interface CaptureChunk {
  samples: Float32Array;
  endIndex: number;
  endTime: number;
}

/** Worker -> playback worklet. `capTime` = context time at which the input matching samples[0] was captured. */
export interface PlaybackChunk {
  samples: Float32Array;
  capTime: number;
}
