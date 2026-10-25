/**
 * Engine worker: owns the model (ORT wasm, single-threaded) and runs all DSP + inference off the
 * main thread. Live audio flows capture-worklet -> (MessagePort) -> here -> (MessagePort) -> playback
 * worklet without touching the main thread.
 */
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { HOP, N_BINS, specToDb } from './dsp';
import { Enhancer, enhanceOffline } from './enhancer';
import { BypassModel, lookaheadFrames, OnnxModel, type FrameModel, type Manifest, type OrtLike } from './model';
import { runSelfChecks, runVectorChecks, type TestVectors } from './diagnostics';
import type { CaptureChunk, FromWorker, ModelStatus, PlaybackChunk, ToWorker } from './protocol';
import { fetchBytesOrNull, fetchJsonOrNull } from '../lib/fetchers';

interface WorkerScope {
  postMessage(msg: unknown, transfer: Transferable[]): void;
  onmessage: ((ev: MessageEvent<ToWorker>) => void) | null;
  location: { href: string };
}
const scope = self as unknown as WorkerScope;

function post(msg: FromWorker, transfer: Transferable[] = []): void {
  scope.postMessage(msg, transfer);
}

let baseUrl = '';
let model: FrameModel = new BypassModel();
let status: ModelStatus = { kind: 'loading' };
let manifest: Manifest | null = null;
let ready: Promise<void> | null = null;
const cancelled = new Set<number>();

const url = (path: string) => new URL(path, baseUrl).href;

async function init(): Promise<void> {
  manifest = await fetchJsonOrNull<Manifest>(url('models/manifest.json'));
  if (!manifest) {
    status = { kind: 'bypass', reason: 'missing', message: 'No model found at models/manifest.json' };
    model = new BypassModel();
    return;
  }
  try {
    const t0 = performance.now();
    const bytes = await fetchBytesOrNull(url(`models/${manifest.onnx || 'ans_stream.onnx'}`));
    if (!bytes) throw new Error(`models/${manifest.onnx} is missing`);
    const ort = await import('onnxruntime-web/wasm');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, scope.location.href).href };
    ort.env.logLevel = 'error';
    model = await OnnxModel.create(ort as unknown as OrtLike, bytes, manifest);
    // Warm-up run so the first real frame is not penalised by lazy initialisation.
    await model.processFrame(new Float32Array(N_BINS * 2), model.initialState());
    status = {
      kind: 'onnx',
      name: manifest.name,
      version: manifest.version,
      trained: Boolean(manifest.trained),
      backend: 'wasm · 1 thread',
      loadMs: performance.now() - t0,
      modelBytes: bytes.length,
      attenSupported: model.supportsAttenLimit,
    };
  } catch (e) {
    console.error('[engine] failed to load ONNX model, falling back to bypass', e);
    status = { kind: 'bypass', reason: 'error', message: e instanceof Error ? e.message : String(e) };
    model = new BypassModel(manifest.state_shape ?? [1]);
  }
}

// ---------------------------------------------------------------------------------------------
// Live streaming
// ---------------------------------------------------------------------------------------------

interface LiveSession {
  enhancer: Enhancer;
  capture: MessagePort;
  playback: MessagePort;
  lastEndIndex: number;
  lastEndTime: number;
  backlog: number;
  dropped: number;
  frameTimes: number[];
  vizIn: Float32Array;
  vizOut: Float32Array;
  vizCount: number;
  statsTimer: ReturnType<typeof setInterval>;
}

let live: LiveSession | null = null;
const VIZ_BATCH = 4;
const MAX_BACKLOG = 40; // ~ capture blocks queued before we start dropping input

function stopLive(): void {
  if (!live) return;
  clearInterval(live.statsTimer);
  live.capture.close();
  live.playback.close();
  live = null;
}

function startLive(capture: MessagePort, playback: MessagePort, bypass: boolean, attenLimDb: number | null): void {
  stopLive();
  const enhancer = new Enhancer(model);
  enhancer.bypass = bypass;
  enhancer.attenLimDb = attenLimDb;
  const delay = HOP * (1 + lookaheadFrames(manifest));
  const session: LiveSession = {
    enhancer,
    capture,
    playback,
    lastEndIndex: 0,
    lastEndTime: 0,
    backlog: 0,
    dropped: 0,
    frameTimes: [],
    vizIn: new Float32Array(VIZ_BATCH * N_BINS),
    vizOut: new Float32Array(VIZ_BATCH * N_BINS),
    vizCount: 0,
    statsTimer: setInterval(() => {
      const ft = session.frameTimes.splice(0);
      const sorted = ft.slice().sort((a, b) => a - b);
      const mean = ft.length ? ft.reduce((a, b) => a + b, 0) / ft.length : 0;
      const p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : 0;
      post({
        type: 'live-stats',
        stats: { frames: enhancer.stats.frames, msPerFrame: mean, p95MsPerFrame: p95, backlog: session.backlog, dropped: session.dropped },
      });
    }, 500),
  };
  let lastTotal = 0;
  let outCount = 0;
  enhancer.onFrame = (inSpec, outSpec) => {
    const s = session;
    specToDb(inSpec, s.vizIn, s.vizCount * N_BINS);
    specToDb(outSpec, s.vizOut, s.vizCount * N_BINS);
    s.vizCount += 1;
    const total = enhancer.stats.totalMs;
    s.frameTimes.push(total - lastTotal);
    lastTotal = total;
    if (s.vizCount === VIZ_BATCH) {
      const inDb = s.vizIn;
      const outDb = s.vizOut;
      post({ type: 'live-viz', frames: VIZ_BATCH, inDb, outDb }, [inDb.buffer, outDb.buffer]);
      s.vizIn = new Float32Array(VIZ_BATCH * N_BINS);
      s.vizOut = new Float32Array(VIZ_BATCH * N_BINS);
      s.vizCount = 0;
    }
  };
  capture.onmessage = (ev: MessageEvent<CaptureChunk>) => {
    const chunk = ev.data;
    if (session !== live) return;
    if (session.backlog > MAX_BACKLOG) {
      session.dropped += 1;
      return;
    }
    session.lastEndIndex = chunk.endIndex;
    session.lastEndTime = chunk.endTime;
    session.backlog += 1;
    void enhancer.push(chunk.samples).then((out) => {
      session.backlog -= 1;
      if (out.length === 0 || session !== live) return;

      // Output stream index j reconstructs input index j - delay (one hop, plus one per lookahead
      // frame); find when that input was captured.
      const firstOut = outCount;
      outCount += out.length;
      const inIndex = firstOut - delay;
      const capTime = session.lastEndTime - (session.lastEndIndex - inIndex) / 16000;
      const msg: PlaybackChunk = { samples: out, capTime };
      session.playback.postMessage(msg, [out.buffer]);
    });
  };
  live = session;
}

// ---------------------------------------------------------------------------------------------
// Message loop
// ---------------------------------------------------------------------------------------------

async function handle(msg: ToWorker): Promise<void> {
  switch (msg.type) {
    case 'init': {
      baseUrl = msg.baseUrl;
      ready ??= init().then(() => post({ type: 'status', status, manifest }));
      await ready;
      return;
    }
    case 'enhance': {
      await ready;
      try {
        const res = await enhanceOffline(model, msg.audio, {
          onProgress: (done, total) => post({ type: 'progress', id: msg.id, done, total }),
          shouldCancel: () => cancelled.has(msg.id),
          chunkFrames: 50,
          attenLimDb: msg.attenLimDb ?? null,
          lookaheadFrames: lookaheadFrames(manifest),
        });
        post({ type: 'enhanced', id: msg.id, audio: res.audio, stats: res.stats, elapsedMs: res.elapsedMs, rtf: res.rtf }, [res.audio.buffer]);
      } catch (e) {
        const isCancel = e instanceof DOMException && e.name === 'AbortError';
        post({ type: 'error', id: msg.id, message: e instanceof Error ? e.message : String(e), cancelled: isCancel });
      } finally {
        cancelled.delete(msg.id);
      }
      return;
    }
    case 'cancel':
      cancelled.add(msg.id);
      return;
    case 'diagnostics': {
      await ready;
      const self = await runSelfChecks();
      const tv = await fetchJsonOrNull<TestVectors>(url('models/test_vectors.json'));
      const vectors = tv ? await runVectorChecks(tv, model) : null;
      post({
        type: 'diagnostics',
        id: msg.id,
        report: { self, vectors, vectorsInfo: tv ? (tv.description ?? 'test_vectors.json') : null, model: status },
      });
      return;
    }
    case 'benchmark': {
      await ready;
      const enh = new Enhancer(model);
      const n = Math.max(50, msg.frames);
      const x = new Float32Array(HOP);
      const times: number[] = [];
      for (let i = 0; i < 20; i++) await enh.push(x.map(() => Math.random() * 0.1 - 0.05));
      enh.reset();
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < HOP; j++) x[j] = Math.random() * 0.1 - 0.05;
        const t0 = performance.now();
        await enh.push(x);
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      post({
        type: 'benchmark',
        id: msg.id,
        frames: n,
        msPerFrame: enh.stats.totalMs / n,
        modelMsPerFrame: enh.stats.modelMs / n,
        p50: times[Math.floor(n * 0.5)],
        p95: times[Math.floor(n * 0.95)],
      });
      return;
    }
    case 'live-start':
      await ready;
      startLive(msg.capture, msg.playback, msg.bypass, msg.attenLimDb ?? null);
      return;
    case 'live-stop':
      stopLive();
      return;
    case 'live-bypass':
      if (live) live.enhancer.bypass = msg.bypass;
      return;
    case 'live-atten':
      if (live) live.enhancer.attenLimDb = msg.attenLimDb;
      return;
  }
}

scope.onmessage = (ev: MessageEvent<ToWorker>) => {
  handle(ev.data).catch((e: unknown) => {
    console.error('[engine]', e);
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  });
};
