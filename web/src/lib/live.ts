/**
 * Live pipeline: mic (raw: no AEC / NS / AGC) -> capture worklet (-> 16 kHz) -> engine worker (ORT)
 * -> playback worklet (16 kHz -> device rate, jitter buffer) -> output gain -> speakers.
 */
import captureUrl from '../engine/worklets/capture.worklet.ts?worker&url';
import playbackUrl from '../engine/worklets/playback.worklet.ts?worker&url';
import { engine } from '../engine/client';
import { RAW_MIC_CONSTRAINTS } from './recorder';

export interface Level {
  peak: number;
  rms: number;
}

export interface PlaybackStats {
  latencyMs: number | null;
  bufferedMs: number;
  underruns: number;
}

export interface LiveHandlers {
  onInputLevel?: (l: Level) => void;
  onOutputLevel?: (l: Level) => void;
  onPlaybackStats?: (s: PlaybackStats) => void;
}

export interface LiveSession {
  ctx: AudioContext;
  sampleRate: number;
  baseLatencyMs: number;
  outputLatencyMs: number;
  inputLatencyMs: number | null;
  deviceLabel: string;
  setGainDb(db: number): void;
  setBypass(b: boolean): void;
  /** Attenuation limit in dB (null = model default); applies from the next frame. */
  setAttenLimit(db: number | null): void;
  setMonitor(on: boolean): void;
  stop(): Promise<void>;
}

export function liveSupport(): string | null {
  if (!window.isSecureContext) return 'Live mode needs a secure context (HTTPS or localhost).';
  if (!navigator.mediaDevices?.getUserMedia) return 'This browser does not expose the microphone (getUserMedia).';
  const AC = window.AudioContext;
  if (!AC || !('audioWorklet' in AC.prototype)) return 'This browser does not support AudioWorklet.';
  if (typeof Worker === 'undefined') return 'This browser does not support Web Workers.';
  return null;
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = setTimeout(() => reject(new Error(`${what} timed out after ${ms / 1000} s`)), ms);
    p.then(
      (v) => {
        clearTimeout(id);
        resolve(v);
      },
      (e: unknown) => {
        clearTimeout(id);
        reject(e instanceof Error ? e : new Error(String(e)));
      },
    );
  });
}

export async function startLive(opts: { bypass: boolean; gainDb: number; monitor: boolean; attenLimDb?: number | null }, h: LiveHandlers): Promise<LiveSession> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW_MIC_CONSTRAINTS, video: false });
  let ctx: AudioContext | null = null;
  try {
    ctx = new AudioContext({ latencyHint: 'interactive' });
    await withTimeout(Promise.all([ctx.audioWorklet.addModule(captureUrl), ctx.audioWorklet.addModule(playbackUrl)]), 10_000, 'Loading the audio worklets');
    if (ctx.state === 'suspended') await withTimeout(ctx.resume(), 5_000, 'Starting the audio context');

    const src = ctx.createMediaStreamSource(stream);
    const capture = new AudioWorkletNode(ctx, 'ans-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const playback = new AudioWorkletNode(ctx, 'ans-playback', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
    const gain = ctx.createGain();
    const monitor = ctx.createGain();
    const sink = ctx.createGain();
    sink.gain.value = 0;
    gain.gain.value = Math.pow(10, opts.gainDb / 20);
    monitor.gain.value = opts.monitor ? 1 : 0;

    src.connect(capture);
    capture.connect(sink).connect(ctx.destination); // keeps the capture node pulled by the graph
    playback.connect(gain).connect(monitor).connect(ctx.destination);

    // Direct worklet <-> worker ports: audio never touches the main thread.
    const capCh = new MessageChannel();
    const playCh = new MessageChannel();
    capture.port.postMessage({ type: 'port', port: capCh.port1 }, [capCh.port1]);
    playback.port.postMessage({ type: 'port', port: playCh.port1 }, [playCh.port1]);
    await engine.startLive(capCh.port2, playCh.port2, opts.bypass, opts.attenLimDb ?? null);

    capture.port.onmessage = (e: MessageEvent<{ type: string } & Level>) => {
      if (e.data.type === 'level') h.onInputLevel?.(e.data);
    };
    playback.port.onmessage = (e: MessageEvent<{ type: string } & Level & PlaybackStats>) => {
      if (e.data.type === 'level') h.onOutputLevel?.(e.data);
      else if (e.data.type === 'stats') h.onPlaybackStats?.(e.data);
    };

    const track = stream.getAudioTracks()[0];
    const settings = track?.getSettings() as MediaTrackSettings & { latency?: number };
    const c = ctx;
    return {
      ctx: c,
      sampleRate: c.sampleRate,
      baseLatencyMs: (c.baseLatency || 0) * 1000,
      outputLatencyMs: ((c as AudioContext & { outputLatency?: number }).outputLatency || 0) * 1000,
      inputLatencyMs: typeof settings?.latency === 'number' ? settings.latency * 1000 : null,
      deviceLabel: track?.label || 'Default microphone',
      setGainDb(db) {
        gain.gain.setTargetAtTime(Math.pow(10, db / 20), c.currentTime, 0.02);
      },
      setBypass(b) {
        engine.setLiveBypass(b);
      },
      setAttenLimit(db) {
        engine.setLiveAttenLimit(db);
      },
      setMonitor(on) {
        monitor.gain.setTargetAtTime(on ? 1 : 0, c.currentTime, 0.02);
      },
      async stop() {
        engine.stopLive();
        stream.getTracks().forEach((t) => t.stop());
        capture.port.onmessage = null;
        playback.port.onmessage = null;
        try {
          src.disconnect();
          capture.disconnect();
          playback.disconnect();
        } catch {
          /* already disconnected */
        }
        await c.close();
      },
    };
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    await ctx?.close().catch(() => undefined);
    throw e;
  }
}
