/** Main-thread audio helpers: decode -> 16 kHz mono, WAV encode, SI-SDR, shared AudioContext. */
import { resampleOffline } from '../engine/resample';

export const TARGET_RATE = 16000;
export const MAX_SECONDS = 60;

export interface DecodedClip {
  audio: Float32Array; // 16 kHz mono
  originalRate: number;
  channels: number;
  originalSeconds: number;
  truncated: boolean;
}

type AudioContextCtor = typeof AudioContext;

function audioContextCtor(): AudioContextCtor | null {
  const w = window as unknown as { AudioContext?: AudioContextCtor; webkitAudioContext?: AudioContextCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

export function audioSupported(): boolean {
  return audioContextCtor() !== null && typeof OfflineAudioContext !== 'undefined';
}

let shared: AudioContext | null = null;
/** One AudioContext for Studio playback (created on first user gesture). */
export function getPlaybackContext(): AudioContext {
  const Ctor = audioContextCtor();
  if (!Ctor) throw new Error('Web Audio is not supported in this browser');
  if (!shared || shared.state === 'closed') shared = new Ctor({ latencyHint: 'playback' });
  return shared;
}

function mixdown(buf: AudioBuffer): Float32Array {
  const n = buf.length;
  if (buf.numberOfChannels === 1) return buf.getChannelData(0).slice();
  const out = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += ch[i];
  }
  const s = 1 / buf.numberOfChannels;
  for (let i = 0; i < n; i++) out[i] *= s;
  return out;
}

/** Sniff the sample rate of a RIFF/WAVE file (to report the original rate). */
function wavRate(bytes: ArrayBuffer): number | null {
  if (bytes.byteLength < 28) return null;
  const v = new DataView(bytes);
  if (v.getUint32(0, false) !== 0x52494646 || v.getUint32(8, false) !== 0x57415645) return null;
  return v.getUint32(24, true);
}

/**
 * Decode any browser-supported file (wav/mp3/m4a/ogg/webm) to 16 kHz mono, capped at 60 s.
 * Primary path: decodeAudioData on a 16 kHz OfflineAudioContext (the browser resamples).
 * Fallback: decode at the device rate and resample with our windowed-sinc resampler.
 */
export async function decodeTo16k(data: ArrayBuffer): Promise<DecodedClip> {
  const sniffedRate = wavRate(data);
  let buf: AudioBuffer;
  let audio: Float32Array;
  try {
    const ctx = new OfflineAudioContext(1, 1, TARGET_RATE);
    buf = await ctx.decodeAudioData(data.slice(0));
    audio = mixdown(buf);
  } catch {
    const ctx = new OfflineAudioContext(1, 1, 44100);
    buf = await ctx.decodeAudioData(data.slice(0));
    audio = resampleOffline(mixdown(buf), buf.sampleRate, TARGET_RATE);
  }
  const originalSeconds = audio.length / TARGET_RATE;
  const max = MAX_SECONDS * TARGET_RATE;
  const truncated = audio.length > max;
  if (truncated) audio = audio.slice(0, max);
  return { audio, originalRate: sniffedRate ?? buf.sampleRate, channels: buf.numberOfChannels, originalSeconds, truncated };
}

/** Build a playable AudioBuffer from 16 kHz samples (falls back to upsampling on old Safari). */
export function toAudioBuffer(ctx: BaseAudioContext, x: Float32Array, rate = TARGET_RATE): AudioBuffer {
  try {
    const b = ctx.createBuffer(1, Math.max(1, x.length), rate);
    b.getChannelData(0).set(x);
    return b;
  } catch {
    const y = resampleOffline(x, rate, ctx.sampleRate);
    const b = ctx.createBuffer(1, Math.max(1, y.length), ctx.sampleRate);
    b.getChannelData(0).set(y);
    return b;
  }
}

/** 16-bit PCM mono WAV. */
export function encodeWav(x: Float32Array, rate = TARGET_RATE): Blob {
  const n = x.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  v.setUint32(4, 36 + n * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, x[i]));
    v.setInt16(44 + i * 2, s < 0 ? Math.round(s * 32768) : Math.round(s * 32767), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/** Scale-invariant SDR in dB (zero-mean, Le Roux et al. 2019). */
export function siSdr(est: Float32Array, ref: Float32Array): number {
  const n = Math.min(est.length, ref.length);
  if (n === 0) return NaN;
  let me = 0;
  let mr = 0;
  for (let i = 0; i < n; i++) {
    me += est[i];
    mr += ref[i];
  }
  me /= n;
  mr /= n;
  let dot = 0;
  let rr = 0;
  for (let i = 0; i < n; i++) {
    const r = ref[i] - mr;
    dot += (est[i] - me) * r;
    rr += r * r;
  }
  if (rr === 0) return NaN;
  const a = dot / rr;
  let sig = 0;
  let err = 0;
  for (let i = 0; i < n; i++) {
    const s = a * (ref[i] - mr);
    const e = est[i] - me - s;
    sig += s * s;
    err += e * e;
  }
  return 10 * Math.log10((sig + 1e-12) / (err + 1e-12));
}

export function formatTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec < 10 ? '0' : ''}${sec.toFixed(1)}`;
}
