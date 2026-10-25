/**
 * Synchronised A/B/C player: all tracks play from one clock, each through its own GainNode, so
 * switching Noisy / Enhanced / Clean never restarts playback, and replacing a track (e.g. after a
 * re-process at a new attenuation limit) keeps the playhead.
 */
import { getPlaybackContext, toAudioBuffer } from './audio';

export type Track = 'noisy' | 'enhanced' | 'clean';
export const TRACKS: Track[] = ['noisy', 'enhanced', 'clean'];

type Listener = () => void;

export class AbPlayer {
  private ctx: AudioContext | null = null;
  private buffers: Partial<Record<Track, AudioBuffer>> = {};
  private data: Partial<Record<Track, Float32Array>> = {};
  private gains: Partial<Record<Track, GainNode>> = {};
  private sources: AudioBufferSourceNode[] = [];
  private startedAt = 0;
  private offset = 0;
  private session = 0;
  private listeners = new Set<Listener>();
  playing = false;
  mode: Track = 'enhanced';
  duration = 0;

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  private emit(): void {
    this.listeners.forEach((l) => l());
  }

  private context(): AudioContext {
    if (!this.ctx) {
      this.ctx = getPlaybackContext();
      for (const t of TRACKS) {
        const g = this.ctx.createGain();
        g.gain.value = 0;
        g.connect(this.ctx.destination);
        this.gains[t] = g;
      }
      this.applyGains(true);
    }
    return this.ctx;
  }

  has(t: Track): boolean {
    return this.data[t] !== undefined;
  }

  /** Replace one or more tracks. Keeps the playhead; restarts sources if playing. */
  setTracks(tracks: Partial<Record<Track, Float32Array | null>>, rate = 16000): void {
    const wasPlaying = this.playing;
    const pos = this.currentTime();
    if (wasPlaying) this.stopSources();
    for (const t of TRACKS) {
      if (!(t in tracks)) continue;
      const v = tracks[t];
      if (v) this.data[t] = v;
      else delete this.data[t];
      delete this.buffers[t];
    }
    const lens = TRACKS.map((t) => this.data[t]?.length ?? 0);
    this.duration = Math.max(0, ...lens) / rate;
    this.offset = Math.min(pos, this.duration);
    if (wasPlaying) this.startSources();
    this.applyGains(true);
    this.emit();
  }

  clear(): void {
    this.pause();
    this.data = {};
    this.buffers = {};
    this.duration = 0;
    this.offset = 0;
    this.emit();
  }

  private buffer(t: Track): AudioBuffer | null {
    const d = this.data[t];
    if (!d) return null;
    this.buffers[t] ??= toAudioBuffer(this.context(), d);
    return this.buffers[t]!;
  }

  currentTime(): number {
    if (!this.playing || !this.ctx) return this.offset;
    return Math.min(this.duration, this.offset + (this.ctx.currentTime - this.startedAt));
  }

  private startSources(): void {
    const ctx = this.context();
    const id = ++this.session;
    const when = ctx.currentTime + 0.03;
    let first = true;
    for (const t of TRACKS) {
      const buf = this.buffer(t);
      if (!buf) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.gains[t]!);
      if (first) {
        src.onended = () => {
          if (this.session === id && this.playing) {
            this.playing = false;
            this.offset = 0;
            this.emit();
          }
        };
        first = false;
      }
      src.start(when, Math.min(this.offset, buf.duration));
      this.sources.push(src);
    }
    this.startedAt = when;
  }

  private stopSources(): void {
    this.session++;
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
      s.disconnect();
    }
    this.sources = [];
  }

  async play(): Promise<void> {
    if (this.playing || this.duration === 0) return;
    const ctx = this.context();
    if (ctx.state === 'suspended') await ctx.resume();
    if (this.offset >= this.duration - 0.01) this.offset = 0;
    this.startSources();
    this.playing = true;
    this.emit();
  }

  pause(): void {
    if (!this.playing) return;
    this.offset = this.currentTime();
    this.stopSources();
    this.playing = false;
    this.emit();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else void this.play();
  }

  seek(t: number): void {
    const target = Math.max(0, Math.min(this.duration, t));
    if (this.playing) {
      this.stopSources();
      this.offset = target;
      this.startSources();
    } else {
      this.offset = target;
    }
    this.emit();
  }

  setMode(m: Track): void {
    this.mode = m;
    this.applyGains();
    this.emit();
  }

  /** The track actually heard (falls back to noisy if the selected one is unavailable). */
  effectiveMode(): Track {
    return this.has(this.mode) ? this.mode : 'noisy';
  }

  private applyGains(immediate = false): void {
    if (!this.ctx) return;
    const m = this.effectiveMode();
    const target: Record<Track, number> = {
      noisy: m === 'noisy' ? 1 : 0,
      enhanced: m === 'enhanced' ? 1 : 0,
      clean: m === 'clean' ? 1 : 0,
    };
    const now = this.ctx.currentTime;
    for (const t of TRACKS) {
      const g = this.gains[t];
      if (!g) continue;
      g.gain.cancelScheduledValues(now);
      if (immediate) g.gain.setValueAtTime(target[t], now);
      else g.gain.setTargetAtTime(target[t], now, 0.012); // ~40 ms click-free crossfade
    }
  }

  dispose(): void {
    this.pause();
    for (const t of TRACKS) this.gains[t]?.disconnect();
    this.gains = {};
    this.ctx = null;
    this.listeners.clear();
  }
}
