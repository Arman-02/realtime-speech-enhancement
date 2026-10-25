/**
 * Playback worklet: enhanced 16 kHz hops from the engine worker -> device rate, through a small
 * jitter buffer. Measures pipeline latency (capture -> render, same AudioContext clock), underruns
 * and output level, and reports them on its own port.
 */
import { StreamResampler } from '../resample';
import { Processor, scope } from './worklet-env';
import type { PlaybackChunk } from '../protocol';

const STATS_INTERVAL_S = 0.25;

class PlaybackProcessor extends Processor {
  private ring: Float32Array;
  private readPos = 0; // absolute output-rate sample index
  private writePos = 0;
  private buffering = true;
  private targetFrames: number;
  private maxFrames: number;
  private underruns = 0;
  private resampler = new StreamResampler(16000, scope.sampleRate);
  private resamplerDelay: number;
  private markers: { pos: number; capTime: number }[] = [];
  private latency = 0;
  private latencyValid = false;
  private peak = 0;
  private sumSq = 0;
  private count = 0;
  private statsCount = 0;
  private chunks = 0;

  constructor() {
    super();
    const sr = scope.sampleRate;
    this.ring = new Float32Array(Math.ceil(sr * 2));
    this.targetFrames = Math.round(sr * 0.02); // 20 ms jitter buffer
    this.maxFrames = Math.round(sr * 0.15);
    this.resamplerDelay = this.resampler.delayOut;
    this.port.onmessage = (ev: MessageEvent<{ type: 'port'; port: MessagePort } | { type: 'config'; jitterMs: number }>) => {
      const d = ev.data;
      if (d?.type === 'port') d.port.onmessage = (e: MessageEvent<PlaybackChunk>) => this.onChunk(e.data);
      else if (d?.type === 'config') this.targetFrames = Math.round((sr * d.jitterMs) / 1000);
    };
  }

  private onChunk(chunk: PlaybackChunk): void {
    const y = this.resampler.process(chunk.samples);
    if ((this.chunks++ & 7) === 0) this.markers.push({ pos: this.writePos + this.resamplerDelay, capTime: chunk.capTime });
    const ring = this.ring;
    const len = ring.length;
    for (let i = 0; i < y.length; i++) ring[(this.writePos + i) % len] = y[i];
    this.writePos += y.length;
    // Overflow: drop the oldest audio so latency stays bounded.
    const buffered = this.writePos - this.readPos;
    if (buffered > this.maxFrames) this.readPos = this.writePos - this.targetFrames;
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0]?.[0];
    if (!out) return true;
    const n = out.length;
    const sr = scope.sampleRate;
    const buffered = this.writePos - this.readPos;
    if (this.buffering && buffered >= this.targetFrames + n) this.buffering = false;
    if (!this.buffering && buffered < n) {
      this.buffering = true;
      this.underruns += 1;
    }
    if (this.buffering) {
      out.fill(0);
    } else {
      const ring = this.ring;
      const len = ring.length;
      for (let i = 0; i < n; i++) out[i] = ring[(this.readPos + i) % len];
      // Latency: render time of a marked sample minus its capture time.
      while (this.markers.length && this.markers[0].pos < this.readPos + n) {
        const m = this.markers.shift()!;
        if (m.pos >= this.readPos) {
          const playTime = scope.currentTime + (m.pos - this.readPos) / sr;
          const l = playTime - m.capTime;
          this.latency = this.latencyValid ? this.latency * 0.8 + l * 0.2 : l;
          this.latencyValid = true;
        }
      }
      this.readPos += n;
    }
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    for (let i = 0; i < n; i++) {
      const a = Math.abs(out[i]);
      if (a > this.peak) this.peak = a;
      this.sumSq += out[i] * out[i];
    }
    this.count += n;
    this.statsCount += n;
    if (this.statsCount >= STATS_INTERVAL_S * sr) {
      this.port.postMessage({
        type: 'stats',
        latencyMs: this.latencyValid ? this.latency * 1000 : null,
        bufferedMs: ((this.writePos - this.readPos) / sr) * 1000,
        underruns: this.underruns,
      });
      this.statsCount = 0;
    }
    if (this.count >= 0.05 * sr) {
      this.port.postMessage({ type: 'level', peak: this.peak, rms: Math.sqrt(this.sumSq / this.count) });
      this.peak = 0;
      this.sumSq = 0;
      this.count = 0;
    }
    return true;
  }
}

scope.registerProcessor('ans-playback', PlaybackProcessor);
