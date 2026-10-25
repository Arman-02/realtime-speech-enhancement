/**
 * Capture worklet: mic (device rate) -> mono -> 16 kHz -> engine worker (via a transferred MessagePort).
 * Also reports input levels to the main thread on its own port.
 */
import { StreamResampler } from '../resample';
import { Processor, scope } from './worklet-env';
import type { CaptureChunk } from '../protocol';

const LEVEL_INTERVAL_S = 0.05;

class CaptureProcessor extends Processor {
  private worker: MessagePort | null = null;
  private resampler = new StreamResampler(scope.sampleRate, 16000);
  private index = 0;
  private peak = 0;
  private sumSq = 0;
  private count = 0;
  private mono = new Float32Array(128);

  constructor() {
    super();
    this.port.onmessage = (ev: MessageEvent<{ type: 'port'; port: MessagePort }>) => {
      if (ev.data?.type === 'port') this.worker = ev.data.port;
    };
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const n = input[0].length;
    if (this.mono.length !== n) this.mono = new Float32Array(n);
    const mono = this.mono;
    const chans = input.length;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let c = 0; c < chans; c++) s += input[c][i];
      mono[i] = s / chans;
    }
    for (let i = 0; i < n; i++) {
      const a = Math.abs(mono[i]);
      if (a > this.peak) this.peak = a;
      this.sumSq += mono[i] * mono[i];
    }
    this.count += n;
    if (this.count >= LEVEL_INTERVAL_S * scope.sampleRate) {
      this.port.postMessage({ type: 'level', peak: this.peak, rms: Math.sqrt(this.sumSq / this.count) });
      this.peak = 0;
      this.sumSq = 0;
      this.count = 0;
    }
    if (this.worker) {
      const out = this.resampler.process(mono);
      if (out.length > 0) {
        this.index += out.length;
        // Context time of the last input sample, minus the resampler's group delay.
        const endTime = scope.currentTime + n / scope.sampleRate - this.resampler.delayIn / scope.sampleRate;
        const chunk: CaptureChunk = { samples: out, endIndex: this.index, endTime };
        this.worker.postMessage(chunk, [out.buffer]);
      }
    }
    return true;
  }
}

scope.registerProcessor('ans-capture', CaptureProcessor);
