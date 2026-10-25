/**
 * Band-limited streaming resampler (windowed-sinc, Blackman-Harris window, tabulated kernel).
 * Used by the capture/playback AudioWorklets (device rate <-> 16 kHz) and as an offline fallback
 * when OfflineAudioContext cannot decode straight to 16 kHz.
 */

const TABLE_RES = 512; // kernel samples per input-sample unit

export class StreamResampler {
  readonly inRate: number;
  readonly outRate: number;
  /** Group delay introduced by the filter, in input samples (may be fractional). */
  readonly delayIn: number;
  /** Group delay in output samples (always an integer, so offline alignment is exact). */
  readonly delayOut: number;
  private readonly step: number;
  private readonly halfWidth: number; // kernel half-width in input samples
  private readonly table: Float32Array;
  private readonly passthrough: boolean;
  private buf: Float32Array;
  private bufLen: number;
  private bufStart: number; // absolute input index of buf[0]
  private t: number; // absolute input position of the next output sample

  constructor(inRate: number, outRate: number, zeroCrossings = 16, rolloff = 0.94) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.step = inRate / outRate;
    this.passthrough = inRate === outRate;
    const fc = 0.5 * Math.min(1, outRate / inRate) * rolloff; // cycles per input sample
    this.halfWidth = Math.ceil(zeroCrossings / (2 * fc));
    const w = this.halfWidth;
    this.table = new Float32Array(w * TABLE_RES + 2);
    for (let i = 0; i < this.table.length; i++) {
      const d = i / TABLE_RES;
      if (d >= w) {
        this.table[i] = 0;
        continue;
      }
      const x = 2 * fc * d;
      const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      // Blackman-Harris over [-w, w]
      const z = (Math.PI * (d + w)) / w; // 0..2π across the full support
      const win = 0.35875 - 0.48829 * Math.cos(z) + 0.14128 * Math.cos(2 * z) - 0.01168 * Math.cos(3 * z);
      this.table[i] = 2 * fc * sinc * win;
    }
    this.delayOut = this.passthrough ? 0 : Math.ceil((w * outRate) / inRate);
    this.delayIn = (this.delayOut * inRate) / outRate; // >= w, keeps the filter causal
    this.buf = new Float32Array(4096);
    // Input sample i lives at absolute index i; indices < 0 are implicit zeros.
    this.bufLen = 0;
    this.bufStart = 0;
    this.t = 0;
  }

  private kernel(d: number): number {
    const a = Math.abs(d) * TABLE_RES;
    const i = Math.floor(a);
    const f = a - i;
    const t = this.table;
    if (i + 1 >= t.length) return 0;
    return t[i] + (t[i + 1] - t[i]) * f;
  }

  /** Feed input samples; returns the output samples that are now computable. */
  process(input: Float32Array): Float32Array {
    if (this.passthrough) return input.slice();
    const need = this.bufLen + input.length;
    if (need > this.buf.length) {
      const grown = new Float32Array(Math.max(need, this.buf.length * 2));
      grown.set(this.buf.subarray(0, this.bufLen));
      this.buf = grown;
    }
    this.buf.set(input, this.bufLen);
    this.bufLen = need;

    const w = this.halfWidth;
    const bufEnd = this.bufStart + this.bufLen; // exclusive absolute index
    // Output k sits at input position t = k·step and is evaluated centred at c = t - delayIn
    // (delayIn >= w), so it only needs input samples <= t.
    const maxOut = Math.max(0, Math.floor((bufEnd - 1 - this.t) / this.step) + 1);
    const out = new Float32Array(maxOut);
    let n = 0;
    while (n < maxOut) {
      const c = this.t - this.delayIn; // centre (absolute input index, fractional)
      const lo = Math.ceil(c - w);
      const hi = Math.floor(c + w);
      if (hi >= bufEnd) break;
      let acc = 0;
      for (let j = Math.max(lo, this.bufStart); j <= hi; j++) acc += this.buf[j - this.bufStart] * this.kernel(c - j);
      out[n++] = acc;
      this.t += this.step;
    }
    // Drop samples that can no longer contribute.
    const keepFrom = Math.floor(this.t - 2 * w) - 1;
    const drop = Math.max(0, Math.min(this.bufLen, keepFrom - this.bufStart));
    if (drop > 0) {
      this.buf.copyWithin(0, drop, this.bufLen);
      this.bufLen -= drop;
      this.bufStart += drop;
    }
    return n === maxOut ? out : out.subarray(0, n).slice();
  }
}

/** Resample a whole signal, compensating the filter delay so the output aligns with the input. */
export function resampleOffline(x: Float32Array, inRate: number, outRate: number): Float32Array {
  if (inRate === outRate) return x.slice();
  const r = new StreamResampler(inRate, outRate);
  const outLen = Math.round((x.length * outRate) / inRate);
  const delayOut = r.delayOut;
  const flush = new Float32Array(Math.ceil(r.delayIn * 2 + 8));
  const a = r.process(x);
  const b = r.process(flush);
  const all = new Float32Array(a.length + b.length);
  all.set(a);
  all.set(b, a.length);
  const out = new Float32Array(outLen);
  out.set(all.subarray(delayOut, delayOut + outLen));
  return out;
}
