import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { N_BINS } from '../engine/dsp';
import { MAGMA } from '../lib/colormap';
import { cssVar, useThemeVersion } from '../lib/theme';

export interface MeterHandle {
  set(peak: number, rms: number): void;
  reset(): void;
}

const toDb = (v: number) => (v > 0 ? 20 * Math.log10(v) : -Infinity);
const FLOOR = -60;

/** Horizontal dBFS meter (RMS bar + peak-hold tick), updated imperatively (no React re-render). */
export const LevelMeter = forwardRef<MeterHandle, { label: string; color: string }>(function LevelMeter({ label, color }, ref) {
  const barRef = useRef<HTMLDivElement>(null);
  const peakRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const hold = useRef({ db: FLOOR, t: 0 });

  useImperativeHandle(ref, () => ({
    set(peak, rms) {
      const rDb = Math.max(FLOOR, toDb(rms));
      const pDb = Math.max(FLOOR, toDb(peak));
      const now = performance.now();
      if (pDb > hold.current.db || now - hold.current.t > 1200) hold.current = { db: pDb, t: now };
      const pct = (d: number) => `${((d - FLOOR) / -FLOOR) * 100}%`;
      if (barRef.current) {
        barRef.current.style.width = pct(rDb);
        barRef.current.style.background = pDb > -1 ? 'var(--danger)' : color;
      }
      if (peakRef.current) peakRef.current.style.left = pct(hold.current.db);
      if (textRef.current) textRef.current.textContent = rDb <= FLOOR ? '−∞' : rDb.toFixed(0);
    },
    reset() {
      hold.current = { db: FLOOR, t: 0 };
      if (barRef.current) barRef.current.style.width = '0%';
      if (peakRef.current) peakRef.current.style.left = '0%';
      if (textRef.current) textRef.current.textContent = '−∞';
    },
  }));

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[12px]">
        <span className="font-medium text-fg-2">{label}</span>
        <span className="tnum font-mono text-fg-3">
          <span ref={textRef}>−∞</span> dBFS
        </span>
      </div>
      <div className="relative h-2 overflow-hidden rounded-full bg-surface-3" role="presentation">
        <div ref={barRef} className="h-full w-0 rounded-full transition-[width] duration-75 ease-out" style={{ background: color }} />
        <div ref={peakRef} className="absolute top-0 h-full w-0.5 bg-fg/70" style={{ left: 0 }} />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[9px] text-fg-3" aria-hidden>
        <span>−60</span>
        <span>−40</span>
        <span>−20</span>
        <span>0</span>
      </div>
    </div>
  );
});

export interface ScrollSpecHandle {
  push(db: Float32Array, frames: number): void;
  clear(): void;
}

const COL_W = 2; // css px per 10 ms frame
const LO = -45;
const HI = 35;

/** Scrolling spectrogram for the live view. */
export const ScrollingSpectrogram = forwardRef<ScrollSpecHandle, { label: string; height?: number }>(function ScrollingSpectrogram({ label, height = 120 }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const themeV = useThemeVersion();

  const clear = () => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = cssVar('--surface-2');
    ctx.fillRect(0, 0, c.width, c.height);
  };

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ro = new ResizeObserver(() => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      c.width = Math.round(c.clientWidth * dpr);
      c.height = Math.round(height * dpr);
      clear();
    });
    ro.observe(c);
    return () => ro.disconnect();
  }, [height]);

  useEffect(clear, [themeV]);

  useImperativeHandle(ref, () => ({
    push(db, frames) {
      const c = canvasRef.current;
      if (!c || c.width === 0) return;
      const ctx = c.getContext('2d')!;
      const dpr = c.height / height;
      const colW = Math.max(1, Math.round(COL_W * dpr));
      const shift = colW * frames;
      ctx.drawImage(c, -shift, 0);
      const img = ctx.createImageData(colW * frames, c.height);
      const h = c.height;
      for (let f = 0; f < frames; f++) {
        for (let y = 0; y < h; y++) {
          const k = Math.min(N_BINS - 1, Math.floor(((h - 1 - y) / h) * N_BINS));
          const v = db[f * N_BINS + k];
          const n = Math.max(0, Math.min(1, (v - LO) / (HI - LO)));
          const li = Math.round(n * 255) * 3;
          for (let x = 0; x < colW; x++) {
            const o = (y * colW * frames + f * colW + x) * 4;
            img.data[o] = MAGMA[li];
            img.data[o + 1] = MAGMA[li + 1];
            img.data[o + 2] = MAGMA[li + 2];
            img.data[o + 3] = 255;
          }
        }
      }
      ctx.putImageData(img, c.width - shift, 0);
    },
    clear,
  }));

  return (
    <figure className="relative overflow-hidden rounded-lg border border-line">
      <canvas ref={canvasRef} className="block w-full" style={{ height }} role="img" aria-label={`${label} live spectrogram`} />
      <figcaption className="absolute top-1.5 left-2 rounded bg-black/45 px-1.5 py-0.5 text-[11px] font-medium text-white/90 backdrop-blur-sm">{label}</figcaption>
      <div className="pointer-events-none absolute top-1 right-1.5 bottom-1 flex flex-col justify-between text-right font-mono text-[9px] text-white/60" aria-hidden>
        <span>8k</span>
        <span>4k</span>
        <span>0 Hz</span>
      </div>
    </figure>
  );
});
