import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { N_BINS } from '../engine/dsp';
import { MAGMA } from '../lib/colormap';
import { cssVar, useThemeVersion } from '../lib/theme';
import { formatTime } from '../lib/audio';
import { cx } from '../lib/format';

export interface SpecTrack {
  key: string;
  label: string;
  color: string;
  audio: Float32Array | null;
  db: Float32Array | null;
  frames: number;
  active: boolean;
  placeholder?: string;
  onSelect?: () => void;
  shortcut?: string;
}

const WAVE_H = 34;
const DYN_RANGE = 80;

function renderSpec(db: Float32Array, frames: number, ref: number): HTMLCanvasElement {
  // One column per frame (capped), low frequencies at the bottom.
  const maxCols = 2400;
  const step = Math.max(1, Math.ceil(frames / maxCols));
  const cols = Math.ceil(frames / step);
  const c = document.createElement('canvas');
  c.width = cols;
  c.height = N_BINS;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(cols, N_BINS);
  const floor = ref - DYN_RANGE;
  for (let x = 0; x < cols; x++) {
    for (let k = 0; k < N_BINS; k++) {
      let v = -Infinity;
      for (let s = 0; s < step; s++) {
        const t = x * step + s;
        if (t < frames) v = Math.max(v, db[t * N_BINS + k]);
      }
      const n = Math.max(0, Math.min(1, (v - floor) / DYN_RANGE));
      const li = Math.round(Math.pow(n, 1.15) * 255) * 3;
      const o = ((N_BINS - 1 - k) * cols + x) * 4;
      img.data[o] = MAGMA[li];
      img.data[o + 1] = MAGMA[li + 1];
      img.data[o + 2] = MAGMA[li + 2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function TrackCanvas({ track, refDb, height }: { track: SpecTrack; refDb: number; height: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const themeV = useThemeVersion();
  const specImg = useRef<{ src: Float32Array; ref: number; canvas: HTMLCanvasElement } | null>(null);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => setWidth(Math.floor(entries[0].contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el || width === 0) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    el.width = Math.round(width * dpr);
    el.height = Math.round(height * dpr);
    const ctx = el.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const surface2 = cssVar('--surface-2');
    const grid = cssVar('--border');
    ctx.fillStyle = surface2;
    ctx.fillRect(0, 0, width, height);
    if (!track.db || !track.audio) return;

    // Waveform strip (min/max envelope)
    const x = track.audio;
    const mid = WAVE_H / 2;
    const m = /^var\((--[\w-]+)\)$/.exec(track.color);
    ctx.fillStyle = m ? cssVar(m[1]) : track.color;
    ctx.globalAlpha = 0.9;
    const per = x.length / width;
    let peak = 1e-6;
    for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    const scale = (WAVE_H / 2 - 3) / Math.max(peak, 0.05);
    for (let px = 0; px < width; px++) {
      const a = Math.floor(px * per);
      const b = Math.min(x.length, Math.floor((px + 1) * per) + 1);
      let lo = 0;
      let hi = 0;
      for (let i = a; i < b; i++) {
        const v = x[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      const y0 = mid - hi * scale;
      const y1 = mid - lo * scale;
      ctx.fillRect(px, y0, 1, Math.max(1, y1 - y0));
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = grid;
    ctx.fillRect(0, WAVE_H, width, 1);

    // Spectrogram
    if (!specImg.current || specImg.current.src !== track.db || specImg.current.ref !== refDb) {
      specImg.current = { src: track.db, ref: refDb, canvas: renderSpec(track.db, track.frames, refDb) };
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(specImg.current.canvas, 0, WAVE_H + 1, width, height - WAVE_H - 1);
  }, [track.db, track.audio, track.frames, track.color, refDb, width, height, themeV]);

  return <canvas ref={canvasRef} className="block h-full w-full" style={{ height }} aria-hidden="true" />;
}

export function StackedSpectrograms({
  tracks,
  duration,
  getTime,
  onSeek,
  rowHeight = 132,
}: {
  tracks: SpecTrack[];
  duration: number;
  getTime: () => number;
  onSeek: (t: number) => void;
  rowHeight?: number;
}) {
  const areaRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<HTMLDivElement>(null);
  const timeRef = useRef<HTMLSpanElement>(null);
  const dragging = useRef(false);
  const dbs = tracks.map((t) => t.db);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const refDb = useMemo(() => Math.max(...dbs.map((d) => (d ? maxOf(d) : -Infinity)), -60), dbs);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const area = areaRef.current;
      const head = headRef.current;
      if (area && head && duration > 0) {
        const t = getTime();
        const w = area.clientWidth;
        head.style.transform = `translateX(${(t / duration) * w}px)`;
        if (timeRef.current) timeRef.current.textContent = formatTime(t);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [duration, getTime]);

  const seekFromEvent = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      const area = areaRef.current;
      if (!area || duration <= 0) return;
      const r = area.getBoundingClientRect();
      onSeek(((e.clientX - r.left) / r.width) * duration);
    },
    [duration, onSeek],
  );

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      onSeek(getTime() + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 5 : 1));
    } else if (e.key === 'Home') {
      e.preventDefault();
      onSeek(0);
    }
  };

  const ticks = timeTicks(duration);

  return (
    <div>
      <div className="flex gap-3">
        {/* Left labels */}
        <div className="flex w-[74px] shrink-0 flex-col gap-2 sm:w-[96px]">
          {tracks.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={t.onSelect}
              disabled={!t.audio}
              aria-pressed={t.active}
              className={cx(
                'group flex flex-col items-start justify-between rounded-lg border px-2 py-2 text-left transition-colors disabled:cursor-not-allowed',
                t.active ? 'border-line-strong bg-surface-2' : 'border-transparent hover:bg-surface-2',
              )}
              style={{ height: rowHeight }}
            >
              <span className="flex items-center gap-1.5">
                <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: t.color, opacity: t.audio ? 1 : 0.35 }} />
                <span className={cx('text-[13px] font-medium', t.audio ? 'text-fg' : 'text-fg-3')}>{t.label}</span>
              </span>
              <span className="flex w-full items-center justify-between text-[11px] text-fg-3">
                {t.active && t.audio ? <span className="text-accent">Listening</span> : <span>{t.audio ? 'Listen' : ''}</span>}
                {t.shortcut && <span className="hidden font-mono sm:inline">{t.shortcut}</span>}
              </span>
            </button>
          ))}
        </div>
        {/* Plot area */}
        <div
          ref={areaRef}
          role="slider"
          tabIndex={0}
          aria-label="Playback position (click to seek, arrow keys to step)"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration * 10) / 10}
          aria-valuenow={0}
          aria-valuetext="Use arrow keys to seek"
          onKeyDown={onKey}
          onPointerDown={(e) => {
            dragging.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            seekFromEvent(e);
          }}
          onPointerMove={(e) => dragging.current && seekFromEvent(e)}
          onPointerUp={() => (dragging.current = false)}
          className="relative min-w-0 flex-1 cursor-crosshair touch-none rounded-lg"
        >
          <div className="flex flex-col gap-2">
            {tracks.map((t) => (
              <div
                key={t.key}
                className={cx('relative overflow-hidden rounded-lg border transition-[border-color,opacity] duration-200', t.active ? 'border-line-strong' : 'border-line')}
                style={{ height: rowHeight, boxShadow: t.active ? `inset 3px 0 0 ${t.color}` : undefined }}
              >
                {t.db ? (
                  <TrackCanvas track={t} refDb={refDb} height={rowHeight - 2} />
                ) : (
                  <div className="grid h-full place-items-center bg-surface-2 text-[12px] text-fg-3">{t.placeholder ?? 'Not available'}</div>
                )}
                {t.db && (
                  <div className="pointer-events-none absolute top-[36px] right-1.5 bottom-1 flex flex-col justify-between text-right font-mono text-[9px] leading-none text-white/60">
                    <span>8k</span>
                    <span>4k</span>
                    <span>0 Hz</span>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div ref={headRef} aria-hidden className="pointer-events-none absolute top-0 bottom-0 left-0 w-px bg-fg shadow-[0_0_0_1px_rgb(0_0_0/0.25)]" style={{ transform: 'translateX(0px)' }}>
            <span className="absolute -top-1 -left-[3px] h-[7px] w-[7px] rounded-full bg-fg" />
          </div>
        </div>
      </div>
      {/* Time axis */}
      <div className="mt-1.5 flex gap-3">
        <div className="w-[74px] shrink-0 sm:w-[96px]">
          <span ref={timeRef} className="tnum font-mono text-[11px] text-fg-2">
            0:00.0
          </span>
        </div>
        <div className="relative h-4 min-w-0 flex-1 font-mono text-[10px] text-fg-3">
          {ticks.map((t) => (
            <span key={t} className="tnum absolute -translate-x-1/2" style={{ left: `${(t / duration) * 100}%` }}>
              {t}s
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function maxOf(a: Float32Array): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

function timeTicks(duration: number): number[] {
  if (duration <= 0) return [];
  const steps = [0.5, 1, 2, 5, 10, 15, 30];
  const step = steps.find((s) => duration / s <= 8) ?? 60;
  const out: number[] = [];
  for (let t = step; t < duration - step * 0.3; t += step) out.push(Math.round(t * 10) / 10);
  return out;
}
