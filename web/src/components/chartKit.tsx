/** Shared chart plumbing: responsive width, hover tooltip, nice ticks, log ticks. */
import { useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react';

/** Width of an element in CSS px (charts render 1:1, so text and marks keep their specified sizes). */
export function useWidth<T extends HTMLElement>(fallback: number): [RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(Math.max(280, Math.floor(el.getBoundingClientRect().width)));
    const ro = new ResizeObserver((e) => setW(Math.max(280, Math.floor(e[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

interface Tip {
  x: number;
  y: number;
  content: ReactNode;
}

export function useTooltip() {
  const [tip, setTip] = useState<Tip | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const show = (e: { clientX: number; clientY: number }, content: ReactNode) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, content });
  };
  const showAt = (x: number, y: number, content: ReactNode) => setTip({ x, y, content });
  const hide = () => setTip(null);
  const node = tip ? (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[calc(100%+10px)] rounded-lg border border-line bg-surface px-2.5 py-1.5 text-[12px] whitespace-nowrap shadow-lg"
      style={{ left: tip.x, top: tip.y }}
    >
      {tip.content}
    </div>
  ) : null;
  return { box, show, showAt, hide, node };
}

export function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  const n = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
  return n * p;
}

export function ticks(min: number, max: number, count = 4): number[] {
  const step = niceMax((max - min) / count);
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

/** 1-2-5 ticks per decade inside [lo, hi] (both > 0). */
export function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out;
}
