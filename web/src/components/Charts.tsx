import { useId } from 'react';
import { niceMax, ticks, useTooltip, useWidth } from './chartKit';

export interface BarDatum {
  label: string;
  value: number | null;
  color: string;
  emphasis?: boolean;
}

/** Horizontal bars, one per model, value at the tip. */
export function HBarChart({ data, unit, digits = 2, domain, title }: { data: BarDatum[]; unit?: string; digits?: number; domain?: [number, number]; title: string }) {
  const tt = useTooltip();
  const titleId = useId();
  const [wrap, W] = useWidth<HTMLDivElement>(300);
  const vals = data.map((d) => d.value).filter((v): v is number => v !== null && Number.isFinite(v));
  const lo = domain?.[0] ?? Math.min(0, ...vals);
  const hi = domain?.[1] ?? niceMax(Math.max(...vals, 0) * 1.08);
  const labelW = Math.min(210, Math.max(120, W * 0.4));
  const valueW = 52;
  const row = 30;
  const barH = 16;
  const top = 6;
  const H = top + data.length * row + 22;
  const plotW = W - labelW - valueW;
  const sx = (v: number) => labelW + ((v - lo) / (hi - lo || 1)) * plotW;
  const tks = ticks(lo, hi);
  const maxChars = Math.floor((labelW - 12) / 6.4);
  return (
    <div ref={wrap} className="w-full min-w-0">
    <div ref={tt.box} className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block" role="img" aria-labelledby={titleId}>
        <title id={titleId}>{title}</title>
        {tks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={top - 2} y2={H - 20} stroke="var(--grid)" strokeWidth={1} />
            <text x={sx(t)} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="tnum">
              {t}
            </text>
          </g>
        ))}
        <line x1={sx(Math.max(lo, 0))} x2={sx(Math.max(lo, 0))} y1={top - 2} y2={H - 20} stroke="var(--border-strong)" strokeWidth={1} />
        {data.map((d, i) => {
          const y = top + i * row;
          const v = d.value;
          const has = v !== null && Number.isFinite(v);
          const x0 = sx(Math.max(lo, 0));
          const x1 = has ? sx(v as number) : x0;
          const w = Math.abs(x1 - x0);
          const left = Math.min(x0, x1);
          const r = Math.min(4, w / 2);
          // Rounded data-end, square at the baseline.
          const pos = (v ?? 0) >= 0;
          const path = has
            ? pos
              ? `M${left},${y + (row - barH) / 2} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${barH - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-(w - r)} z`
              : `M${left + w},${y + (row - barH) / 2} h${-(w - r)} a${r},${r} 0 0 0 ${-r},${r} v${barH - 2 * r} a${r},${r} 0 0 0 ${r},${r} h${w - r} z`
            : '';
          return (
            <g key={d.label}>
              <text x={labelW - 10} y={y + row / 2 + 4} textAnchor="end" fontSize={12} fill={d.emphasis ? 'var(--text)' : 'var(--text-2)'} fontWeight={d.emphasis ? 600 : 400}>
                {d.label.length > maxChars ? `${d.label.slice(0, maxChars - 1)}…` : d.label}
              </text>
              {has && <path d={path} fill={d.color} />}
              <text x={has ? Math.max(x1, x0) + 6 : x0 + 6} y={y + row / 2 + 4} fontSize={11} fill={has ? 'var(--text)' : 'var(--text-3)'} className="tnum">
                {has ? (v as number).toFixed(digits) : 'n/a'}
              </text>
              <rect
                x={0}
                y={y}
                width={W}
                height={row}
                fill="transparent"
                onMouseMove={(e) =>
                  tt.show(
                    e,
                    <span>
                      <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: d.color }} />
                      {d.label}: <strong className="tnum">{has ? (v as number).toFixed(digits) : 'n/a'}</strong>
                      {unit ? ` ${unit}` : ''}
                    </span>,
                  )
                }
                onMouseLeave={tt.hide}
              />
            </g>
          );
        })}
      </svg>
      {tt.node}
    </div>
    </div>
  );
}

export interface Series {
  name: string;
  color: string;
}

/** Grouped columns: categories on x, one column per series. */
export function GroupedColumns({
  categories,
  categoryLabel = (c: string) => c,
  series,
  value,
  digits = 2,
  title,
}: {
  categories: string[];
  categoryLabel?: (c: string) => string;
  series: Series[];
  value: (category: string, series: string) => number | null;
  digits?: number;
  title: string;
}) {
  const tt = useTooltip();
  const titleId = useId();
  const [wrap, cw] = useWidth<HTMLDivElement>(300);
  const all: number[] = [];
  for (const c of categories) for (const s of series) {
    const v = value(c, s.name);
    if (v !== null && Number.isFinite(v)) all.push(v);
  }
  const lo = Math.min(0, ...all);
  const hi = niceMax(Math.max(...all, 0) * 1.05);
  const W = Math.max(cw, categories.length * (series.length * 14 + 24) + 50);
  const H = 260;
  const left = 36;
  const bottom = 34;
  const top = 10;
  const plotH = H - top - bottom;
  const band = (W - left - 8) / Math.max(1, categories.length);
  const colW = Math.max(6, Math.min(24, (band * 0.7) / Math.max(1, series.length) - 2));
  const sy = (v: number) => top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;
  const tks = ticks(lo, hi);
  return (
    <div ref={wrap} className="w-full min-w-0 overflow-x-auto">
    <div ref={tt.box} className="relative" style={{ width: W }}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block" role="img" aria-labelledby={titleId}>
        <title id={titleId}>{title}</title>
        {tks.map((t) => (
          <g key={t}>
            <line x1={left} x2={W - 4} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={left - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="var(--text-3)" className="tnum">
              {t}
            </text>
          </g>
        ))}
        <line x1={left} x2={W - 4} y1={sy(Math.max(lo, 0))} y2={sy(Math.max(lo, 0))} stroke="var(--border-strong)" />
        {categories.map((c, ci) => {
          const groupW = series.length * (colW + 2) - 2;
          const gx = left + ci * band + (band - groupW) / 2;
          return (
            <g key={c}>
              {series.map((s, si) => {
                const v = value(c, s.name);
                if (v === null || !Number.isFinite(v)) return null;
                const x = gx + si * (colW + 2);
                const y0 = sy(Math.max(lo, 0));
                const y1 = sy(v);
                const h = Math.abs(y0 - y1);
                const r = Math.min(4, h / 2, colW / 2);
                const top0 = Math.min(y0, y1);
                const d = `M${x},${top0 + h} v${-(h - r)} a${r},${r} 0 0 1 ${r},${-r} h${colW - 2 * r} a${r},${r} 0 0 1 ${r},${r} v${h - r} z`;
                return (
                  <path
                    key={s.name}
                    d={d}
                    fill={s.color}
                    onMouseMove={(e) =>
                      tt.show(
                        e,
                        <span>
                          <span className="text-fg-3">{categoryLabel(c)}</span>
                          <br />
                          <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: s.color }} />
                          {s.name}: <strong className="tnum">{v.toFixed(digits)}</strong>
                        </span>,
                      )
                    }
                    onMouseLeave={tt.hide}
                  />
                );
              })}
              <text x={left + ci * band + band / 2} y={H - bottom + 16} textAnchor="middle" fontSize={11} fill="var(--text-2)">
                {categoryLabel(c)}
              </text>
            </g>
          );
        })}
      </svg>
      {tt.node}
    </div>
    </div>
  );
}

export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg-2" aria-label="Legend">
      {series.map((s) => (
        <li key={s.name} className="flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: s.color }} />
          {s.name}
        </li>
      ))}
    </ul>
  );
}
