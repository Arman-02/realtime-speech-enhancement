/**
 * Speech distortion vs. noise reduction (§4.6): x = noise reduction (dB), y = speech-distortion SI-SDR
 * (dB); both higher-is-better, so the ideal corner is top right. One connected series per model through
 * the input-SNR buckets (or one point per model when only overall numbers exist).
 */
import { useId } from 'react';
import { ticks, useTooltip, useWidth } from './chartKit';
import { fmt } from '../lib/format';

export interface TradeoffPoint {
  x: number;
  y: number;
  /** e.g. "2.5 dB" (SNR bucket); shown on the endpoints of the first series only (selective labels). */
  label: string;
}

export interface TradeoffSeries {
  name: string;
  color: string;
  points: TradeoffPoint[];
}

const H = 320;
const M = { left: 48, right: 18, top: 14, bottom: 42 };

export function TradeoffChart({ series, pointLabel }: { series: TradeoffSeries[]; pointLabel: string }) {
  const tt = useTooltip();
  const titleId = useId();
  const [wrap, W] = useWidth<HTMLDivElement>(600);
  const all = series.flatMap((s) => s.points);
  const pad = (lo: number, hi: number) => {
    const p = Math.max(1, (hi - lo) * 0.1);
    return [Math.floor(lo - p), Math.ceil(hi + p)] as const;
  };
  const [x0, x1] = pad(Math.min(...all.map((p) => p.x)), Math.max(...all.map((p) => p.x)));
  const [y0, y1] = pad(Math.min(...all.map((p) => p.y)), Math.max(...all.map((p) => p.y)));
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const sx = (v: number) => M.left + ((v - x0) / (x1 - x0 || 1)) * plotW;
  const sy = (v: number) => M.top + plotH - ((v - y0) / (y1 - y0 || 1)) * plotH;
  return (
    <div ref={wrap} className="w-full min-w-0">
      <div ref={tt.box} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block" role="img" aria-labelledby={titleId}>
          <title id={titleId}>Speech distortion (SI-SDR of the masked clean speech) vs. noise reduction, per model{pointLabel ? ` and ${pointLabel}` : ''}</title>
          {ticks(y0, y1).map((t) => (
            <g key={`y${t}`}>
              <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
              <text x={M.left - 8} y={sy(t) + 3.5} textAnchor="end" fontSize={10} fill="var(--text-3)" className="tnum">
                {t}
              </text>
            </g>
          ))}
          {ticks(x0, x1).map((t) => (
            <g key={`x${t}`}>
              <line x1={sx(t)} x2={sx(t)} y1={M.top} y2={H - M.bottom} stroke="var(--grid)" />
              <text x={sx(t)} y={H - M.bottom + 15} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="tnum">
                {t}
              </text>
            </g>
          ))}
          <text x={M.left + plotW / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-2)">
            Noise reduction (dB) → more noise removed
          </text>
          <text x={12} y={M.top + plotH / 2} textAnchor="middle" fontSize={11} fill="var(--text-2)" transform={`rotate(-90 12 ${M.top + plotH / 2})`}>
            Speech SI-SDR (dB) ↑ less voice damage
          </text>
          {series.map((s, si) => (
            <g key={s.name}>
              {s.points.length > 1 && <polyline points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />}
              {s.points.map((p, i) => (
                <g key={i}>
                  <circle cx={sx(p.x)} cy={sy(p.y)} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                  {si === 0 && s.points.length > 1 && (i === 0 || i === s.points.length - 1) && (
                    <text x={sx(p.x) + 7} y={sy(p.y) - 6} fontSize={9.5} fill="var(--text-3)" className="tnum">
                      {p.label}
                    </text>
                  )}
                  <circle
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={11}
                    fill="transparent"
                    onMouseMove={(e) =>
                      tt.show(
                        e,
                        <span>
                          <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: s.color }} />
                          <strong>{s.name}</strong>
                          {p.label && <span className="text-fg-3"> · {p.label}</span>}
                          <br />
                          Noise reduction <strong className="tnum">{fmt(p.x, 1)} dB</strong> · speech SI-SDR <strong className="tnum">{fmt(p.y, 1)} dB</strong>
                        </span>,
                      )
                    }
                    onMouseLeave={tt.hide}
                  />
                </g>
              ))}
            </g>
          ))}
        </svg>
        {tt.node}
      </div>
    </div>
  );
}
