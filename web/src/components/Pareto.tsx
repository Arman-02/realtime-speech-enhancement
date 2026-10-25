/**
 * Quality-vs-compute Pareto chart (§4.9 headline figure): x = MACs per frame on a log scale, y = PESQ or
 * DNSMOS OVRL. Two lines (plain vs. distilled students) through the per-size seed means, individual seeds
 * as faint dots with a min–max range bar, and the teacher / reference systems as labelled points.
 */
import { useId, useMemo } from 'react';
import { logTicks, useTooltip, useWidth } from './chartKit';
import { compact, fmt } from '../lib/format';

export interface ParetoInput {
  model: string;
  hidden: number | null;
  distilled: boolean;
  seed: number | null;
  macs: number;
  y: number;
}

export interface ParetoRef {
  label: string;
  kind: 'teacher' | 'reference';
  macs: number;
  y: number;
}

interface ArmPoint {
  key: string;
  label: string;
  macs: number;
  mean: number;
  min: number;
  max: number;
  seeds: ParetoInput[];
}

const ARM_COLORS = { plain: 'var(--s1)', kd: 'var(--s2)' } as const;

/** Group seeds by student size (hidden, else MACs) and average them. */
function armPoints(points: ParetoInput[]): ArmPoint[] {
  const groups = new Map<string, ParetoInput[]>();
  for (const p of points) {
    const k = p.hidden !== null ? `h${p.hidden}` : `m${p.macs}`;
    groups.set(k, [...(groups.get(k) ?? []), p]);
  }
  return [...groups.entries()]
    .map(([key, ps]) => {
      const ys = ps.map((p) => p.y);
      return {
        key,
        label: ps[0].hidden !== null ? String(ps[0].hidden) : ps[0].model,
        macs: ps.reduce((a, p) => a + p.macs, 0) / ps.length,
        mean: ys.reduce((a, b) => a + b, 0) / ys.length,
        min: Math.min(...ys),
        max: Math.max(...ys),
        seeds: ps,
      };
    })
    .sort((a, b) => a.macs - b.macs);
}

const H = 340;
const M = { left: 48, right: 20, top: 18, bottom: 42 };

export function ParetoChart({ points, refs, metricLabel, digits }: { points: ParetoInput[]; refs: ParetoRef[]; metricLabel: string; digits: number }) {
  const tt = useTooltip();
  const titleId = useId();
  const descId = useId();
  const [wrap, W] = useWidth<HTMLDivElement>(640);
  const arms = useMemo(
    () => [
      { id: 'plain' as const, name: 'Plain', color: ARM_COLORS.plain, pts: armPoints(points.filter((p) => !p.distilled)) },
      { id: 'kd' as const, name: 'Distilled (KD)', color: ARM_COLORS.kd, pts: armPoints(points.filter((p) => p.distilled)) },
    ],
    [points],
  );

  const xs = [...points.map((p) => p.macs), ...refs.map((r) => r.macs)];
  const ys = [...points.map((p) => p.y), ...refs.map((r) => r.y)];
  const lx0 = Math.log10(Math.min(...xs)) - 0.18;
  const lx1 = Math.log10(Math.max(...xs)) + 0.18;
  const yPad = Math.max(0.05, (Math.max(...ys) - Math.min(...ys)) * 0.12);
  const y0 = Math.floor((Math.min(...ys) - yPad) * 10) / 10;
  const y1 = Math.ceil((Math.max(...ys) + yPad) * 10) / 10;
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const sx = (v: number) => M.left + ((Math.log10(v) - lx0) / (lx1 - lx0 || 1)) * plotW;
  const sy = (v: number) => M.top + plotH - ((v - y0) / (y1 - y0 || 1)) * plotH;
  const xt = logTicks(Math.pow(10, lx0), Math.pow(10, lx1));
  const yStep = (y1 - y0) / 4 > 0.15 ? 0.25 : 0.1;
  const yt: number[] = [];
  for (let v = Math.ceil(y0 / yStep) * yStep; v <= y1 + 1e-9; v += yStep) yt.push(Math.round(v * 100) / 100);

  const desc = arms
    .map((a) => `${a.name}: ${a.pts.map((p) => `${p.label} at ${compact(p.macs)} MACs = ${fmt(p.mean, digits)}`).join('; ')}`)
    .concat(refs.map((r) => `${r.label} (${r.kind}) at ${compact(r.macs)} MACs = ${fmt(r.y, digits)}`))
    .join('. ');

  return (
    <div ref={wrap} className="w-full min-w-0">
      <div ref={tt.box} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block" role="img" aria-labelledby={titleId} aria-describedby={descId}>
          <title id={titleId}>{`${metricLabel} vs. MACs per frame (log scale), plain vs. distilled students`}</title>
          <desc id={descId}>{desc}</desc>
          {yt.map((t) => (
            <g key={`y${t}`}>
              <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
              <text x={M.left - 8} y={sy(t) + 3.5} textAnchor="end" fontSize={10} fill="var(--text-3)" className="tnum">
                {t.toFixed(yStep < 0.25 ? 1 : 2)}
              </text>
            </g>
          ))}
          {xt.map((t) => (
            <g key={`x${t}`}>
              <line x1={sx(t)} x2={sx(t)} y1={M.top} y2={H - M.bottom} stroke="var(--grid)" />
              <text x={sx(t)} y={H - M.bottom + 15} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="tnum">
                {compact(t)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="var(--border-strong)" />
          <text x={M.left + plotW / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-2)">
            MACs per 10 ms frame (log scale) →
          </text>
          <text x={12} y={M.top + plotH / 2} textAnchor="middle" fontSize={11} fill="var(--text-2)" transform={`rotate(-90 12 ${M.top + plotH / 2})`}>
            {metricLabel} ↑
          </text>

          {/* reference systems and teacher: neutral ink, distinct shapes, direct labels */}
          {refs.map((r) => {
            const x = sx(r.macs);
            const y = sy(r.y);
            const right = x > W - M.right - 130;
            return (
              <g key={`r-${r.label}`}>
                {r.kind === 'teacher' ? (
                  <path d={`M${x},${y - 6} L${x + 6},${y} L${x},${y + 6} L${x - 6},${y} Z`} fill="var(--surface)" stroke="var(--text-2)" strokeWidth={1.6} />
                ) : (
                  <rect x={x - 5} y={y - 5} width={10} height={10} rx={2} fill="var(--surface)" stroke="var(--text-2)" strokeWidth={1.6} />
                )}
                <text x={right ? x - 10 : x + 10} y={y + 3.5} textAnchor={right ? 'end' : 'start'} fontSize={10.5} fill="var(--text-2)">
                  {r.label}
                </text>
                <circle
                  cx={x}
                  cy={y}
                  r={12}
                  fill="transparent"
                  onMouseMove={(e) =>
                    tt.show(
                      e,
                      <span>
                        <strong>{r.label}</strong> <span className="text-fg-3">({r.kind})</span>
                        <br />
                        {metricLabel} <strong className="tnum">{fmt(r.y, digits)}</strong> · {compact(r.macs)} MACs
                      </span>,
                    )
                  }
                  onMouseLeave={tt.hide}
                />
              </g>
            );
          })}

          {arms.map((arm) => (
            <g key={arm.id}>
              {arm.pts.length > 1 && (
                <polyline points={arm.pts.map((p) => `${sx(p.macs)},${sy(p.mean)}`).join(' ')} fill="none" stroke={arm.color} strokeWidth={2} strokeLinejoin="round" />
              )}
              {arm.pts.map((p) => {
                const x = sx(p.macs);
                const y = sy(p.mean);
                return (
                  <g key={p.key}>
                    {p.seeds.length > 1 && (
                      <g stroke={arm.color} strokeWidth={1.5}>
                        <line x1={x} x2={x} y1={sy(p.min)} y2={sy(p.max)} />
                        <line x1={x - 4} x2={x + 4} y1={sy(p.min)} y2={sy(p.min)} />
                        <line x1={x - 4} x2={x + 4} y1={sy(p.max)} y2={sy(p.max)} />
                      </g>
                    )}
                    {p.seeds.length > 1 &&
                      p.seeds.map((s, i) => <circle key={i} cx={x + (arm.id === 'kd' ? 7 : -7)} cy={sy(s.y)} r={2.5} fill={arm.color} opacity={0.55} />)}
                    {arm.id === 'kd' ? (
                      <rect x={x - 5} y={y - 5} width={10} height={10} rx={2} fill={arm.color} stroke="var(--surface)" strokeWidth={2} />
                    ) : (
                      <circle cx={x} cy={y} r={5.5} fill={arm.color} stroke="var(--surface)" strokeWidth={2} />
                    )}
                    <text x={x} y={arm.id === 'kd' ? y - 11 : y + 18} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="tnum">
                      {p.label}
                    </text>
                    <circle
                      cx={x}
                      cy={y}
                      r={12}
                      fill="transparent"
                      onMouseMove={(e) =>
                        tt.show(
                          e,
                          <span>
                            <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: arm.color }} />
                            <strong>{arm.name}</strong> · hidden {p.label}
                            <br />
                            {metricLabel} <strong className="tnum">{fmt(p.mean, digits)}</strong>
                            {p.seeds.length > 1 && (
                              <span className="text-fg-3">
                                {' '}
                                (mean of {p.seeds.length} seeds, {fmt(p.min, digits)}–{fmt(p.max, digits)})
                              </span>
                            )}
                            <br />
                            {compact(p.macs)} MACs / frame
                          </span>,
                        )
                      }
                      onMouseLeave={tt.hide}
                    />
                  </g>
                );
              })}
              {arm.pts.length > 0 &&
                (() => {
                  const last = arm.pts[arm.pts.length - 1];
                  const x = sx(last.macs);
                  const right = x > W - M.right - 90;
                  return (
                    <text x={right ? x - 12 : x + 12} y={sy(last.mean) + (arm.id === 'kd' ? -2 : 12)} textAnchor={right ? 'end' : 'start'} fontSize={11} fontWeight={600} fill="var(--text)">
                      {arm.id === 'kd' ? 'KD' : 'Plain'}
                    </text>
                  );
                })()}
            </g>
          ))}
        </svg>
        {tt.node}
      </div>
    </div>
  );
}

export function ParetoLegend({ hasTeacher, hasRef, hasSeeds }: { hasTeacher: boolean; hasRef: boolean; hasSeeds: boolean }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg-2" aria-label="Legend">
      <li className="flex items-center gap-1.5">
        <svg width="12" height="12" aria-hidden>
          <circle cx="6" cy="6" r="5" fill={ARM_COLORS.plain} />
        </svg>
        Plain student
      </li>
      <li className="flex items-center gap-1.5">
        <svg width="12" height="12" aria-hidden>
          <rect x="1" y="1" width="10" height="10" rx="2" fill={ARM_COLORS.kd} />
        </svg>
        Distilled student (KD)
      </li>
      {hasSeeds && (
        <li className="flex items-center gap-1.5">
          <svg width="12" height="12" aria-hidden>
            <line x1="6" x2="6" y1="1" y2="11" stroke="var(--text-2)" strokeWidth="1.5" />
            <line x1="3" x2="9" y1="1" y2="1" stroke="var(--text-2)" strokeWidth="1.5" />
            <line x1="3" x2="9" y1="11" y2="11" stroke="var(--text-2)" strokeWidth="1.5" />
          </svg>
          Seed range
        </li>
      )}
      {hasTeacher && (
        <li className="flex items-center gap-1.5">
          <svg width="12" height="12" aria-hidden>
            <path d="M6,1 L11,6 L6,11 L1,6 Z" fill="none" stroke="var(--text-2)" strokeWidth="1.5" />
          </svg>
          Teacher (non-causal)
        </li>
      )}
      {hasRef && (
        <li className="flex items-center gap-1.5">
          <svg width="12" height="12" aria-hidden>
            <rect x="1.5" y="1.5" width="9" height="9" rx="2" fill="none" stroke="var(--text-2)" strokeWidth="1.5" />
          </svg>
          Reference system
        </li>
      )}
    </ul>
  );
}
