/** Inline-SVG explainers for How it works: knowledge distillation and the attenuation limit (theme-aware via CSS vars). */
import { useId, useMemo, useState, type ReactNode } from 'react';
import { Slider } from './ui';

function Pill({ x, y, w, h, children, stroke = 'var(--border-strong)', fill = 'var(--surface)' }: { x: number; y: number; w: number; h: number; children: ReactNode; stroke?: string; fill?: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={h / 2} fill={fill} stroke={stroke} strokeWidth={1.3} />
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize={11.5} fill="var(--text)" fontFamily="var(--font-mono)">
        {children}
      </text>
    </g>
  );
}

export function DistillationDiagram() {
  const id = useId().replace(/:/g, '');
  const arrow = `dist-arrow-${id}`;
  const arrowKd = `dist-arrow-kd-${id}`;
  return (
    <svg viewBox="0 0 960 330" className="w-full min-w-[760px]" role="img" aria-labelledby={`${id}-t ${id}-d`}>
      <title id={`${id}-t`}>Knowledge distillation from a non-causal teacher into causal students</title>
      <desc id={`${id}-d`}>
        The noisy spectrum feeds both a frozen non-causal teacher (bidirectional GRU, hidden size 512, training only) and a causal student (GRU with hidden size 64, 128 or 256). The student is trained on the target loss against the clean speech plus beta times a spectral loss against the teacher output with stop-gradient, and optionally gamma times a feature loss between the projected student and teacher hidden states. Only the student is exported.
      </desc>
      <defs>
        <marker id={arrow} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--text-3)" />
        </marker>
        <marker id={arrowKd} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--s2)" />
        </marker>
      </defs>

      {/* input */}
      <rect x={16} y={133} width={120} height={64} rx={10} fill="var(--surface-2)" stroke="var(--noisy)" />
      <text x={76} y={161} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--text)">
        Noisy X
      </text>
      <text x={76} y={178} textAnchor="middle" fontSize={10.5} fill="var(--text-3)">
        STFT frames
      </text>
      <path d="M136,155 C170,155 165,70 196,70" fill="none" stroke="var(--text-3)" strokeWidth={1.4} markerEnd={`url(#${arrow})`} />
      <path d="M136,175 C170,175 165,260 196,260" fill="none" stroke="var(--text-3)" strokeWidth={1.4} markerEnd={`url(#${arrow})`} />

      {/* teacher */}
      <rect x={200} y={30} width={250} height={80} rx={12} fill="var(--surface-2)" stroke="var(--border-strong)" strokeDasharray="5 4" />
      <text x={325} y={58} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--text)">
        Teacher · frozen
      </text>
      <text x={325} y={76} textAnchor="middle" fontSize={10.5} fill="var(--text-2)">
        bi-GRU 512 · sees past and future frames
      </text>
      <text x={325} y={93} textAnchor="middle" fontSize={10} fill="var(--text-3)">
        training only · never exported
      </text>

      {/* students (three sizes, stacked) */}
      <rect x={212} y={232} width={250} height={80} rx={12} fill="var(--surface-2)" stroke="var(--border)" />
      <rect x={206} y={226} width={250} height={80} rx={12} fill="var(--surface-2)" stroke="var(--border)" />
      <rect x={200} y={220} width={250} height={80} rx={12} fill="var(--surface)" />
      <rect x={200} y={220} width={250} height={80} rx={12} fill="var(--accent-soft)" stroke="var(--accent)" />
      <text x={325} y={248} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--text)">
        Causal students
      </text>
      <text x={325} y={266} textAnchor="middle" fontSize={10.5} fill="var(--text-2)">
        GRU 64 / 128 / 256 · past frames only
      </text>
      <text x={325} y={283} textAnchor="middle" fontSize={10} fill="var(--text-3)">
        what ships: ONNX, one frame per call
      </text>

      {/* optional feature distillation */}
      <line x1={400} y1={110} x2={400} y2={218} stroke="var(--s2)" strokeWidth={1.3} strokeDasharray="2 4" />
      <text x={408} y={160} fontSize={10.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        γ·L_feat
      </text>
      <text x={408} y={174} fontSize={10} fill="var(--text-3)">
        optional
      </text>

      {/* outputs */}
      <line x1={450} y1={70} x2={516} y2={70} stroke="var(--text-3)" strokeWidth={1.4} markerEnd={`url(#${arrow})`} />
      <line x1={450} y1={260} x2={516} y2={260} stroke="var(--text-3)" strokeWidth={1.4} markerEnd={`url(#${arrow})`} />
      <rect x={520} y={50} width={100} height={40} rx={8} fill="var(--surface-2)" stroke="var(--border-strong)" />
      <text x={570} y={75} textAnchor="middle" fontSize={13} fill="var(--text)" fontFamily="var(--font-mono)">
        Y_t
      </text>
      <rect x={520} y={240} width={100} height={40} rx={8} fill="var(--surface-2)" stroke="var(--accent)" />
      <text x={570} y={265} textAnchor="middle" fontSize={13} fill="var(--text)" fontFamily="var(--font-mono)">
        Y_s
      </text>

      {/* KD loss */}
      <line x1={570} y1={90} x2={570} y2={141} stroke="var(--s2)" strokeWidth={1.6} markerEnd={`url(#${arrowKd})`} />
      <text x={578} y={118} fontSize={10} fill="var(--text-3)">
        stop-grad
      </text>
      <Pill x={480} y={145} w={180} h={30} stroke="var(--s2)">
        β·L_spec(Y_s, sg(Y_t))
      </Pill>
      <line x1={570} y1={238} x2={570} y2={177} stroke="var(--s2)" strokeWidth={1.6} markerEnd={`url(#${arrowKd})`} />

      {/* target loss */}
      <rect x={820} y={240} width={124} height={40} rx={8} fill="var(--surface-2)" stroke="var(--clean)" />
      <text x={882} y={265} textAnchor="middle" fontSize={12.5} fill="var(--text)">
        Clean S
      </text>
      <line x1={622} y1={260} x2={662} y2={260} stroke="var(--text-3)" strokeWidth={1.4} markerEnd={`url(#${arrow})`} />
      <Pill x={666} y={245} w={128} h={30}>
        L_target(Y_s, S)
      </Pill>
      <line x1={818} y1={260} x2={798} y2={260} stroke="var(--text-3)" strokeWidth={1.4} markerEnd={`url(#${arrow})`} />

      {/* total */}
      <rect x={690} y={22} width={254} height={150} rx={12} fill="var(--surface-2)" fillOpacity={0.6} stroke="var(--border)" />
      <text x={706} y={46} fontSize={12} fontWeight={600} fill="var(--text)">
        Student loss
      </text>
      <text x={706} y={68} fontSize={11.5} fill="var(--text)" fontFamily="var(--font-mono)">
        L = L_target + β·L_spec
      </text>
      <text x={706} y={86} fontSize={11.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        {'    [+ γ·L_feat]'}
      </text>
      <text x={706} y={108} fontSize={10.5} fill="var(--text-2)" fontFamily="var(--font-mono)">
        L_feat = MSE(h_s W, sg(h_t))
      </text>
      <text x={706} y={142} fontSize={10.5} fill="var(--text-3)">
        β ∈ {'{'}0.5, 1.0{'}'} · gradients reach the
      </text>
      <text x={706} y={156} fontSize={10.5} fill="var(--text-3)">
        student only (teacher in no_grad)
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------------------------

/** Synthetic mask gain (dB) across 161 bins: a smooth speech envelope with harmonic ripple, deep cuts elsewhere. */
function syntheticMaskDb(): number[] {
  const out: number[] = [];
  for (let k = 0; k < 161; k++) {
    const f = (k / 160) * 8000;
    const env = Math.exp(-(((f - 600) / 500) ** 2)) + 0.8 * Math.exp(-(((f - 1900) / 650) ** 2)) + 0.45 * Math.exp(-(((f - 3200) / 600) ** 2));
    const ripple = 0.55 + 0.45 * Math.cos((Math.PI * f) / 200) ** 2;
    const presence = Math.min(1, env * ripple * 1.1);
    out.push(-38 * (1 - presence) + 2.5 * Math.sin(k * 2.3));
  }
  return out.map((v) => Math.max(-42, Math.min(0, v)));
}

export function AttenLimitPlot() {
  const [lim, setLim] = useState(12);
  const id = useId().replace(/:/g, '');
  const mask = useMemo(syntheticMaskDb, []);
  const W = 520;
  const H = 210;
  const L = 40;
  const R = 12;
  const T = 12;
  const B = 30;
  const sx = (k: number) => L + (k / 160) * (W - L - R);
  const sy = (db: number) => T + (-db / 45) * (H - T - B);
  const path = (vals: number[]) => vals.map((v, k) => `${k ? 'L' : 'M'}${sx(k).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
  const clamped = mask.map((v) => Math.max(v, -lim));
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-labelledby={`${id}-t`}>
        <title id={`${id}-t`}>{`Mask gain across frequency for one frame, with the attenuation floor at minus ${lim} dB`}</title>
        {[0, -10, -20, -30, -40].map((d) => (
          <g key={d}>
            <line x1={L} x2={W - R} y1={sy(d)} y2={sy(d)} stroke="var(--grid)" />
            <text x={L - 6} y={sy(d) + 3.5} textAnchor="end" fontSize={10} fill="var(--text-3)" className="tnum">
              {d}
            </text>
          </g>
        ))}
        {[0, 2, 4, 6, 8].map((k) => (
          <text key={k} x={sx((k / 8) * 160)} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--text-3)">
            {k} kHz
          </text>
        ))}
        <text x={12} y={T + (H - T - B) / 2} textAnchor="middle" fontSize={10.5} fill="var(--text-2)" transform={`rotate(-90 12 ${T + (H - T - B) / 2})`}>
          mask gain (dB)
        </text>
        <path d={path(mask)} fill="none" stroke="var(--text-3)" strokeWidth={1.2} strokeDasharray="3 3" />
        <path d={path(clamped)} fill="none" stroke="var(--enhanced)" strokeWidth={2} strokeLinejoin="round" />
        <line x1={L} x2={W - R} y1={sy(-lim)} y2={sy(-lim)} stroke="var(--warn)" strokeWidth={1.4} strokeDasharray="6 4" />
        <text x={W - R - 4} y={sy(-lim) + 15} textAnchor="end" fontSize={10.5} fill="var(--text)" stroke="var(--surface)" strokeWidth={4} paintOrder="stroke" strokeLinejoin="round">
          floor −{lim} dB: |M| ≥ {Math.pow(10, -lim / 20).toFixed(2)}
        </text>
      </svg>
      <figcaption className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-fg-3">
        <span className="flex items-center gap-1.5">
          <svg width="18" height="6" aria-hidden>
            <line x1="0" x2="18" y1="3" y2="3" stroke="var(--text-3)" strokeDasharray="3 3" />
          </svg>
          network’s mask
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="18" height="6" aria-hidden>
            <line x1="0" x2="18" y1="3" y2="3" stroke="var(--enhanced)" strokeWidth="2" />
          </svg>
          after the floor
        </span>
        <span>Illustrative frame, not model output.</span>
      </figcaption>
      <Slider className="mt-3 max-w-xs" label="Attenuation limit" min={3} max={40} step={1} value={lim} onChange={setLim} format={(v) => `${v} dB`} />
    </figure>
  );
}
