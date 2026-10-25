import { useMemo, useState, type ReactNode } from 'react';
import { engine, type BenchmarkResult } from '../engine/client';
import { algorithmicLatencyMs, attenSpec, lookaheadFrames, manifestDeclaresAtten, shapeSize, type Manifest } from '../engine/model';
import { useFailures, useManifestFixture, useResults, type MetricMap, type Results as ResultsData, type Row, type Significance, type TableRow, type TestSet } from '../lib/data';
import { USE_FIXTURES } from '../lib/fixtures';
import { useEngineStatus } from '../lib/useEngine';
import { GroupedColumns, HBarChart, Legend } from '../components/Charts';
import { ParetoChart, ParetoLegend, type ParetoInput, type ParetoRef } from '../components/Pareto';
import { TradeoffChart, type TradeoffSeries } from '../components/Tradeoff';
import { DataTable, type Column } from '../components/DataTable';
import { FailureGallery } from '../components/FailureGallery';
import { Badge, Button, Callout, Card, SectionTitle, Segmented } from '../components/ui';
import { compact, cx, displayModelName, fmt, formatAtten, SERIES } from '../lib/format';

// ---- metric vocabulary ---------------------------------------------------------------------------

interface MetricDef {
  label: string;
  digits: number;
  unit?: string;
  domain?: [number, number];
}

const METRICS: Record<string, MetricDef> = {
  pesq: { label: 'PESQ-WB', digits: 2, domain: [1, 4.5] },
  stoi: { label: 'STOI', digits: 3 },
  estoi: { label: 'ESTOI', digits: 3 },
  si_sdr: { label: 'SI-SDR', digits: 1, unit: 'dB' },
  dnsmos_sig: { label: 'DNSMOS SIG', digits: 2, domain: [1, 5] },
  dnsmos_bak: { label: 'DNSMOS BAK', digits: 2, domain: [1, 5] },
  dnsmos_ovrl: { label: 'DNSMOS OVRL', digits: 2, domain: [1, 5] },
  speech_distortion_sisdr: { label: 'Speech SI-SDR', digits: 1, unit: 'dB' },
  noise_reduction_db: { label: 'Noise reduction', digits: 1, unit: 'dB' },
  pesq_ood: { label: 'PESQ (OOD)', digits: 2 },
};
const SUMMARY_ORDER = ['pesq', 'stoi', 'estoi', 'si_sdr', 'dnsmos_sig', 'dnsmos_bak', 'dnsmos_ovrl', 'speech_distortion_sisdr', 'noise_reduction_db'];
const BAR_METRICS = ['pesq', 'stoi', 'si_sdr', 'dnsmos_ovrl'];

const WORDS: Record<string, string> = { ms: 'ms', kb: 'kB', db: 'dB', ort: 'ORT', wasm: 'WASM', m4: 'M4', fp32: 'fp32', int8: 'int8', torch: 'PyTorch', rpi4: 'Raspberry Pi 4', rpi5: 'Raspberry Pi 5', mmacs: 'MMAC', ood: 'OOD', cpu: 'CPU' };
function prettify(key: string): string {
  const s = key
    .split('_')
    .map((w) => WORDS[w.toLowerCase()] ?? w)
    .join(' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}
const metricLabel = (k: string) => METRICS[k]?.label ?? prettify(k);
const metricDigits = (k: string) => METRICS[k]?.digits ?? 2;

type Kind = 'ours' | 'teacher' | 'reference' | 'baseline';
function kindOf(r: Pick<Row, 'model' | 'family'>): Kind {
  const f = r.family?.toLowerCase();
  if (f === 'teacher' || /teacher/i.test(r.model)) return 'teacher';
  if (f === 'reference' || /rnnoise|deepfilter/i.test(r.model)) return 'reference';
  if (f === 'ours' || (!f && /crn|ours/i.test(r.model) && !/noisy/i.test(r.model))) return 'ours';
  return 'baseline';
}

function setBadge(s: TestSet): { text: string; tone: 'accent' | 'warn' | 'neutral' } {
  if (s.noiseSeen === true) return { text: 'seen noise', tone: 'warn' };
  if (s.noiseSeen === false) return { text: 'unseen noise', tone: 'accent' };
  return { text: s.intrusive ? 'noise not stated' : 'real-world, DNSMOS only', tone: 'neutral' };
}

/** Stable per-entity colours (first appearance order), capped at the 8 validated slots. */
function colorMap(names: string[]): (n: string) => string {
  const m = new Map<string, string>();
  for (const n of names) if (!m.has(n) && m.size < SERIES.length) m.set(n, SERIES[m.size]);
  return (n) => m.get(n) ?? 'var(--text-3)';
}

// ---- page ------------------------------------------------------------------------------------------

export default function Results() {
  const results = useResults();
  const failures = useFailures();
  const fixtureManifest = useManifestFixture();
  const { status, manifest } = useEngineStatus();
  const cardManifest = USE_FIXTURES && fixtureManifest.state === 'ready' ? fixtureManifest.data : manifest;

  return (
    <div className="space-y-8">
      <header className="max-w-2xl">
        <div className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">Results</div>
        <h1 className="text-[28px] font-semibold tracking-tight sm:text-[34px]">Measured, not claimed</h1>
        <p className="mt-2 text-[15px] text-fg-2">
          Objective metrics on VoiceBank+DEMAND and, where available, on out-of-domain noise, a competing talker and real recordings. Every claim that one model beats another is backed by a paired Wilcoxon test.
        </p>
      </header>

      {results.state === 'loading' && <div className="h-40 animate-pulse rounded-2xl bg-surface-2" />}
      {results.state === 'missing' && (
        <Card className="p-6">
          <Badge tone="warn" className="mb-2">
            Pending
          </Badge>
          <h2 className="text-[17px] font-semibold tracking-tight">Evaluation results are pending</h2>
          <p className="mt-1 max-w-prose text-sm text-fg-2">
            The tables and charts fill in automatically once <code className="font-mono text-[13px]">results.json</code> is produced by <code className="font-mono text-[13px]">ans.evaluate</code>. Reference point for the noisy input from the literature: PESQ 1.97, STOI 0.921.
          </p>
        </Card>
      )}
      {results.state === 'ready' && <ResultsBody data={results.data} />}

      {failures.state === 'ready' && failures.data.length > 0 && <FailureGallery cases={failures.data} />}

      <ModelCard manifest={cardManifest} loaded={status.kind === 'onnx'} />
    </div>
  );
}

function ResultsBody({ data }: { data: ResultsData }) {
  const [setKey, setSetKey] = useState(data.sets[0]?.key ?? 'vbd');
  const set = data.sets.find((s) => s.key === setKey) ?? data.sets[0];
  const rows = useMemo(() => data.rows.filter((r) => r.set === set.key), [data.rows, set.key]);
  const sig = useMemo(() => data.significance.filter((s) => s.set === null || s.set === set.key), [data.significance, set.key]);
  // One colour per model across every multi-series chart of this set (colour follows the entity).
  const colorOf = useMemo(() => {
    const names: string[] = [];
    for (const t of [data.perSnr[set.key], data.perNoiseType[set.key]]) for (const cat of Object.values(t ?? {})) names.push(...Object.keys(cat));
    names.push(...rows.filter((r) => typeof r.values.noise_reduction_db === 'number').map((r) => r.model));
    return colorMap(names);
  }, [data.perSnr, data.perNoiseType, rows, set.key]);
  const bySet = <T extends Record<string, unknown>>(entries: T[]) => {
    const keyed = entries.some((e) => typeof e.set === 'string');
    return keyed ? entries.filter((e) => e.set === set.key || e.set == null) : entries;
  };

  return (
    <>
      {data.pareto.length > 0 && <ParetoSection data={data} />}

      <div className="space-y-4">
        {data.sets.length > 1 ? (
          <SetSelector sets={data.sets} value={set.key} onChange={setSetKey} />
        ) : (
          <p className="flex flex-wrap items-center gap-2 text-[13px] text-fg-2">
            Test set: <strong className="font-medium text-fg">{set.label}</strong>
            <Badge tone={setBadge(set).tone}>{setBadge(set).text}</Badge>
          </p>
        )}
        <SummaryTable rows={rows} set={set} sig={sig} generatedAt={data.generatedAt} gitHash={data.gitHash} />
      </div>

      <MetricBars rows={rows} />
      <TradeoffSection key={`t-${set.key}`} table={data.perSnr[set.key]} rows={rows} colorOf={colorOf} />
      {data.perNoiseType[set.key] && (
        <Breakdown
          key={`n-${set.key}`}
          title="By noise type"
          note={set.noiseSeen === false ? 'None of these noise types occur in training, so every column is an unseen condition.' : undefined}
          table={data.perNoiseType[set.key]}
          colorOf={colorOf}
        />
      )}
      {data.perSnr[set.key] && <Breakdown key={`s-${set.key}`} title="By input SNR" unit="dB" table={data.perSnr[set.key]} colorOf={colorOf} />}

      <div className="grid gap-4 xl:grid-cols-2">
        {data.attenSweep.length > 0 && <AttenSweep rows={bySet(data.attenSweep)} />}
        {data.lookahead.length > 0 && <Lookahead rows={bySet(data.lookahead)} />}
      </div>
      {data.quantization.length > 0 && <Quantization rows={bySet(data.quantization)} />}
      {data.significance.length > 0 && <SignificanceTable entries={sig} sets={data.sets} setLabel={set.label} />}
    </>
  );
}

// ---- test-set selector ---------------------------------------------------------------------------

function SetSelector({ sets, value, onChange }: { sets: TestSet[]; value: string; onChange: (k: string) => void }) {
  return (
    <div>
      <div className="mb-2 text-[12px] font-medium text-fg-2" id="set-label">
        Test set
      </div>
      <div role="radiogroup" aria-labelledby="set-label" className="flex flex-wrap gap-2" data-testid="set-selector">
        {sets.map((s) => {
          const active = s.key === value;
          const b = setBadge(s);
          return (
            <button
              key={s.key}
              type="button"
              role="radio"
              aria-checked={active}
              tabIndex={active ? 0 : -1}
              onClick={() => onChange(s.key)}
              onKeyDown={(e) => {
                if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
                e.preventDefault();
                const i = sets.findIndex((x) => x.key === value);
                const next = sets[(i + (e.key === 'ArrowRight' ? 1 : sets.length - 1)) % sets.length];
                onChange(next.key);
                (e.currentTarget.parentElement?.querySelector(`[data-set="${next.key}"]`) as HTMLElement | null)?.focus();
              }}
              data-set={s.key}
              className={cx(
                'flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-[13px] transition-colors',
                active ? 'border-accent bg-accent-soft text-fg' : 'border-line bg-surface text-fg-2 hover:border-line-strong hover:text-fg',
              )}
            >
              <span className="font-medium">{s.label}</span>
              <Badge tone={b.tone}>{b.text}</Badge>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---- Pareto hero -----------------------------------------------------------------------------------

function sizeOf(name: string): number | null {
  const m = /(\d+)/.exec(name);
  return m ? Number(m[1]) : null;
}

function ParetoSection({ data }: { data: ResultsData }) {
  const metrics = ['pesq', 'dnsmos_ovrl'].filter((k) => data.pareto.some((p) => typeof p.values[k] === 'number'));
  const [metric, setMetric] = useState(metrics[0] ?? 'pesq');
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const paretoSets = Array.from(new Set(data.pareto.map((p) => p.set).filter((s): s is string => s !== null)));
  // Out-of-domain noise separates model sizes better than VoiceBank+DEMAND, so it is the default view.
  const defaultSet = ['ood_esc', 'vbd'].find((k) => paretoSets.includes(k)) ?? paretoSets[0] ?? 'vbd';
  const [pSet, setPSet] = useState(defaultSet);
  // §3.3 pareto entries carry no set; they are assumed to be VoiceBank+DEMAND test numbers.
  const setKey = paretoSets.length ? pSet : 'vbd';
  const setLabel = data.sets.find((s) => s.key === setKey)?.label ?? setKey;
  const def = METRICS[metric];

  const points: ParetoInput[] = data.pareto
    .filter((p) => (paretoSets.length ? p.set === setKey : true) && typeof p.values[metric] === 'number')
    .map((p) => ({ model: p.model, hidden: p.hidden, distilled: p.distilled, seed: p.seed, macs: p.macs, y: p.values[metric] }));
  const refRows = data.rows.filter((r) => r.set === setKey && (kindOf(r) === 'teacher' || kindOf(r) === 'reference') && typeof r.values[metric] === 'number');
  const refs: ParetoRef[] = refRows
    .filter((r) => r.macs !== null && r.macs > 0)
    .map((r) => ({ label: displayModelName(r.model), kind: kindOf(r) === 'teacher' ? 'teacher' : 'reference', macs: r.macs!, y: r.values[metric] }));
  // Several runs at the same size and arm = seeds.
  const hasSeeds = new Set(points.map((p) => `${p.hidden ?? p.macs}-${p.distilled}`)).size < points.length;
  const unplotted = refRows.filter((r) => !(r.macs !== null && r.macs > 0)).map((r) => displayModelName(r.model));

  // KD vs plain at each student size, for the chosen metric and set.
  const sizes = Array.from(new Set(points.map((p) => p.hidden).filter((h): h is number => h !== null))).sort((a, b) => a - b);
  const kdTests = sizes
    .map((h) => ({
      h,
      s: data.significance.find((s) => s.metric === metric && (s.set === null || s.set === setKey) && /kd|distil/i.test(s.a) && !/kd|distil/i.test(s.b) && sizeOf(s.a) === h && sizeOf(s.b) === h),
    }))
    .filter((x): x is { h: number; s: Significance } => !!x.s);

  return (
    <Card className="p-5 sm:p-6" aria-labelledby="pareto-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <SectionTitle id="pareto-title" eyebrow="Headline" title="Quality vs. compute: does distillation help small models?">
          Causal students at three sizes, trained with and without a non-causal teacher under identical settings. Up and to the left is better. {setLabel}.
        </SectionTitle>
        <div className="flex flex-wrap gap-2">
          {paretoSets.length > 1 && (
            <Segmented size="sm" label="Pareto test set" value={pSet} onChange={setPSet} options={paretoSets.map((k) => ({ value: k, label: data.sets.find((s) => s.key === k)?.label.replace(/\s*\(.*\)$/, '') ?? k }))} />
          )}
          {metrics.length > 1 && <Segmented size="sm" label="Quality metric" value={metric} onChange={setMetric} options={metrics.map((k) => ({ value: k, label: metricLabel(k) }))} />}
          <Segmented<'chart' | 'table'> size="sm" label="View" value={view} onChange={setView} options={[{ value: 'chart', label: 'Chart' }, { value: 'table', label: 'Table' }]} />
        </div>
      </div>
      {points.length === 0 ? (
        <p className="text-sm text-fg-3">No {metricLabel(metric)} values in the Pareto runs yet.</p>
      ) : view === 'chart' ? (
        <>
          <div className="mb-3">
            <ParetoLegend hasTeacher={refs.some((r) => r.kind === 'teacher')} hasRef={refs.some((r) => r.kind === 'reference')} hasSeeds={hasSeeds} />
          </div>
          <div data-testid="pareto-chart">
            <ParetoChart points={points} refs={refs} metricLabel={metricLabel(metric)} digits={def?.digits ?? 2} />
          </div>
        </>
      ) : (
        <DataTable
          caption="Pareto runs"
          rows={points.map((p) => ({ model: displayModelName(p.model), hidden: p.hidden, kd: p.distilled, seed: p.seed, macs: p.macs, y: p.y }))}
          columns={[
            { key: 'model', label: 'Run', render: (r) => r.model },
            { key: 'hidden', label: 'Hidden', render: (r) => r.hidden ?? '—' },
            { key: 'kd', label: 'KD', render: (r) => (r.kd ? 'yes' : 'no') },
            { key: 'seed', label: 'Seed', render: (r) => r.seed ?? '—' },
            { key: 'macs', label: 'MACs / frame', render: (r) => compact(r.macs as number) },
            { key: 'y', label: metricLabel(metric), render: (r) => fmt(r.y as number, def?.digits ?? 2) },
          ]}
        />
      )}
      {(kdTests.length > 0 || unplotted.length > 0) && (
        <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-line pt-4 text-[12px] text-fg-2" data-testid="pareto-significance">
          {kdTests.length > 0 && <span className="text-fg-3">KD vs. plain ({metricLabel(metric)}, paired Wilcoxon):</span>}
          {kdTests.map(({ h, s }) => (
            <SigBadge key={h} s={s} label={`${h}: ${s.medianDiff !== null ? `${s.medianDiff >= 0 ? '+' : ''}${fmt(s.medianDiff, metricDigits(metric))}` : ''}`} />
          ))}
          {unplotted.length > 0 && <span className="text-fg-3">Not plotted (no MAC count): {unplotted.join(', ')}.</span>}
        </div>
      )}
    </Card>
  );
}

function pText(p: number | null): string {
  if (p === null) return 'p n/a';
  if (p < 0.001) return 'p < 0.001';
  return `p = ${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}`;
}

function SigBadge({ s, label }: { s: Significance; label?: string }) {
  return (
    <span title={`${s.a} vs ${s.b} on ${metricLabel(s.metric)}: median difference ${fmt(s.medianDiff, 3)}, ${pText(s.p)}${s.p !== s.pRaw ? ' (Holm-corrected)' : ''}`}>
      <Badge tone={s.significant ? 'accent' : 'neutral'}>
        {label && <span className="tnum">{label}</span>}
        {s.significant ? `✓ ${pText(s.p)}` : `n.s. (${pText(s.p)})`}
      </Badge>
    </span>
  );
}

// ---- summary --------------------------------------------------------------------------------------

function SummaryTable({ rows, set, sig, generatedAt, gitHash }: { rows: Row[]; set: TestSet; sig: Significance[]; generatedAt: string | null; gitHash: string | null }) {
  const metricKeys = SUMMARY_ORDER.filter((k) => rows.some((r) => typeof r.values[k] === 'number'));
  const best = (k: string) => {
    const vals = rows.map((r) => r.values[k]).filter((v): v is number => typeof v === 'number');
    return vals.length > 1 ? Math.max(...vals) : null;
  };
  const hasMacs = rows.some((r) => r.macs !== null);
  return (
    <Card className="overflow-hidden" aria-labelledby="table-title">
      <div className="px-5 pt-5">
        <SectionTitle id="table-title" title="Summary">
          {set.label}
          {generatedAt && <span className="text-fg-3"> · generated {new Date(generatedAt).toLocaleDateString()}</span>}
          {gitHash && <span className="text-fg-3"> · {gitHash.slice(0, 8)}</span>}
        </SectionTitle>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[13px]" style={{ minWidth: 420 + metricKeys.length * 92 }} data-testid="summary-table">
          <thead>
            <tr className="border-y border-line bg-surface-2/60 text-left text-[11px] tracking-wide text-fg-3 uppercase">
              <th scope="col" className="px-5 py-2 font-medium">Model</th>
              <th scope="col" className="px-3 py-2 text-center font-medium">Causal</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Params</th>
              {hasMacs && <th scope="col" className="px-3 py-2 text-right font-medium">MACs</th>}
              {metricKeys.map((k) => (
                <th key={k} scope="col" className="px-3 py-2 text-right font-medium whitespace-nowrap">
                  {metricLabel(k)} ↑
                </th>
              ))}
              <th scope="col" className="px-3 py-2 text-right font-medium">ms / frame</th>
              <th scope="col" className="px-5 py-2 text-right font-medium">RTF</th>
            </tr>
          </thead>
          <tbody className="tnum">
            {rows.map((r) => {
              const kind = kindOf(r);
              const wins = sig.filter((s) => s.a === r.model && s.significant && (s.medianDiff ?? 0) > 0);
              return (
                <tr key={r.model} className={cx('border-b border-line last:border-0', kind === 'ours' && 'bg-accent-soft')}>
                  <th scope="row" className="min-w-[230px] px-5 py-2.5 text-left font-medium">
                    <span className="flex flex-wrap items-center gap-1.5 [&_span]:whitespace-nowrap">
                      <span className={kind === 'ours' ? 'text-fg' : 'text-fg-2'}>{displayModelName(r.model)}</span>
                      {kind === 'teacher' && <Badge>teacher</Badge>}
                      {kind === 'reference' && <Badge>reference</Badge>}
                      {wins.length > 0 && (
                        <span title={wins.map((s) => `Better than ${s.b} on ${metricLabel(s.metric)} (median +${fmt(s.medianDiff, 3)}, ${pText(s.p)})`).join('\n')}>
                          <Badge tone="accent">p &lt; 0.05 vs {wins.length === 1 ? displayModelName(wins[0].b).replace(/\s*\(.*\)$/, '') : `${wins.length} models`}</Badge>
                        </span>
                      )}
                    </span>
                  </th>
                  <td className="px-3 py-2.5 text-center text-fg-2">{r.causal === true ? '✓' : r.causal === false ? '✗' : '—'}</td>
                  <td className="px-3 py-2.5 text-right text-fg-2">{compact(r.params)}</td>
                  {hasMacs && <td className="px-3 py-2.5 text-right text-fg-2">{compact(r.macs)}</td>}
                  {metricKeys.map((k) => {
                    const v = r.values[k];
                    const isBest = typeof v === 'number' && v === best(k);
                    const ci = r.ci[k];
                    return (
                      <td key={k} className={cx('px-3 py-2.5 text-right whitespace-nowrap', isBest ? 'font-semibold text-fg' : 'text-fg-2')}>
                        {fmt(v, metricDigits(k))}
                        {typeof ci === 'number' && <span className="ml-1 text-[11px] font-normal text-fg-3">±{fmt(ci, metricDigits(k))}</span>}
                      </td>
                    );
                  })}
                  <td className="px-3 py-2.5 text-right text-fg-2">{fmt(r.msPerFrame, 3)}</td>
                  <td className="px-5 py-2.5 text-right text-fg-2">{fmt(r.rtf, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="px-5 py-3 text-[11px] text-fg-3">
        Best value per column in bold; ± is the 95 % confidence interval of the mean. Speech SI-SDR and noise reduction come from applying each model’s mask separately to the clean speech and the noise, so they exist only for mask-based models. ms/frame and RTF are single-thread CPU numbers from the Python benchmark; the Studio and Live pages measure your own device.
      </p>
    </Card>
  );
}

function MetricBars({ rows }: { rows: Row[] }) {
  const keys = BAR_METRICS.filter((k) => rows.filter((r) => typeof r.values[k] === 'number').length > 1);
  if (!keys.length) return null;
  return (
    <section aria-labelledby="charts-title">
      <h2 id="charts-title" className="sr-only">
        Metric charts
      </h2>
      <div className="grid gap-4 md:grid-cols-2">
        {keys.map((k) => {
          const d = METRICS[k];
          return (
            <Card key={k} className="min-w-0 p-5">
              <div className="mb-2 text-[13px] font-medium text-fg">
                {d.label} {d.unit && <span className="text-fg-3">({d.unit})</span>} <span className="text-fg-3">· higher is better</span>
              </div>
              <HBarChart
                title={`${d.label} by model`}
                digits={d.digits}
                domain={d.domain}
                data={rows
                  .filter((r) => typeof r.values[k] === 'number')
                  .map((r) => ({ label: displayModelName(r.model), value: r.values[k], color: kindOf(r) === 'ours' ? 'var(--s1)' : 'var(--text-3)', emphasis: kindOf(r) === 'ours' }))}
              />
            </Card>
          );
        })}
      </div>
    </section>
  );
}

// ---- speech distortion vs noise reduction ------------------------------------------------------

function TradeoffSection({ table, rows, colorOf }: { table: Record<string, Record<string, MetricMap>> | undefined; rows: Row[]; colorOf: (model: string) => string }) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const bySnr = useMemo(() => {
    if (!table) return [];
    const cats = Object.keys(table).sort((a, b) => Number(a) - Number(b));
    const models = Array.from(new Set(cats.flatMap((c) => Object.keys(table[c]))));
    return models
      .map((m) => ({
        key: m,
        name: displayModelName(m),
        points: cats
          .map((c) => ({ x: table[c][m]?.noise_reduction_db, y: table[c][m]?.speech_distortion_sisdr, label: `${c} dB` }))
          .filter((p): p is { x: number; y: number; label: string } => typeof p.x === 'number' && typeof p.y === 'number'),
      }))
      .filter((s) => s.points.length > 0);
  }, [table]);
  const overall = rows
    .filter((r) => typeof r.values.noise_reduction_db === 'number' && typeof r.values.speech_distortion_sisdr === 'number')
    .map((r) => ({ key: r.model, name: displayModelName(r.model), points: [{ x: r.values.noise_reduction_db, y: r.values.speech_distortion_sisdr, label: '' }] }));
  const raw = bySnr.length ? bySnr : overall;
  if (!raw.length) return null;
  const series: TradeoffSeries[] = raw.filter((s) => colorOf(s.key) !== 'var(--text-3)').map((s) => ({ name: s.name, points: s.points, color: colorOf(s.key) }));
  return (
    <Card className="p-5" aria-labelledby="tradeoff-title">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <SectionTitle id="tradeoff-title" title="Speech distortion vs. noise reduction">
          The predicted mask applied separately to the clean speech and to the noise. Further right removes more noise; higher keeps the voice intact. {bySnr.length ? 'Each line runs from the lowest to the highest input SNR.' : 'Overall numbers per model.'}
        </SectionTitle>
        <Segmented<'chart' | 'table'> size="sm" label="View" value={view} onChange={setView} options={[{ value: 'chart', label: 'Chart' }, { value: 'table', label: 'Table' }]} />
      </div>
      {view === 'chart' ? (
        <>
          <div className="mb-3">
            <Legend series={series} />
          </div>
          <div data-testid="tradeoff-chart">
            <TradeoffChart series={series} pointLabel={bySnr.length ? 'input SNR' : ''} />
          </div>
          {raw.length > series.length && <p className="mt-2 text-[11px] text-fg-3">{raw.length - series.length} more models in the table view.</p>}
        </>
      ) : (
        <DataTable
          caption="Speech distortion and noise reduction"
          rows={raw.flatMap((s) => s.points.map((p) => ({ model: s.name, snr: p.label || 'overall', nr: p.x, sd: p.y })))}
          columns={[
            { key: 'model', label: 'Model', render: (r) => r.model },
            { key: 'snr', label: 'Input SNR', render: (r) => r.snr },
            { key: 'nr', label: 'Noise reduction (dB)', render: (r) => fmt(r.nr as number, 1) },
            { key: 'sd', label: 'Speech SI-SDR (dB)', render: (r) => fmt(r.sd as number, 1) },
          ]}
        />
      )}
    </Card>
  );
}

// ---- breakdowns -----------------------------------------------------------------------------------

const NON_METRIC = new Set(['n', 'count', 'num', 'ci95']);

function Breakdown({ title, table, unit, note, colorOf }: { title: string; table: Record<string, Record<string, MetricMap>>; unit?: string; note?: string; colorOf: (model: string) => string }) {
  const keys = Object.keys(table);
  const numeric = keys.every((k) => Number.isFinite(Number(k)));
  const categories = numeric ? keys.slice().sort((a, b) => Number(a) - Number(b)) : keys;
  const catLabel = (c: string) => (unit ? `${c} ${unit}` : c);
  const models = Array.from(new Set(categories.flatMap((c) => Object.keys(table[c]))));
  const metricKeys = Array.from(new Set(categories.flatMap((c) => Object.values(table[c]).flatMap((m) => Object.keys(m ?? {}))))).filter(
    (k) => !NON_METRIC.has(k) && categories.some((c) => models.some((m) => typeof table[c][m]?.[k] === 'number')),
  );
  metricKeys.sort((a, b) => (SUMMARY_ORDER.indexOf(a) + 1 || 99) - (SUMMARY_ORDER.indexOf(b) + 1 || 99));
  const [metric, setMetric] = useState(metricKeys.includes('pesq') ? 'pesq' : (metricKeys[0] ?? 'pesq'));
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const digits = metricDigits(metric);
  const charted = models.filter((m) => colorOf(m) !== 'var(--text-3)');
  const series = charted.map((m) => ({ name: displayModelName(m), color: colorOf(m), key: m }));
  const value = (c: string, m: string) => {
    const v = table[c]?.[m]?.[metric];
    return typeof v === 'number' ? v : null;
  };
  return (
    <Card className="p-5" aria-label={title}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-[17px] font-semibold tracking-tight">{title}</h2>
          {note && <p className="mt-1 text-[13px] text-fg-2">{note}</p>}
        </div>
        <div className="flex flex-wrap gap-2">
          {metricKeys.length > 1 && <Segmented size="sm" label="Metric" value={metric} onChange={setMetric} options={metricKeys.map((k) => ({ value: k, label: metricLabel(k) }))} />}
          <Segmented<'chart' | 'table'> size="sm" label="View" value={view} onChange={setView} options={[{ value: 'chart', label: 'Chart' }, { value: 'table', label: 'Table' }]} />
        </div>
      </div>
      {view === 'chart' ? (
        <>
          <div className="mb-3">
            <Legend series={series} />
          </div>
          <GroupedColumns
            title={`${metricLabel(metric)} ${title.toLowerCase()}`}
            categories={categories}
            categoryLabel={catLabel}
            series={series}
            value={(c, name) => value(c, series.find((s) => s.name === name)?.key ?? name)}
            digits={digits}
          />
          {models.length > charted.length && <p className="mt-2 text-[11px] text-fg-3">{models.length - charted.length} more models in the table view.</p>}
        </>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-[11px] tracking-wide text-fg-3 uppercase">
                <th className="py-2 pr-3 font-medium">{title.replace(/^By /, '')}</th>
                {models.map((m) => (
                  <th key={m} className="px-3 py-2 text-right font-medium">
                    {displayModelName(m)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tnum">
              {categories.map((c) => (
                <tr key={c} className="border-b border-line last:border-0">
                  <th scope="row" className="py-2 pr-3 text-left font-medium text-fg-2">
                    {catLabel(c)}
                  </th>
                  {models.map((m) => (
                    <td key={m} className="px-3 py-2 text-right text-fg-2">
                      {fmt(value(c, m), digits)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

// ---- ablation tables -------------------------------------------------------------------------------

const isInf = (v: unknown) => v === null || v === 'inf' || v === 'Infinity' || v === '∞' || (typeof v === 'number' && (!Number.isFinite(v) || v >= 100));

function cell(key: string, v: TableRow[string]): ReactNode {
  if (key === 'atten_lim_db') return isInf(v) ? '∞' : `${fmt(v as number, 0)} dB`;
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'string') return v;
  if (key === 'size_kb') return `${Math.round(v).toLocaleString()} kB`;
  if (key === 'latency_ms') return `${fmt(v, 0)} ms`;
  if (key === 'lookahead_frames') return String(v);
  if (key.startsWith('ms_per_frame')) return fmt(v, 3);
  return fmt(v, metricDigits(key));
}

const COL_LABEL: Record<string, string> = {
  atten_lim_db: 'Max. attenuation',
  lookahead_frames: 'Lookahead (frames)',
  latency_ms: 'Algorithmic latency',
  variant: 'Variant',
  size_kb: 'File size',
  ms_per_frame: 'ms / frame',
  model: 'Model',
};

/** Columns: the preferred keys first (when present), then any other keys, never `set`. */
function autoColumns(rows: TableRow[], first: string[]): Column[] {
  const present = new Set(rows.flatMap((r) => Object.keys(r)));
  present.delete('set');
  const order = [...first.filter((k) => present.has(k)), ...[...present].filter((k) => !first.includes(k) && !k.startsWith('_'))];
  return order.map((key) => ({ key, label: COL_LABEL[key] ?? metricLabel(key), render: (r: TableRow) => cell(key, r[key]) }));
}

function TableCard({ id, eyebrow, title, children, table }: { id: string; eyebrow: string; title: string; children: ReactNode; table: ReactNode }) {
  return (
    <Card className="overflow-hidden" aria-labelledby={id}>
      <div className="px-5 pt-5">
        <SectionTitle id={id} eyebrow={eyebrow} title={title}>
          {children}
        </SectionTitle>
      </div>
      {table}
    </Card>
  );
}

function AttenSweep({ rows }: { rows: TableRow[] }) {
  const sorted = rows.slice().sort((a, b) => (isInf(a.atten_lim_db) ? 1e9 : Number(a.atten_lim_db)) - (isInf(b.atten_lim_db) ? 1e9 : Number(b.atten_lim_db)));
  return (
    <TableCard id="atten-title" eyebrow="Ablation" title="Attenuation limit sweep" table={<div data-testid="atten-sweep"><DataTable caption="Attenuation limit sweep" rows={sorted} columns={autoColumns(sorted, ['model', 'atten_lim_db', 'pesq', 'speech_distortion_sisdr', 'noise_reduction_db', 'dnsmos_bak'])} /></div>}>
      Flooring the mask trades noise reduction for a more natural voice, with no retraining. This is the Max. attenuation control in the Studio.
    </TableCard>
  );
}

function Lookahead({ rows }: { rows: TableRow[] }) {
  return (
    <TableCard id="lookahead-title" eyebrow="Ablation" title="Lookahead vs. quality" table={<div data-testid="lookahead"><DataTable caption="Lookahead ablation" rows={rows} columns={autoColumns(rows, ['lookahead_frames', 'latency_ms', 'pesq'])} highlight={(r) => r.lookahead_frames === 0} /></div>}>
      Each frame of lookahead adds 10 ms of algorithmic latency. The shipped model uses none (highlighted).
    </TableCard>
  );
}

function Quantization({ rows }: { rows: TableRow[] }) {
  const fp32 = rows.find((r) => typeof r.variant === 'string' && /fp32/i.test(r.variant));
  const base = typeof fp32?.pesq === 'number' ? fp32.pesq : null;
  const cols = autoColumns(rows, ['variant', 'size_kb', 'ms_per_frame', 'pesq']);
  if (base !== null) {
    cols.push({
      key: '_delta',
      label: 'Δ PESQ vs fp32',
      render: (r) => {
        if (typeof r.pesq !== 'number' || r === fp32) return '—';
        const d = r.pesq - base;
        return (
          <span className="inline-flex items-center gap-1.5">
            {`${d >= 0 ? '+' : ''}${fmt(d, 3)}`}
            <Badge tone={Math.abs(d) < 0.05 ? 'accent' : 'warn'}>{Math.abs(d) < 0.05 ? 'within 0.05' : 'over budget'}</Badge>
          </span>
        );
      },
    });
  }
  return (
    <TableCard id="quant-title" eyebrow="Deployment" title="Quantization" table={<div data-testid="quantization"><DataTable caption="Quantization variants" rows={rows} columns={cols} minWidth={560} /></div>}>
      int8 ships only if PESQ drops by less than 0.05. Dynamic quantization covers MatMul/Gemm only, so the GRU stays fp32.
    </TableCard>
  );
}

function SignificanceTable({ entries, sets, setLabel }: { entries: Significance[]; sets: TestSet[]; setLabel: string }) {
  const label = (k: string | null) => (k === null ? 'all' : (sets.find((s) => s.key === k)?.label ?? k));
  const rows: TableRow[] = entries.map((_s, i) => ({ i }));
  return (
    <TableCard
      id="sig-title"
      eyebrow="Statistics"
      title="Significance of comparisons"
      table={
        entries.length ? (
          <div data-testid="significance">
            <DataTable
              caption="Paired Wilcoxon signed-rank tests"
              rows={rows}
              minWidth={640}
              columns={[
                { key: 'cmp', label: 'Comparison', render: (r) => `${displayModelName(entries[r.i as number].a)} vs. ${displayModelName(entries[r.i as number].b)}` },
                { key: 'metric', label: 'Metric', render: (r) => metricLabel(entries[r.i as number].metric) },
                { key: 'set', label: 'Set', render: (r) => label(entries[r.i as number].set) },
                { key: 'diff', label: 'Median Δ', render: (r) => { const d = entries[r.i as number].medianDiff; return d === null ? '—' : `${d >= 0 ? '+' : ''}${fmt(d, 3)}`; } },
                { key: 'p', label: 'Result', render: (r) => <SigBadge s={entries[r.i as number]} /> },
              ]}
            />
          </div>
        ) : (
          <p className="px-5 pb-5 text-[13px] text-fg-3">No comparisons were run on {setLabel}.</p>
        )
      }
    >
      Paired Wilcoxon signed-rank tests on per-utterance scores, Holm-corrected within each family where given. A claim is only made when p &lt; 0.05.
    </TableCard>
  );
}

// ---- model card -------------------------------------------------------------------------------------

const BUDGET_LABEL: Record<string, string> = {
  weights_kb_fp32: 'Weights (fp32)',
  weights_kb_int8: 'Weights (int8)',
  peak_activation_kb: 'Peak activations / frame',
  state_kb: 'Recurrent state',
  mmacs_per_s: 'Compute',
  ms_per_frame: 'ms / frame',
};

function budgetValue(key: string, v: number): string {
  if (key.endsWith('_kb') || key.includes('_kb_')) return `${v >= 100 ? Math.round(v).toLocaleString() : fmt(v, 1)} kB`;
  if (key === 'mmacs_per_s') return `${fmt(v, 1)} MMAC/s`;
  if (key.includes('ms')) return `${fmt(v, 3)} ms`;
  return fmt(v, 2);
}

/** Flatten manifest.budget into label/value rows (nested objects such as per-device ms/frame included). */
function budgetRows(m: Manifest): [string, string][] {
  const b = m.budget;
  if (!b || typeof b !== 'object') return [];
  const out: [string, string][] = [];
  for (const [k, v] of Object.entries(b)) {
    if (typeof v === 'number' && Number.isFinite(v)) out.push([BUDGET_LABEL[k] ?? prettify(k), budgetValue(k, v)]);
    else if (v && typeof v === 'object' && !Array.isArray(v)) {
      for (const [sub, sv] of Object.entries(v as Record<string, unknown>)) {
        if (typeof sv === 'number' && Number.isFinite(sv)) out.push([`${BUDGET_LABEL[k] ?? prettify(k)} · ${prettify(sub)}`, budgetValue(k.includes('ms') ? 'ms' : k, sv)]);
      }
    }
  }
  if (!('state_kb' in b) && m.state_shape?.length) out.push(['Recurrent state', `${fmt((shapeSize(m.state_shape) * 4) / 1024, 1)} kB (from state_shape)`]);
  return out;
}

function ModelCard({ manifest, loaded }: { manifest: Manifest | null; loaded: boolean }) {
  const [bench, setBench] = useState<BenchmarkResult | null>(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    setRunning(true);
    setErr(null);
    try {
      setBench(await engine.benchmark(1000));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };
  const m = manifest;
  const metrics = m?.metrics && typeof m.metrics === 'object' ? Object.entries(m.metrics).filter(([, v]) => typeof v === 'number') : [];
  const training = m?.training && typeof m.training === 'object' ? m.training : null;
  const k = lookaheadFrames(m);
  const at = attenSpec(m);
  const specs: [string, string][] = m
    ? [
        ['Model', `${displayModelName(m.name)} v${m.version}${m.variant ? ` · ${m.variant}` : ''}`],
        ['Weights', m.trained ? 'Trained' : 'Untrained (random init, pipeline test)'],
        ['Parameters', m.params ? `${compact(m.params)} (${m.params.toLocaleString()})` : '—'],
        ['MACs / 10 ms frame', m.macs_per_frame ? `${compact(m.macs_per_frame)} (${fmt((m.macs_per_frame * 100) / 1e6, 1)} MMAC/s)` : '—'],
        ['STFT', `${m.n_fft}-pt (${(m.n_fft / m.sample_rate) * 1000} ms) · hop ${m.hop} (${(m.hop / m.sample_rate) * 1000} ms) · ${m.window.replace(/_/g, ' ')}`],
        ['Algorithmic latency', `${fmt(algorithmicLatencyMs(m), 0)} ms${k ? ` (window + ${k} frame${k > 1 ? 's' : ''} lookahead)` : ' (one window, no lookahead)'}`],
        ['Sample rate', `${m.sample_rate / 1000} kHz mono`],
        ['Streaming state', `[${m.state_shape.join(' × ')}] float32`],
        ['Attenuation limit', manifestDeclaresAtten(m) ? `${at.min}–${at.max} dB input, default ${formatAtten(at.default, at.max)}` : 'Not in this export (contract v0.1)'],
        ['ONNX', `${m.onnx}${m.onnx_int8 ? ` · int8: ${m.onnx_int8}` : ''} · opset 17, one frame per call`],
      ]
    : [];
  if (training) {
    if (typeof training.data === 'string') specs.push(['Training data', training.data]);
    if (typeof training.distilled === 'boolean') specs.push(['Distilled', training.distilled ? 'Yes (from a non-causal teacher)' : 'No']);
    if (typeof training.git_hash === 'string') specs.push(['Git', training.git_hash.slice(0, 10)]);
  }
  const budget = m ? budgetRows(m) : [];
  return (
    <Card className="p-5 sm:p-6" aria-labelledby="model-card-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <SectionTitle id="model-card-title" eyebrow="Model card" title={m ? displayModelName(m.name) : 'No model exported yet'}>
          A causal convolutional-recurrent network with a bounded complex ratio mask. Frequency-only convolutions plus a 2-layer GRU, so streaming inference matches offline inference exactly.
        </SectionTitle>
        {m && (
          <div className="flex gap-1.5">
            {m.variant && <Badge>{m.variant}</Badge>}
            <Badge tone={m.trained ? 'accent' : 'warn'}>{m.trained ? 'Trained' : 'Untrained'}</Badge>
          </div>
        )}
      </div>
      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <div>
          {m ? (
            <dl className="divide-y divide-[var(--border)] text-[13px]">
              {specs.map(([key, v]) => (
                <div key={key} className="grid grid-cols-[140px_1fr] gap-3 py-2 sm:grid-cols-[170px_1fr]">
                  <dt className="text-fg-3">{key}</dt>
                  <dd className="tnum min-w-0 break-words text-fg-2">{v}</dd>
                </div>
              ))}
              {metrics.map(([key, v]) => (
                <div key={key} className="grid grid-cols-[140px_1fr] gap-3 py-2 sm:grid-cols-[170px_1fr]">
                  <dt className="text-fg-3">Val. {key}</dt>
                  <dd className="tnum text-fg-2">{fmt(v as number, 3)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="text-sm text-fg-2">The model card fills in from models/manifest.json once the ONNX export lands.</p>
          )}
          <div className="mt-5 grid gap-4 text-[13px] sm:grid-cols-2">
            <div>
              <h3 className="mb-1 font-medium text-fg">Training data</h3>
              <p className="text-fg-2">VoiceBank+DEMAND (28 speakers, about 11.5k utterances), with dynamic remixing of noise across utterances at −5 to 20 dB SNR and random gain{training && typeof training.data === 'string' && /musan/i.test(training.data) ? ', plus MUSAN noise and reverb, mic and level augmentation for the demo model' : ''}.</p>
            </div>
            <div>
              <h3 className="mb-1 font-medium text-fg">Limitations</h3>
              <p className="text-fg-2">16 kHz only. It keeps competing voices (it was never trained to separate talkers) and does no echo cancellation or full dereverberation. Music and very low SNR degrade; see Known limitations above when available.</p>
            </div>
          </div>
        </div>
        <div className="space-y-4 self-start">
          {budget.length > 0 && (
            <div className="rounded-xl border border-line p-4" data-testid="budget-table">
              <h3 className="text-[13px] font-medium text-fg">Deployment budget</h3>
              <p className="mt-1 text-[12px] text-fg-3">From the export manifest: memory and compute per stream.</p>
              <table className="mt-3 w-full text-[12px]">
                <caption className="sr-only">Deployment budget</caption>
                <tbody className="tnum">
                  {budget.map(([key, v]) => (
                    <tr key={key} className="border-t border-line first:border-0">
                      <th scope="row" className="py-1.5 pr-3 text-left font-normal text-fg-3">
                        {key}
                      </th>
                      <td className="py-1.5 text-right text-fg-2">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="rounded-xl border border-line bg-surface-2/60 p-4">
            <h3 className="text-[13px] font-medium text-fg">Benchmark on this device</h3>
            <p className="mt-1 text-[12px] text-fg-3">Runs 1,000 frames of noise through the full streaming path (STFT, model, iSTFT) in the worker, single-threaded WebAssembly.</p>
            <Button className="mt-3" size="sm" onClick={() => void run()} disabled={running || !loaded}>
              {running ? 'Running…' : bench ? 'Run again' : 'Run benchmark'}
            </Button>
            {!loaded && <p className="mt-2 text-[12px] text-fg-3">Available once the ONNX model has loaded.</p>}
            {err && (
              <Callout tone="danger" className="mt-3">
                {err}
              </Callout>
            )}
            {bench && (
              <dl className="tnum mt-4 grid grid-cols-2 gap-3 text-[12px]">
                <div>
                  <dt className="text-fg-3">Mean / frame</dt>
                  <dd className="text-[18px] font-semibold text-fg">{fmt(bench.msPerFrame, 3)} ms</dd>
                </div>
                <div>
                  <dt className="text-fg-3">Real-time factor</dt>
                  <dd className="text-[18px] font-semibold text-fg">{fmt(bench.msPerFrame / 10, 3)}</dd>
                </div>
                <div>
                  <dt className="text-fg-3">Model only</dt>
                  <dd className="text-fg-2">{fmt(bench.modelMsPerFrame, 3)} ms</dd>
                </div>
                <div>
                  <dt className="text-fg-3">p50 / p95</dt>
                  <dd className="text-fg-2">
                    {fmt(bench.p50, 3)} / {fmt(bench.p95, 3)} ms
                  </dd>
                </div>
              </dl>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
