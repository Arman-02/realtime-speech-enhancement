import { useEffect, useState } from 'react';
import type { Manifest } from '../engine/model';
import { assetUrl } from './fetchers';
import { loadDataFile, USE_FIXTURES } from './fixtures';

// ---- §3.3 results.json (v0.1 and v0.2) ----------------------------------------------------------
// Unknown keys are ignored. Each metric value is either a number or {"mean", "ci95", "n"}.

export type MetricValue = number | { mean: number; ci95?: number | null; n?: number | null } | null | undefined;

export interface ResultRow {
  model: string;
  /** v0.2: test-set key; absent in v0.1 (treated as "vbd"). */
  set?: string;
  family?: string;
  causal?: boolean | null;
  distilled?: boolean | null;
  params?: number | null;
  macs_per_frame?: number | null;
  ms_per_frame_cpu?: number | null;
  rtf?: number | null;
  /** v0.1: per-metric CI dict on the row. */
  ci95?: Record<string, number | null>;
  n?: number | null;
  [metric: string]: unknown;
}

export interface TestSetMeta {
  label?: string;
  intrusive?: boolean;
  noise_seen_in_training?: boolean;
  [extra: string]: unknown;
}

/** Raw file shape (loosely typed; see `normalizeResults`). */
export interface RawResults {
  dataset?: string;
  generated_at?: string;
  git_hash?: string;
  test_sets?: Record<string, TestSetMeta>;
  rows: ResultRow[];
  [extra: string]: unknown;
}

export type MetricMap = Record<string, number | null | undefined>;
/** set -> category (noise type / SNR) -> model -> metric -> value */
export type SetBreakdown = Record<string, Record<string, Record<string, MetricMap>>>;

export interface TestSet {
  key: string;
  label: string;
  intrusive: boolean;
  /** null = not stated (e.g. real recordings). */
  noiseSeen: boolean | null;
}

export interface Row {
  model: string;
  set: string;
  family: string | null;
  causal: boolean | null;
  distilled: boolean | null;
  params: number | null;
  macs: number | null;
  msPerFrame: number | null;
  rtf: number | null;
  /** metric -> mean */
  values: Record<string, number>;
  /** metric -> 95 % CI half-width */
  ci: Record<string, number>;
  n: number | null;
}

export interface ParetoPoint {
  model: string;
  hidden: number | null;
  distilled: boolean;
  seed: number | null;
  macs: number;
  set: string | null;
  values: Record<string, number>;
}

export interface Significance {
  a: string;
  b: string;
  metric: string;
  set: string | null;
  medianDiff: number | null;
  /** Holm-corrected p if given, else the raw Wilcoxon p. */
  p: number | null;
  pRaw: number | null;
  significant: boolean;
}

export type TableRow = Record<string, string | number | boolean | null>;

export interface Results {
  version: '0.1' | '0.2';
  generatedAt: string | null;
  gitHash: string | null;
  dataset: string | null;
  sets: TestSet[];
  rows: Row[];
  perNoiseType: SetBreakdown;
  perSnr: SetBreakdown;
  pareto: ParetoPoint[];
  attenSweep: TableRow[];
  lookahead: TableRow[];
  quantization: TableRow[];
  significance: Significance[];
  /** sample id -> label (model, or "" for a flat per-sample map) -> metrics */
  perSample: Record<string, Record<string, MetricMap>>;
}

const META_KEYS = new Set(['model', 'set', 'family', 'causal', 'distilled', 'params', 'macs_per_frame', 'ms_per_frame_cpu', 'rtf', 'ci95', 'n', 'hidden', 'seed']);

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** Mean of a metric value (number or {mean}); null otherwise. */
export function metricMean(v: unknown): number | null {
  if (finite(v)) return v;
  if (isObj(v) && finite(v.mean)) return v.mean;
  return null;
}

export function metricCi(v: unknown): number | null {
  return isObj(v) && finite(v.ci95) ? v.ci95 : null;
}

const isMetricValue = (v: unknown) => metricMean(v) !== null;
const numOrNull = (v: unknown) => (finite(v) ? v : null);
const boolOrNull = (v: unknown) => (typeof v === 'boolean' ? v : null);
const strOrNull = (v: unknown) => (typeof v === 'string' && v ? v : null);

function toMetricMap(o: unknown): MetricMap {
  const out: MetricMap = {};
  if (!isObj(o)) return out;
  for (const [k, v] of Object.entries(o)) {
    const m = metricMean(v);
    if (m !== null) out[k] = m;
  }
  return out;
}

function normRow(r: ResultRow): Row {
  const values: Record<string, number> = {};
  const ci: Record<string, number> = {};
  const legacyCi = isObj(r.ci95) ? r.ci95 : {};
  for (const [k, v] of Object.entries(r)) {
    if (META_KEYS.has(k)) continue;
    const m = metricMean(v);
    if (m === null) continue;
    values[k] = m;
    const c = metricCi(v) ?? numOrNull(legacyCi[k]);
    if (c !== null) ci[k] = c;
  }
  return {
    model: String(r.model),
    set: strOrNull(r.set) ?? 'vbd',
    family: strOrNull(r.family),
    causal: boolOrNull(r.causal),
    distilled: boolOrNull(r.distilled),
    params: numOrNull(r.params),
    macs: numOrNull(r.macs_per_frame),
    msPerFrame: numOrNull(r.ms_per_frame_cpu),
    rtf: numOrNull(r.rtf),
    values,
    ci,
    n: numOrNull(r.n),
  };
}

/** Depth-3 value of a nested table, used to tell v0.1 (category→model→metric) from v0.2 (set→category→model). */
function firstLeaf(t: Record<string, unknown>): unknown {
  const a = Object.values(t)[0];
  if (!isObj(a)) return undefined;
  const b = Object.values(a)[0];
  if (!isObj(b)) return undefined;
  return Object.values(b)[0];
}

function normBreakdown(raw: unknown): SetBreakdown {
  if (!isObj(raw) || Object.keys(raw).length === 0) return {};
  const leaf = firstLeaf(raw);
  const v01 = isMetricValue(leaf);
  const bySet: Record<string, unknown> = v01 ? { vbd: raw } : raw;
  const out: SetBreakdown = {};
  for (const [set, cats] of Object.entries(bySet)) {
    if (!isObj(cats)) continue;
    const c: Record<string, Record<string, MetricMap>> = {};
    for (const [cat, models] of Object.entries(cats)) {
      if (!isObj(models)) continue;
      const mm: Record<string, MetricMap> = {};
      for (const [model, metrics] of Object.entries(models)) {
        const m = toMetricMap(metrics);
        if (Object.keys(m).length) mm[model] = m;
      }
      if (Object.keys(mm).length) c[cat] = mm;
    }
    if (Object.keys(c).length) out[set] = c;
  }
  return out;
}

function normPerSample(raw: unknown): Results['perSample'] {
  const out: Results['perSample'] = {};
  if (!isObj(raw)) return out;
  for (const [id, v] of Object.entries(raw)) {
    if (!isObj(v)) continue;
    const flat = Object.values(v).some(isMetricValue);
    if (flat) {
      const m = toMetricMap(v);
      if (Object.keys(m).length) out[id] = { '': m };
    } else {
      const byModel: Record<string, MetricMap> = {};
      for (const [model, metrics] of Object.entries(v)) {
        const m = toMetricMap(metrics);
        if (Object.keys(m).length) byModel[model] = m;
      }
      if (Object.keys(byModel).length) out[id] = byModel;
    }
  }
  return out;
}

function normTable(raw: unknown): TableRow[] {
  if (!Array.isArray(raw)) return [];
  const out: TableRow[] = [];
  for (const e of raw) {
    if (!isObj(e)) continue;
    const row: TableRow = {};
    for (const [k, v] of Object.entries(e)) {
      if (typeof v === 'string' || typeof v === 'boolean' || v === null) row[k] = v;
      else {
        const m = metricMean(v);
        if (m !== null) row[k] = m;
        // Infinity cannot be expressed in JSON; some writers use "inf".
      }
    }
    if (Object.keys(row).length) out.push(row);
  }
  return out;
}

function normPareto(raw: unknown): ParetoPoint[] {
  if (!Array.isArray(raw)) return [];
  const out: ParetoPoint[] = [];
  for (const e of raw) {
    if (!isObj(e)) continue;
    const macs = numOrNull(e.macs_per_frame);
    if (macs === null || macs <= 0) continue;
    const values: Record<string, number> = {};
    for (const [k, v] of Object.entries(e)) {
      if (META_KEYS.has(k)) continue;
      const m = metricMean(v);
      if (m !== null) values[k] = m;
    }
    out.push({
      model: strOrNull(e.model) ?? `hidden ${String(e.hidden)}`,
      hidden: numOrNull(e.hidden),
      distilled: e.distilled === true,
      seed: numOrNull(e.seed),
      macs,
      set: strOrNull(e.set),
      values,
    });
  }
  return out;
}

function normSignificance(raw: unknown): Significance[] {
  if (!Array.isArray(raw)) return [];
  const out: Significance[] = [];
  for (const e of raw) {
    if (!isObj(e) || typeof e.a !== 'string' || typeof e.b !== 'string') continue;
    const pRaw = numOrNull(e.p_wilcoxon) ?? numOrNull(e.p);
    const p = numOrNull(e.p_holm) ?? pRaw;
    out.push({
      a: e.a,
      b: e.b,
      metric: strOrNull(e.metric) ?? 'pesq',
      set: strOrNull(e.set),
      medianDiff: numOrNull(e.median_diff),
      p,
      pRaw,
      significant: p !== null && p < 0.05,
    });
  }
  return out;
}

const DEFAULT_SET_LABELS: Record<string, string> = {
  vbd: 'VoiceBank+DEMAND test',
  ood_esc: 'VB-D speech + ESC-50 noise',
  talker: 'Competing talker',
  real: 'Real recordings',
};

/** Normalise results.json v0.1 or v0.2 into one shape. Returns null if there is nothing usable. */
export function normalizeResults(raw: unknown): Results | null {
  if (!isObj(raw) || !Array.isArray(raw.rows)) return null;
  const r = raw as RawResults;
  const rows = r.rows.filter((x) => isObj(x) && typeof x.model === 'string').map(normRow);
  const testSets = isObj(r.test_sets) ? r.test_sets : null;
  const sets: TestSet[] = [];
  const seen = new Set<string>();
  const addSet = (key: string, meta: TestSetMeta | null) => {
    if (seen.has(key)) return;
    seen.add(key);
    sets.push({
      key,
      label: strOrNull(meta?.label) ?? (key === 'vbd' && !testSets ? (strOrNull(r.dataset) ?? DEFAULT_SET_LABELS.vbd) : (DEFAULT_SET_LABELS[key] ?? key)),
      intrusive: typeof meta?.intrusive === 'boolean' ? meta.intrusive : key !== 'real',
      // §4.6: the VB-D test noise types never appear in training.
      noiseSeen: typeof meta?.noise_seen_in_training === 'boolean' ? meta.noise_seen_in_training : key === 'vbd' ? false : null,
    });
  };
  if (testSets) for (const [k, v] of Object.entries(testSets)) addSet(k, isObj(v) ? (v as TestSetMeta) : null);
  for (const row of rows) addSet(row.set, null);
  if (sets.length === 0) addSet('vbd', null);
  // Sets with no rows at all are of no use in the selector.
  const withRows = sets.filter((s) => rows.some((x) => x.set === s.key));

  return {
    version: testSets || rows.some((x) => x.family !== null) ? '0.2' : '0.1',
    generatedAt: strOrNull(r.generated_at),
    gitHash: strOrNull(r.git_hash),
    dataset: strOrNull(r.dataset),
    sets: withRows.length ? withRows : sets,
    rows,
    perNoiseType: normBreakdown(r.per_noise_type),
    perSnr: normBreakdown(r.per_snr),
    pareto: normPareto(r.pareto),
    attenSweep: normTable(r.atten_sweep),
    lookahead: normTable(r.lookahead),
    quantization: normTable(r.quantization),
    significance: normSignificance(r.significance),
    perSample: normPerSample(r.per_sample),
  };
}

// ---- §3.4 samples.json -------------------------------------------------------------------------
export interface Sample {
  id: string;
  title: string;
  noise_type?: string;
  snr_db?: number;
  noisy: string;
  clean?: string | null;
}

// ---- §3.6 failures/index.json ------------------------------------------------------------------
export interface FailureCase {
  id: string;
  title: string;
  category: string | null;
  explanation: string;
  noisy: string;
  enhanced: string;
  clean: string | null;
  model: string | null;
}

export function normalizeFailures(raw: unknown): FailureCase[] | null {
  if (!Array.isArray(raw)) return null;
  const out: FailureCase[] = [];
  for (const e of raw) {
    if (!isObj(e) || typeof e.noisy !== 'string' || typeof e.enhanced !== 'string') continue;
    out.push({
      id: strOrNull(e.id) ?? `case-${out.length + 1}`,
      title: strOrNull(e.title) ?? strOrNull(e.id) ?? 'Failure case',
      category: strOrNull(e.category),
      explanation: typeof e.explanation === 'string' ? e.explanation : '',
      noisy: e.noisy,
      enhanced: e.enhanced,
      clean: strOrNull(e.clean),
      model: strOrNull(e.model),
    });
  }
  return out;
}

// ---- loading ------------------------------------------------------------------------------------

export type Loadable<T> = { state: 'loading' } | { state: 'missing' } | { state: 'ready'; data: T };

const cache = new Map<string, Promise<unknown>>();

function load<T>(key: string, fetcher: () => Promise<unknown>, normalize: (v: unknown) => T | null): Promise<T | null> {
  if (!cache.has(key)) cache.set(key, fetcher().then((v) => (v === null ? null : normalize(v))));
  return cache.get(key) as Promise<T | null>;
}

function useLoad<T>(key: string, fetcher: () => Promise<unknown>, normalize: (v: unknown) => T | null): Loadable<T> {
  const [s, setS] = useState<Loadable<T>>({ state: 'loading' });
  useEffect(() => {
    let alive = true;
    void load(key, fetcher, normalize).then((d) => alive && setS(d ? { state: 'ready', data: d } : { state: 'missing' }));
    return () => {
      alive = false;
    };
    // fetcher/normalize are module-level constants for each key
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return s;
}

const isSamples = (v: unknown): Sample[] | null =>
  Array.isArray(v) && v.every((s) => isObj(s) && typeof s.noisy === 'string' && typeof s.id === 'string') ? (v as Sample[]) : null;

const fetchResults = () => loadDataFile<unknown>('results.json', 'results');
const fetchSamples = () => loadDataFile<unknown>('samples/samples.json');
const fetchFailures = () => loadDataFile<unknown>('failures/index.json', 'failures');
const fetchManifestFixture = () => (USE_FIXTURES ? loadDataFile<unknown>('models/manifest.json', 'manifest') : Promise.resolve(null));
const asManifest = (v: unknown): Manifest | null => (isObj(v) && Array.isArray(v.state_shape) ? (v as unknown as Manifest) : null);

export const useResults = () => useLoad<Results>('results', fetchResults, normalizeResults);
export const useSamples = () => useLoad<Sample[]>('samples', fetchSamples, isSamples);
export const useFailures = () => useLoad<FailureCase[]>('failures', fetchFailures, normalizeFailures);
/** Fixture builds only: the v0.2 manifest used to render the model card (the engine keeps the real model). */
export const useManifestFixture = () => useLoad<Manifest>('manifest-fixture', fetchManifestFixture, asManifest);

/** Resolve a data-file path (relative to the site root, e.g. "samples/x.wav"). */
export function sampleUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return assetUrl(path);
}
