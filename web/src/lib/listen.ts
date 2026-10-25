/**
 * Blind listening test (PLAN.md §3.6, §5.1.6): index parsing, seeded per-rater randomisation,
 * resumable session state and the response export. Pure logic only (no DOM), so it is unit-tested.
 *
 * Response export (`listen-<rater>-<timestamp>.json`), §3.6 plus a few extra keys:
 *   { version: 1, rater_id, started_at, device_note,
 *     ratings: [{ trial, condition, score, position, ms }], ...extras }
 * - `score`    ACR 1–5 (1 Bad, 2 Poor, 3 Fair, 4 Good, 5 Excellent).
 * - `position` 0-based index of the condition in the order it was presented in that trial
 *              (0 = "A", 1 = "B", ...), so position bias can be checked.
 * - `ms`       milliseconds from the moment the trial was first shown to the rater's final rating of
 *              that condition (a re-rating overwrites it). If the page is reloaded mid-trial the
 *              clock keeps the original start, so the value includes the time away.
 * Practice-trial ratings are never exported.
 */

export const ACR_LABELS = ['Bad', 'Poor', 'Fair', 'Good', 'Excellent'] as const;
export const MAX_TRIALS = 20;
export const STORAGE_PREFIX = 'listen-session-v1:';

// ---- §3.6 listening/index.json -----------------------------------------------------------------

export interface ListeningTrial {
  id: string;
  utterance?: string;
  set?: string;
  noise_type?: string;
  snr_db?: number;
  files: Record<string, string>;
}

export interface ListeningIndex {
  version: number;
  conditions: string[];
  hidden_reference?: string;
  low_anchor?: string;
  practice?: { id: string; files: Record<string, string> } | null;
  trials: ListeningTrial[];
}

export type ParseResult = { ok: true; index: ListeningIndex } | { ok: false; reason: string };

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

function cleanFiles(v: unknown): Record<string, string> | null {
  if (!isObj(v)) return null;
  const out: Record<string, string> = {};
  for (const [k, p] of Object.entries(v)) if (typeof p === 'string' && p.trim() && p !== '…') out[k] = p;
  return out;
}

/** Validate an index loosely (unknown keys are ignored); trials with fewer than 2 playable conditions are dropped. */
export function parseIndex(v: unknown): ParseResult {
  if (!isObj(v)) return { ok: false, reason: 'index.json is not an object' };
  const conditions = Array.isArray(v.conditions) ? v.conditions.filter((c): c is string => typeof c === 'string' && c.length > 0) : [];
  if (conditions.length < 2) return { ok: false, reason: 'index.json lists fewer than two conditions' };
  if (!Array.isArray(v.trials)) return { ok: false, reason: 'index.json has no trials array' };
  const trials: ListeningTrial[] = [];
  for (const t of v.trials) {
    if (!isObj(t) || typeof t.id !== 'string') continue;
    const files = cleanFiles(t.files);
    if (!files) continue;
    const trial: ListeningTrial = { id: t.id, files };
    if (typeof t.utterance === 'string') trial.utterance = t.utterance;
    if (typeof t.set === 'string') trial.set = t.set;
    if (typeof t.noise_type === 'string') trial.noise_type = t.noise_type;
    if (typeof t.snr_db === 'number') trial.snr_db = t.snr_db;
    if (trialConditions(conditions, trial.files).length >= 2) trials.push(trial);
  }
  if (trials.length === 0) return { ok: false, reason: 'index.json has no playable trials' };
  let practice: ListeningIndex['practice'] = null;
  if (isObj(v.practice)) {
    const files = cleanFiles(v.practice.files);
    if (files && trialConditions(conditions, files).length >= 1) practice = { id: typeof v.practice.id === 'string' ? v.practice.id : 'practice', files };
  }
  return {
    ok: true,
    index: {
      version: typeof v.version === 'number' ? v.version : 1,
      conditions,
      hidden_reference: typeof v.hidden_reference === 'string' ? v.hidden_reference : undefined,
      low_anchor: typeof v.low_anchor === 'string' ? v.low_anchor : undefined,
      practice,
      trials,
    },
  };
}

/** The index's conditions that this trial actually has a file for (index order). */
export function trialConditions(conditions: readonly string[], files: Record<string, string>): string[] {
  return conditions.filter((c) => typeof files[c] === 'string' && files[c].length > 0);
}

// ---- seeded randomness -------------------------------------------------------------------------

/** FNV-1a 32-bit string hash. */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seededShuffle<T>(items: readonly T[], seed: string): T[] {
  const rnd = mulberry32(hashString(seed));
  const a = items.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Presentation order of one trial's conditions for one rater. */
export function conditionOrder(raterSeed: string, trialId: string, conditions: readonly string[]): string[] {
  return seededShuffle(conditions, `${raterSeed}|order|${trialId}`);
}

/** Up to `max` trials in a per-rater random order (all of them when there are fewer). */
export function selectTrials(trials: readonly ListeningTrial[], raterSeed: string, max = MAX_TRIALS): ListeningTrial[] {
  return seededShuffle(trials, `${raterSeed}|trials`).slice(0, max);
}

export function positionLabel(i: number): string {
  return String.fromCharCode(65 + i);
}

/** A stable short key for an index, so a re-rendered index invalidates saved progress. */
export function indexKey(index: ListeningIndex): string {
  return hashString(JSON.stringify({ c: index.conditions, p: index.practice ?? null, t: index.trials })).toString(36);
}

export function randomSeed(): string {
  const a = new Uint32Array(2);
  try {
    crypto.getRandomValues(a);
  } catch {
    a[0] = Math.floor(Math.random() * 2 ** 32);
    a[1] = Math.floor(Math.random() * 2 ** 32);
  }
  return `${a[0].toString(36)}${a[1].toString(36)}`;
}

export function slugify(s: string): string {
  return s
    .normalize('NFKD')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 24);
}

// ---- session -----------------------------------------------------------------------------------

export type Phase = 'consent' | 'headphones' | 'details' | 'practice' | 'trials' | 'done';

export interface PlannedTrial {
  id: string;
  /** Condition names in presentation order (index = position, label A, B, ...). */
  order: string[];
}

export interface Rating {
  score: number;
  position: number;
  ms: number;
}

export interface Session {
  version: 1;
  index_key: string;
  rater_seed: string;
  rater_id: string;
  nickname: string;
  device_note: string;
  started_at: string;
  finished_at: string | null;
  phase: Phase;
  headphone_check: { passed: boolean; attempts: number };
  practice_order: string[];
  plan: PlannedTrial[];
  current: number;
  trial_started_at: Record<string, number>;
  /** Conditions the rater has listened to in full, per trial id ("practice" for the practice trial). */
  listened: Record<string, string[]>;
  ratings: Record<string, Record<string, Rating>>;
  practice_ratings: Record<string, number>;
  submitted: boolean;
}

export function newSession(index: ListeningIndex, seed: string = randomSeed(), now: Date = new Date()): Session {
  const plan = selectTrials(index.trials, seed).map((t) => ({ id: t.id, order: conditionOrder(seed, t.id, trialConditions(index.conditions, t.files)) }));
  return {
    version: 1,
    index_key: indexKey(index),
    rater_seed: seed,
    rater_id: `r-${seed.slice(0, 8)}`,
    nickname: '',
    device_note: '',
    started_at: now.toISOString(),
    finished_at: null,
    phase: 'consent',
    headphone_check: { passed: false, attempts: 0 },
    practice_order: index.practice ? conditionOrder(seed, index.practice.id, trialConditions(index.conditions, index.practice.files)) : [],
    plan,
    current: 0,
    trial_started_at: {},
    listened: {},
    ratings: {},
    practice_ratings: {},
    submitted: false,
  };
}

/** rater_id = "<nickname slug>-<seed prefix>" (or "r-<seed prefix>"), unique even if two raters share a nickname. */
export function raterIdFor(nickname: string, seed: string): string {
  const slug = slugify(nickname);
  return `${slug || 'r'}-${seed.slice(0, 8)}`;
}

export function storageKey(index: ListeningIndex): string {
  return `${STORAGE_PREFIX}${indexKey(index)}`;
}

export function serializeSession(s: Session): string {
  return JSON.stringify(s);
}

/** Parse a saved session; null when it is malformed or belongs to a different index. */
export function parseSession(raw: string | null, index: ListeningIndex): Session | null {
  if (!raw) return null;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isObj(v) || v.version !== 1 || v.index_key !== indexKey(index)) return null;
  const s = v as unknown as Session;
  if (typeof s.rater_seed !== 'string' || !Array.isArray(s.plan) || typeof s.current !== 'number' || !isObj(s.ratings)) return null;
  const known = new Set(index.trials.map((t) => t.id));
  if (!s.plan.every((p) => isObj(p) && known.has(p.id) && Array.isArray(p.order))) return null;
  s.current = Math.max(0, Math.min(s.plan.length - 1, Math.floor(s.current)));
  s.listened ??= {};
  s.trial_started_at ??= {};
  s.practice_ratings ??= {};
  s.practice_order ??= [];
  return s;
}

export function loadSession(index: ListeningIndex): Session | null {
  try {
    return parseSession(localStorage.getItem(storageKey(index)), index);
  } catch {
    return null;
  }
}

export function saveSession(index: ListeningIndex, s: Session): void {
  try {
    localStorage.setItem(storageKey(index), serializeSession(s));
  } catch {
    /* storage unavailable (private mode, quota): progress just isn't persisted */
  }
}

export function clearSession(index: ListeningIndex): void {
  try {
    localStorage.removeItem(storageKey(index));
  } catch {
    /* ignore */
  }
}

export function trialComplete(s: Session, t: PlannedTrial): boolean {
  const r = s.ratings[t.id] ?? {};
  return t.order.every((c) => typeof r[c]?.score === 'number');
}

/** Record (or overwrite) a rating. Returns a new session object. */
export function rate(s: Session, trialId: string, condition: string, score: number, now: number = Date.now()): Session {
  const t = s.plan.find((p) => p.id === trialId);
  if (!t) return s;
  const position = t.order.indexOf(condition);
  if (position < 0 || !(score >= 1 && score <= 5)) return s;
  const started = s.trial_started_at[trialId] ?? now;
  return {
    ...s,
    ratings: { ...s.ratings, [trialId]: { ...(s.ratings[trialId] ?? {}), [condition]: { score: Math.round(score), position, ms: Math.max(0, Math.round(now - started)) } } },
  };
}

// ---- export ------------------------------------------------------------------------------------

export interface ExportRating {
  trial: string;
  condition: string;
  score: number;
  position: number;
  ms: number;
}

export interface ResponseExport {
  version: 1;
  rater_id: string;
  started_at: string;
  device_note: string;
  ratings: ExportRating[];
  nickname: string;
  finished_at: string | null;
  index_key: string;
  conditions: string[];
  headphone_check: { passed: boolean; attempts: number };
  user_agent?: string;
}

export function buildExport(s: Session, conditions: readonly string[], userAgent?: string): ResponseExport {
  const ratings: ExportRating[] = [];
  for (const t of s.plan) {
    const r = s.ratings[t.id] ?? {};
    t.order.forEach((c) => {
      const x = r[c];
      if (x) ratings.push({ trial: t.id, condition: c, score: x.score, position: x.position, ms: x.ms });
    });
  }
  const out: ResponseExport = {
    version: 1,
    rater_id: s.rater_id,
    started_at: s.started_at,
    device_note: s.device_note,
    ratings,
    nickname: s.nickname,
    finished_at: s.finished_at,
    index_key: s.index_key,
    conditions: [...conditions],
    headphone_check: s.headphone_check,
  };
  if (userAgent) out.user_agent = userAgent;
  return out;
}

export function exportFilename(s: Session): string {
  const ts = (s.finished_at ?? new Date().toISOString()).replace(/\.\d+Z$/, 'Z').replace(/[:]/g, '-');
  return `listen-${s.rater_id}-${ts}.json`;
}
