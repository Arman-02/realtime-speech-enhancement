import { ATTEN_UNLIMITED_DB } from '../engine/model';

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}

export function fmt(v: number | null | undefined, digits = 2, fallback = '—'): string {
  return v === null || v === undefined || !Number.isFinite(v) ? fallback : v.toFixed(digits);
}

export function compact(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 1e9) return `${(v / 1e9).toFixed(2)}G`;
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
}

/**
 * User-facing model name: the internal package prefix ("ans-crn-small", "ANS-CRN (ours)") is an
 * implementation detail and never shown.
 */
export function displayModelName(name: string): string {
  return name.replace(/^ans[-_ ]+/i, '').trim() || name;
}

/** Categorical slots (validated palette), assigned to entities in fixed order. */
export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)'];

/** Attenuation limit for display: the top of the range (100 dB by contract) means "no limit". */
export function formatAtten(db: number | null | undefined, max = ATTEN_UNLIMITED_DB): string {
  if (db === null || db === undefined || !Number.isFinite(db) || db >= max) return '∞ (no limit)';
  return `${Math.round(db)} dB`;
}
