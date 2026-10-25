/** results.json v0.1 / v0.2 normalisation and the failures index (§3.3, §3.6). */
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { metricMean, normalizeFailures, normalizeResults } from '../lib/data';
import fixture from '../fixtures/results_v02.json';
import failuresFixture from '../fixtures/failures_v1.json';
import listeningFixture from '../fixtures/listening_v1.json';

const realPath = fileURLToPath(new URL('../../public/results.json', import.meta.url));

const V01 = {
  dataset: 'VoiceBank+DEMAND test set (824 utterances, 16 kHz)',
  rows: [
    { model: 'Noisy (unprocessed)', pesq: 1.97, stoi: 0.921, si_sdr: 8.4, ci95: { pesq: 0.05 }, n: 824 },
    { model: 'ANS-CRN (ours)', pesq: 2.76, stoi: 0.937, si_sdr: 16.7, params: 1214038, macs_per_frame: 1957456, ci95: { pesq: 0.047 } },
  ],
  per_noise_type: { bus: { 'Noisy (unprocessed)': { pesq: 2.48, n: 164 }, 'ANS-CRN (ours)': { pesq: 3.17 } } },
  per_sample: { p232_032: { 'ANS-CRN (ours)': { pesq: 2.5, stoi: 0.93 } } },
};

describe('normalizeResults', () => {
  it('reads v0.1: rows without set become "vbd", legacy ci95 dict, per_noise_type wrapped by set', () => {
    const r = normalizeResults(V01)!;
    expect(r.version).toBe('0.1');
    expect(r.sets.map((s) => s.key)).toEqual(['vbd']);
    expect(r.sets[0].label).toBe(V01.dataset);
    expect(r.sets[0].noiseSeen).toBe(false);
    expect(r.rows.every((x) => x.set === 'vbd')).toBe(true);
    expect(r.rows[1].values.pesq).toBe(2.76);
    expect(r.rows[1].ci.pesq).toBe(0.047);
    expect(r.rows[1].macs).toBe(1957456);
    expect(Object.keys(r.perNoiseType)).toEqual(['vbd']);
    expect(r.perNoiseType.vbd.bus['ANS-CRN (ours)'].pesq).toBe(3.17);
    expect(r.perSample.p232_032['ANS-CRN (ours)'].stoi).toBe(0.93);
    expect(r.pareto).toEqual([]);
    expect(r.significance).toEqual([]);
  });

  it.skipIf(!existsSync(realPath))('accepts the real public/results.json', () => {
    const r = normalizeResults(JSON.parse(readFileSync(realPath, 'utf8')));
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    expect(r!.sets.length).toBeGreaterThan(0);
  });

  it('reads the v0.2 fixture: every section present, {mean,ci95,n} values, unknown keys ignored', () => {
    const r = normalizeResults(fixture)!;
    expect(r.version).toBe('0.2');
    expect(r.sets.map((s) => s.key)).toEqual(['vbd', 'ood_esc', 'talker', 'real']);
    expect(r.sets.find((s) => s.key === 'real')!.intrusive).toBe(false);
    const crn = r.rows.find((x) => x.model === 'CRN-256' && x.set === 'vbd')!;
    expect(crn.values.pesq).toBeGreaterThan(2);
    expect(crn.ci.pesq).toBeGreaterThan(0);
    expect(crn.values).not.toHaveProperty('notes');
    expect(r.rows.find((x) => x.set === 'real')!.values).not.toHaveProperty('pesq');
    expect(Object.keys(r.perSnr)).toEqual(['vbd', 'ood_esc']);
    expect(r.pareto.filter((p) => p.hidden === 128).length).toBe(4);
    expect(r.attenSweep.find((x) => x.atten_lim_db === null)).toBeTruthy();
    expect(r.lookahead).toHaveLength(3);
    expect(r.quantization).toHaveLength(3);
    const sig = r.significance.find((s) => s.a === 'CRN-128 (KD)' && s.set === 'vbd')!;
    expect(sig.p).toBe(0.012); // Holm-corrected p preferred
    expect(sig.significant).toBe(true);
    expect(r.significance.find((s) => s.a === 'CRN-256 (KD)' && s.set === 'vbd')!.significant).toBe(false);
    expect(Object.keys(r.perSample)).toHaveLength(6);
    expect(r.perSample.p232_032['']).toHaveProperty('pesq');
  });

  it('rejects things that are not results', () => {
    expect(normalizeResults(null)).toBeNull();
    expect(normalizeResults({ rows: 'nope' })).toBeNull();
    expect(normalizeResults([])).toBeNull();
  });

  it('metricMean handles numbers and {mean}', () => {
    expect(metricMean(2)).toBe(2);
    expect(metricMean({ mean: 3, ci95: 0.1, n: 5 })).toBe(3);
    expect(metricMean(null)).toBeNull();
    expect(metricMean('2')).toBeNull();
  });
});

describe('normalizeFailures', () => {
  it('reads the fixture and skips malformed entries', () => {
    const f = normalizeFailures([...(failuresFixture as unknown[]), { id: 'bad' }, 'x'])!;
    expect(f).toHaveLength((failuresFixture as unknown[]).length);
    expect(f[0].clean).toBeNull();
    expect(f[1].clean).toMatch(/\.wav$/);
    expect(normalizeFailures({})).toBeNull();
  });
});

describe('fixtures point at files that exist', () => {
  const pub = fileURLToPath(new URL('../../public/', import.meta.url));
  it('every wav referenced by the failure and listening fixtures is in public/', () => {
    const paths = new Set<string>();
    for (const c of failuresFixture as { noisy: string; enhanced: string; clean: string | null }[]) [c.noisy, c.enhanced, c.clean].forEach((p) => p && paths.add(p));
    const walk = (v: unknown) => {
      if (typeof v === 'string' && v.endsWith('.wav')) paths.add(v);
      else if (v && typeof v === 'object') Object.values(v).forEach(walk);
    };
    walk(listeningFixture);
    expect(paths.size).toBeGreaterThan(3);
    for (const p of paths) expect(existsSync(pub + p), p).toBe(true);
  });
});
