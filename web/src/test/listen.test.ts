import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/listening_v1.json';
import {
  buildExport,
  conditionOrder,
  exportFilename,
  indexKey,
  newSession,
  parseIndex,
  parseSession,
  rate,
  raterIdFor,
  selectTrials,
  serializeSession,
  trialComplete,
  trialConditions,
  type ListeningIndex,
} from '../lib/listen';

function idx(): ListeningIndex {
  const r = parseIndex(fixture);
  if (!r.ok) throw new Error(r.reason);
  return r.index;
}

function bigIndex(n: number): ListeningIndex {
  const files = { noisy: 'a.wav', wiener: 'b.wav', crn: 'c.wav', crn_kd: 'd.wav', clean: 'e.wav' };
  return {
    version: 1,
    conditions: ['noisy', 'wiener', 'crn', 'crn_kd', 'clean'],
    trials: Array.from({ length: n }, (_, i) => ({ id: `t${String(i + 1).padStart(2, '0')}`, files })),
  };
}

describe('listening index', () => {
  it('parses the fixture and keeps the practice trial', () => {
    const i = idx();
    expect(i.conditions).toEqual(['noisy', 'wiener', 'crn', 'crn_kd', 'clean']);
    expect(i.trials).toHaveLength(4);
    expect(i.practice?.id).toBe('practice');
  });

  it('rejects malformed indexes with a reason', () => {
    expect(parseIndex(null).ok).toBe(false);
    expect(parseIndex({ conditions: ['noisy'], trials: [] }).ok).toBe(false);
    expect(parseIndex({ conditions: ['a', 'b'], trials: [{ id: 'x', files: { a: 'a.wav' } }] }).ok).toBe(false);
  });

  it('filters conditions to the files a trial has (missing crn_kd shortens the trial)', () => {
    const i = idx();
    const t3 = i.trials.find((t) => t.id === 't03')!;
    expect(trialConditions(i.conditions, t3.files)).toEqual(['noisy', 'wiener', 'crn', 'clean']);
    expect(trialConditions(['noisy', 'crn'], { noisy: 'x.wav', crn: '', clean: 'y.wav' })).toEqual(['noisy']);
  });
});

describe('seeded randomisation', () => {
  const conds = ['noisy', 'wiener', 'crn', 'crn_kd', 'clean'];

  it('is deterministic per (rater, trial)', () => {
    expect(conditionOrder('rater-a', 't01', conds)).toEqual(conditionOrder('rater-a', 't01', conds));
    expect(conditionOrder('rater-a', 't01', conds).slice().sort()).toEqual(conds.slice().sort());
  });

  it('differs across raters and across trials', () => {
    const raters = Array.from({ length: 12 }, (_, i) => conditionOrder(`r${i}`, 't01', conds).join(','));
    expect(new Set(raters).size).toBeGreaterThan(6);
    const trials = Array.from({ length: 12 }, (_, i) => conditionOrder('r0', `t${i}`, conds).join(','));
    expect(new Set(trials).size).toBeGreaterThan(6);
  });

  it('selects at most 20 trials, all of them when fewer', () => {
    expect(selectTrials(bigIndex(30).trials, 's')).toHaveLength(20);
    expect(new Set(selectTrials(bigIndex(30).trials, 's').map((t) => t.id)).size).toBe(20);
    expect(selectTrials(bigIndex(7).trials, 's')).toHaveLength(7);
    expect(selectTrials(bigIndex(30).trials, 's').map((t) => t.id)).toEqual(selectTrials(bigIndex(30).trials, 's').map((t) => t.id));
  });

  it('builds a session plan whose orders match conditionOrder', () => {
    const i = idx();
    const s = newSession(i, 'seed123');
    expect(s.plan).toHaveLength(4);
    for (const p of s.plan) {
      const t = i.trials.find((x) => x.id === p.id)!;
      expect(p.order).toEqual(conditionOrder('seed123', p.id, trialConditions(i.conditions, t.files)));
    }
    expect(s.plan.find((p) => p.id === 't03')!.order).toHaveLength(4);
  });
});

describe('ratings, export and resume', () => {
  it('records position and ms, and exports the §3.6 shape', () => {
    const i = idx();
    let s = newSession(i, 'abcdefghij', new Date('2026-10-23T10:00:00Z'));
    s = { ...s, rater_id: raterIdFor('Maple Leaf', s.rater_seed), device_note: 'AirPods', finished_at: '2026-10-23T10:15:00.123Z' };
    expect(s.rater_id).toBe('maple-leaf-abcdefgh');
    const t = s.plan[0];
    s = { ...s, trial_started_at: { [t.id]: 1000 } };
    t.order.forEach((c, k) => {
      s = rate(s, t.id, c, k + 1, 2000 + k * 100);
    });
    s = rate(s, t.id, t.order[0], 5, 9000); // re-rating overwrites
    expect(s.ratings[t.id][t.order[0]]).toEqual({ score: 5, position: 0, ms: 8000 });
    expect(rate(s, t.id, t.order[0], 7)).toBe(s); // out of range is ignored
    expect(trialComplete(s, t)).toBe(true);
    expect(trialComplete(s, s.plan[1])).toBe(false);

    const out = buildExport(s, i.conditions, 'test-agent');
    expect(out).toMatchObject({ version: 1, rater_id: 'maple-leaf-abcdefgh', started_at: '2026-10-23T10:00:00.000Z', device_note: 'AirPods' });
    expect(out.ratings).toHaveLength(t.order.length);
    for (const r of out.ratings) {
      expect(Object.keys(r).sort()).toEqual(['condition', 'ms', 'position', 'score', 'trial']);
      expect(r.score).toBeGreaterThanOrEqual(1);
      expect(r.score).toBeLessThanOrEqual(5);
      expect(t.order[r.position]).toBe(r.condition);
    }
    expect(out.ratings[1]).toEqual({ trial: t.id, condition: t.order[1], score: 2, position: 1, ms: 1100 });
    expect(exportFilename(s)).toBe('listen-maple-leaf-abcdefgh-2026-10-23T10-15-00Z.json');
  });

  it('round-trips a session and rejects one from a different index', () => {
    const i = idx();
    let s = newSession(i, 'resume-seed');
    s = { ...s, phase: 'trials', current: 2 };
    s = rate(s, s.plan[2].id, s.plan[2].order[0], 3, Date.now());
    const back = parseSession(serializeSession(s), i);
    expect(back).toEqual(s);

    const other = bigIndex(5);
    expect(indexKey(other)).not.toBe(indexKey(i));
    expect(parseSession(serializeSession(s), other)).toBeNull();
    expect(parseSession('{not json', i)).toBeNull();
    expect(parseSession(null, i)).toBeNull();
  });
});
