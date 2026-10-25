/**
 * Hidden blind listening test (#/listen), PLAN.md §3.6 and §5.1.6: ACR (1–5) ratings of anonymised
 * conditions, P.808 in spirit. Logic (randomisation, persistence, export) lives in lib/listen.ts.
 */
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { decodeTo16k, getPlaybackContext, toAudioBuffer } from '../lib/audio';
import { assetUrl } from '../lib/fetchers';
import { loadDataFile } from '../lib/fixtures';
import {
  ACR_LABELS,
  buildExport,
  clearSession,
  exportFilename,
  loadSession,
  newSession,
  parseIndex,
  positionLabel,
  raterIdFor,
  rate,
  saveSession,
  trialComplete,
  type ListeningIndex,
  type Phase,
  type Session,
} from '../lib/listen';
import { IconCheck, IconDownload, IconHeadphones, IconPause, IconPlay } from '../components/icons';
import { Badge, Button, Callout, Card, ProgressBar, SectionTitle } from '../components/ui';
import { cx } from '../lib/format';

type IndexState = { kind: 'loading' } | { kind: 'missing' } | { kind: 'invalid'; reason: string } | { kind: 'ready'; index: ListeningIndex };

export default function Listen() {
  const [state, setState] = useState<IndexState>({ kind: 'loading' });
  useEffect(() => {
    let alive = true;
    void loadDataFile<unknown>('listening/index.json', 'listening').then((raw) => {
      if (!alive) return;
      if (raw === null) return setState({ kind: 'missing' });
      const r = parseIndex(raw);
      setState(r.ok ? { kind: 'ready', index: r.index } : { kind: 'invalid', reason: r.reason });
    });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <header>
        <div className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">Listening test</div>
        <h1 className="text-[28px] font-semibold tracking-tight sm:text-[34px]">How good does it sound?</h1>
        <p className="mt-2 text-[15px] text-fg-2">A blind comparison of speech recordings. It takes about 15 minutes and helps measure how natural the processed speech sounds to people, not just to metrics.</p>
      </header>
      {state.kind === 'loading' && (
        <div role="status" className="flex items-center gap-3 text-sm text-fg-2">
          <span className="pulse-dot h-2 w-2 rounded-full bg-accent" /> Loading the test…
        </div>
      )}
      {(state.kind === 'missing' || state.kind === 'invalid') && (
        <Card className="p-6">
          <Badge tone="warn" className="mb-2">
            Not available yet
          </Badge>
          <h2 className="text-[17px] font-semibold tracking-tight">The listening test isn’t available yet</h2>
          <p className="mt-1 max-w-prose text-sm text-fg-2">
            The test clips haven’t been published. Thanks for your interest; please check back later or ask whoever sent you this link.
          </p>
          {state.kind === 'invalid' && <p className="mt-2 text-[12px] text-fg-3">Details: {state.reason}.</p>}
        </Card>
      )}
      {state.kind === 'ready' && <Runner index={state.index} />}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// Clip player: one clip at a time, decoded to 16 kHz once and cached.

type Listener = () => void;

class ClipPlayer {
  private pcm = new Map<string, Promise<Float32Array>>();
  private buffers = new Map<string, AudioBuffer>();
  private src: AudioBufferSourceNode | null = null;
  private startedAt = 0;
  private listeners = new Set<Listener>();
  playing: string | null = null;
  duration = 0;

  subscribe = (l: Listener): (() => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };

  getPlaying = (): string | null => this.playing;

  private emit(): void {
    this.listeners.forEach((l) => l());
  }

  /** Fetch + decode without creating an AudioContext (no user gesture needed). */
  load(url: string): Promise<Float32Array> {
    let p = this.pcm.get(url);
    if (!p) {
      p = fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.arrayBuffer();
        })
        .then((b) => decodeTo16k(b))
        .then((d) => d.audio);
      p.catch(() => this.pcm.delete(url));
      this.pcm.set(url, p);
    }
    return p;
  }

  async play(key: string, url: string, onEnded: () => void): Promise<void> {
    this.stop();
    const x = await this.load(url);
    const ctx = getPlaybackContext();
    if (ctx.state === 'suspended') await ctx.resume();
    let buf = this.buffers.get(url);
    if (!buf) {
      buf = toAudioBuffer(ctx, x);
      this.buffers.set(url, buf);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => {
      if (this.src !== src) return;
      this.src = null;
      this.playing = null;
      this.emit();
      onEnded();
    };
    this.src = src;
    this.playing = key;
    this.duration = buf.duration;
    this.startedAt = ctx.currentTime;
    src.start();
    this.emit();
  }

  stop(): void {
    const s = this.src;
    this.src = null;
    if (s) {
      try {
        s.stop();
      } catch {
        /* not started */
      }
      s.disconnect();
    }
    if (this.playing !== null) {
      this.playing = null;
      this.emit();
    }
  }

  progress(): number {
    if (!this.src || !this.duration) return 0;
    const ctx = getPlaybackContext();
    return Math.min(1, (ctx.currentTime - this.startedAt) / this.duration);
  }
}

function usePlayer(): ClipPlayer {
  const ref = useRef<ClipPlayer | null>(null);
  ref.current ??= new ClipPlayer();
  const p = ref.current;
  useEffect(() => () => p.stop(), [p]);
  return p;
}

// ---------------------------------------------------------------------------------------------

const STEPS: { phase: Phase; label: string }[] = [
  { phase: 'consent', label: 'Consent' },
  { phase: 'headphones', label: 'Headphones' },
  { phase: 'details', label: 'About you' },
  { phase: 'practice', label: 'Practice' },
  { phase: 'trials', label: 'Test' },
];

function Runner({ index }: { index: ListeningIndex }) {
  const [session, setSession] = useState<Session>(() => loadSession(index) ?? newSession(index));
  const player = usePlayer();

  useEffect(() => saveSession(index, session), [index, session]);

  const update = useCallback((f: (s: Session) => Session) => setSession((s) => f(s)), []);
  const go = useCallback(
    (phase: Phase) => {
      player.stop();
      update((s) => ({ ...s, phase, finished_at: phase === 'done' ? (s.finished_at ?? new Date().toISOString()) : s.finished_at }));
      window.scrollTo({ top: 0 });
    },
    [player, update],
  );

  const restart = () => {
    if (!window.confirm('Start over? Your ratings so far will be discarded.')) return;
    player.stop();
    clearSession(index);
    setSession(newSession(index));
  };

  const steps = index.practice ? STEPS : STEPS.filter((s) => s.phase !== 'practice');
  const stepIdx = session.phase === 'done' ? steps.length : steps.findIndex((s) => s.phase === session.phase);

  return (
    <>
      {session.phase !== 'done' && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <ol className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]" aria-label="Progress">
            {steps.map((s, i) => (
              <li key={s.phase} className={cx('flex items-center gap-1.5', i === stepIdx ? 'font-medium text-fg' : i < stepIdx ? 'text-fg-2' : 'text-fg-3')} aria-current={i === stepIdx ? 'step' : undefined}>
                <span className={cx('grid h-5 w-5 place-items-center rounded-full text-[10px]', i < stepIdx ? 'bg-accent-soft text-accent' : i === stepIdx ? 'bg-accent text-accent-ink' : 'bg-surface-3')}>
                  {i < stepIdx ? <IconCheck size={10} /> : i + 1}
                </span>
                {s.label}
              </li>
            ))}
          </ol>
          {session.phase !== 'consent' && (
            <Button size="sm" variant="ghost" onClick={restart}>
              Start over
            </Button>
          )}
        </div>
      )}

      {session.phase === 'consent' && <Consent trials={session.plan.length} onAgree={() => go('headphones')} />}
      {session.phase === 'headphones' && (
        <HeadphoneCheck
          onDone={(attempts) => {
            update((s) => ({ ...s, headphone_check: { passed: true, attempts } }));
            go('details');
          }}
          onFail={() => update((s) => ({ ...s, headphone_check: { passed: false, attempts: s.headphone_check.attempts + 1 } }))}
          priorAttempts={session.headphone_check.attempts}
        />
      )}
      {session.phase === 'details' && (
        <Details
          session={session}
          onSubmit={(nickname, device) => {
            update((s) => ({ ...s, nickname, device_note: device, rater_id: raterIdFor(nickname, s.rater_seed) }));
            go(index.practice ? 'practice' : 'trials');
          }}
        />
      )}
      {session.phase === 'practice' && index.practice && (
        <TrialView
          key="practice"
          player={player}
          title="Practice trial"
          subtitle="This one isn’t recorded. Use it to set a comfortable volume and get a feel for the range of quality you’ll hear."
          order={session.practice_order}
          files={index.practice.files}
          listened={session.listened.practice ?? []}
          scores={session.practice_ratings}
          onListened={(c) => update((s) => ({ ...s, listened: { ...s.listened, practice: Array.from(new Set([...(s.listened.practice ?? []), c])) } }))}
          onRate={(c, v) => update((s) => ({ ...s, practice_ratings: { ...s.practice_ratings, [c]: v } }))}
          next={{ label: 'Start the test', onClick: () => go('trials') }}
        />
      )}
      {session.phase === 'trials' && <Trials index={index} session={session} update={update} player={player} onFinish={() => go('done')} />}
      {session.phase === 'done' && <Done index={index} session={session} update={update} onRestart={restart} />}
    </>
  );
}

function Consent({ trials, onAgree }: { trials: number; onAgree: () => void }) {
  const [ok, setOk] = useState(false);
  return (
    <Card className="p-5 sm:p-6">
      <SectionTitle eyebrow="Before you start" title="Instructions and consent" />
      <div className="space-y-3 text-[14px] leading-relaxed text-fg-2">
        <p>
          You’ll hear <strong className="text-fg">{trials} short sentences</strong>, each in several versions labelled A, B, C… in a random order. Listen to every version, then rate its <strong className="text-fg">overall quality</strong> from 1 (Bad) to 5 (Excellent): how clear and natural the voice sounds and how annoying any remaining noise or distortion is. Some versions may sound identical; that’s fine.
        </p>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            <strong className="text-fg">Wear headphones</strong> and sit somewhere quiet. A quick check comes next.
          </li>
          <li>Keep the volume fixed once you’ve set it during the practice trial.</li>
          <li>There are no right answers. Go with your first impression; each trial takes about 30–45 seconds.</li>
          <li>Progress is saved in this browser, so you can take a break and come back to this page.</li>
        </ul>
        <p className="text-[13px] text-fg-3">
          Your ratings, an optional nickname and an optional device note are stored for a research project on speech enhancement. No audio is recorded and no personal data is required. Participation is voluntary and you can stop at any time.
        </p>
      </div>
      <label className="mt-5 flex cursor-pointer items-start gap-3 text-[14px] text-fg">
        <input type="checkbox" className="mt-1 h-4 w-4 accent-[var(--accent)]" checked={ok} onChange={(e) => setOk(e.target.checked)} />
        <span>I’m 18 or older and I agree to take part.</span>
      </label>
      <Button className="mt-5" variant="primary" disabled={!ok} onClick={onAgree}>
        Continue
      </Button>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------

type Side = 'left' | 'right';
const CHECK_BEEPS = 4;

function randomSides(): Side[] {
  return Array.from({ length: CHECK_BEEPS }, () => (Math.random() < 0.5 ? 'left' : 'right'));
}

async function playBeep(side: Side): Promise<void> {
  const ctx = getPlaybackContext();
  if (ctx.state === 'suspended') await ctx.resume();
  const t = ctx.currentTime + 0.05;
  const osc = ctx.createOscillator();
  osc.frequency.value = 660;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(0.25, t + 0.02);
  g.gain.setValueAtTime(0.25, t + 0.38);
  g.gain.linearRampToValueAtTime(0, t + 0.4);
  const merger = ctx.createChannelMerger(2);
  osc.connect(g);
  g.connect(merger, 0, side === 'left' ? 0 : 1);
  merger.connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 0.42);
  await new Promise<void>((r) => {
    osc.onended = () => {
      merger.disconnect();
      r();
    };
  });
}

function HeadphoneCheck({ onDone, onFail, priorAttempts }: { onDone: (attempts: number) => void; onFail: () => void; priorAttempts: number }) {
  const [sides, setSides] = useState<Side[]>(randomSides);
  const [answers, setAnswers] = useState<Side[]>([]);
  const [played, setPlayed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const step = answers.length;
  const finished = step === sides.length;
  const passed = finished && answers.every((a, i) => a === sides[i]);

  const play = async () => {
    setBusy(true);
    setErr(null);
    try {
      await playBeep(sides[step]);
      setPlayed(true);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const answer = (s: Side) => {
    const next = [...answers, s];
    setAnswers(next);
    setPlayed(false);
    if (next.length === sides.length && !next.every((a, i) => a === sides[i])) onFail();
  };
  const retry = () => {
    setSides(randomSides());
    setAnswers([]);
    setPlayed(false);
  };

  return (
    <Card className="p-5 sm:p-6">
      <SectionTitle eyebrow="Step 2" title="Headphone check">
        Put your headphones on. You’ll hear {CHECK_BEEPS} beeps, each in one ear only. After each one, say which ear it was in.
      </SectionTitle>
      {!finished && (
        <div className="space-y-4">
          <div className="text-[13px] text-fg-3">
            Beep {step + 1} of {sides.length}
          </div>
          <Button onClick={() => void play()} disabled={busy}>
            <IconHeadphones size={16} /> {busy ? 'Playing…' : played ? 'Play again' : 'Play beep'}
          </Button>
          <div role="group" aria-label="Which ear did you hear the beep in?" className="flex flex-wrap items-center gap-2">
            <span className="mr-1 text-[13px] text-fg-2">I heard it in my</span>
            <Button variant="secondary" disabled={!played || busy} onClick={() => answer('left')}>
              Left ear
            </Button>
            <Button variant="secondary" disabled={!played || busy} onClick={() => answer('right')}>
              Right ear
            </Button>
          </div>
          {err && <Callout tone="danger">Audio could not play: {err}</Callout>}
        </div>
      )}
      {finished && passed && (
        <div className="space-y-4">
          <Callout title="All correct">Your headphones are working and on the right way round.</Callout>
          <Button variant="primary" onClick={() => onDone(priorAttempts + 1)}>
            Continue
          </Button>
        </div>
      )}
      {finished && !passed && (
        <div className="space-y-4">
          <Callout tone="warn" title="That didn’t match">
            Make sure you’re wearing headphones (not speakers), both sides are on, and left is on your left ear, then try again.
          </Callout>
          <Button onClick={retry}>Try again</Button>
        </div>
      )}
    </Card>
  );
}

function Details({ session, onSubmit }: { session: Session; onSubmit: (nickname: string, device: string) => void }) {
  const [nick, setNick] = useState(session.nickname);
  const [device, setDevice] = useState(session.device_note);
  return (
    <Card className="p-5 sm:p-6">
      <SectionTitle eyebrow="Step 3" title="About you (optional)">
        A nickname lets us tell your responses apart if you send them in; the device note helps interpret the results. Both can be left empty.
      </SectionTitle>
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit(nick.trim().slice(0, 40), device.trim().slice(0, 200));
        }}
      >
        <label className="block">
          <span className="text-[12px] font-medium text-fg-2">Nickname</span>
          <input
            className="mt-1 block h-10 w-full max-w-sm rounded-lg border border-line bg-surface-2 px-3 text-[14px] text-fg placeholder:text-fg-3"
            value={nick}
            maxLength={40}
            autoComplete="nickname"
            placeholder="e.g. maple"
            onChange={(e) => setNick(e.target.value)}
          />
        </label>
        <label className="block">
          <span className="text-[12px] font-medium text-fg-2">Headphones and device</span>
          <input
            className="mt-1 block h-10 w-full rounded-lg border border-line bg-surface-2 px-3 text-[14px] text-fg placeholder:text-fg-3"
            value={device}
            maxLength={200}
            placeholder="e.g. AirPods Pro on a MacBook, quiet room"
            onChange={(e) => setDevice(e.target.value)}
          />
        </label>
        <Button type="submit" variant="primary">
          Continue
        </Button>
      </form>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------

function Trials({ index, session, update, player, onFinish }: { index: ListeningIndex; session: Session; update: (f: (s: Session) => Session) => void; player: ClipPlayer; onFinish: () => void }) {
  const t = session.plan[session.current];
  const trial = useMemo(() => index.trials.find((x) => x.id === t?.id), [index, t]);

  // Start the trial clock the first time a trial is shown.
  useEffect(() => {
    if (t && session.trial_started_at[t.id] === undefined) update((s) => ({ ...s, trial_started_at: { ...s.trial_started_at, [t.id]: Date.now() } }));
  }, [t, session.trial_started_at, update]);

  // Prefetch the next trial's clips.
  const upcoming = session.plan[session.current + 1];
  useEffect(() => {
    const nt = upcoming && index.trials.find((x) => x.id === upcoming.id);
    if (nt) for (const c of upcoming.order) void player.load(assetUrl(nt.files[c])).catch(() => undefined);
  }, [index, upcoming, player]);

  if (!t || !trial) return <Callout tone="danger">This trial could not be found. Use “Start over” to reset.</Callout>;
  const last = session.current === session.plan.length - 1;
  const scores = Object.fromEntries(Object.entries(session.ratings[t.id] ?? {}).map(([c, r]) => [c, r.score]));
  return (
    <TrialView
      key={t.id}
      player={player}
      title={`Trial ${session.current + 1} of ${session.plan.length}`}
      progress={(session.current + (trialComplete(session, t) ? 1 : 0)) / session.plan.length}
      order={t.order}
      files={trial.files}
      listened={session.listened[t.id] ?? []}
      scores={scores}
      onListened={(c) => update((s) => ({ ...s, listened: { ...s.listened, [t.id]: Array.from(new Set([...(s.listened[t.id] ?? []), c])) } }))}
      onRate={(c, v) => update((s) => rate(s, t.id, c, v))}
      prev={session.current > 0 ? () => update((s) => ({ ...s, current: s.current - 1 })) : undefined}
      next={{
        label: last ? 'Finish' : 'Next trial',
        onClick: () => {
          if (last) onFinish();
          else {
            update((s) => ({ ...s, current: s.current + 1 }));
            window.scrollTo({ top: 0 });
          }
        },
      }}
    />
  );
}

function PlayProgress({ player }: { player: ClipPlayer }) {
  const bar = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let id = 0;
    const tick = () => {
      if (bar.current) bar.current.style.width = `${player.progress() * 100}%`;
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [player]);
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-surface-3" aria-hidden>
      <div ref={bar} className="h-full rounded-full bg-accent" style={{ width: 0 }} />
    </div>
  );
}

function TrialView({
  player,
  title,
  subtitle,
  progress,
  order,
  files,
  listened,
  scores,
  onListened,
  onRate,
  prev,
  next,
}: {
  player: ClipPlayer;
  title: string;
  subtitle?: string;
  progress?: number;
  order: string[];
  files: Record<string, string>;
  listened: string[];
  scores: Record<string, number>;
  onListened: (condition: string) => void;
  onRate: (condition: string, score: number) => void;
  prev?: () => void;
  next: { label: string; onClick: () => void };
}) {
  const playing = useSyncExternalStore(player.subscribe, player.getPlaying);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let alive = true;
    setReady(false);
    setLoadErr(null);
    Promise.all(order.map((c) => player.load(assetUrl(files[c]))))
      .then(() => alive && setReady(true))
      .catch((e: unknown) => alive && setLoadErr(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
      player.stop();
    };
  }, [order, files, player]);

  const toggle = (c: string) => {
    if (playing === c) return player.stop();
    void player.play(c, assetUrl(files[c]), () => onListened(c)).catch((e: unknown) => setLoadErr(e instanceof Error ? e.message : String(e)));
  };

  const done = order.every((c) => typeof scores[c] === 'number');
  const remaining = order.filter((c) => typeof scores[c] !== 'number').length;

  return (
    <Card className="p-4 sm:p-6" aria-label={title}>
      <div className="mb-4">
        <h2 className="text-[17px] font-semibold tracking-tight">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-fg-2">{subtitle}</p>}
        {progress !== undefined && (
          <div className="mt-3">
            <ProgressBar value={progress} label="Test progress" />
          </div>
        )}
        <p className="mt-3 text-[13px] text-fg-3">Play each version to the end, then rate its overall quality. You can replay and change ratings as often as you like.</p>
      </div>
      {loadErr && (
        <Callout tone="danger" className="mb-4" title="A clip failed to load">
          {loadErr}. Check your connection and reload the page; your progress is saved.
        </Callout>
      )}
      <ul className="space-y-3">
        {order.map((c, i) => {
          const label = positionLabel(i);
          const heard = listened.includes(c);
          const isPlaying = playing === c;
          return (
            <li key={c} className={cx('rounded-xl border p-3 sm:p-4', isPlaying ? 'border-accent bg-accent-soft' : 'border-line bg-surface-2/50')}>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => toggle(c)}
                  disabled={!ready && !loadErr}
                  className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-fg text-bg transition-transform hover:scale-105 active:scale-95 disabled:opacity-40"
                  aria-label={isPlaying ? `Stop version ${label}` : `Play version ${label}`}
                >
                  {isPlaying ? <IconPause size={16} /> : <IconPlay size={16} className="translate-x-px" />}
                </button>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[15px] font-semibold">Version {label}</span>
                    {heard ? (
                      <Badge tone="accent">
                        <IconCheck size={10} /> heard
                      </Badge>
                    ) : (
                      <span className="text-[12px] text-fg-3">{ready ? 'not played yet' : 'loading…'}</span>
                    )}
                  </div>
                  <div className="mt-1.5 h-1">{isPlaying && <PlayProgress player={player} />}</div>
                </div>
              </div>
              <fieldset disabled={!heard} className="mt-3 min-w-0">
                <legend className="sr-only">Overall quality of version {label}</legend>
                <div className="grid grid-cols-5 gap-1.5">
                  {ACR_LABELS.map((name, k) => {
                    const v = k + 1;
                    const sel = scores[c] === v;
                    return (
                      <label
                        key={v}
                        className={cx(
                          'flex cursor-pointer flex-col items-center rounded-lg border px-1 py-1.5 text-center transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent',
                          sel ? 'border-accent bg-accent text-accent-ink' : 'border-line bg-surface text-fg-2 hover:border-line-strong',
                          !heard && 'cursor-not-allowed opacity-45',
                        )}
                      >
                        <input type="radio" className="sr-only" name={`rate-${title}-${c}`} value={v} checked={sel} onChange={() => onRate(c, v)} />
                        <span className="tnum text-[15px] font-semibold">{v}</span>
                        <span className="text-[10.5px] leading-tight sm:text-[11px]">{name}</span>
                      </label>
                    );
                  })}
                </div>
                {!heard && <p className="mt-1.5 text-[11px] text-fg-3">Listen to the whole clip to unlock the rating.</p>}
              </fieldset>
            </li>
          );
        })}
      </ul>
      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        {prev ? (
          <Button variant="ghost" onClick={prev}>
            Previous
          </Button>
        ) : (
          <span />
        )}
        <div className="flex items-center gap-3">
          {!done && <span className="text-[12px] text-fg-3">{remaining} left to rate</span>}
          <Button variant="primary" disabled={!done} onClick={next.onClick}>
            {next.label}
          </Button>
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------

type SendState = { kind: 'idle' } | { kind: 'sending' } | { kind: 'sent' } | { kind: 'failed'; message: string };

function Done({ index, session, update, onRestart }: { index: ListeningIndex; session: Session; update: (f: (s: Session) => Session) => void; onRestart: () => void }) {
  const endpoint = import.meta.env.VITE_LISTEN_ENDPOINT;
  const payload = useMemo(() => buildExport(session, index.conditions, typeof navigator !== 'undefined' ? navigator.userAgent : undefined), [session, index.conditions]);
  const [send, setSend] = useState<SendState>(session.submitted ? { kind: 'sent' } : { kind: 'idle' });
  const attempted = useRef(false);

  const submit = useCallback(async () => {
    if (!endpoint) return;
    setSend({ kind: 'sending' });
    try {
      const r = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(payload) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setSend({ kind: 'sent' });
      update((s) => ({ ...s, submitted: true }));
    } catch (e) {
      setSend({ kind: 'failed', message: e instanceof Error ? e.message : String(e) });
    }
  }, [endpoint, payload, update]);

  useEffect(() => {
    if (!endpoint || session.submitted || attempted.current) return;
    attempted.current = true;
    void submit();
  }, [endpoint, session.submitted, submit]);

  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = exportFilename(session);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  return (
    <Card className="p-5 sm:p-6">
      <Badge tone="accent" className="mb-3">
        <IconCheck size={12} /> Complete
      </Badge>
      <h2 className="text-[22px] font-semibold tracking-tight">Thank you!</h2>
      <p className="mt-2 max-w-prose text-[14px] text-fg-2">
        You rated {payload.ratings.length} clips across {session.plan.length} trials. {endpoint ? 'Your responses are sent automatically; you can also download a copy.' : 'Please download your responses and send the file to the person who invited you.'}
      </p>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={download}>
          <IconDownload size={14} /> Download responses
        </Button>
        {endpoint && send.kind === 'failed' && <Button onClick={() => void submit()}>Retry sending</Button>}
      </div>
      {endpoint && (
        <p className="mt-3 text-[13px]" role="status">
          {send.kind === 'sending' && <span className="text-fg-2">Sending…</span>}
          {send.kind === 'sent' && <span className="text-ok">Responses sent. Thanks!</span>}
          {send.kind === 'failed' && <span className="text-danger">Sending failed ({send.message}). Please retry or download the file instead.</span>}
        </p>
      )}
      <p className="mt-6 text-[12px] text-fg-3">
        Rater ID <code className="font-mono">{session.rater_id}</code>.{' '}
        <button type="button" className="underline underline-offset-2 hover:text-fg" onClick={onRestart}>
          Start a new session
        </button>{' '}
        (for another listener on this device).
      </p>
    </Card>
  );
}
