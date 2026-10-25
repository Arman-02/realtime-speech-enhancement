import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { algorithmicLatencyMs, attenSpec, lookaheadFrames } from '../engine/model';
import { AttenSlider } from '../components/AttenSlider';
import { engine } from '../engine/client';
import type { LiveStats } from '../engine/protocol';
import { liveSupport, startLive, type LiveSession, type PlaybackStats } from '../lib/live';
import { micErrorMessage } from '../lib/recorder';
import { useEngineStatus } from '../lib/useEngine';
import { LevelMeter, ScrollingSpectrogram, type MeterHandle, type ScrollSpecHandle } from '../components/LiveViz';
import { IconHeadphones, IconMic, IconStop } from '../components/icons';
import { Badge, Button, Callout, Card, SectionTitle, Segmented, Slider, Stat } from '../components/ui';
import { cx, fmt } from '../lib/format';

type Phase = { kind: 'idle' } | { kind: 'starting' } | { kind: 'running'; session: LiveSession } | { kind: 'error'; message: string; denied: boolean };

export default function Live() {
  const { status, manifest } = useEngineStatus();
  const algoMs = algorithmicLatencyMs(manifest);
  const lookahead = lookaheadFrames(manifest);
  const atSpec = useMemo(() => attenSpec(manifest), [manifest]);
  const attenSupported = status.kind === 'onnx' && status.attenSupported;
  const [attenUser, setAttenUser] = useState<number | null>(null);
  const atten = attenUser ?? atSpec.default;
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [mode, setMode] = useState<'enhance' | 'bypass'>('enhance');
  const [gainDb, setGainDb] = useState(0);
  const [monitor, setMonitor] = useState(true);
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [pstats, setPstats] = useState<PlaybackStats | null>(null);
  const [hpAck, setHpAck] = useState(false);
  const inMeter = useRef<MeterHandle>(null);
  const outMeter = useRef<MeterHandle>(null);
  const inSpec = useRef<ScrollSpecHandle>(null);
  const outSpec = useRef<ScrollSpecHandle>(null);
  const sessionRef = useRef<LiveSession | null>(null);
  const unsupported = liveSupport();

  useEffect(() => {
    const offViz = engine.onLiveViz((v) => {
      inSpec.current?.push(v.inDb, v.frames);
      outSpec.current?.push(v.outDb, v.frames);
    });
    const offStats = engine.onLiveStats(setStats);
    return () => {
      offViz();
      offStats();
    };
  }, []);

  useEffect(
    () => () => {
      void sessionRef.current?.stop();
      sessionRef.current = null;
    },
    [],
  );

  const start = useCallback(async () => {
    setPhase({ kind: 'starting' });
    setStats(null);
    setPstats(null);
    inSpec.current?.clear();
    outSpec.current?.clear();
    try {
      await engine.ensure();
      const session = await startLive(
        { bypass: mode === 'bypass', gainDb, monitor, attenLimDb: attenSupported ? atten : null },
        {
          onInputLevel: (l) => inMeter.current?.set(l.peak, l.rms),
          onOutputLevel: (l) => outMeter.current?.set(l.peak, l.rms),
          onPlaybackStats: setPstats,
        },
      );
      sessionRef.current = session;
      setPhase({ kind: 'running', session });
    } catch (e) {
      setPhase({ kind: 'error', ...micErrorMessage(e) });
    }
  }, [mode, gainDb, monitor, attenSupported, atten]);

  const stop = useCallback(async () => {
    const s = sessionRef.current;
    sessionRef.current = null;
    setPhase({ kind: 'idle' });
    inMeter.current?.reset();
    outMeter.current?.reset();
    await s?.stop();
  }, []);

  const running = phase.kind === 'running' ? phase.session : null;
  const e2e = running && pstats?.latencyMs != null ? pstats.latencyMs + running.outputLatencyMs + (running.inputLatencyMs ?? 0) : null;
  const budgetPct = stats ? (stats.msPerFrame / 10) * 100 : null;

  return (
    <div className="space-y-6">
      <header className="max-w-2xl">
        <div className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">Live</div>
        <h1 className="text-[28px] font-semibold tracking-tight sm:text-[34px]">Real-time mic denoising</h1>
        <p className="mt-2 text-[15px] text-fg-2">
          Your microphone feeds an AudioWorklet, which resamples to 16 kHz and streams 10 ms hops to a Web Worker running the network. The enhanced audio comes straight back to your ears. Browser echo cancellation, noise suppression and auto-gain are switched off so you hear only this model.
        </p>
      </header>

      {unsupported && (
        <Callout tone="danger" title="Live mode isn’t available here">
          {unsupported} Try a recent Chrome, Edge, Firefox or Safari.
        </Callout>
      )}

      {!hpAck && !unsupported && (
        <Callout
          tone="warn"
          title="Use headphones"
          action={
            <Button size="sm" variant="ghost" onClick={() => setHpAck(true)}>
              Got it
            </Button>
          }
        >
          Without headphones your speakers feed back into the mic. The model removes noise, not echo, so you will get a howl.
        </Callout>
      )}

      {phase.kind === 'error' && (
        <Callout tone={phase.denied ? 'warn' : 'danger'} title={phase.denied ? 'Microphone access denied' : 'Could not start the microphone'}>
          {phase.message}
        </Callout>
      )}

      <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
        <Card className="p-5" aria-label="Controls">
          <div className="flex items-center gap-3">
            {running ? (
              <Button variant="danger" size="lg" className="w-full" onClick={() => void stop()}>
                <IconStop size={16} /> Stop
              </Button>
            ) : (
              <Button variant="primary" size="lg" className="w-full" disabled={!!unsupported || phase.kind === 'starting' || status.kind === 'loading'} onClick={() => void start()}>
                <IconMic size={18} /> {phase.kind === 'starting' ? 'Starting…' : status.kind === 'loading' ? 'Loading model…' : 'Start microphone'}
              </Button>
            )}
          </div>
          {running && (
            <p className="mt-3 flex items-center gap-2 truncate text-[12px] text-fg-3">
              <span className="pulse-dot h-1.5 w-1.5 shrink-0 rounded-full bg-danger" /> {running.deviceLabel} · {running.sampleRate / 1000} kHz
            </p>
          )}

          <div className="mt-6 space-y-5">
            <div>
              <div className="mb-2 text-[12px] font-medium text-fg-2">Processing</div>
              <Segmented<'enhance' | 'bypass'>
                label="Processing mode"
                value={mode}
                onChange={(m) => {
                  setMode(m);
                  running?.setBypass(m === 'bypass');
                }}
                options={[
                  { value: 'enhance', label: 'Enhance', swatch: 'var(--enhanced)' },
                  { value: 'bypass', label: 'Bypass', swatch: 'var(--noisy)' },
                ]}
              />
              <p className="mt-2 text-[12px] text-fg-3">Bypass keeps the model running (same latency) but plays the unprocessed signal, so the switch is seamless.</p>
            </div>
            <AttenSlider
              spec={atSpec}
              supported={attenSupported}
              reason={status.kind === 'loading' ? 'Loading the model…' : status.kind === 'bypass' ? 'No model is loaded (bypass mode).' : undefined}
              value={atten}
              onChange={(db) => {
                setAttenUser(db);
                running?.setAttenLimit(db);
              }}
            />
            <Slider
              label="Output gain"
              min={-24}
              max={12}
              step={1}
              value={gainDb}
              onChange={(v) => {
                setGainDb(v);
                running?.setGainDb(v);
              }}
              format={(v) => `${v > 0 ? '+' : ''}${v} dB`}
            />
            <label className="flex cursor-pointer items-center justify-between gap-3 text-[13px]">
              <span className="flex items-center gap-2 text-fg-2">
                <IconHeadphones size={16} /> Monitor output
              </span>
              <input
                type="checkbox"
                className="peer sr-only"
                checked={monitor}
                onChange={(e) => {
                  setMonitor(e.target.checked);
                  running?.setMonitor(e.target.checked);
                }}
              />
              <span aria-hidden className={cx('relative h-5 w-9 rounded-full transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent', monitor ? 'bg-accent' : 'bg-surface-3')}>
                <span className={cx('absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform', monitor ? 'translate-x-4' : 'translate-x-0.5')} />
              </span>
            </label>
            <div className="space-y-4 border-t border-line pt-5">
              <LevelMeter ref={inMeter} label="Input" color="var(--noisy)" />
              <LevelMeter ref={outMeter} label="Output" color="var(--enhanced)" />
            </div>
          </div>
        </Card>

        <div className="min-w-0 space-y-4">
          <Card className="p-4 sm:p-5" aria-label="Live spectrograms">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-[15px] font-semibold tracking-tight">Spectrogram</h2>
              {running ? <Badge tone={mode === 'enhance' ? 'accent' : 'warn'}>{mode === 'enhance' ? 'Enhancing' : 'Bypassed'}</Badge> : <Badge>Idle</Badge>}
            </div>
            <div className="space-y-2">
              <ScrollingSpectrogram ref={inSpec} label="Mic input" />
              <ScrollingSpectrogram ref={outSpec} label={mode === 'enhance' ? 'Enhanced output' : 'Output (bypass)'} />
            </div>
            <p className="mt-2 text-right text-[11px] text-fg-3">Time scrolls right to left, about 2 px per 10 ms</p>
          </Card>

          <Card className="p-4 sm:p-5" aria-label="Measured performance">
            <SectionTitle title="Measured on this device">Numbers update every half second while the mic is running.</SectionTitle>
            <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-4">
              <Stat label="Inference" value={stats ? fmt(stats.msPerFrame, 2) : '—'} unit="ms/frame" hint="Mean worker time per 10 ms hop (STFT + model + iSTFT)" />
              <Stat label="p95" value={stats ? fmt(stats.p95MsPerFrame, 2) : '—'} unit="ms" />
              <Stat label="End-to-end" value={e2e != null ? fmt(e2e, 0) : '—'} unit="ms" hint={`Measured capture→render time inside the AudioContext (includes the ${fmt(algoMs, 0)} ms algorithmic latency, worker hop and jitter buffer), plus the output and input latency the browser reports`} />
              <Stat label="Underruns" value={pstats ? pstats.underruns : '—'} tone={pstats && pstats.underruns > 0 ? 'bad' : undefined} hint="Times the playback buffer ran dry" />
            </div>
            <div className="mt-5">
              <div className="mb-1.5 flex justify-between text-[12px] text-fg-2">
                <span>Compute budget per 10 ms hop</span>
                <span className="tnum font-mono text-fg-3">{budgetPct != null ? `${fmt(budgetPct, 0)}%` : '—'}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-3">
                <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${Math.min(100, budgetPct ?? 0)}%`, background: (budgetPct ?? 0) > 80 ? 'var(--danger)' : 'var(--accent)' }} />
              </div>
            </div>
            {running && (
              <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-line pt-4 text-[12px] sm:grid-cols-3">
                <Row k={lookahead ? `Algorithmic (window + ${lookahead} lookahead)` : 'Algorithmic (window)'} v={`${fmt(algoMs, 0)} ms`} />
                <Row k="Pipeline (measured)" v={pstats?.latencyMs != null ? `${fmt(pstats.latencyMs, 1)} ms` : '…'} />
                <Row k="Jitter buffer" v={pstats ? `${fmt(pstats.bufferedMs, 1)} ms` : '…'} />
                <Row k="Output (reported)" v={`${fmt(running.outputLatencyMs, 1)} ms`} />
                <Row k="Input (reported)" v={running.inputLatencyMs != null ? `${fmt(running.inputLatencyMs, 1)} ms` : 'n/a'} />
                <Row k="Dropped blocks" v={stats ? String(stats.dropped) : '0'} />
              </dl>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2 py-0.5">
      <dt className="text-fg-3">{k}</dt>
      <dd className="tnum font-mono text-fg-2">{v}</dd>
    </div>
  );
}
