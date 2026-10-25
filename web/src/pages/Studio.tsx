import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type DragEvent } from 'react';
import { engine } from '../engine/client';
import { logSpectrogram } from '../engine/dsp';
import { algorithmicLatencyMs, attenSpec } from '../engine/model';
import { decodeTo16k, encodeWav, formatTime, MAX_SECONDS, siSdr, audioSupported } from '../lib/audio';
import { sampleUrl, useResults, useSamples, type MetricMap, type Sample } from '../lib/data';
import { AttenSlider } from '../components/AttenSlider';
import { AbPlayer, type Track } from '../lib/player';
import { useRecorder } from '../lib/recorder';
import { useEngineStatus } from '../lib/useEngine';
import { StackedSpectrograms, type SpecTrack } from '../components/Spectrogram';
import { IconDownload, IconMic, IconPause, IconPlay, IconStop, IconUpload, IconWave } from '../components/icons';
import { Badge, Button, Callout, Card, Kbd, ProgressBar, SectionTitle, Segmented, Stat } from '../components/ui';
import { compact, cx, displayModelName, fmt, formatAtten } from '../lib/format';

interface Clip {
  name: string;
  source: 'upload' | 'record' | 'sample';
  noisy: Float32Array;
  clean: Float32Array | null;
  sample?: Sample;
  originalRate: number;
  originalSeconds: number;
  truncated: boolean;
}

interface Processed {
  enhanced: Float32Array;
  msPerFrame: number;
  modelMsPerFrame: number;
  rtf: number;
  frames: number;
  elapsedMs: number;
  model: string;
  /** Attenuation limit used (null = model without the input). */
  attenLimDb: number | null;
}

/** Debounce for re-processing while the attenuation slider moves. */
const REPROCESS_DEBOUNCE_MS = 250;

const COLORS: Record<Track, string> = { noisy: 'var(--noisy)', enhanced: 'var(--enhanced)', clean: 'var(--clean)' };
const LABELS: Record<Track, string> = { noisy: 'Noisy', enhanced: 'Enhanced', clean: 'Clean' };

function usePlayer(): AbPlayer {
  const ref = useRef<AbPlayer | null>(null);
  ref.current ??= new AbPlayer();
  useEffect(() => {
    const p = ref.current;
    return () => p?.pause();
  }, []);
  return ref.current;
}

function usePlayerState(player: AbPlayer) {
  const subscribe = useCallback((l: () => void) => player.subscribe(l), [player]);
  const playing = useSyncExternalStore(subscribe, () => player.playing);
  const mode = useSyncExternalStore(subscribe, () => player.mode);
  const duration = useSyncExternalStore(subscribe, () => player.duration);
  return { playing, mode, duration };
}

export default function Studio() {
  const { status, manifest } = useEngineStatus();
  const samples = useSamples();
  const results = useResults();
  const player = usePlayer();
  const { playing, mode, duration } = usePlayerState(player);
  const atSpec = useMemo(() => attenSpec(manifest), [manifest]);
  const attenSupported = status.kind === 'onnx' && status.attenSupported;
  const attenReason =
    status.kind === 'loading'
      ? 'Loading the model…'
      : status.kind === 'bypass'
        ? 'No model is loaded (bypass mode).'
        : 'This model was exported without the attenuation-limit input (manifest v0.1), so the limit cannot be changed.';
  const [attenUser, setAttenUser] = useState<number | null>(null);
  const atten = attenUser ?? atSpec.default;
  const attenRef = useRef(atten);
  attenRef.current = atten;
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [clip, setClip] = useState<Clip | null>(null);
  const [processed, setProcessed] = useState<Processed | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const workspaceRef = useRef<HTMLDivElement>(null);

  /**
   * Run the clip through the worker. With `keep`, the previous result stays audible (and the playhead
   * where it is) until the new one replaces it, e.g. while the attenuation limit is being dragged.
   */
  const process = useCallback(
    async (c: Clip, keep = false) => {
      cancelRef.current?.();
      if (!keep) setProcessed(null);
      setProgress(0);
      setError(null);
      const st0 = engine.status;
      const attenLimDb = st0.kind === 'onnx' && st0.attenSupported ? attenRef.current : null;
      const job = engine.enhance(c.noisy, (done, total) => setProgress(total ? done / total : 0), attenLimDb);
      const cancel = job.cancel;
      cancelRef.current = cancel;
      try {
        const r = await job.promise;
        const st = engine.status;
        setProcessed({
          enhanced: r.audio,
          msPerFrame: r.stats.frames ? r.stats.totalMs / r.stats.frames : 0,
          modelMsPerFrame: r.stats.frames ? r.stats.modelMs / r.stats.frames : 0,
          rtf: r.rtf,
          frames: r.stats.frames,
          elapsedMs: r.elapsedMs,
          model: st.kind === 'onnx' ? `${displayModelName(st.name)} v${st.version}` : 'Bypass (identity)',
          attenLimDb,
        });
        player.setTracks({ enhanced: r.audio });
      } catch (e) {
        if (!(e instanceof Error && e.name === 'AbortError')) setError(`Processing failed: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        // A newer job (started by the attenuation slider) may already own the progress bar.
        if (cancelRef.current === cancel) {
          setProgress(null);
          cancelRef.current = null;
        }
      }
    },
    [player],
  );

  const onAttenChange = useCallback(
    (db: number) => {
      setAttenUser(db);
      attenRef.current = db;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (!clip) return;
      debounceRef.current = setTimeout(() => {
        debounceRef.current = null;
        void process(clip, true);
      }, REPROCESS_DEBOUNCE_MS);
    },
    [clip, process],
  );
  useEffect(
    () => () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    },
    [],
  );

  const loadClip = useCallback(
    async (name: string, source: Clip['source'], data: ArrayBuffer, cleanData?: ArrayBuffer | null, sample?: Sample) => {
      setError(null);
      setLoading(`Decoding ${name}…`);
      try {
        if (!audioSupported()) throw new Error('Web Audio is not supported in this browser.');
        const dec = await decodeTo16k(data);
        if (dec.audio.length < 1600) throw new Error('The clip is too short (under 0.1 s).');
        let clean: Float32Array | null = null;
        if (cleanData) {
          const dc = await decodeTo16k(cleanData);
          clean = dc.audio.slice(0, dec.audio.length);
        }
        const c: Clip = { name, source, noisy: dec.audio, clean, sample, originalRate: dec.originalRate, originalSeconds: dec.originalSeconds, truncated: dec.truncated };
        player.pause();
        player.clear();
        player.setTracks({ noisy: c.noisy, clean: c.clean, enhanced: null });
        player.setMode('enhanced');
        setClip(c);
        setLoading(null);
        requestAnimationFrame(() => workspaceRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
        await process(c);
      } catch (e) {
        setLoading(null);
        setError(e instanceof Error ? (e.name === 'EncodingError' ? 'This file could not be decoded. Try WAV, MP3, M4A or OGG.' : e.message) : String(e));
      }
    },
    [player, process],
  );

  const onFiles = useCallback(
    async (files: FileList | null) => {
      const f = files?.[0];
      if (!f) return;
      if (f.size > 200 * 1024 * 1024) {
        setError('That file is larger than 200 MB. Please trim it first.');
        return;
      }
      await loadClip(f.name, 'upload', await f.arrayBuffer());
    },
    [loadClip],
  );

  const recorder = useRecorder((blob) => {
    void blob.arrayBuffer().then((buf) => loadClip(`Recording ${new Date().toLocaleTimeString()}`, 'record', buf));
  });

  const loadSample = useCallback(
    async (s: Sample) => {
      setError(null);
      setLoading(`Fetching ${s.title}…`);
      try {
        const [n, c] = await Promise.all([
          fetch(sampleUrl(s.noisy)).then((r) => {
            if (!r.ok) throw new Error(`Could not fetch ${s.noisy} (${r.status})`);
            return r.arrayBuffer();
          }),
          s.clean ? fetch(sampleUrl(s.clean)).then((r) => (r.ok ? r.arrayBuffer() : null)) : Promise.resolve(null),
        ]);
        await loadClip(s.title, 'sample', n, c, s);
      } catch (e) {
        setLoading(null);
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [loadClip],
  );

  // Keyboard shortcuts: space = play/pause, 1/2/3 = track, ←/→ handled by the plot.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (!clip || (t && (t.closest('input, textarea, select, [contenteditable]') || (e.key === ' ' && t.closest('button, a, [role="slider"]'))))) return;
      if (e.key === ' ') {
        e.preventDefault();
        player.toggle();
      } else if (e.key === '1') player.setMode('noisy');
      else if (e.key === '2' && player.has('enhanced')) player.setMode('enhanced');
      else if (e.key === '3' && player.has('clean')) player.setMode('clean');
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [clip, player]);

  // Spectrograms
  const specNoisy = useMemo(() => (clip ? logSpectrogram(clip.noisy) : null), [clip]);
  const specClean = useMemo(() => (clip?.clean ? logSpectrogram(clip.clean) : null), [clip]);
  const specEnh = useMemo(() => (processed ? logSpectrogram(processed.enhanced) : null), [processed]);

  const metrics = useMemo(() => {
    if (!clip?.clean || !processed) return null;
    const n = siSdr(clip.noisy, clip.clean);
    const e = siSdr(processed.enhanced, clip.clean);
    return { noisy: n, enhanced: e, delta: e - n };
  }, [clip, processed]);

  const sampleMetrics: Record<string, MetricMap> | null = useMemo(() => {
    const id = clip?.sample?.id;
    if (!id || results.state !== 'ready') return null;
    return results.data.perSample[id] ?? null;
  }, [clip, results]);

  const download = () => {
    if (!clip || !processed) return;
    const url = URL.createObjectURL(encodeWav(processed.enhanced));
    const a = document.createElement('a');
    a.href = url;
    const lim = processed.attenLimDb !== null && processed.attenLimDb < atSpec.max ? `_limit${Math.round(processed.attenLimDb)}dB` : '';
    a.download = `${clip.name.replace(/\.[a-z0-9]+$/i, '').replace(/[^\w.-]+/g, '_')}_enhanced${lim}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  const getTime = useCallback(() => player.currentTime(), [player]);
  const onSeek = useCallback((t: number) => player.seek(t), [player]);

  const effMode = clip ? (player.has(mode) ? mode : 'noisy') : mode;
  const specTracks: SpecTrack[] = clip
    ? (['noisy', 'enhanced', 'clean'] as Track[]).map((t, i) => {
        const s = t === 'noisy' ? specNoisy : t === 'enhanced' ? specEnh : specClean;
        const audio = t === 'noisy' ? clip.noisy : t === 'enhanced' ? (processed?.enhanced ?? null) : clip.clean;
        return {
          key: t,
          label: LABELS[t],
          color: COLORS[t],
          audio,
          db: s?.db ?? null,
          frames: s?.frames ?? 0,
          active: effMode === t,
          onSelect: () => player.setMode(t),
          shortcut: String(i + 1),
          placeholder: t === 'enhanced' ? (progress !== null ? 'Enhancing…' : 'Not processed') : 'No clean reference for this clip',
        };
      })
    : [];

  const recState = recorder.state;
  const params = manifest?.params ?? null;
  const macs = manifest?.macs_per_frame ?? null;

  return (
    <div className="space-y-8">
      {/* Hero */}
      <section className="grid gap-6 lg:grid-cols-[1.25fr_1fr] lg:items-end">
        <div>
          <Badge tone="accent" className="mb-4">
            <IconWave size={12} /> Runs locally · WebAssembly
          </Badge>
          <h1 className="text-[32px] leading-[1.1] font-semibold tracking-tight text-balance sm:text-[44px]">
            Hear the noise <span className="text-fg-3">disappear.</span>
          </h1>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-fg-2">
            A causal convolutional-recurrent network that strips background noise from speech, 10 ms at a time. Drop in a clip, record yourself, or pick a demo, then A/B the result against the original.
          </p>
        </div>
        <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-line bg-line">
          {[
            { k: 'Algorithmic latency', v: fmt(algorithmicLatencyMs(manifest), 0), u: 'ms' },
            { k: 'Parameters', v: params ? compact(params) : '—', u: '' },
            { k: 'MACs / frame', v: macs ? compact(macs) : '—', u: '' },
          ].map((s) => (
            <div key={s.k} className="bg-surface px-4 py-3">
              <dt className="text-[11px] text-fg-3">{s.k}</dt>
              <dd className="tnum mt-1 text-[22px] font-semibold tracking-tight">
                {s.v}
                {s.u && <span className="ml-1 text-[12px] font-medium text-fg-3">{s.u}</span>}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      {/* Inputs */}
      <section aria-labelledby="input-title" className="grid gap-4 lg:grid-cols-[1fr_1.35fr]">
        <Card className="p-5">
          <SectionTitle id="input-title" eyebrow="Input" title="Your audio">
            WAV, MP3, M4A or OGG, up to {MAX_SECONDS} s. Everything is resampled to 16 kHz mono in the browser.
          </SectionTitle>
          <div
            onDragOver={(e: DragEvent) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e: DragEvent) => {
              e.preventDefault();
              setDragOver(false);
              void onFiles(e.dataTransfer.files);
            }}
            className={cx(
              'relative flex flex-col items-center justify-center rounded-xl border border-dashed px-4 py-8 text-center transition-colors',
              dragOver ? 'border-accent bg-accent-soft' : 'border-line-strong bg-surface-2/60',
            )}
          >
            <div className="mb-3 grid h-10 w-10 place-items-center rounded-full bg-surface-3 text-fg-2">
              <IconUpload size={18} />
            </div>
            <p className="text-sm text-fg">
              Drop a file here or{' '}
              <button type="button" className="font-medium text-accent underline-offset-2 hover:underline" onClick={() => fileInput.current?.click()}>
                browse
              </button>
            </p>
            <p className="mt-1 text-[12px] text-fg-3">Files never leave your device.</p>
            <input
              ref={fileInput}
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.oga,.webm,.flac,.aac"
              className="sr-only"
              aria-label="Choose an audio file"
              onChange={(e) => {
                void onFiles(e.target.files);
                e.target.value = '';
              }}
            />
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            {recState.kind === 'recording' ? (
              <Button variant="danger" onClick={recorder.stop}>
                <IconStop size={14} /> Stop recording
                <RecTimer startedAt={recState.startedAt} />
              </Button>
            ) : (
              <Button onClick={() => void recorder.start()} disabled={recState.kind === 'requesting'}>
                <IconMic size={16} /> {recState.kind === 'requesting' ? 'Waiting for permission…' : 'Record'}
              </Button>
            )}
            <span className="text-[12px] text-fg-3">Browser noise suppression is disabled so you hear the raw mic.</span>
          </div>
          {recState.kind === 'error' && (
            <Callout tone={recState.denied ? 'warn' : 'danger'} className="mt-3" title={recState.denied ? 'Microphone blocked' : 'Recording unavailable'}>
              {recState.message}
            </Callout>
          )}
        </Card>

        <Card className="p-5" aria-labelledby="samples-title">
          <SectionTitle id="samples-title" eyebrow="Demo clips" title="VoiceBank+DEMAND test set">
            Real noisy recordings with their clean references, so SI-SDR can be measured right here.
          </SectionTitle>
          <SampleGallery state={samples} onPick={(s) => void loadSample(s)} activeId={clip?.sample?.id} disabled={loading !== null || progress !== null} />
        </Card>
      </section>

      {error && (
        <Callout tone="danger" title="Something went wrong" action={<Button size="sm" variant="ghost" onClick={() => setError(null)}>Dismiss</Button>}>
          {error}
        </Callout>
      )}
      {loading && (
        <div role="status" className="flex items-center gap-3 text-sm text-fg-2">
          <span className="pulse-dot h-2 w-2 rounded-full bg-accent" /> {loading}
        </div>
      )}

      {/* Workspace */}
      <div ref={workspaceRef} className="scroll-mt-24">
        {clip ? (
          <Card className="fade-up p-4 sm:p-5" aria-label="Playback and analysis">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">Clip</div>
                <h2 className="truncate text-[17px] font-semibold tracking-tight">{clip.name}</h2>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  <Badge>{formatTime(clip.noisy.length / 16000)}</Badge>
                  <Badge>{clip.originalRate === 16000 ? '16 kHz' : `${(clip.originalRate / 1000).toFixed(1)} → 16 kHz`}</Badge>
                  {clip.sample?.noise_type && <Badge>{clip.sample.noise_type}</Badge>}
                  {clip.sample?.snr_db !== undefined && <Badge>{clip.sample.snr_db} dB SNR</Badge>}
                  {clip.truncated && <Badge tone="warn">Trimmed to first {MAX_SECONDS} s (of {Math.round(clip.originalSeconds)} s)</Badge>}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {progress === null && (
                  <Button size="sm" variant="ghost" onClick={() => void process(clip)} disabled={status.kind === 'loading'}>
                    Re-run
                  </Button>
                )}
                <Button size="sm" onClick={download} disabled={!processed}>
                  <IconDownload size={14} /> Download WAV
                </Button>
              </div>
            </div>

            {progress !== null && (
              <div className="mt-4 flex items-center gap-3">
                <div className="flex-1">
                  <ProgressBar value={progress} label="Enhancement progress" />
                </div>
                <span className="tnum w-10 text-right font-mono text-[12px] text-fg-2">{Math.round(progress * 100)}%</span>
                <Button size="sm" variant="ghost" onClick={() => cancelRef.current?.()}>
                  Cancel
                </Button>
              </div>
            )}

            {/* Transport */}
            <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-line pt-4">
              <button
                type="button"
                onClick={() => player.toggle()}
                className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-fg text-bg transition-transform hover:scale-105 active:scale-95"
                aria-label={playing ? 'Pause' : 'Play'}
              >
                {playing ? <IconPause size={18} /> : <IconPlay size={18} className="translate-x-px" />}
              </button>
              <Segmented<Track>
                label="Which signal to hear"
                value={effMode}
                onChange={(v) => player.setMode(v)}
                options={(['noisy', 'enhanced', 'clean'] as Track[]).map((t) => ({
                  value: t,
                  label: LABELS[t],
                  swatch: COLORS[t],
                  disabled: !player.has(t),
                  hint: t === 'clean' && !clip.clean ? 'No clean reference for this clip' : undefined,
                }))}
              />
              <AttenSlider
                className="w-full max-w-[260px] min-w-[180px] flex-1"
                spec={atSpec}
                supported={attenSupported}
                reason={attenReason}
                value={atten}
                onChange={onAttenChange}
              />
              <div className="ml-auto hidden items-center gap-1.5 text-[11px] text-fg-3 lg:flex">
                <Kbd>Space</Kbd> play <Kbd>1</Kbd>
                <Kbd>2</Kbd>
                <Kbd>3</Kbd> switch <Kbd>←</Kbd>
                <Kbd>→</Kbd> seek
              </div>
            </div>

            <div className="mt-5">
              <StackedSpectrograms tracks={specTracks} duration={duration} getTime={getTime} onSeek={onSeek} rowHeight={typeof window !== 'undefined' && window.innerWidth < 640 ? 104 : 132} />
            </div>

            {/* Metrics */}
            <div className="mt-6 grid gap-4 border-t border-line pt-5 md:grid-cols-2">
              <div>
                <div className="mb-3 text-[12px] font-medium text-fg-2">Performance on this device</div>
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Per frame" value={processed ? fmt(processed.msPerFrame, 2) : '—'} unit="ms" hint="Average wall time per 10 ms hop (STFT + model + iSTFT) in the worker" />
                  <Stat label="RTF" value={processed ? fmt(processed.rtf, 3) : '—'} hint="Real-time factor: processing time ÷ audio duration (lower is faster; < 1 is real-time)" tone={processed ? (processed.rtf < 1 ? 'good' : 'bad') : undefined} />
                  <Stat label="Frames" value={processed ? processed.frames.toLocaleString() : '—'} hint="10 ms hops processed" />
                </div>
                {processed && (
                  <p className="mt-3 text-[12px] text-fg-3">
                    {processed.model} · model {fmt(processed.modelMsPerFrame, 2)} ms/frame · total {Math.round(processed.elapsedMs)} ms for {fmt(clip.noisy.length / 16000, 1)} s of audio
                  </p>
                )}
              </div>
              <div>
                <div className="mb-3 text-[12px] font-medium text-fg-2">Quality vs. clean reference</div>
                {clip.clean ? (
                  <>
                    <div className="grid grid-cols-3 gap-4">
                      <Stat label="SI-SDR noisy" value={metrics ? fmt(metrics.noisy, 1) : '—'} unit="dB" />
                      <Stat label="SI-SDR enhanced" value={metrics ? fmt(metrics.enhanced, 1) : '—'} unit="dB" />
                      <Stat label="Improvement" value={metrics ? `${metrics.delta >= 0 ? '+' : ''}${fmt(metrics.delta, 1)}` : '—'} unit="dB" tone={metrics ? (metrics.delta > 0.05 ? 'good' : metrics.delta < -0.05 ? 'bad' : 'neutral') : undefined} />
                    </div>
                    {sampleMetrics ? (
                      <SampleMetricTable metrics={sampleMetrics} note={attenSupported && atten !== atSpec.default ? `Offline scores are for the default setting (${formatAtten(atSpec.default, atSpec.max)}); the slider is at ${formatAtten(atten, atSpec.max)}.` : null} />
                    ) : (
                      <p className="mt-3 text-[12px] text-fg-3">
                        SI-SDR is computed in your browser. PESQ and STOI need the reference implementations; see{' '}
                        <a href="#/results" className="text-accent underline-offset-2 hover:underline">
                          Results
                        </a>{' '}
                        for test-set scores.
                      </p>
                    )}
                  </>
                ) : (
                  <p className="text-[13px] text-fg-3">No clean reference for this clip, so objective metrics are unavailable. Pick a demo clip to see SI-SDR.</p>
                )}
              </div>
            </div>
          </Card>
        ) : (
          <EmptyWorkspace />
        )}
      </div>
    </div>
  );
}

function RecTimer({ startedAt }: { startedAt: number }) {
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((x) => x + 1), 200);
    return () => clearInterval(id);
  }, []);
  const s = (performance.now() - startedAt) / 1000;
  return (
    <span className="tnum ml-1 font-mono text-[12px] opacity-90">
      {formatTime(s)} / {MAX_SECONDS}s
    </span>
  );
}

function SampleGallery({
  state,
  onPick,
  activeId,
  disabled,
}: {
  state: ReturnType<typeof useSamples>;
  onPick: (s: Sample) => void;
  activeId?: string;
  disabled: boolean;
}) {
  if (state.state === 'loading') {
    return (
      <div className="grid gap-2 sm:grid-cols-2" aria-busy="true">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-[62px] animate-pulse rounded-lg bg-surface-2" />
        ))}
      </div>
    );
  }
  if (state.state === 'missing' || state.data.length === 0) {
    return (
      <div className="grid min-h-[150px] place-items-center rounded-xl border border-dashed border-line-strong px-4 py-6 text-center">
        <div>
          <p className="text-sm text-fg-2">Demo clips haven’t been generated yet.</p>
          <p className="mt-1 text-[12px] text-fg-3">
            They appear here once <code className="font-mono">samples/samples.json</code> is published. Upload or record in the meantime.
          </p>
        </div>
      </div>
    );
  }
  return (
    <ul className="grid max-h-[288px] gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
      {state.data.map((s) => (
        <li key={s.id}>
          <button
            type="button"
            disabled={disabled}
            onClick={() => onPick(s)}
            aria-pressed={activeId === s.id}
            className={cx(
              'flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors disabled:cursor-wait disabled:opacity-60',
              activeId === s.id ? 'border-accent bg-accent-soft' : 'border-line bg-surface-2/50 hover:border-line-strong hover:bg-surface-2',
            )}
          >
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-surface-3 text-fg-2">
              <IconPlay size={12} />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-medium text-fg">{s.title}</span>
              <span className="block truncate text-[11px] text-fg-3">
                {[s.noise_type, s.snr_db !== undefined ? `${s.snr_db} dB` : null, s.id].filter(Boolean).join(' · ')}
              </span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function SampleMetricTable({ metrics, note }: { metrics: Record<string, MetricMap>; note: string | null }) {
  const models = Object.keys(metrics);
  return (
    <div className="mt-4">
      <table className="w-full text-[12px]" data-testid="per-sample-metrics">
        <caption className="mb-1 text-left text-[11px] text-fg-3">This clip in the offline evaluation (Python reference metrics)</caption>
        <thead>
          <tr className="text-left text-fg-3">
            <th className="py-1 font-medium">Model</th>
            <th className="py-1 text-right font-medium">PESQ</th>
            <th className="py-1 text-right font-medium">STOI</th>
            <th className="py-1 text-right font-medium">SI-SDR</th>
          </tr>
        </thead>
        <tbody className="tnum">
          {models.map((m) => (
            <tr key={m} className="border-t border-line">
              <td className="py-1 text-fg-2">{m ? displayModelName(m) : 'Shipped model'}</td>
              <td className="py-1 text-right">{fmt(metrics[m]?.pesq, 2)}</td>
              <td className="py-1 text-right">{fmt(metrics[m]?.stoi, 3)}</td>
              <td className="py-1 text-right">{metrics[m]?.si_sdr !== undefined ? `${fmt(metrics[m]?.si_sdr, 1)} dB` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {note && <p className="mt-1.5 text-[11px] text-fg-3">{note}</p>}
    </div>
  );
}

function EmptyWorkspace() {
  return (
    <div className="card relative overflow-hidden p-6 sm:p-8">
      <div className="pointer-events-none absolute inset-0 opacity-[0.55]" aria-hidden>
        <svg className="h-full w-full" preserveAspectRatio="none" viewBox="0 0 600 160">
          {Array.from({ length: 120 }, (_, i) => {
            const x = i * 5 + 2;
            const h = 8 + Math.abs(Math.sin(i * 0.37) * 34 + Math.sin(i * 0.11) * 22) * (i > 60 ? 0.55 : 1);
            return <rect key={i} x={x} y={80 - h / 2} width="2" height={h} rx="1" fill={i > 60 ? 'var(--enhanced)' : 'var(--border-strong)'} opacity={i > 60 ? 0.45 : 0.7} />;
          })}
        </svg>
      </div>
      <div className="relative flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-[17px] font-semibold tracking-tight">No clip loaded</h2>
          <p className="mt-1 text-sm text-fg-2">Upload, record or choose a demo clip above. Spectrograms, A/B playback and metrics appear here.</p>
        </div>
      </div>
    </div>
  );
}
