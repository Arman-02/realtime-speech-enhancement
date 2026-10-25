import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { analysisFrame, HOP, N_BINS, N_FFT } from '../engine/dsp';
import { engine } from '../engine/client';
import { MAGMA } from '../lib/colormap';
import { useEngineStatus } from '../lib/useEngine';
import { lookaheadFrames } from '../engine/model';
import { AttenLimitPlot, DistillationDiagram } from '../components/Distillation';
import { Card, SectionTitle } from '../components/ui';
import { fmt } from '../lib/format';

export default function HowItWorks() {
  return (
    <div className="space-y-8">
      <header className="max-w-2xl">
        <div className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">How it works</div>
        <h1 className="text-[28px] font-semibold tracking-tight sm:text-[34px]">Ten milliseconds at a time</h1>
        <p className="mt-2 text-[15px] text-fg-2">
          The enhancer never sees the future. Every 10 ms it takes the newest 20 ms of audio, predicts a complex mask for that one spectral frame, applies it, and hands back 10 ms of clean audio. All of its memory lives in one small recurrent state tensor.
        </p>
      </header>

      <Card className="p-4 sm:p-6" aria-labelledby="arch-title">
        <SectionTitle id="arch-title" eyebrow="Architecture" title="Signal path and the CRN">
          The top row runs per 10 ms hop in the browser. The lower panel opens up the network: frequency-only convolutions, a GRU over time, and a mirrored decoder with skip connections.
        </SectionTitle>
        <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
          <ArchitectureDiagram />
        </div>
      </Card>

      <Card className="p-4 sm:p-6" aria-labelledby="kd-title">
        <SectionTitle id="kd-title" eyebrow="Training" title="Distillation: borrowing hindsight from a teacher">
          A causal model has to decide about each frame without hearing what comes next. A much larger bidirectional teacher can listen ahead, so it makes better masks. Its outputs become a second training target for the small causal students, which keep their causality and size.
        </SectionTitle>
        <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
          <DistillationDiagram />
        </div>
        <p className="mt-4 max-w-3xl text-[13px] text-fg-2">
          Each student size is trained twice, with and without the teacher, under the same config and seeds, so the gain from distillation can be measured. The Results page plots quality against compute for both arms and reports a gain only where a paired Wilcoxon test gives p &lt; 0.05.
        </p>
      </Card>

      <Card className="p-5 sm:p-6" aria-labelledby="atten-title">
        <SectionTitle id="atten-title" eyebrow="Control" title="The attenuation limit: removing less to sound better" />
        <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
          <div className="space-y-3 text-[14px] leading-relaxed text-fg-2">
            <p>
              Where the network is confident that a time-frequency bin is noise, it pushes the mask toward zero. Between speech harmonics that means deep cuts that change every 10 ms, and the leftover noise turns into a smeared, watery residue: the familiar <strong className="text-fg">“underwater”</strong> sound of aggressive suppression.
            </p>
            <p>
              The attenuation limit puts a <strong className="text-fg">floor under the mask magnitude</strong>: |<Var>M</Var>| ≥ 10<sup>−lim/20</sup>, with the phase kept. At 12 dB no bin is turned down by more than a factor of four. A little steady background survives, which masks the artifacts and keeps the voice natural, at the cost of some noise reduction.
            </p>
            <p>
              The floor is a parameter-free clamp after the network, so it needs no retraining. It is an input of the exported model, which is why the <strong className="text-fg">Max. attenuation</strong> slider in the Studio and Live pages works instantly. The Results page measures the trade-off as an attenuation sweep.
            </p>
          </div>
          <AttenLimitPlot />
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-5 sm:p-6" aria-labelledby="mask-title">
          <SectionTitle id="mask-title" eyebrow="STFT → mask" title="Masking in the time-frequency domain" />
          <div className="space-y-3 text-[14px] leading-relaxed text-fg-2">
            <p>
              Each hop, a 320-sample (20 ms) frame is multiplied by a square-root Hann window and transformed with a real FFT into <strong className="text-fg">161 complex bins</strong> from 0 to 8 kHz. Speech energy sits in harmonics and formants; most noise is spread out.
            </p>
            <p>
              The network predicts a <strong className="text-fg">complex ratio mask</strong> <Var>M</Var> per bin, bounded so that |<Var>M</Var>| ≤ 1.2, and the output is the complex product <Var>Y = M ⊙ X</Var>. Being complex, the mask can correct phase as well as attenuate. An inverse FFT, the same window, and 50 % overlap-add turn <Var>Y</Var> back into audio; the squared window sums to exactly one, so a mask of 1 reconstructs the input perfectly.
            </p>
          </div>
          <MaskIllustration />
        </Card>

        <Card className="p-5 sm:p-6" aria-labelledby="latency-title">
          <SectionTitle id="latency-title" eyebrow="Latency budget" title="Where the milliseconds go" />
          <LatencyBudget />
          <ul className="mt-5 space-y-2 text-[13px] text-fg-2">
            <li>
              <strong className="text-fg">Algorithmic, 20 ms.</strong> One window: a sample can only leave once the frame that completes its overlap-add has been seen. A model with lookahead adds 10 ms per future frame it waits for; the shipped model uses none.
            </li>
            <li>
              <strong className="text-fg">Inference.</strong> Must finish inside the 10 ms hop. The number shown is measured on your device.
            </li>
            <li>
              <strong className="text-fg">Audio I/O.</strong> 128-sample render quanta for capture and playback, a small jitter buffer between the worker and the output, and whatever your OS and device add.
            </li>
          </ul>
        </Card>
      </div>

      <Card className="p-5 sm:p-6" aria-labelledby="parity-title">
        <SectionTitle id="parity-title" eyebrow="Engineering" title="One DSP contract, two languages" />
        <div className="grid gap-5 text-[13px] text-fg-2 md:grid-cols-3">
          <div>
            <h3 className="mb-1 font-medium text-fg">Streaming equals offline</h3>
            <p>Convolutions only run along frequency, so the only temporal memory is the GRU state. Frame-by-frame inference reproduces whole-utterance inference exactly, and tests enforce it in Python and here.</p>
          </div>
          <div>
            <h3 className="mb-1 font-medium text-fg">Python ↔ JS parity</h3>
            <p>
              The export writes test vectors (STFT frames, model outputs, the full streamed waveform). The JS DSP is a hand-written mixed-radix FFT, since 320 = 4·4·4·5, and it matches numpy to about 1e-6. Check it yourself on the{' '}
              <a className="text-accent underline-offset-2 hover:underline" href="#/diagnostics">
                diagnostics page
              </a>
              .
            </p>
          </div>
          <div>
            <h3 className="mb-1 font-medium text-fg">Off the main thread</h3>
            <p>Audio flows AudioWorklet → Web Worker (ONNX Runtime, WebAssembly, one thread) → AudioWorklet over direct message ports, so the UI never sits in the audio path. No server, no uploads.</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

function Var({ children }: { children: ReactNode }) {
  return <span className="font-serif text-[15px] text-fg italic">{children}</span>;
}

// ---------------------------------------------------------------------------------------------

function Box({ x, y, w, h, title, sub, accent }: { x: number; y: number; w: number; h: number; title: string; sub?: string; accent?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={accent ? 'var(--accent-soft)' : 'var(--surface-2)'} stroke={accent ? 'var(--accent)' : 'var(--border-strong)'} strokeWidth={1} />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 3 : h / 2 + 4)} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--text)">
        {title}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={10.5} fill="var(--text-3)">
          {sub}
        </text>
      )}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, dashed }: { x1: number; y1: number; x2: number; y2: number; dashed?: boolean }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--text-3)" strokeWidth={1.4} markerEnd="url(#arrow)" strokeDasharray={dashed ? '4 4' : undefined} />;
}

function ArchitectureDiagram() {
  const top = [
    { t: 'Microphone', s: 'device rate' },
    { t: 'Resample', s: '→ 16 kHz mono' },
    { t: 'STFT', s: '320 win · 160 hop' },
    { t: 'CRN', s: 'ONNX · 1 frame/call', accent: true },
    { t: 'Mask ⊙ X', s: 'complex multiply' },
    { t: 'iSTFT + OLA', s: '10 ms out / hop' },
    { t: 'Speaker', s: 'device rate' },
  ];
  const bw = 116;
  const gap = 20;
  const x0 = 12;
  const by = 54;
  const bh = 56;
  const bx = (i: number) => x0 + i * (bw + gap);
  const fh = (f: number) => 18 + f * 0.62; // block height for a frequency size
  const cy = 300;
  const enc = [
    { x: 120, f: 81 },
    { x: 172, f: 41 },
    { x: 224, f: 21 },
    { x: 276, f: 11 },
  ];
  const dec = [
    { x: 640, f: 21 },
    { x: 692, f: 41 },
    { x: 744, f: 81 },
    { x: 796, f: 161 },
  ];
  const bwid = 24;
  return (
    <svg viewBox="0 0 960 440" className="w-full min-w-[760px]" role="img" aria-labelledby="arch-svg-title arch-svg-desc">
      <title id="arch-svg-title">Speech enhancer architecture</title>
      <desc id="arch-svg-desc">
        Microphone, resample to 16 kHz, STFT, CRN network, complex mask multiplied with the noisy spectrum, inverse STFT with overlap-add, speaker. The CRN consists of feature compression, four frequency-only convolution encoder blocks reducing 161 bins to 11, a linear layer and two-layer GRU with recurrent state, and four transposed-convolution decoder blocks back to 161 bins with skip connections, followed by a mask head.
      </desc>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--text-3)" />
        </marker>
        <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>

      {/* noisy spectrum bypass X */}
      <path d={`M${bx(2) + bw / 2},${by} C${bx(2) + bw / 2},${by - 34} ${bx(4) + bw / 2},${by - 34} ${bx(4) + bw / 2},${by - 2}`} fill="none" stroke="var(--noisy)" strokeWidth={1.4} markerEnd="url(#arrow)" />
      <text x={(bx(2) + bx(4) + bw) / 2} y={by - 30} textAnchor="middle" fontSize={11} fill="var(--text-2)">
        X — noisy spectrum (161 complex bins)
      </text>

      {top.map((b, i) => (
        <g key={b.t}>
          <Box x={bx(i)} y={by} w={bw} h={bh} title={b.t} sub={b.s} accent={b.accent} />
          {i < top.length - 1 && <Arrow x1={bx(i) + bw + 2} y1={by + bh / 2} x2={bx(i + 1) - 3} y2={by + bh / 2} />}
        </g>
      ))}
      <text x={bx(3) + bw / 2} y={by + bh + 16} textAnchor="middle" fontSize={10.5} fill="var(--accent)">
        M — complex ratio mask
      </text>

      {/* zoom lines */}
      <path d={`M${bx(3)},${by + bh} L14,170 M${bx(3) + bw},${by + bh} L946,170`} stroke="var(--border-strong)" strokeDasharray="3 4" fill="none" />
      <rect x={14} y={170} width={932} height={262} rx={14} fill="var(--surface-2)" fillOpacity={0.45} stroke="var(--border)" />
      <text x={30} y={194} fontSize={12} fontWeight={600} fill="var(--text)">
        CRN · one frame in, one frame out
      </text>
      <text x={30} y={210} fontSize={10.5} fill="var(--text-3)">
        block height ∝ frequency resolution
      </text>

      {/* input features */}
      <rect x={52} y={cy - fh(161) / 2} width={bwid} height={fh(161)} rx={5} fill="var(--noisy)" fillOpacity={0.85} />
      <text x={64} y={cy + fh(161) / 2 + 16} textAnchor="middle" fontSize={10.5} fill="var(--text-2)">
        features
      </text>
      <text x={64} y={cy + fh(161) / 2 + 29} textAnchor="middle" fontSize={9.5} fill="var(--text-3)">
        |X|^0.3, X/|X|^0.7
      </text>

      {/* encoder */}
      {enc.map((b, i) => (
        <g key={`e${i}`}>
          <Arrow x1={i === 0 ? 78 : enc[i - 1].x + bwid + 2} y1={cy} x2={b.x - 3} y2={cy} />
          <rect x={b.x} y={cy - fh(b.f) / 2} width={bwid} height={fh(b.f)} rx={5} fill="var(--clean)" fillOpacity={0.75 + i * 0.06} />
          <text x={b.x + bwid / 2} y={cy - fh(b.f) / 2 - 6} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="tnum">
            {b.f}
          </text>
        </g>
      ))}
      <text x={210} y={420} textAnchor="middle" fontSize={11} fill="var(--text-2)">
        Encoder · Conv (1×3), stride 2 in freq
      </text>

      {/* bottleneck */}
      <Arrow x1={302} y1={cy} x2={346} y2={cy} />
      <rect x={350} y={cy - 38} width={250} height={76} rx={12} fill="var(--accent-soft)" stroke="var(--accent)" />
      <text x={475} y={cy - 12} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--text)">
        GRU 256 × 2 layers
      </text>
      <text x={475} y={cy + 6} textAnchor="middle" fontSize={10.5} fill="var(--text-2)">
        Linear 704→256 · GRU · Linear 256→704
      </text>
      <text x={475} y={cy + 22} textAnchor="middle" fontSize={10} fill="var(--text-3)">
        the only memory across time
      </text>
      {/* state loop */}
      <path d={`M560,${cy - 38} C560,${cy - 88} 390,${cy - 88} 390,${cy - 42}`} fill="none" stroke="var(--accent)" strokeWidth={1.5} markerEnd="url(#arrow-accent)" />
      <text x={475} y={cy - 80} textAnchor="middle" fontSize={10.5} fill="var(--accent)">
        state_out → state_in · [2, 1, 256]
      </text>
      <Arrow x1={602} y1={cy} x2={dec[0].x - 3} y2={cy} />

      {/* decoder */}
      {dec.map((b, i) => (
        <g key={`d${i}`}>
          {i > 0 && <Arrow x1={dec[i - 1].x + bwid + 2} y1={cy} x2={b.x - 3} y2={cy} />}
          <rect x={b.x} y={cy - fh(b.f) / 2} width={bwid} height={fh(b.f)} rx={5} fill="var(--s7)" fillOpacity={0.75 + i * 0.06} />
          <text x={b.x + bwid / 2} y={cy - fh(b.f) / 2 - 6} textAnchor="middle" fontSize={10} fill="var(--text-3)" className="tnum">
            {b.f}
          </text>
        </g>
      ))}
      <text x={730} y={420} textAnchor="middle" fontSize={11} fill="var(--text-2)">
        Decoder · ConvTranspose (1×3)
      </text>

      {/* skips: enc k (size f) -> decoder block that outputs the same size */}
      {[
        [enc[0], dec[2]],
        [enc[1], dec[1]],
        [enc[2], dec[0]],
      ].map(([a, b], i) => {
        const ya = cy + fh(a.f) / 2 + 2;
        const yb = cy + fh(b.f) / 2 + 2;
        const dip = 392 + i * 0;
        return (
          <path
            key={`s${i}`}
            d={`M${a.x + bwid / 2},${ya} C${a.x + bwid / 2},${dip + 10 - i * 14} ${b.x + bwid / 2},${dip + 10 - i * 14} ${b.x + bwid / 2},${yb + 3}`}
            fill="none"
            stroke="var(--text-3)"
            strokeDasharray="3 3"
            strokeWidth={1.1}
            markerEnd="url(#arrow)"
          />
        );
      })}
      <text x={475} y={420} textAnchor="middle" fontSize={10.5} fill="var(--text-3)">
        ⤷ skip connections (concat) ⤴
      </text>

      {/* head */}
      <Arrow x1={822} y1={cy} x2={854} y2={cy} />
      <rect x={858} y={cy - 36} width={76} height={72} rx={10} fill="var(--surface)" stroke="var(--accent)" />
      <text x={896} y={cy - 10} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text)">
        Mask
      </text>
      <text x={896} y={cy + 7} textAnchor="middle" fontSize={9.5} fill="var(--text-3)">
        tanh(|m|)·1.2
      </text>
      <text x={896} y={cy + 21} textAnchor="middle" fontSize={9.5} fill="var(--text-3)">
        ∠m
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------------------------

/** Synthetic voiced "speech" + noise, its noisy spectrogram, the ideal Wiener mask and the masked result. */
function MaskIllustration() {
  const data = useMemo(() => {
    const sr = 16000;
    const n = Math.round(sr * 1.4);
    const s = new Float32Array(n);
    const z = new Float32Array(n);
    let seed = 7;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 2 ** 32 - 0.5;
    };
    let phase = 0;
    for (let i = 0; i < n; i++) {
      const t = i / sr;
      const syll = Math.max(0, Math.sin(Math.PI * ((t * 3.2) % 1))) ** 1.5 * (t % 0.625 < 0.5 ? 1 : 0.15);
      const f0 = 120 + 40 * Math.sin(2 * Math.PI * 0.9 * t);
      phase += (2 * Math.PI * f0) / sr;
      let v = 0;
      for (let h = 1; h <= 30; h++) {
        const fh = f0 * h;
        if (fh > 7800) break;
        const formant = Math.exp(-(((fh - 600) / 300) ** 2)) + 0.6 * Math.exp(-(((fh - 1700) / 400) ** 2)) + 0.3 * Math.exp(-(((fh - 2800) / 500) ** 2));
        v += (formant / h ** 0.3) * Math.sin(h * phase);
      }
      s[i] = 0.12 * syll * v;
      z[i] = 0.05 * rnd() + 0.03 * Math.sin(2 * Math.PI * 3100 * t + rnd() * 0.2);
    }
    const frames = Math.floor((n - N_FFT) / HOP) + 1;
    const X = new Float32Array(frames * N_BINS);
    const M = new Float32Array(frames * N_BINS);
    const Y = new Float32Array(frames * N_BINS);
    const x = s.map((v, i) => v + z[i]);
    for (let t = 0; t < frames; t++) {
      const a = analysisFrame(s.subarray(t * HOP, t * HOP + N_FFT));
      const b = analysisFrame(z.subarray(t * HOP, t * HOP + N_FFT));
      const c = analysisFrame(x.subarray(t * HOP, t * HOP + N_FFT));
      for (let k = 0; k < N_BINS; k++) {
        const ps = a[2 * k] ** 2 + a[2 * k + 1] ** 2;
        const pn = b[2 * k] ** 2 + b[2 * k + 1] ** 2;
        const px = c[2 * k] ** 2 + c[2 * k + 1] ** 2;
        const m = ps / (ps + pn + 1e-12);
        X[t * N_BINS + k] = 10 * Math.log10(px + 1e-10);
        M[t * N_BINS + k] = m;
        Y[t * N_BINS + k] = 10 * Math.log10(m * m * px + 1e-10);
      }
    }
    return { frames, X, M, Y };
  }, []);
  return (
    <div className="mt-5 grid grid-cols-3 gap-2">
      <Tile label="Noisy |X|" data={data.X} frames={data.frames} kind="db" />
      <Tile label="Mask |M|" data={data.M} frames={data.frames} kind="mask" />
      <Tile label="|M ⊙ X|" data={data.Y} frames={data.frames} kind="db" />
      <p className="col-span-3 text-[11px] text-fg-3">Illustration on a synthetic voiced signal with the ideal (oracle) mask. The network learns to predict this from the noisy frame and its state.</p>
    </div>
  );
}

function Tile({ label, data, frames, kind }: { label: string; data: Float32Array; frames: number; kind: 'db' | 'mask' }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = frames;
    c.height = N_BINS;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(frames, N_BINS);
    let max = -Infinity;
    if (kind === 'db') for (let i = 0; i < data.length; i++) max = Math.max(max, data[i]);
    for (let t = 0; t < frames; t++) {
      for (let k = 0; k < N_BINS; k++) {
        const v = data[t * N_BINS + k];
        const n = kind === 'db' ? Math.max(0, Math.min(1, (v - (max - 70)) / 70)) : v;
        const li = Math.round(n * 255) * 3;
        const o = ((N_BINS - 1 - k) * frames + t) * 4;
        img.data[o] = MAGMA[li];
        img.data[o + 1] = MAGMA[li + 1];
        img.data[o + 2] = MAGMA[li + 2];
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, frames, kind]);
  return (
    <figure>
      <canvas ref={ref} className="block aspect-[4/3] w-full rounded-md border border-line" style={{ imageRendering: 'auto' }} role="img" aria-label={label} />
      <figcaption className="mt-1 text-center font-mono text-[11px] text-fg-2">{label}</figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------

function LatencyBudget() {
  const { status, manifest } = useEngineStatus();
  const k = lookaheadFrames(manifest);
  const [inferMs, setInferMs] = useState<number | null>(null);
  useEffect(() => {
    if (status.kind !== 'onnx') return;
    let alive = true;
    void engine
      .benchmark(300)
      .then((b) => alive && setInferMs(b.msPerFrame))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [status.kind]);
  const infer = inferMs ?? 1;
  const segs = [
    { k: 'Algorithmic window', v: 20, color: 'var(--s1)' },
    ...(k > 0 ? [{ k: `Lookahead (${k} frame${k > 1 ? 's' : ''})`, v: 10 * k, color: 'var(--s2)' }] : []),
    { k: inferMs !== null ? 'Inference (measured here)' : 'Inference (target ≤ 1 ms)', v: infer, color: 'var(--s3)' },
    { k: 'Capture + playback quanta', v: (2 * 128 * 1000) / 48000, color: 'var(--s4)' },
    { k: 'Jitter buffer', v: 20, color: 'var(--s7)' },
    { k: 'Device output (typical)', v: 10, color: 'var(--s5)' },
  ];
  const total = segs.reduce((a, s) => a + s.v, 0);
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] text-fg-3">Estimated mouth-to-ear</span>
        <span className="tnum text-[28px] font-semibold tracking-tight">
          ≈ {fmt(total, 0)}
          <span className="ml-1 text-[13px] font-medium text-fg-3">ms</span>
        </span>
      </div>
      <div className="mt-3 flex h-4 w-full gap-[2px] overflow-hidden rounded-[5px]" role="img" aria-label={segs.map((s) => `${s.k} ${fmt(s.v, 1)} ms`).join(', ')}>
        {segs.map((s) => (
          <div key={s.k} title={`${s.k}: ${fmt(s.v, 1)} ms`} style={{ width: `${(s.v / total) * 100}%`, background: s.color, minWidth: 3 }} />
        ))}
      </div>
      <ul className="mt-3 grid gap-1.5 text-[12px] sm:grid-cols-2">
        {segs.map((s) => (
          <li key={s.k} className="flex items-center justify-between gap-2">
            <span className="flex min-w-0 items-center gap-2 text-fg-2">
              <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: s.color }} />
              <span className="truncate">{s.k}</span>
            </span>
            <span className="tnum font-mono text-fg-3">{fmt(s.v, s.v < 5 ? 2 : 0)} ms</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[12px] text-fg-3">The Live page measures the actual capture-to-render latency on your machine.</p>
    </div>
  );
}
