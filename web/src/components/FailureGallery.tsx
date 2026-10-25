/** "Known limitations" (§3.6 failures/index.json, §5.3): playable noisy / enhanced (/ clean) clips. */
import { useEffect, useRef } from 'react';
import { sampleUrl, type FailureCase } from '../lib/data';
import { displayModelName } from '../lib/format';
import { Badge, Card, SectionTitle } from './ui';

const CATEGORY_LABEL: Record<string, string> = {
  music: 'Music',
  silence: 'Silence / pumping',
  low_snr: 'Very low SNR',
  talker: 'Competing talker',
  reverb: 'Reverberation',
  clipping: 'Clipped input',
};

function Clip({ label, src, color }: { label: string; src: string; color: string }) {
  return (
    <div className="grid grid-cols-[72px_1fr] items-center gap-2">
      <span className="flex items-center gap-1.5 text-[12px] text-fg-2">
        <span aria-hidden className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
        {label}
      </span>
      <audio className="h-9 w-full min-w-0" controls preload="none" src={sampleUrl(src)} aria-label={label} />
    </div>
  );
}

export function FailureGallery({ cases }: { cases: FailureCase[] }) {
  const root = useRef<HTMLDivElement>(null);
  // One clip at a time: starting a player pauses the others.
  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const onPlay = (e: Event) => {
      el.querySelectorAll('audio').forEach((a) => {
        if (a !== e.target) a.pause();
      });
    };
    el.addEventListener('play', onPlay, true);
    return () => el.removeEventListener('play', onPlay, true);
  }, []);
  return (
    <section aria-labelledby="limits-title" ref={root}>
      <SectionTitle id="limits-title" eyebrow="Honest reporting" title="Known limitations">
        Cases where the model struggles, rendered offline from the same network. Listen to the input and the output side by side.
      </SectionTitle>
      <ul className="grid gap-4 md:grid-cols-2" data-testid="failure-gallery">
        {cases.map((c) => (
          <li key={c.id}>
            <Card as="article" className="flex h-full flex-col p-5" aria-label={c.title}>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-[15px] font-semibold tracking-tight">{c.title}</h3>
                {c.category && <Badge tone="warn">{CATEGORY_LABEL[c.category] ?? c.category}</Badge>}
              </div>
              {c.explanation && <p className="mt-1.5 text-[13px] leading-relaxed text-fg-2">{c.explanation}</p>}
              <div className="mt-4 space-y-2">
                <Clip label="Noisy" src={c.noisy} color="var(--noisy)" />
                <Clip label="Enhanced" src={c.enhanced} color="var(--enhanced)" />
                {c.clean && <Clip label="Clean" src={c.clean} color="var(--clean)" />}
              </div>
              {c.model && <p className="mt-auto pt-3 text-[11px] text-fg-3">Model: {displayModelName(c.model)}</p>}
            </Card>
          </li>
        ))}
      </ul>
    </section>
  );
}
