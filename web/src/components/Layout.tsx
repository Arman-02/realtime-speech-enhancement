import type { ReactNode } from 'react';
import { hrefFor, type Route } from '../lib/router';
import { useTheme } from '../lib/theme';
import { useEngineStatus } from '../lib/useEngine';
import type { ModelStatus } from '../engine/protocol';
import { IconGithub, IconMoon, IconSun } from './icons';
import { cx, displayModelName } from '../lib/format';
import { USE_FIXTURES } from '../lib/fixtures';

const NAV: { route: Route; label: string }[] = [
  { route: 'studio', label: 'Studio' },
  { route: 'live', label: 'Live' },
  { route: 'results', label: 'Results' },
  { route: 'how', label: 'How it works' },
];

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--surface-3)" />
      <g strokeLinecap="round" strokeWidth="2.4">
        <path d="M7 13v6" stroke="var(--text-3)" />
        <path d="M11.5 9v14" stroke="var(--text-3)" />
        <path d="M16 6v20" stroke="var(--accent)" />
        <path d="M20.5 10v12" stroke="var(--accent)" />
        <path d="M25 14v4" stroke="var(--accent)" />
      </g>
    </svg>
  );
}

export function StatusPill({ status }: { status: ModelStatus }) {
  let dot = 'var(--text-3)';
  let text = 'Loading model…';
  let title = 'Loading the ONNX model in a Web Worker';
  if (status.kind === 'onnx') {
    dot = status.trained ? 'var(--ok)' : 'var(--warn)';
    text = `${displayModelName(status.name)} · wasm`;
    title = `${displayModelName(status.name)} v${status.version} (${status.trained ? 'trained' : 'untrained'}) · ${status.backend} · loaded in ${Math.round(status.loadMs)} ms`;
  } else if (status.kind === 'bypass') {
    dot = 'var(--danger)';
    text = 'Bypass mode';
    title = status.message;
  }
  return (
    <span title={title} className="hidden items-center gap-2 rounded-full border border-line bg-surface px-2.5 py-1 text-[12px] text-fg-2 sm:inline-flex">
      <span aria-hidden className={cx('h-1.5 w-1.5 rounded-full', status.kind === 'loading' && 'pulse-dot')} style={{ background: dot }} />
      <span className="max-w-[180px] truncate">{text}</span>
    </span>
  );
}

function ModelBanner({ status }: { status: ModelStatus }) {
  if (status.kind === 'bypass') {
    return (
      <div className="border-b border-line bg-[color-mix(in_srgb,var(--warn)_8%,var(--bg))]">
        <div className="mx-auto flex max-w-6xl items-center gap-2 px-4 py-2 text-[13px] text-fg-2 sm:px-6">
          <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-warn" />
          <span>
            <strong className="font-medium text-fg">Bypass mode.</strong>{' '}
            {status.reason === 'missing' ? 'No model has been exported yet, so audio passes through the full DSP pipeline unchanged.' : `The model failed to load (${status.message}); audio passes through unchanged.`}
          </span>
        </div>
      </div>
    );
  }
  if (status.kind === 'onnx' && !status.trained) {
    return (
      <div className="border-b border-line bg-[color-mix(in_srgb,var(--warn)_6%,var(--bg))]">
        <div className="mx-auto flex max-w-6xl items-center gap-2 px-4 py-2 text-[13px] text-fg-2 sm:px-6">
          <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-warn" />
          <span>
            <strong className="font-medium text-fg">Untrained model.</strong> This build ships a randomly initialised export to exercise the pipeline end to end, so the output is not denoised yet.
          </span>
        </div>
      </div>
    );
  }
  return null;
}

export function Layout({ route, children }: { route: Route; children: ReactNode }) {
  const [theme, toggle] = useTheme();
  const { status } = useEngineStatus();
  const repo = import.meta.env.VITE_REPO_URL;
  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-line bg-[color-mix(in_srgb,var(--bg)_82%,transparent)] backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6">
          <a href={hrefFor('studio')} className="flex shrink-0 items-center gap-2.5" aria-label="Real-Time Speech Enhancement, home">
            <Logo />
            <span className="leading-none">
              <span className="block text-[14px] font-semibold tracking-tight whitespace-nowrap">Real-Time Speech Enhancement</span>
              <span className="hidden text-[11px] whitespace-nowrap text-fg-3 lg:block">Causal neural noise suppression for voice pickup</span>
            </span>
          </a>
          <nav aria-label="Main" className="ml-2 hidden md:block">
            <ul className="flex items-center gap-1">
              {NAV.map((n) => (
                <li key={n.route}>
                  <a
                    href={hrefFor(n.route)}
                    aria-current={route === n.route ? 'page' : undefined}
                    className={cx(
                      'relative rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors',
                      route === n.route ? 'text-fg' : 'text-fg-3 hover:text-fg',
                    )}
                  >
                    {n.label}
                    {route === n.route && <span aria-hidden className="absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full bg-accent" />}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <StatusPill status={status} />
            {repo && (
              <a href={repo} target="_blank" rel="noreferrer" className="grid h-9 w-9 place-items-center rounded-lg text-fg-2 hover:bg-surface-2 hover:text-fg" aria-label="Source code on GitHub">
                <IconGithub size={18} />
              </a>
            )}
            <button
              type="button"
              onClick={toggle}
              className="grid h-9 w-9 place-items-center rounded-lg text-fg-2 transition-colors hover:bg-surface-2 hover:text-fg"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            >
              {theme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
            </button>
          </div>
        </div>
        <nav aria-label="Main (mobile)" className="border-t border-line md:hidden">
          <ul className="mx-auto flex max-w-6xl gap-1 overflow-x-auto px-3 py-1.5">
            {NAV.map((n) => (
              <li key={n.route} className="shrink-0">
                <a
                  href={hrefFor(n.route)}
                  aria-current={route === n.route ? 'page' : undefined}
                  className={cx('block rounded-md px-3 py-1.5 text-[13px] font-medium', route === n.route ? 'bg-surface-2 text-fg' : 'text-fg-3')}
                >
                  {n.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      {USE_FIXTURES && (
        <div className="border-b border-line bg-[color-mix(in_srgb,var(--danger)_10%,var(--bg))]" data-testid="fixture-banner">
          <div className="mx-auto max-w-6xl px-4 py-2 text-[13px] text-fg-2 sm:px-6">
            <strong className="font-medium text-fg">Fixture build.</strong> Results, limitations and listening data come from <code className="font-mono">src/fixtures</code> and are fake. Never deploy this build.
          </div>
        </div>
      )}
      <ModelBanner status={status} />
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pt-8 pb-16 sm:px-6">
        {children}
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-[12px] text-fg-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <p>
            Inference runs entirely in your browser (ONNX Runtime Web, WebAssembly). No audio leaves your device.
          </p>
          <p>
            Demo clips: VoiceBank+DEMAND (Valentini-Botinhao, Univ. of Edinburgh, 2017),{' '}
            <a className="underline decoration-line-strong underline-offset-2 hover:text-fg" href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">
              CC BY 4.0
            </a>
            .
          </p>
        </div>
      </footer>
    </div>
  );
}
