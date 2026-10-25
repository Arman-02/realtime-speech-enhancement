import { useId, type ButtonHTMLAttributes, type CSSProperties, type ReactNode } from 'react';
import { IconAlert, IconInfo } from './icons';
import { cx } from '../lib/format';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: 'sm' | 'md' | 'lg' }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-[background,color,box-shadow,transform] duration-150 select-none disabled:opacity-45 disabled:cursor-not-allowed active:translate-y-px whitespace-nowrap';
  const sizes = { sm: 'h-8 px-3 text-[13px]', md: 'h-10 px-4 text-sm', lg: 'h-12 px-5 text-[15px]' }[size];
  const variants: Record<ButtonVariant, string> = {
    primary: 'bg-accent text-accent-ink hover:brightness-110 shadow-[0_1px_0_rgb(255_255_255/0.2)_inset]',
    secondary: 'bg-surface-2 text-fg border border-line hover:bg-surface-3 hover:border-line-strong',
    ghost: 'text-fg-2 hover:text-fg hover:bg-surface-2',
    danger: 'bg-danger text-white hover:brightness-110',
  };
  return (
    <button type="button" className={cx(base, sizes, variants[variant], className)} {...rest}>
      {children}
    </button>
  );
}

export function Card({ className, children, as: As = 'section', ...rest }: { className?: string; children: ReactNode; as?: 'section' | 'div' | 'article'; 'aria-label'?: string; 'aria-labelledby'?: string }) {
  return (
    <As className={cx('card', className)} {...rest}>
      {children}
    </As>
  );
}

export function SectionTitle({ eyebrow, title, children, id }: { eyebrow?: string; title: ReactNode; children?: ReactNode; id?: string }) {
  return (
    <div className="mb-4">
      {eyebrow && <div className="mb-1 text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">{eyebrow}</div>}
      <h2 id={id} className="text-[17px] font-semibold tracking-tight text-fg">
        {title}
      </h2>
      {children && <p className="mt-1 max-w-prose text-sm text-fg-2">{children}</p>}
    </div>
  );
}

export interface SegOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  swatch?: string;
  hint?: string;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size = 'md',
}: {
  value: T;
  options: SegOption<T>[];
  onChange: (v: T) => void;
  label: string;
  size?: 'sm' | 'md';
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line bg-surface-2 p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            title={o.hint}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => {
              if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
              e.preventDefault();
              const enabled = options.filter((x) => !x.disabled);
              const i = enabled.findIndex((x) => x.value === value);
              const next = enabled[(i + (e.key === 'ArrowRight' ? 1 : enabled.length - 1)) % enabled.length];
              onChange(next.value);
              const el = (e.currentTarget.parentElement?.querySelector(`[data-v="${next.value}"]`) as HTMLElement | null) ?? null;
              el?.focus();
            }}
            data-v={o.value}
            tabIndex={active ? 0 : -1}
            className={cx(
              'inline-flex items-center gap-2 rounded-md font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40',
              size === 'sm' ? 'h-7 px-2.5 text-[12px]' : 'h-8 px-3 text-[13px]',
              active ? 'bg-surface text-fg shadow-sm ring-1 ring-line-strong' : 'text-fg-2 hover:text-fg',
            )}
          >
            {o.swatch && <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: o.swatch }} />}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  disabled,
  className,
  hint,
  description,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  disabled?: boolean;
  className?: string;
  /** Tooltip on the whole control (also announced via aria-describedby). */
  hint?: string;
  /** Visible help text under the track. */
  description?: ReactNode;
}) {
  const id = useId();
  const descId = useId();
  const fill = `${((value - min) / (max - min)) * 100}%`;
  const shown = format ? format(value) : String(value);
  return (
    <div className={cx('min-w-0', className)} title={hint}>
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-[12px] font-medium text-fg-2">
          {label}
        </label>
        <span className="tnum font-mono text-[12px] text-fg-3" aria-hidden>
          {shown}
        </span>
      </div>
      <input
        id={id}
        className="range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-valuetext={shown}
        aria-describedby={hint || description ? descId : undefined}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ '--fill': fill } as CSSProperties}
      />
      {(hint || description) && (
        <div id={descId} className={cx('mt-1 text-[11px] leading-snug text-fg-3', !description && 'sr-only')}>
          {description ?? hint}
        </div>
      )}
    </div>
  );
}

export function Stat({ label, value, unit, hint, tone }: { label: string; value: ReactNode; unit?: string; hint?: string; tone?: 'good' | 'bad' | 'neutral' }) {
  return (
    <div className="min-w-0" title={hint}>
      <div className="truncate text-[11px] font-medium tracking-wide text-fg-3 uppercase">{label}</div>
      <div className={cx('tnum mt-0.5 truncate text-[20px] font-semibold tracking-tight', tone === 'good' ? 'text-ok' : tone === 'bad' ? 'text-danger' : 'text-fg')}>
        {value}
        {unit && <span className="ml-1 text-[12px] font-medium text-fg-3">{unit}</span>}
      </div>
    </div>
  );
}

export function Callout({ tone = 'info', title, children, className, action }: { tone?: 'info' | 'warn' | 'danger'; title?: ReactNode; children?: ReactNode; className?: string; action?: ReactNode }) {
  const color = tone === 'warn' ? 'var(--warn)' : tone === 'danger' ? 'var(--danger)' : 'var(--accent)';
  const Icon = tone === 'info' ? IconInfo : IconAlert;
  return (
    <div
      role={tone === 'danger' ? 'alert' : 'status'}
      className={cx('flex items-start gap-3 rounded-xl border px-4 py-3 text-sm', className)}
      style={{ borderColor: `color-mix(in srgb, ${color} 35%, var(--border))`, background: `color-mix(in srgb, ${color} 7%, var(--surface))` }}
    >
      <Icon size={18} className="mt-0.5 shrink-0" style={{ color }} />
      <div className="min-w-0 flex-1">
        {title && <div className="font-medium text-fg">{title}</div>}
        {children && <div className="text-fg-2">{children}</div>}
      </div>
      {action}
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)} className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
      <div className="h-full rounded-full bg-accent transition-[width] duration-150 ease-out" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Badge({ children, tone = 'neutral', className }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'warn' | 'danger'; className?: string }) {
  const tones = {
    neutral: 'border-line text-fg-2 bg-surface-2',
    accent: 'border-transparent text-accent bg-accent-soft',
    warn: 'border-transparent text-warn bg-[color-mix(in_srgb,var(--warn)_12%,transparent)]',
    danger: 'border-transparent text-danger bg-[color-mix(in_srgb,var(--danger)_12%,transparent)]',
  }[tone];
  return <span className={cx('inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium', tones, className)}>{children}</span>;
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line bg-surface-2 px-1.5 py-px font-mono text-[11px] text-fg-2">{children}</kbd>;
}

