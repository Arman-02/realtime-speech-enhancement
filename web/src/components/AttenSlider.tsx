import type { AttenLimitSpec } from '../engine/model';
import { formatAtten } from '../lib/format';
import { Slider } from './ui';

/** Values within this many dB of the top snap to "∞" (a detent at the end of the travel). */
const DETENT_DB = 4;

/**
 * Maximum-attenuation control for models with the v0.2 `atten_lim_db` input. Disabled (with a tooltip
 * saying why) for older exports, while loading, and in bypass mode.
 */
export function AttenSlider({
  spec,
  supported,
  reason,
  value,
  onChange,
  className,
  compact,
}: {
  spec: AttenLimitSpec;
  supported: boolean;
  /** Why it is disabled (shown as a tooltip); ignored when supported. */
  reason?: string;
  value: number;
  onChange: (db: number) => void;
  className?: string;
  compact?: boolean;
}) {
  const hint = supported
    ? 'Caps how far the network may turn any time-frequency bin down. Lower keeps more of the original sound (and noise); ∞ lets the model remove as much as it wants.'
    : (reason ?? 'This model was exported without the attenuation-limit input (manifest v0.1), so the limit cannot be changed.');
  return (
    <Slider
      className={className}
      label="Max. attenuation"
      min={spec.min}
      max={spec.max}
      step={1}
      value={supported ? value : spec.max}
      disabled={!supported}
      hint={hint}
      description={compact ? undefined : supported ? 'Lower = more natural voice, more residual noise.' : 'Not available for this model export.'}
      format={(v) => formatAtten(v, spec.max)}
      onChange={(v) => onChange(v >= spec.max - DETENT_DB ? spec.max : v)}
    />
  );
}
