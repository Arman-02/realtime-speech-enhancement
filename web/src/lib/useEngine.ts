import { useEffect, useState } from 'react';
import { engine } from '../engine/client';
import type { ModelStatus } from '../engine/protocol';
import type { Manifest } from '../engine/model';

/** Starts the engine worker (idempotent) and tracks the model status. */
export function useEngineStatus(): { status: ModelStatus; manifest: Manifest | null } {
  const [status, setStatus] = useState<ModelStatus>(engine.status);
  useEffect(() => {
    const off = engine.onStatus(setStatus);
    // Start the worker (and the ~5 MB model + wasm download) once the page is idle, so it never
    // competes with first paint. Any engine call (process, live, diagnostics) starts it immediately.
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number };
    const start = () => void engine.ensure().then(setStatus);
    let t: ReturnType<typeof setTimeout> | null = null;
    if (w.requestIdleCallback) w.requestIdleCallback(start, { timeout: 1500 });
    else t = setTimeout(start, 300);
    return () => {
      off();
      if (t) clearTimeout(t);
    };
  }, []);
  return { status, manifest: engine.manifest };
}
