import { useCallback, useEffect, useRef, useState } from 'react';
import { MAX_SECONDS } from './audio';

export type RecorderState =
  | { kind: 'idle' }
  | { kind: 'requesting' }
  | { kind: 'recording'; startedAt: number }
  | { kind: 'error'; message: string; denied: boolean };

export function micErrorMessage(e: unknown): { message: string; denied: boolean } {
  const name = e instanceof DOMException ? e.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return { denied: true, message: 'Microphone access was blocked. Allow it from the address bar (site settings) and try again.' };
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return { denied: false, message: 'No microphone was found on this device.' };
  if (name === 'NotReadableError') return { denied: false, message: 'The microphone is in use by another application.' };
  if (!window.isSecureContext) return { denied: false, message: 'Microphone access requires HTTPS (or localhost).' };
  return { denied: false, message: e instanceof Error ? e.message : String(e) };
}

export const RAW_MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
};

/** MediaRecorder-based clip recorder for the Studio (auto-stops at 60 s). */
export function useRecorder(onDone: (blob: Blob) => void) {
  const [state, setState] = useState<RecorderState>({ kind: 'idle' });
  const rec = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const cleanup = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    rec.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setState({ kind: 'error', denied: false, message: 'Recording is not supported in this browser.' });
      return;
    }
    setState({ kind: 'requesting' });
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: RAW_MIC_CONSTRAINTS });
      stream.current = s;
      const r = new MediaRecorder(s);
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
      r.onstop = () => {
        const blob = new Blob(chunks, { type: r.mimeType || 'audio/webm' });
        cleanup();
        setState({ kind: 'idle' });
        if (blob.size > 0) doneRef.current(blob);
      };
      rec.current = r;
      r.start(250);
      setState({ kind: 'recording', startedAt: performance.now() });
      timer.current = setTimeout(() => r.state !== 'inactive' && r.stop(), MAX_SECONDS * 1000);
    } catch (e) {
      cleanup();
      setState({ kind: 'error', ...micErrorMessage(e) });
    }
  }, [cleanup]);

  const stop = useCallback(() => {
    if (rec.current && rec.current.state !== 'inactive') rec.current.stop();
  }, []);

  return { state, start, stop };
}
