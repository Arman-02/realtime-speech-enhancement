import { useCallback, useEffect, useState } from 'react';
import { engine } from '../engine/client';
import type { CheckResult } from '../engine/diagnostics';
import type { DiagnosticsReport } from '../engine/protocol';
import { IconCheck, IconMinus, IconX } from '../components/icons';
import { Badge, Button, Callout, Card } from '../components/ui';
import { cx, displayModelName } from '../lib/format';

export default function Diagnostics() {
  const [report, setReport] = useState<DiagnosticsReport | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      setReport(await engine.diagnostics());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    void run();
  }, [run]);

  const all = report ? [...report.self, ...(report.vectors ?? [])] : [];
  const failed = all.filter((c) => c.status === 'fail').length;
  const passed = all.filter((c) => c.status === 'pass').length;
  const overall = !report ? 'running' : failed > 0 ? 'fail' : 'pass';

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-2xl">
          <div className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-fg-3 uppercase">Diagnostics</div>
          <h1 className="text-[28px] font-semibold tracking-tight">Python ↔ JS parity</h1>
          <p className="mt-2 text-[14px] text-fg-2">
            Replays <code className="font-mono text-[13px]">models/test_vectors.json</code> from the Python export through this browser’s DSP and ONNX Runtime Web, in the same worker that processes your audio.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span data-testid="diag-overall" data-status={overall}>
            {overall === 'running' ? (
              <Badge>Running…</Badge>
            ) : overall === 'pass' ? (
              <Badge tone="accent">
                <IconCheck size={12} /> All {passed} checks passed
              </Badge>
            ) : (
              <Badge tone="danger">
                <IconX size={12} /> {failed} failed
              </Badge>
            )}
          </span>
          <Button size="sm" onClick={() => void run()} disabled={running}>
            {running ? 'Running…' : 'Run again'}
          </Button>
        </div>
      </header>

      {error && <Callout tone="danger" title="Diagnostics failed to run">{error}</Callout>}

      {report && (
        <>
          <Card className="p-4 text-[13px] text-fg-2">
            <div className="flex flex-wrap gap-x-6 gap-y-1">
              <span>
                Model:{' '}
                <strong className="font-medium text-fg">
                  {report.model.kind === 'onnx' ? `${displayModelName(report.model.name)} v${report.model.version} (${report.model.trained ? 'trained' : 'untrained'})` : report.model.kind === 'bypass' ? `Bypass: ${report.model.message}` : 'loading'}
                </strong>
              </span>
              {report.model.kind === 'onnx' && (
                <span>
                  Runtime: <strong className="font-medium text-fg">onnxruntime-web · {report.model.backend}</strong> · {(report.model.modelBytes / 1e6).toFixed(2)} MB · loaded in {Math.round(report.model.loadMs)} ms
                </span>
              )}
              {report.model.kind === 'onnx' && (
                <span data-testid="diag-atten" data-atten={report.model.attenSupported ? 'yes' : 'no'}>
                  atten_lim_db input: <strong className="font-medium text-fg">{report.model.attenSupported ? 'yes (contract v0.2)' : 'no (contract v0.1)'}</strong>
                </span>
              )}
              <span>
                Vectors: <strong className="font-medium text-fg">{report.vectorsInfo ?? 'not found'}</strong>
              </span>
            </div>
          </Card>
          <CheckTable title="Self-checks (no external files)" checks={report.self} />
          {report.vectors ? (
            <CheckTable title="Test vectors (from ml/ans/export_onnx.py)" checks={report.vectors} />
          ) : (
            <Callout tone="warn" title="No test vectors found">
              models/test_vectors.json is missing, so Python parity can’t be checked yet. The self-checks above still validate the JS DSP.
            </Callout>
          )}
        </>
      )}
    </div>
  );
}

function CheckTable({ title, checks }: { title: string; checks: CheckResult[] }) {
  return (
    <Card className="overflow-hidden">
      <h2 className="px-5 pt-4 pb-2 text-[15px] font-semibold tracking-tight">{title}</h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-[13px]">
          <thead>
            <tr className="border-y border-line bg-surface-2/60 text-left text-[11px] tracking-wide text-fg-3 uppercase">
              <th className="w-24 px-5 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Check</th>
              <th className="px-3 py-2 text-right font-medium">Max abs error</th>
              <th className="px-3 py-2 text-right font-medium">Tolerance</th>
              <th className="px-5 py-2 text-right font-medium">Time</th>
            </tr>
          </thead>
          <tbody className="tnum">
            {checks.map((c) => (
              <tr key={c.id} className="border-b border-line last:border-0" data-check={c.id} data-status={c.status}>
                <td className="px-5 py-2.5">
                  <span
                    className={cx(
                      'inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase',
                      c.status === 'pass' && 'bg-accent-soft text-accent',
                      c.status === 'fail' && 'bg-[color-mix(in_srgb,var(--danger)_14%,transparent)] text-danger',
                      c.status === 'skip' && 'bg-surface-3 text-fg-3',
                    )}
                  >
                    {c.status === 'pass' ? <IconCheck size={11} /> : c.status === 'fail' ? <IconX size={11} /> : <IconMinus size={11} />}
                    {c.status}
                  </span>
                </td>
                <td className="px-3 py-2.5 text-fg">
                  {c.name}
                  {c.detail && <div className="text-[11px] text-fg-3">{c.detail}</div>}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-[12px] text-fg-2">{c.maxAbsErr !== undefined ? c.maxAbsErr.toExponential(2) : '—'}</td>
                <td className="px-3 py-2.5 text-right font-mono text-[12px] text-fg-3">{c.tol !== undefined ? c.tol.toExponential(0) : '—'}</td>
                <td className="px-5 py-2.5 text-right font-mono text-[12px] text-fg-3">{c.ms !== undefined ? `${c.ms.toFixed(0)} ms` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
