/**
 * Python/JS parity against web/public/models/test_vectors.json (written by ml/ans/export_onnx.py).
 * Skipped when the ML workstream has not exported a model yet.
 */
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { runVectorChecks, type TestVectors } from '../engine/diagnostics';
import { OnnxModel, type Manifest, type OrtLike } from '../engine/model';

// MODELS_DIR (absolute, with trailing slash) points the parity test at another export, e.g. a candidate model.
const modelsDir = process.env.MODELS_DIR ?? fileURLToPath(new URL('../../public/models/', import.meta.url));
const tvPath = `${modelsDir}test_vectors.json`;
const manifestPath = `${modelsDir}manifest.json`;
const hasVectors = existsSync(tvPath);

function load<T>(p: string): T {
  return JSON.parse(readFileSync(p, 'utf8')) as T;
}

describe.skipIf(!hasVectors)('test_vectors.json parity', () => {
  it('JS STFT matches stft_frames (tol 1e-4)', async () => {
    const tv = load<TestVectors>(tvPath);
    const [params, frames] = await runVectorChecks(tv, null);
    expect(params.status, params.detail).toBe('pass');
    expect(frames.status, `max abs err ${frames.maxAbsErr}`).toBe('pass');
  });

  const manifest = existsSync(manifestPath) ? load<Manifest>(manifestPath) : null;
  const onnxPath = manifest ? `${modelsDir}${manifest.onnx}` : '';

  it.skipIf(!manifest || !existsSync(onnxPath))('ORT (onnxruntime-web, wasm in node) matches model_out_frames and output_wav', async () => {
    const tv = load<TestVectors>(tvPath);
    const ort = await import('onnxruntime-web');
    ort.env.wasm.numThreads = 1;
    const model = await OnnxModel.create(ort as unknown as OrtLike, new Uint8Array(readFileSync(onnxPath)), manifest!);
    try {
      const results = await runVectorChecks(tv, model);
      for (const r of results) {
        expect(r.status, `${r.name}: max abs err ${r.maxAbsErr} (tol ${r.tol}) ${r.detail ?? ''}`).toBe('pass');
      }
      console.info(results.map((r) => `${r.id}: ${r.status} maxAbsErr=${r.maxAbsErr?.toExponential(2) ?? '-'}`).join('\n'));
      if (manifest!.inputs && 'atten_lim_db' in manifest!.inputs) expect(model.supportsAttenLimit).toBe(true);
      const byId = Object.fromEntries(results.map((r) => [r.id, r]));
      expect(byId['model-frames']?.maxAbsErr).toBeLessThan(tv.tolerances.model);
      expect(byId['output-wav']?.maxAbsErr).toBeLessThan(tv.tolerances.waveform);
    } finally {
      await model.dispose();
    }
  });
});
