/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// BASE_PATH lets the same build be hosted at a sub-path (e.g. GitHub Pages: BASE_PATH=/repo-name/).
function normalizeBase(raw: string | undefined): string {
  if (!raw || raw === '/') return '/';
  if (raw === '.' || raw === './') return './';
  let b = raw.trim();
  if (!b.startsWith('/') && !b.startsWith('.') && !/^https?:\/\//.test(b)) b = `/${b}`;
  if (!b.endsWith('/')) b = `${b}/`;
  return b;
}

export default defineConfig({
  base: normalizeBase(process.env.BASE_PATH),
  plugins: [react(), tailwindcss()],
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    assetsInlineLimit: (file) => (file.endsWith('.wasm') ? false : undefined),
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    testTimeout: 60_000,
  },
});
