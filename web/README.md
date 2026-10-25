# Real-Time Speech Enhancement: web app

*Causal neural noise suppression for voice pickup.* This is the in-browser front end: a static Vite + React + TypeScript + Tailwind site. The causal speech-enhancement model runs on the viewer's own CPU through `onnxruntime-web` (WebAssembly, single-threaded), so there is no backend.

| Route | What it does |
|---|---|
| `#/` **Studio** | Upload, drag-and-drop or record a clip (up to 60 s), or pick a demo clip. The clip is processed in a Web Worker, then you can A/B/C the noisy, enhanced and clean signals without restarting playback, view spectrograms with a shared playhead, see SI-SDR, ms/frame and RTF, and download a WAV. The **Max. attenuation** slider (0–100 dB, ∞ at the top) re-processes the clip in the worker, debounced, keeping the playhead. Demo clips also show their offline PESQ/STOI/SI-SDR from `results.json → per_sample`. |
| `#/live` | Mic → AudioWorklet (resample to 16 kHz) → Worker (ORT) → AudioWorklet → speakers, with bypass, the attenuation limit (applied from the next frame, no restart), output gain, level meters, scrolling spectrograms and measured latency. |
| `#/results` | From `results.json` (v0.1 or v0.2): the quality-vs-compute Pareto chart (plain vs. distilled students, seeds, teacher and reference systems), a test-set selector labelled seen/unseen noise, the summary table with significance badges, speech distortion vs. noise reduction by SNR, breakdowns by noise type and SNR, the attenuation-sweep, lookahead, quantization and significance tables, the **Known limitations** gallery from `failures/index.json`, and the model card with the deployment budget from `models/manifest.json`. Each section hides when its data is absent. |
| `#/how-it-works` | Architecture, distillation and attenuation-limit diagrams, STFT/mask explainer and latency budget. |
| `#/diagnostics` | Hidden page. Replays `models/test_vectors.json` through the JS DSP and ORT, including the `atten_lim_db = 12` case when present, and shows pass/fail with max abs error for each check. |
| `#/listen` | Hidden page. Blind MOS listening test (ITU-T P.808 / ACR in spirit) over `listening/index.json`: consent, headphone check, optional nickname and device note, a practice trial, then up to 20 trials. Each trial plays its conditions in a seeded per-rater random order as anonymous versions A–E, and each version is rated 1–5 after being heard in full. Progress is kept in localStorage, so a reload resumes. The end screen downloads `listen-<rater>-<timestamp>.json` (PLAN.md §3.6) and, if configured, POSTs it. |

## Local development

Requires Node 20 or newer.

```bash
npm install
npm run dev         # http://localhost:5173
npm test            # Vitest: FFT vs naive DFT and numpy, STFT round trip, streaming == offline,
                    # resampler, and Python parity against public/models/test_vectors.json if present
npm run lint
npm run typecheck
npm run build       # -> dist/
npm run preview     # serves dist/ at http://localhost:4173
```

The app reads files that the other workstreams produce, and degrades gracefully when they are missing:

| Path (under `public/`) | Producer | If missing |
|---|---|---|
| `models/manifest.json`, `models/ans_stream.onnx`, `models/test_vectors.json` | `ml/ans/export_onnx.py` | Bypass (identity) model plus a banner; diagnostics skip the model checks |
| `results.json` | `ml/ans/evaluate.py` | "Results pending" state |
| `samples/samples.json` and the WAVs | `scripts/prepare_vbd.py` | Empty gallery; upload and record still work |
| `failures/index.json` and WAVs | `ml/ans/render.py` | The Known limitations section is hidden |
| `listening/index.json` and WAVs | `ml/ans/render.py` | `#/listen` shows "not available yet" |

**Model contract v0.2.** The engine reads `manifest.inputs`. It feeds `atten_lim_db` (a `[1]` float32 tensor) only when the graph has that input. The default and range come from `manifest.atten_lim_db`, and the limit changes from the next frame. v0.1 exports keep working, with the slider disabled and a tooltip explaining why. `manifest.lookahead_frames` is included in the latency figures and in the offline alignment.

### Environment variables (build time)

| Variable | Effect |
|---|---|
| `BASE_PATH` | Sub-path deploys, e.g. `/repo-name/` for GitHub Pages. |
| `VITE_REPO_URL` | Shows a GitHub link in the header. |
| `VITE_LISTEN_ENDPOINT` | Optional URL that receives each completed listening test as a JSON `POST` (`Content-Type: application/json`). Use a form backend that accepts JSON, such as Formspree or a Google Apps Script web app. Responses are always downloadable as well, so this is optional. |
| `VITE_USE_FIXTURES` | `1` makes the app read `src/fixtures/*.json` (fake v0.2 results, failures, listening index and model card) instead of `public/`. A red banner marks this mode. It is **for development and tests only; never deploy it.** Normal builds don't contain the fixtures; CI checks this. |

```bash
VITE_USE_FIXTURES=1 npm run dev     # every Results section, the gallery and #/listen populated with fake data
```

## Engine layout (`src/engine/`)

- `dsp.ts`: the §3.1 contract. A mixed-radix FFT (320 = 4·4·4·5) that matches numpy's `rfft`/`irfft`, analysis and synthesis frames, offline `stft`/`istft`, and `StreamingSTFT` (160-sample history plus a 160-sample OLA tail).
- `model.ts`: `OnnxModel`, which is generic over `manifest.state_shape`, and `BypassModel`.
- `enhancer.ts`: `Enhancer.push()` (streaming) and `enhanceOffline()`. Both use the same per-hop code path.
- `worker.ts` and `client.ts`: the engine Web Worker and its main-thread handle.
- `worklets/`: the capture worklet (device rate → 16 kHz) and the playback worklet (16 kHz → device rate, jitter buffer, latency measurement). They talk to the worker over direct `MessagePort`s.
- `resample.ts`: a windowed-sinc streaming resampler, shared by the worklets and the decode fallback.
- `diagnostics.ts`: the parity checks, shared by the Diagnostics page and the Vitest suite.

## Deploying

The build is fully static (`dist/`) and uses hash routing, so no rewrite rules are needed. Single-threaded WASM means COOP/COEP headers are **not** required. The ORT `.wasm` file (about 14 MB, about 3.7 MB compressed) is fetched lazily by the worker.

| Host | Settings |
|---|---|
| **Vercel** | Root directory `web`, framework "Vite", build `npm run build`, output `dist`. |
| **Netlify** | Base directory `web`, build `npm run build`, publish `web/dist`. `public/_headers` sets long-lived caching for hashed assets. |
| **Cloudflare Pages** | Root `web`, build `npm run build`, output `dist`. `_headers` applies here too. Every file is under the 25 MiB per-file limit. |
| **GitHub Pages** | Build with the repo sub-path: `BASE_PATH=/<repo-name>/ npm run build`, then publish `dist/` (for example with `actions/upload-pages-artifact` + `actions/deploy-pages`). `.nojekyll` is included. |

Deploy notes:
- The data files under `public/` (`models/`, `samples/`, `results.json`, `failures/`, `listening/`) are copied into `dist/` as they are. Re-run the build, or just replace those files on the host, after ml re-exports.
- Set `VITE_LISTEN_ENDPOINT` in the host's build environment if listening-test responses should be collected automatically. `#/listen` and `#/diagnostics` are not linked from the navigation, so share their URLs directly.
- Never set `VITE_USE_FIXTURES` on a host.

## CI

`.github/workflows/ci.yml` runs on every push and PR. It has two jobs:
- **`ml`**: `uv sync`, `ruff check`, `pytest` on CPU with synthetic data.
- **`web`** (Node 22): `npm ci`, `typecheck`, `lint`, `vitest`, `build`. The vitest run includes the Python/JS parity test, which replays `public/models/test_vectors.json` through onnxruntime-web's wasm backend under Node. The job then checks that no fixture data reached `dist/` and that the fixture build compiles.

To check a candidate export before it is copied into `public/`, run `MODELS_DIR=/abs/path/to/export/ npx vitest run src/test/vectors.test.ts`.

## Browser support

This targets current Chrome, Edge, Firefox and Safari. Live mode needs AudioWorklet, `getUserMedia` and a secure context (HTTPS or localhost). The app turns off the browser's echo cancellation, noise suppression and auto-gain so that only this model is heard. Use headphones in Live mode.
