# Real-Time Speech Enhancement for Voice Pickup

A causal neural noise suppressor that removes background noise (traffic, cafés, offices, public spaces, transport) from a single microphone signal in real time. It targets voice pickup in headphones and true-wireless earbuds during calls. This is microphone-path speech enhancement, not active noise cancellation (ANC). The model is small enough to run **in your browser**, frame by frame, with 20 ms algorithmic latency.

> **Status:** 🚧 in development. See [PLAN.md](PLAN.md) for the full implementation plan and milestones.
>
> **Live demo:** _coming soon_

---

## Highlights

- **Streaming by design.** A causal convolutional-recurrent network (CRN) predicts a bounded complex ratio mask on 20 ms / 10 ms-hop STFT frames. Frame-by-frame inference is bit-for-bit equivalent to offline inference, and this is enforced by tests.
- **Distillation for small causal models.** A non-causal teacher is distilled into causal students of three sizes. The result is a quality-vs-compute Pareto curve for plain vs. distilled training, backed by paired significance tests.
- **Measured beyond the benchmark.** PESQ-WB, STOI/ESTOI and SI-SDR on VoiceBank+DEMAND, where all test noise types are unseen in training. Also out-of-domain noise, competing-talker and real-recording sets scored with DNSMOS P.835, plus RNNoise and DeepFilterNet3 reference rows.
- **Why it sounds the way it does.** Each model's effect is split into *speech distortion* and *noise reduction*, and an adjustable attenuation limit lets you tune the tradeoff yourself.
- **Deployable.** The model is exported to ONNX (fp32 and int8) with per-frame latency and real-time factor benchmarks, and runs in the browser through `onnxruntime-web`.
- **Interactive frontend.** Upload or record audio, A/B noisy vs. enhanced vs. clean, view spectrograms, or switch on **live mic denoising**. The app is a fully static site with no server-side inference.

## Results

_Evaluated on the VoiceBank+DEMAND test set (824 utterances, 16 kHz). None of its noise types appear in training. Timing is per 10 ms frame on 1 CPU thread (Apple M4 Pro, ONNX Runtime). More rows (distillation, out-of-domain sets, DNSMOS, reference systems) will land as the milestones in [PLAN.md](PLAN.md) progress._

| Model                          | Causal | Params | PESQ ↑ | STOI ↑ | SI-SDR ↑ (dB) | DNSMOS OVRL ↑ | ms / frame |
|--------------------------------|:------:|-------:|-------:|-------:|--------------:|--------------:|-----------:|
| Noisy (unprocessed)            |   —    |    —   |  1.97  | 0.921  |     8.45      |     2.70      |     —      |
| Wiener (min. statistics)       |   ✓    |    0   |  2.20  | 0.899  |    12.23      |     2.76      |   0.02     |
| RNNoise¹                       |   ✓    |    —   |  2.09  | 0.888  |    12.13      |     2.94      |     —      |
| DeepFilterNet3¹                |   ✓    |    —   |  2.70  | 0.925  |    17.24      |   **3.21**    |     —      |
| **CRN-256 (benchmark model)**  |   ✓    |  1.21M | **2.76** | **0.937** |  16.73    |     3.03      | **0.13**   |
| Teacher (bi-GRU 512, offline)  |   ✗    |    —   |  2.93  | 0.946  |    18.16      |     3.15      |     —      |

¹ These are full-band 48 kHz models, run here by resampling 16 → 48 → 16 kHz. They were trained on far more data, and their published numbers come from a different evaluation setup.

**Out-of-domain noise is the harder test.** On VoiceBank+DEMAND speech mixed with ESC-50 noise (500 clips, −5 to 15 dB), the benchmark CRN scores PESQ **1.59**, against **2.15** for DeepFilterNet3. A model trained only on VoiceBank+DEMAND does well on that benchmark but generalizes poorly, which motivates the augmented demo model (in progress). In the competing-talker test, every system keeps the interfering voice.

**Suppression vs. distortion.** The attenuation limit trades noise reduction against speech distortion smoothly:

| Attenuation limit | 6 dB | 12 dB | 18 dB | 24 dB | ∞ |
|---|---|---|---|---|---|
| Noise reduction (dB) | 6.6 | 9.7 | 11.0 | 11.4 | 11.5 |
| Speech distortion SI-SDR (dB, higher = less damage) | 22.2 | 21.0 | 20.6 | 20.5 | 20.5 |
| PESQ | 2.37 | 2.62 | 2.73 | 2.76 | 2.76 |

**Efficiency.** The CRN uses 1.96 M MACs per frame (0.2 GMAC/s) and runs at a real-time factor of 0.013 in ONNX Runtime, so one CPU core spends 1.3 % of real time on it. In the browser it takes about 0.2 ms per frame. Dynamic int8 quantization shrinks it to 3.7 MB with no loss in PESQ (2.76). The GRU stays fp32, so the size saving is modest.

**Distillation (preliminary).** At 128 hidden units, with two seeds per arm, both distilled students beat both plain ones. VB-D PESQ is 2.795 and 2.807 against 2.740 and 2.772 (+0.045 on average), and ESC-50 PESQ improves by +0.016. Full Pareto results across sizes are on the Results page of the demo.

## How it works

```
mic ─▶ 16 kHz ─▶ STFT (20 ms sqrt-Hann, 10 ms hop) ─▶ CRN ─▶ complex mask ⊙ noisy spectrum ─▶ iSTFT/OLA ─▶ speaker
                                      │
          freq-only conv encoder ─▶ 2-layer GRU (time) ─▶ conv decoder w/ skips
```

- **Input features:** power-law compressed magnitude (|X|^0.3) plus normalized real and imaginary parts.
- **Loss:** compressed complex spectral loss (c = 0.3), with optional multi-resolution STFT and SI-SDR terms.
- **Training data:** VoiceBank+DEMAND, augmented with dynamic noise remixing over SNRs from −5 to 20 dB.
- **Latency budget:** 20 ms algorithmic (one window), plus inference (< 1 ms/frame target), plus the audio I/O buffer.

## Repository layout

```
ml/        PyTorch package: data, models, training, evaluation, ONNX export, benchmarks
web/       Vite + React + TypeScript frontend (onnxruntime-web), deployable as a static site
scripts/   Dataset download and preparation
PLAN.md    Detailed design, contracts and milestones
```

## Quickstart

**Requirements:** [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically) and Node 20+.

```bash
# 1. Data (~3 GB download): VoiceBank+DEMAND → 16 kHz
uv run scripts/prepare_vbd.py --out data

# 2. Train and evaluate
cd ml
uv sync
uv run pytest
uv run python -m ans.train --config configs/crn_small.yaml
uv run python -m ans.evaluate --models noisy classical crn:checkpoints/crn_small/best.pt

# 3. Export for the web
uv run python -m ans.export_onnx --ckpt checkpoints/crn_small/best.pt --out ../web/public/models

# 4. Run the frontend
cd ../web
npm install
npm run dev        # http://localhost:5173
```

Training runs on Apple Silicon (`mps`), CUDA or CPU. The device is selected automatically.

## Deploying the frontend

The web app is a static bundle (`web/dist`). Any static host works:

- **Vercel / Netlify / Cloudflare Pages:** root `web`, build command `npm run build`, output directory `dist`.
- **GitHub Pages:** set `BASE_PATH=/<repo-name>/` before building.

No server, GPU or API keys are needed, because inference runs client-side in WebAssembly.

## Datasets and acknowledgements

- **VoiceBank+DEMAND:** C. Valentini-Botinhao, *Noisy speech database for training speech enhancement algorithms and TTS models*, University of Edinburgh, 2017. Licensed CC BY 4.0. Demo clips on the site come from its test set.
- Metrics: [`pesq`](https://github.com/ludlows/PESQ) (ITU-T P.862.2) and [`pystoi`](https://github.com/mpariente/pystoi).
- Architecture inspiration: Tan & Wang (CRN, 2018), Luo & Mesgarani (Conv-TasNet, 2019), and the DNS Challenge baselines.

## License

- **Code:** MIT.
- **Third-party data and models** keep their own terms:
  - VoiceBank+DEMAND and LibriSpeech: CC BY 4.0.
  - MUSAN: mixed CC and public domain.
  - ESC-50: CC BY-NC 3.0. Used for evaluation only and not redistributed.
  - DNSMOS: Microsoft DNS-Challenge terms.
  - RNNoise: BSD.
  - DeepFilterNet: MIT or Apache-2.0.
