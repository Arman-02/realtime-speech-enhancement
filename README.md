# Real-Time Speech Enhancement for Voice Pickup

A causal neural noise suppressor that removes background noise (traffic, cafés, offices, public spaces, transport) from a single microphone signal in real time. It targets voice pickup in headphones and true-wireless earbuds during calls. This is microphone-path speech enhancement, not active noise cancellation (ANC). The model is small enough to run **in your browser**, frame by frame, with 20 ms algorithmic latency.

> **Status:** 🚧 in development. See [PLAN.md](PLAN.md) for the full implementation plan and milestones.
>
> **Live demo:** _coming soon_

---

## Highlights

- **Streaming by design.** A causal convolutional-recurrent network (CRN) predicts a bounded complex ratio mask on 20 ms / 10 ms-hop STFT frames. Frame-by-frame inference is bit-for-bit equivalent to offline inference, and this is enforced by tests.
- **Generalization, measured.** The benchmark-only model looked strong on VoiceBank+DEMAND but fell apart on unseen noise. Training-data augmentation (MUSAN noise, simulated rooms, mic coloration, level range) lifted out-of-domain PESQ by +0.55 and reverberant PESQ by +0.30. It now matches DeepFilterNet3 on unseen noise (PESQ) with a 1.2M-parameter causal model.
- **Distillation, honestly reported.** A non-causal teacher distilled into causal students gives a small, seed-consistent gain on VB-D-only training (+0.045 PESQ at 128 units, two seeds per arm), and **no gain** once the training data is augmented.
- **Measured beyond the benchmark.** PESQ-WB, STOI/ESTOI, SI-SDR and DNSMOS P.835 are reported on four test sets: VoiceBank+DEMAND (all noise types unseen in training), ESC-50 noise, held-out reverberant rooms, and competing talkers. Every comparison uses paired Wilcoxon tests with Holm correction, and RNNoise and DeepFilterNet3 are included as reference rows.
- **Why it sounds the way it does.** Each model's effect is split into *speech distortion* and *noise reduction*, and an adjustable attenuation limit lets you tune the tradeoff yourself.
- **Deployable.** The model is exported to ONNX (fp32 and int8) with per-frame latency and real-time factor benchmarks, and runs in the browser through `onnxruntime-web`.
- **Interactive frontend.** Upload or record audio, A/B noisy vs. enhanced vs. clean, view spectrograms, or switch on **live mic denoising**. The app is a fully static site with no server-side inference.

## Results

There are two CRN-256 models: a **benchmark model**, trained on VoiceBank+DEMAND only for comparability with published results, and a **demo model**, the same architecture trained with augmentation, which is what the website runs. Both are causal, have 1.21M parameters and run at 0.13 ms per 10 ms frame on one CPU thread (Apple M4 Pro, ONNX Runtime).

**PESQ / DNSMOS OVRL on four test sets:**

| Model | VoiceBank+DEMAND (824) | ESC-50 noise (500) | Reverberant rooms (300) | Competing talker (300) |
|---|---|---|---|---|
| Noisy (unprocessed) | 1.97 / 2.70 | 1.31 / 2.09 | 1.59 / 1.94 | 1.25 / 2.64 |
| Wiener (min. statistics) | 2.20 / 2.76 | 1.40 / 2.28 | 1.67 / 2.18 | 1.21 / 2.57 |
| RNNoise¹ | 2.09 / 2.94 | 1.74 / 2.87 | 1.51 / 2.35 | 1.26 / 2.80 |
| DeepFilterNet3¹ | 2.70 / **3.21** | **2.15** / **3.01** | 2.08 / **2.88** | 1.31 / **2.85** |
| CRN-256 benchmark | 2.76 / 3.03 | 1.59 / 2.45 | 1.91 / 2.51 | 1.29 / 2.69 |
| **CRN-256 demo** | **2.92** / 3.06 | 2.14 / 2.79 | **2.21** / 2.60 | **1.34** / 2.75 |
| Teacher (bi-GRU 512, offline, augmented) | 3.09 / 3.16 | 2.37 / 2.93 | 2.37 / 2.77 | 1.45 / 2.84 |

¹ These are full-band 48 kHz models, run here by resampling 16 → 48 → 16 kHz. They were trained on far more data, and their published numbers come from a different evaluation setup.

**What the numbers say** (paired Wilcoxon, Holm-corrected):
- **Augmentation is the biggest win.** Compared with the benchmark, the demo model gains +0.55 PESQ on ESC-50 noise, +0.30 on reverberant rooms and +0.16 on VoiceBank+DEMAND (all p < 1e-15). MUSAN noise accounts for most of the out-of-domain gain, and room-impulse augmentation for the reverb gain, at a cost of −0.02 PESQ on VB-D.
- **Against DeepFilterNet3:** the demo model is significantly better on PESQ for VB-D (+0.22), reverb (+0.13) and competing talkers (+0.03), and **tied on ESC-50 noise** (−0.007, p = 0.06). DeepFilterNet3 is consistently better on DNSMOS, by 0.15–0.28, so it likely *sounds* cleaner. A blind listening test (planned, via the site's hidden `#/listen` page) will show which metric listeners agree with.
- **The competing-talker test is a known limitation.** Every system keeps the interfering voice, because none of them is trained to tell speakers apart.
- **The reverb reference** is the direct path plus 50 ms of early reflections, which matches the reverb-augmentation target and therefore favours the demo model on intrusive metrics. DNSMOS, which uses no reference, shows the same ranking among the CRNs.

**Suppression vs. distortion.** The attenuation limit trades noise reduction against speech distortion smoothly:

| Attenuation limit | 6 dB | 12 dB | 18 dB | 24 dB | ∞ |
|---|---|---|---|---|---|
| Noise reduction (dB) | 6.6 | 9.7 | 11.0 | 11.4 | 11.5 |
| Speech distortion SI-SDR (dB, higher = less damage) | 22.2 | 21.0 | 20.6 | 20.5 | 20.5 |
| PESQ | 2.37 | 2.62 | 2.73 | 2.76 | 2.76 |

**Efficiency.** The CRN uses 1.96 M MACs per frame (0.2 GMAC/s) and runs at a real-time factor of 0.013 in ONNX Runtime, so one CPU core spends 1.3 % of real time on it. In the browser it takes about 0.2 ms per frame. Dynamic int8 quantization shrinks it to 3.7 MB with no loss in PESQ (2.76). The GRU stays fp32, so the size saving is modest.

**Distillation.**
- **VB-D-only training:** a non-causal bi-GRU-512 teacher (2.93 PESQ) distilled into causal students gives a small gain at 128 units. With two seeds per arm, both distilled students beat both plain ones: VB-D PESQ 2.795 and 2.807 against 2.740 and 2.772 (+0.045), and ESC-50 PESQ +0.016.
- **Augmented training:** with a teacher trained on the same augmented data, distillation gives **no significant gain** on any test set (|Δ PESQ| ≤ 0.01, p > 0.2). The shipped demo model is therefore trained without it.
- **Pareto curve:** quality vs. compute across 64/128/256 units is on the demo's Results page. On VB-D, size barely matters (CRN-64 scores 2.78, CRN-256 2.76), so the chart defaults to the out-of-domain set.

**Latency.** Allowing 10 or 20 ms of lookahead adds only +0.03 and +0.05 PESQ on VB-D (+0.02 and +0.04 out of domain). The shipped model therefore keeps 20 ms algorithmic latency, which is the right tradeoff for calls.

## How it works

```
mic ─▶ 16 kHz ─▶ STFT (20 ms sqrt-Hann, 10 ms hop) ─▶ CRN ─▶ complex mask ⊙ noisy spectrum ─▶ iSTFT/OLA ─▶ speaker
                                      │
          freq-only conv encoder ─▶ 2-layer GRU (time) ─▶ conv decoder w/ skips
```

- **Input features:** power-law compressed magnitude (|X|^0.3) plus normalized real and imaginary parts.
- **Loss:** compressed complex spectral loss (c = 0.3), with optional multi-resolution STFT and SI-SDR terms.
- **Training data:** VoiceBank+DEMAND with dynamic noise remixing over SNRs from −5 to 20 dB. The demo model also uses:
  - MUSAN noise;
  - 3,000 simulated rooms (the target keeps the direct path plus 50 ms of early reflections);
  - random mic EQ and band-limiting;
  - a −45 to −3 dBFS level range.
- **No leakage:** test-only noise, rooms and talkers come from separate corpora. A cross-correlation audit removed 27 ESC-50 clips that were copies of MUSAN training noise.
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
