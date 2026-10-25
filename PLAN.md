# Implementation Plan — Real-Time Speech Enhancement for Voice Pickup

This is the working plan for the project. It sets the architecture, the **contracts between components** (which let separate workstreams proceed in parallel), the experiments, the milestones, and the acceptance criteria. `README.md` is the public-facing summary; this file is the engineering source of truth.

> **Naming.** This is *speech enhancement / noise suppression for the microphone path*, not active noise cancellation (ANC). Public-facing text uses "Real-Time Speech Enhancement" and never "ANS" or "ANC". The internal Python package keeps the name `ans`.

---

## 0. Goals and non-goals

**Goal.** A causal, low-latency neural speech enhancer that removes background noise (traffic, cafés, offices, public spaces, transport) from a single microphone signal. It must be small enough to run in real time on a CPU and **in the browser**, and it should come with:

1. A reproducible training and evaluation pipeline (PyTorch), with PESQ, STOI and SI-SDR on VoiceBank+DEMAND, plus **out-of-domain and real-recording evaluation** (DNSMOS).
2. A **research contribution:** knowledge distillation from a non-causal teacher into causal students. The headline result is a quality-vs-compute Pareto plot.
3. Exported ONNX streaming models with a measured latency and memory budget.
4. A hostable web app for uploading, recording or live-mic denoising. It includes an A/B comparison, spectrograms, a suppression-strength control, and the results.

**Use case framing.** Clean mic pickup for calls on headphones and earbuds. The constraints that come with that (causality, ≤ ~20–30 ms algorithmic latency, a small compute and memory budget) drive the design decisions below.

**Non-goals for v1.** Acoustic echo cancellation, multi-mic beamforming, full dereverberation, wind noise (see stretch goals) and training on the full DNS Challenge corpus.

---

## 1. Architecture overview

```
                 ┌──────────────────────── Python (ml/) ─────────────────────────┐
 VoiceBank+DEMAND│ prepare_vbd.py → 16 kHz wavs + manifests                       │
 + extra noise / ──▶ Dataset (crops, dynamic remixing, reverb, mic/channel, level) │
   RIRs          │      ▼                                                         │
                 │ STFT (n_fft=320, hop=160, sqrt-Hann) ─▶ Model ─▶ iSTFT         │
                 │   ans_crn (causal students) · crn_teacher (non-causal)         │
                 │   conv_tasnet · classical Wiener · RNNoise/DeepFilterNet (ref) │
                 │      ▼                                                         │
                 │ train.py  (+ distillation)    evaluate.py (intrusive + DNSMOS, │
                 │ export_onnx.py (+atten_lim)    distortion/NR decomposition,    │
                 │ benchmark.py (latency+memory)  OOD sets, significance tests)   │
                 └────────────────────────────────┬──────────────────────────────┘
                                                  │ ans_stream.onnx, manifest.json, test_vectors.json,
                                                  │ results.json, listening/*.wav, failures/*.wav
                 ┌──────────────────────── Web (web/) ────────────────▼──────────┐
                 │ Vite + React + TS + Tailwind, onnxruntime-web (wasm)           │
                 │ Studio · Live · Results (+Pareto, limits) · How it works       │
                 │ hidden: Diagnostics, Listen (blind MOS test)                   │
                 │ Fully static → Vercel / Netlify / Cloudflare Pages / GH Pages  │
                 └────────────────────────────────────────────────────────────────┘
```

**Two shipped models** (§4.12): a **benchmark model** trained only on VoiceBank+DEMAND, for comparison with published results, and a **demo model** trained with extra noise and augmentation that runs on the website. A model trained only on VoiceBank+DEMAND generalizes poorly to real laptop and phone mics, and the live demo would expose that.

---

## 2. Repository layout and ownership

```
├── README.md  PLAN.md  .gitignore  .github/workflows/ci.yml
├── scripts/
│   ├── prepare_vbd.py         # VoiceBank+DEMAND (owner: data) ✅
│   ├── prepare_noise.py       # MUSAN (train) + ESC-50 (OOD test) + RIRs (owner: data)
│   └── build_ood_sets.py      # fixed-seed OOD + competing-talker test sets (owner: data)
├── data/                      # gitignored
├── ml/                        # Python package `ans` (owner: ml)
│   ├── configs/*.yaml         # one file per run in the run registry (§6.2)
│   ├── ans/{dsp,data,augment,losses,metrics,dnsmos,stream,train,distill,
│   │        evaluate,stats,export_onnx,benchmark,enhance,render}.py
│   ├── ans/models/{crn,conv_tasnet,classical,reference}.py
│   └── tests/
├── recordings/                # real-world clips (§4.10.3), small, committed with consent
└── web/                       # frontend (owner: web)
    └── public/
        ├── models/            # written by ml
        ├── samples/           # written by data
        ├── listening/         # written by ml/render.py (§5.1 Listen)
        ├── failures/          # written by ml/render.py (§5.1 Results)
        └── results.json       # written by ml/evaluate.py
```

**Ownership rule.** Each workstream writes only inside the paths it owns. Everything under `web/public/` other than the app's own assets is written by ml or data and only *read* by the web app. The web app must degrade gracefully when any of those files are missing.

---

## 3. Contracts

### 3.1 Audio and DSP contract (shared by Python and JS — must match exactly)

| Parameter       | Value                                                          |
|-----------------|----------------------------------------------------------------|
| Sample rate     | 16 000 Hz, mono, float32 in [-1, 1]                            |
| Frame (n_fft)   | 320 samples (20 ms)                                            |
| Hop             | 160 samples (10 ms), 50 % overlap                              |
| Window          | `sqrt(hann(320, periodic=True))` at **analysis and synthesis** |
| Bins            | 161 (rfft of 320); FFT `norm="backward"`                        |
| Analysis        | frame *t* = `window * x[t*hop : t*hop + 320]`, then `rfft`     |
| Synthesis       | `window * irfft(Y_t, n=320)`, overlap-added at `t*hop`. Hann at 50 % overlap sums to 1, so no extra normalization is needed. |
| Alg. latency    | 20 ms (one window) for the shipped model (lookahead k = 0)     |

**Streaming framing (clarification).** The streaming engine keeps a 160-sample input history and a 160-sample OLA tail, both starting at zeros. Each pushed hop of 160 samples produces one frame made of `[history, new hop]` and emits 160 finished output samples. The streamed output is therefore **the same length as the input and delayed by exactly 160 samples** compared with offline framing of the unpadded signal. `test_vectors.json` follows this convention: `stft_frames` are offline frames of the unpadded input, and `output_wav` is the raw streamed output.

### 3.2 Streaming model contract (ONNX)

One graph call processes **one STFT frame**. The model applies its own mask internally.

| Name           | Dir | dtype   | Shape                  | Meaning |
|----------------|-----|---------|------------------------|---------|
| `spec`         | in  | float32 | `[1, 1, 161, 2]`       | noisy STFT frame (real, imag) |
| `state_in`     | in  | float32 | `manifest.state_shape` | recurrent state; zeros at stream start |
| `atten_lim_db` | in  | float32 | `[1]`                  | **new (v0.2):** maximum attenuation in dB. The mask magnitude is floored at `10^(-lim/20)` and its phase is kept. `100` is effectively unlimited. |
| `spec_out`     | out | float32 | `[1, 1, 161, 2]`       | enhanced STFT frame |
| `state_out`    | out | float32 | `manifest.state_shape` | feed back as `state_in` |

- **Backward compatibility.** The manifest lists its inputs, and the JS runtime sends `atten_lim_db` only when the model declares it. Older models keep working, with the slider disabled.
- **Graph constraints.** Opset 17, static shapes, no `If` or `Loop` ops.
- **Attenuation limit.** It is a parameter-free clamp applied after the network, so it needs **no retraining**. Any trained checkpoint can be re-exported with it.

`manifest.json` (v0.2; keys marked *opt.* may be absent):
```json
{
  "name": "crn-demo", "version": "0.2.0", "variant": "demo | benchmark", "trained": true,
  "sample_rate": 16000, "n_fft": 320, "hop": 160, "window": "sqrt_hann_periodic",
  "lookahead_frames": 0,
  "onnx": "ans_stream.onnx", "onnx_int8": "ans_stream_int8.onnx",
  "inputs":  {"spec": [1,1,161,2], "state_in": [2,1,256], "atten_lim_db": [1]},
  "outputs": {"spec_out": [1,1,161,2], "state_out": [2,1,256]},
  "state_shape": [2,1,256],
  "atten_lim_db": {"default": 100, "min": 0, "max": 100},
  "params": 0, "macs_per_frame": 0,
  "budget": {"weights_kb_fp32": 0, "weights_kb_int8": 0, "peak_activation_kb": 0, "mmacs_per_s": 0},
  "training": {"data": "VoiceBank+DEMAND + MUSAN", "distilled": false, "git_hash": "…"},
  "metrics": null
}
```
`test_vectors.json` is unchanged from v0.1, plus one extra case run at `atten_lim_db = 12`. Tolerances: STFT 1e-4, model 1e-3, waveform 1e-3.

### 3.3 Results contract — `web/public/results.json` (v0.2)

Unknown keys must be ignored by consumers. Each metric value is either a number or `{"mean", "ci95", "n"}`.
```json
{
  "generated_at": "ISO-8601", "git_hash": "…",
  "test_sets": {
    "vbd":     {"label": "VoiceBank+DEMAND test (824)", "intrusive": true,  "noise_seen_in_training": false},
    "ood_esc": {"label": "VB-D speech + ESC-50 noise (500)", "intrusive": true, "noise_seen_in_training": false},
    "talker":  {"label": "Competing talker (300)", "intrusive": true,  "noise_seen_in_training": false},
    "real":    {"label": "Real recordings (n)", "intrusive": false}
  },
  "rows": [
    {"model": "CRN-256 (KD)", "set": "vbd", "family": "ours", "causal": true, "distilled": true,
     "pesq": 0, "stoi": 0, "estoi": 0, "si_sdr": 0, "dnsmos_sig": 0, "dnsmos_bak": 0, "dnsmos_ovrl": 0,
     "speech_distortion_sisdr": 0, "noise_reduction_db": 0,
     "params": 0, "macs_per_frame": 0, "ms_per_frame_cpu": 0, "rtf": 0}
  ],
  "per_noise_type": {"<set>": {"<noise>": {"<model>": {"pesq": 0, "stoi": 0}}}},
  "per_snr":        {"<set>": {"<snr>":   {"<model>": {"pesq": 0, "noise_reduction_db": 0, "speech_distortion_sisdr": 0}}}},
  "pareto":  [{"model": "CRN-64", "hidden": 64, "distilled": false, "seed": 0, "macs_per_frame": 0, "pesq": 0, "dnsmos_ovrl": 0}],
  "atten_sweep": [{"atten_lim_db": 6, "pesq": 0, "speech_distortion_sisdr": 0, "noise_reduction_db": 0}],
  "lookahead":   [{"lookahead_frames": 0, "latency_ms": 20, "pesq": 0}],
  "quantization":[{"variant": "fp32 | int8-dynamic | int8-static", "size_kb": 0, "ms_per_frame": 0, "pesq": 0}],
  "significance":[{"a": "CRN-128 (KD)", "b": "CRN-128", "metric": "pesq", "set": "vbd", "median_diff": 0, "p_wilcoxon": 0}],
  "per_sample":  {"<sample id>": {"pesq": 0, "stoi": 0, "si_sdr": 0}},
  "mos":         null
}
```
The v0.1 single-set rows (no `set` key) are treated as `set: "vbd"`.

### 3.4 Samples contract — `web/public/samples/samples.json` ✅ (unchanged)
The six VoiceBank+DEMAND test clips span the five test noise types (bus, cafe, living room, office, public square). Per-clip metrics live in `results.json → per_sample`.

### 3.6 Listening-test and failure-gallery contracts (written by `ml/ans/render.py`)

`web/public/listening/index.json`:
```json
{
  "version": 1,
  "conditions": ["noisy", "wiener", "crn", "crn_kd", "clean"],
  "hidden_reference": "clean", "low_anchor": "noisy",
  "practice": {"id": "practice", "files": {"noisy": "listening/practice_noisy.wav", "…": "…"}},
  "trials": [{"id": "t01", "utterance": "p232_104", "set": "vbd", "noise_type": "office", "snr_db": 2.5,
              "files": {"noisy": "listening/t01_noisy.wav", "wiener": "…", "crn": "…", "crn_kd": "…", "clean": "…"}}]
}
```
The web app runs the conditions it finds in `conditions`, so a missing `crn_kd` (before distillation lands) just shortens each trial. A response export (`listen-<rater>-<timestamp>.json`) looks like `{"version": 1, "rater_id", "started_at", "device_note", "ratings": [{"trial", "condition", "score" (1–5), "position", "ms"}]}`.

`web/public/failures/index.json`:
```json
[{"id": "music", "title": "Music and singing", "category": "music | silence | low_snr | talker | reverb | clipping",
  "explanation": "One or two sentences on what goes wrong and why.",
  "noisy": "failures/music_noisy.wav", "enhanced": "failures/music_enhanced.wav", "clean": null,
  "model": "CRN-256 (benchmark)"}]
```
All clips are 16 kHz mono PCM16 and at most 10 s long.

### 3.5 Data layout ✅ plus additions
```
data/vbd16k/{train,test}/{clean,noisy}/<id>.wav, manifests/{train,val,test}.csv      ✅ done
data/noise/musan16k/<id>.wav            + manifests/noise_train.csv    (training-only noise)
data/noise/esc50_16k/<id>.wav           + manifests/noise_ood.csv      (test-only noise; disjoint corpus)
data/rir/{sim,real}/<id>.wav            + manifests/rir_{train,test}.csv
data/ood/{esc,talker}/{clean,noisy}/<id>.wav + manifests/ood_{esc,talker}.csv
data/real/<id>.wav                      + manifests/real.csv           (from recordings/)
```
**Corpus separation rule.** Noise, RIRs and interfering speakers used in **training** must never appear in **evaluation** sets. MUSAN noise and simulated RIRs are for training. ESC-50, held-out RIRs and LibriSpeech `test-clean` speakers are for testing.

**As built ✅ (Phase A).**
- **Manifest location:** the new manifests live in `data/manifests/`: `noise_train`, `noise_ood`, `talkers_test`, `rir_train`, `rir_test`, `ood_esc` and `ood_talker` (the `talker` set). Failure sources are in `data/failures_src/manifest.csv`.
- **Leakage removed:** MUSAN and ESC-50 both draw from freesound. Cross-correlation found 27 ESC-50 clips that are excerpts of MUSAN training noise, and they were removed from the test pool (audit: `esc50_musan_overlap.csv`).
- **Measured RIRs are test-only** for licence reasons (OpenSLR 28's RWCP and REVERB sets are research-only). Training reverb uses 3,000 simulated RIRs.
- **SNR and SIR** use the ITU-T P.56 active speech level, matching VoiceBank+DEMAND.

---

## 4. Machine learning design

### 4.1 Data pipeline and augmentation
- **VoiceBank+DEMAND** ✅ is prepared at 16 kHz: 10 802 train, 770 val (p226 and p287) and 824 test utterances.
- **Crops and remixing:** random 2 s crops. With p = 0.5, the clean speech is remixed with another utterance's noise (`noisy − clean`) at an SNR drawn from U[−5, 20] dB. ✅ implemented.
- **New augmentations.** Each one is a config flag, is off for the benchmark model, and is ablated in §6.2:
  - `extra_noise`: with probability p, the noise comes from MUSAN instead (noise subset; training-only corpus).
  - `reverb`: with probability p, the clean speech is convolved with an RIR (pyroomacoustics shoebox simulation with T60 from 0.1 to 0.8 s; measured RIRs are held out for testing, see §3.5). **The training target is the direct path plus early reflections (the first 50 ms)**, so the model learns mild dereverberation without being pushed into artifacts.
  - `mic`: a random high-pass at 80–200 Hz, a ±6 dB spectral tilt, random band-limiting (4–8 kHz low-pass) and occasional soft clipping. Applied to the **input only**.
  - `level`: an extended gain range of [−45, −3] dBFS, so whispered and shouted speech are covered.
- **Dataloader:** soundfile decoding, `num_workers = min(8, cpus − 2)`, persistent workers.

### 4.2 Primary model — `ans_crn` (causal CRN, complex mask) ✅
- **Structure:** frequency-only convolutions (kernel (1,3), stride (1,2), channels 16/32/48/64), then `Linear → GRU(hidden, 2 layers) → Linear`, then a transposed-conv decoder with U-Net skips. The output is a bounded complex ratio mask (|M| ≤ 1.2) and `Y = M ⊙ X`.
- **Size:** the 256-hidden variant has 1.21 M params and 1.96 M MACs per frame.
- `forward(spec, state, atten_lim_db=None, return_mask=False)`. The mask is returned only in Python, for the decomposition in §4.6, and is never part of the ONNX contract.
- `hidden` ∈ {64, 128, 256} defines the student family (§4.9).
- `lookahead` k ∈ {0, 1, 2} is used only in the experiment in §4.11.

### 4.3 Baselines and reference systems
1. **Noisy**, unprocessed ✅.
2. **Classical Wiener filter** with minimum-statistics noise estimation ✅. Scores PESQ 2.20 and STOI 0.899 on VB-D test.
3. **Conv-TasNet (causal, small).** Too slow on the M4 (about 15 h), so it runs on a cloud GPU or with a 50-epoch schedule. Optional for M4.
4. **Off-the-shelf reference rows** (`models/reference.py`):
   - **RNNoise**, via the `pyrnnoise` binding or a built `rnnoise` binary; native 48 kHz.
   - **DeepFilterNet3**, via the `deepfilternet` package; native 48 kHz.
   - Both are run by resampling 16 → 48 → 16 kHz. They are framed honestly as *trained on far more data, at full band*, and are expected to beat us on several sets. We don't need to beat them; we need to show where we stand.

### 4.4 Losses
- **Target loss:** compressed complex spectral loss, `L = α·MSE(|Y|^c e^{jφY}, |S|^c e^{jφS}) + (1−α)·MSE(|Y|^c, |S|^c)`, with c = 0.3 and α = 0.3 ✅.
- **Optional:** multi-resolution STFT loss and −SI-SDR (both weight 0 by default) ✅.
- **Distillation loss (§4.9):** `L_student = L_target(Y_s, S) + β·L_spec(Y_s, sg(Y_t)) [+ γ·MSE(h_s W, sg(h_t))]`.

### 4.5 Training ✅
AdamW (lr 1e-3), a 2-epoch warmup then cosine decay, 100 epochs, batch 32, gradient clip 5, EMA 0.999, MPS. Each run takes about 111 s per epoch for CRN-256 on the M4 Pro, so 3.2 h for 100 epochs. Best checkpoint is chosen by validation PESQ (200-clip subset).
**Every run logs** its resolved config, git hash, seed, data manifest hashes and the pip-freeze digest, to TensorBoard and `checkpoints/<run>/meta.json`.

### 4.6 Evaluation
- **Intrusive metrics:** PESQ-WB, STOI, ESTOI, SI-SDR ✅.
- **Non-intrusive metrics:** **DNSMOS P.835** (SIG, BAK, OVRL) through the ONNX models from Microsoft's DNS-Challenge repository (`ans/dnsmos.py`). Computed on every set, and the only metric available for real recordings.
- **Unseen noise.** The VoiceBank+DEMAND test noise types (bus, cafe, living room, office, public square) **do not appear in training**, which uses babble, cafeteria, car, kitchen, meeting, metro, restaurant, ssn, station and traffic. So the per-noise-type results are entirely unseen conditions, and the Results page says so.
- **Speech distortion vs. noise reduction (new).** Since `Y = M ⊙ X = M ⊙ S + M ⊙ N` and `N = noisy − clean` is known, the predicted mask is applied separately to the clean and noise STFTs:
  - *Speech distortion* = SI-SDR(iSTFT(M ⊙ S), s), in dB; higher means less damage to the voice.
  - *Noise reduction* = 10·log10(‖N‖² / ‖M ⊙ N‖²), in dB; higher means more noise removed.
  - Both are reported per model and per SNR bucket, and plotted as a two-axis tradeoff. This applies only to mask-based models (CRN family and Wiener), not to Conv-TasNet, RNNoise or DeepFilterNet; those cells show "—".
- **Attenuation sweep (new).** `atten_lim_db` ∈ {6, 12, 18, 24, ∞}, each scored on PESQ, speech distortion, noise reduction and DNSMOS BAK.
- **Statistics (`ans/stats.py`).** Mean ± 95 % CI, plus a **paired Wilcoxon signed-rank test** on per-utterance scores for every claim that one model is better than another (Holm-corrected within a family). A claim goes into the README only when p < 0.05.
- **Output.** `evaluate.py --sets vbd ood_esc talker real --models …` writes `ml/exports/results.json` (§3.3), which is copied to `web/public/results.json`.

### 4.7 Export, quantization and deployment budget
- **Export ✅.** Plus the `atten_lim_db` input. Re-export the test vectors, including the `atten_lim_db = 12` case.
- **Quantization.** Inspect which ops become int8 (the export prints an op-type × dtype table). `quantize_dynamic` currently covers only MatMul/Gemm, so **the GRU stays fp32**. Add **`quantize_static`** (QDQ, per-channel, about 100 calibration clips from val) for the convs and linears. Report fp32 vs. int8-dynamic vs. int8-static on file size, ms/frame and PESQ (VB-D plus OOD), and ship int8 only if PESQ drops by less than 0.05.
- **Deployment budget table** (in the manifest and on the model card):
  - weight memory (fp32 and int8);
  - peak activation memory per frame (largest live tensor set, measured via ORT profiling or analytically);
  - state memory;
  - MMAC/s (MACs per frame × 100);
  - ms/frame on M4 (torch, ORT fp32, ORT int8), in the browser (wasm), and **on a Raspberry Pi 4/5 with ORT if one is available** as a stand-in for constrained hardware.
- **Stretch:** TFLite (ai-edge-torch) and CoreML.

### 4.8 Tests ✅ plus additions
- **Existing ✅:** DSP round trip, streaming = offline, ORT = torch, overfit one batch, metric sanity.
- **New:**
  - The attenuation floor is respected (min |M| ≥ 10^(-lim/20) − ε), and `lim = 100` matches the unlimited model.
  - The decomposition is consistent: iSTFT(M⊙S) + iSTFT(M⊙N) ≈ ŷ to 1e-4.
  - The KD loss is zero when student equals teacher.
  - A lookahead-k model's streaming output equals its offline output shifted by k frames.
  - Augmentation preserves alignment: the target and input lengths match, and the reverb target is the early-reflection signal.
  - Corpus separation: the train and test noise manifests have no overlapping file IDs or source corpora.
  - The Wilcoxon wrapper matches scipy.

### 4.9 Knowledge distillation (the main contribution, new)
- **Teacher (`crn_teacher`):** the same CRN topology with a **bidirectional** GRU, hidden size 512, and non-causal. It is trained once on the same data (and a second time for the demo-data variant). It is never exported.
- **Students:** causal `ans_crn` with hidden ∈ {64, 128, 256}, each trained **with and without** distillation under identical configs and seeds.
- **Loss:** as in §4.4, with β swept over {0.5, 1.0} at hidden 128 and the best value used everywhere. Feature distillation (γ, on the GRU output through a learned projection) is an optional ablation.
- **Efficiency:** augmentation makes every crop unique, so teacher outputs can't be cached. The teacher runs online in `torch.no_grad()` in fp16 on MPS, which costs about 1.3–1.5× the student's training time.
- **Headline figure:** **PESQ (and DNSMOS OVRL on the OOD sets) vs. MACs per frame**, with two curves (plain vs. KD) and teacher plus reference systems as points. Hidden 128 is trained with **2 seeds** in each arm, so seed variance is visible as error bars.
- **Honesty rule:** the plot goes at the top of the README only if KD beats plain significantly (Wilcoxon, §4.6) at two or more of the three sizes. Otherwise it is reported as a finding ("distillation gives +x at small sizes, nothing at 256"), which is still useful.

### 4.10 Out-of-domain and real-world evaluation (new)
1. **Mismatched noise (`ood_esc`):** VB-D clean test speech mixed with ESC-50 noise from about 20 relevant classes (rain, wind, traffic, car horn, engine, train, siren, keyboard typing, mouse click, washing machine, vacuum cleaner, footsteps, door knock, clock alarm, crying baby, laughing, coughing, pouring water, dishes/can opening, helicopter) at SNR ∈ {−5, 0, 5, 10, 15} dB. Fixed seed, 500 clips, built by `scripts/build_ood_sets.py`.
2. **Competing talker (`talker`):** VB-D test speech mixed with a LibriSpeech `test-clean` speaker at signal-to-interference ratios of {0, 5, 10} dB, 300 clips. We expect the model to largely **keep** the interfering voice, since it was never trained to remove speech. This is reported as a measured limitation, not something to optimize for.
3. **Real recordings (`real`):** 20–30 clips recorded by the author on an actual earbud and a phone mic, in a café, on a street, in an office, outdoors and in a car. Each is 8–15 s long and logged in `recordings/manifest.csv` (device, place, notes). Scored with DNSMOS P.835 only. Optionally, the real-recording portion of the DNS Challenge blind test set is added.
4. **Reference rows** (RNNoise, DeepFilterNet3) are evaluated on every set.

### 4.11 Latency ablation: lookahead (new; offline experiment)
- **Model:** the CRN with k ∈ {0, 1, 2} frames of lookahead, implemented by training the model to emit frame *t − k* after seeing frames up to *t*. The GRU stays causal; the output and target are shifted by k.
- **Latency:** total algorithmic latency = 20 + 10·k ms.
- **Result:** PESQ / DNSMOS vs. latency.
- **Scope:** the shipped model stays at k = 0. If a lookahead model is ever exported, `manifest.lookahead_frames` tells the JS engine about the extra delay. This replaces the earlier window-size ablation, which is now optional.

### 4.12 Shipped models
| Variant | Data | Purpose |
|---|---|---|
| `benchmark` | VB-D only (+ dynamic remixing) | Published-comparable numbers; shown on the Results page |
| `demo`      | VB-D + MUSAN + reverb + mic + level augmentation, KD if it helps | **Runs on the website** (Studio and Live); scored on all sets |

---

## 5. Web application design

**Stack ✅:** Vite 6, React 18, TypeScript 5.9 (strict), Tailwind 4, onnxruntime-web 1.30 (wasm, single thread), a hand-written mixed-radix FFT (N = 320), Vitest. Python ↔ JS parity is verified to below 1e-5.

### 5.1 Pages
1. **Studio ✅.** Upload, record or sample clips; worker-based processing; A/B/C playback; spectrograms; SI-SDR; download. **Changes:**
   - **The attenuation-limit slider** (0–100 dB, default from the manifest, a "∞" detent at the top) **replaces the dry/wet slider**. Moving it re-processes the clip in the worker (fast enough for 60 s clips).
   - Per-sample PESQ and STOI from `results.json → per_sample`.
   - A model switch between benchmark and demo, if both are shipped.
2. **Live ✅.** Adds the attenuation-limit slider, which applies on the next frame with no restart. Needs testing on real hardware (Chrome, Safari, Firefox; AirPods and laptop mic).
3. **Results ✅.** Additions:
   - The Pareto plot (plain vs. KD) as the hero chart.
   - A test-set selector (vbd / ood_esc / talker / real).
   - The speech-distortion vs. noise-reduction chart by SNR.
   - The attenuation sweep, lookahead and quantization tables.
   - A significance badge on comparisons.
   - The model card with the deployment-budget table.
   - **"Known limitations"**: a failure-case gallery (§5.3) with playable noisy/enhanced pairs.
   - Every set is labelled "seen" or "unseen" noise.
4. **How it works ✅.** Adds a distillation diagram (teacher → students) and a short explanation of the attenuation limit.
5. **Diagnostics ✅** (`#/diagnostics`). Adds the `atten_lim_db = 12` vector case.
6. **Listen (new, hidden `#/listen`).** A blind MOS test, following ITU-T P.808 / ACR in spirit:
   - Each trial is one utterance with **5 conditions in random order**: noisy, Wiener, CRN, CRN + KD, and a **hidden clean reference**. The clean reference and noisy input are the anchors used to screen raters.
   - The listener rates each condition 1–5 for overall quality. Headphones are required, and there is a short training trial first.
   - 20 trials per listener, drawn from pre-rendered clips in `web/public/listening/` (generated by `ml/ans/render.py`, since only one model runs in the browser).
   - Responses are kept in localStorage and exported as JSON. If a form endpoint is set (`VITE_LISTEN_ENDPOINT`, e.g. Formspree or a Google Form), they are POSTed there as well.
   - `ml/ans/stats.py mos` aggregates them into MOS ± 95 % CI per condition. Raters who score the clean reference below the noisy input on more than 30 % of trials are excluded.

### 5.2 Engine changes
- `model.ts` reads `manifest.inputs`. It feeds `atten_lim_db` when the model declares it, exposes `setAttenLimit(db)`, and honours `lookahead_frames` in the latency display.
- `dsp.ts` is unchanged.

### 5.3 Failure-case gallery (rendered by `ml/ans/render.py` into `web/public/failures/`)
Music and singing; long silence with residual noise (to check for pumping); very low SNR (−5 to −10 dB); competing talker; heavy reverb; clipped input. Each entry has a one-line explanation of what goes wrong and why.

### 5.4 Quality bar ✅
Responsive, dark by default, accessible, main bundle 67 kB gzipped (budget 300 kB). Graceful degradation everywhere. Lighthouse ≥ 90 is still to be measured.

---

## 6. Milestones, run registry and schedule

### 6.1 Milestones

| #   | Milestone | Deliverables | Status / exit criteria |
|-----|-----------|--------------|------------------------|
| M0  | Scaffolding | Data, `ml/` package and tests, web app, parity | ✅ done (pytest 36, vitest 26, diagnostics pass) |
| M1  | Baselines | Noisy and Wiener evaluated | ✅ done (noisy 1.97 / 0.921 matches the literature) |
| M2  | Benchmark CRN | R0 trained and evaluated; **`atten_lim_db` in the contract, export and web UI**; contract fixes; per-sample metrics; trained model live on the site | PESQ ≥ 2.7, STOI ≥ 0.93 on VB-D |
| M3  | Evaluation suite | Distortion/NR decomposition, attenuation sweep, DNSMOS, OOD sets (`ood_esc`, `talker`), RNNoise and DeepFilterNet3 rows, Wilcoxon, deployment budget, static int8 | All rows filled for the existing models |
| M4  | Research | Teacher plus 3 students × {plain, KD} (+ seed 2 at 128), Pareto plot, lookahead k = 1, 2, augmentation ablations, demo model | Pareto plot with significance; demo model beats benchmark on OOD sets |
| M4.5| Human evaluation | Real recordings (`real` set), listening test (10–15 listeners × 20 trials), failure gallery | MOS table with CIs; gallery live |
| M5  | Ship | Deployed site, README with final results and hero plot, demo video/GIF, model card, CI green | Public URL |
| S   | Stretch | Wind noise with a dedicated wind test set (wind is turbulence at the mic, so it needs its own data), 48 kHz model, TFLite/CoreML, target-speaker enhancement, Conv-TasNet on a cloud GPU | — |

### 6.2 Run registry (compute budget on the M4 Pro, one run at a time on MPS)

| Run | Config | Purpose | Est. time |
|-----|--------|---------|-----------|
| R0  | CRN-256, VB-D, seed 0 | Benchmark model; plain student at 256 | ✅ 2.9 h. Test PESQ 2.762, STOI 0.937, SI-SDR 16.73 |
| R1  | Teacher, bi-GRU 512, VB-D | Distillation teacher | ~5 h |
| R2, R3 | CRN-64, CRN-128 plain | Pareto, plain arm | ~2.5 h each |
| R4–R6 | CRN-64/128/256 + KD | Pareto, KD arm | ~4 h each |
| R7, R8 | CRN-128 plain / KD, seed 1 | Seed variance | ~2.5 h + ~4 h |
| R9  | CRN-128 KD, β = 0.5 | β sweep (R5 uses β = 1.0) | ~4 h |
| R10, R11 | CRN-256, k = 1 and k = 2 | Lookahead | 3.2 h each |
| R12b | CRN-256 plain, 50 epochs | Matched baseline for R12–R14 (they train for 50 epochs, not 100) | ~1.7 h |
| R12–R14 | CRN-256 50 epochs: +MUSAN, +reverb, +mic/level | Augmentation ablations, judged on OOD sets | ~1.7 h each |
| R15 | Teacher on demo data | KD teacher for the demo model | ~6 h |
| R16 | Demo model (all augmentation, KD if it helped) | Ships on the site | ~5 h |
| —   | Conv-TasNet | Optional baseline | 15 h (cloud GPU or skip) |

**Status (2026-10-24).** R0–R8 are done. The queue was reordered so that R12b and R12–R14 run **before** R10/R11, because the benchmark model generalizes poorly out of domain (ESC-50 PESQ 1.59, against 2.15 for DeepFilterNet3). The Pareto chart now defaults to the OOD set, since VB-D test barely separates model sizes (CRN-64 2.78 vs CRN-256 2.76).

**Total ≈ 70 h of training** (about 55 h without R15). On one Mac that means **about 5–7 days of chained runs**, mostly overnight. Alternatively, a cloud GPU (one A10/L4 at roughly $0.5–1/h) running 3–4 runs in parallel brings it down to 1–2 days for under $50. A queue script (`ml/scripts/run_queue.sh`) chains runs, resumes after interruptions, and evaluates each finished run automatically.

### 6.3 Schedule (what can start when)

| Phase | When | Work | Blocked by |
|-------|------|------|------------|
| **A** | **Now**, in parallel with R0 (CPU, network) | Contract v0.2 + `atten_lim_db` (export + web); decomposition + attenuation sweep in `evaluate.py`; DNSMOS; RNNoise/DeepFilterNet rows; download MUSAN, ESC-50, LibriSpeech test-clean; build OOD sets; augmentation module; distillation + lookahead code; stats; deployment-budget benchmark; static int8; queue script; web additions (slider, Results v2, Listen route, Pareto chart) | Nothing; all built against the contracts and tested on synthetic data and the R0 checkpoints |
| **B** | R0 finishes (~3 h) | Evaluate R0 on every set; export with `atten_lim_db`; trained model live on the site; README numbers | R0 |
| **C** | Days 1–6 | Run queue R1 → R16, each auto-evaluated; Pareto and ablation results fill in as they finish | MPS time |
| **D** | Any time, **needs you** | Record 20–30 real clips; recruit 10–15 listeners; create a GitHub repo for CI and deployment | You |
| **E** | After C and D | Render listening and failure clips, run the listening test, final README, deploy | C, D |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `pesq` fails to build on macOS | ✅ Not an issue (it builds on Python 3.12) |
| MPS ops are unsupported or slow | ✅ Real-valued STFT via matmul; `PYTORCH_ENABLE_MPS_FALLBACK=1`; cloud GPU option |
| Python/JS DSP drift | ✅ Test vectors + Diagnostics page (parity < 1e-5) |
| VB-D is "easy", so the demo disappoints on real mics | Demo model with extra noise, reverb, mic and level augmentation; OOD + real-recording evaluation; honest reporting |
| Distillation gain is within seed noise | 2 seeds at hidden 128; Wilcoxon; honesty rule (§4.9) |
| **The GRU stays fp32 after quantization** | Expected with ORT dynamic/static quantization; report which ops are int8; ship int8 only if PESQ drops < 0.05; note it in the model card |
| Train/test leakage across noise corpora | Corpus separation rule (§3.5) + test |
| RNNoise / DeepFilterNet packages fail to install on Python 3.12 / arm64 | Isolated `uv run --with` environment or a separate venv for reference systems; precompute their outputs once and cache them |
| DNSMOS licence and usage terms | Check the DNS-Challenge repo terms before redistributing scores or models; never redistribute its ONNX files in this repo, download them at runtime |
| Browser audio latency and dropouts | ✅ Worker + ring buffers (54 ms end to end measured); test on real hardware |
| Training queue contention (one MPS) | Strictly serial queue; Phase A work stays on CPU; cloud option |
| Listening test quality | Anchors, headphone requirement, rater screening, CIs |

---

## 8. Reproducibility and hygiene
- **CI (GitHub Actions, `.github/workflows/ci.yml`):**
  - `ml`: `uv sync`, then `ruff` and `pytest` (CPU, synthetic data only).
  - `web`: `typecheck`, `lint`, `vitest` (including the onnxruntime-node parity test against `test_vectors.json`) and `build`.
  - Runs on every push and PR.
- **Experiment tracking:** every run records its config, git hash, seed and data manifest hashes (§4.5). There is one YAML per registry run, and results are traceable from `results.json` back to the run's git hash.
- **Statistics:** 95 % CIs plus paired Wilcoxon with Holm correction. No claim goes in without p < 0.05.
- **Licensing** (README section):

  | Resource | Licence / terms |
  |---|---|
  | VoiceBank+DEMAND | CC BY 4.0 |
  | MUSAN | Mixed (mostly CC / public domain; see its README) |
  | ESC-50 | **CC BY-NC 3.0** (non-commercial; fine for a portfolio, and clips are not redistributed) |
  | LibriSpeech | CC BY 4.0 |
  | DNSMOS | Microsoft DNS-Challenge terms |
  | RNNoise | BSD |
  | DeepFilterNet | MIT / Apache-2.0 |
  | Real recordings | Recorded with consent; the author's own |

---

## 9. Commands

```bash
# data
uv run scripts/prepare_vbd.py --out data                 # ✅
uv run scripts/prepare_noise.py --out data               # MUSAN, ESC-50, LibriSpeech test-clean, RIRs
uv run scripts/build_ood_sets.py --out data --seed 0

# ml (from ml/)
uv sync && uv run pytest && uv run ruff check
uv run python -m ans.train --config configs/crn_small.yaml                    # R0
uv run python -m ans.train --config configs/teacher_bigru512.yaml             # R1
uv run python -m ans.train --config configs/crn128_kd.yaml                    # e.g. R5
bash scripts/run_queue.sh configs/queue_m4.txt                                # chain R1…R16
uv run python -m ans.evaluate --sets vbd ood_esc talker real --models noisy classical rnnoise deepfilternet crn:checkpoints/crn_small/best.pt --decompose --atten-sweep --merge
uv run python -m ans.export_onnx --ckpt checkpoints/<demo>/best.pt --variant demo --out ../web/public/models --int8 --int8-static
uv run python -m ans.benchmark --ckpt … --onnx ../web/public/models/ans_stream.onnx --budget
uv run python -m ans.render --listening --failures --out ../web/public
uv run python -m ans.stats mos responses/*.json

# web (from web/)
npm install && npm run dev
npm run typecheck && npm run lint && npm test && npm run build
```
