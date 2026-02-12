# RVC v2 (Soft-VC) Inference Pipeline — Specification

This document specifies the **Python-based inference pipeline** for converting **Indian English accent → US English accent** using the consensus **RVC v2 (Soft-VC)** architecture.

---

## 1. Architecture

| Component | Choice | Role |
|-----------|--------|------|
| **Content encoder** | **ContentVec** (or HuBERT-Soft) | Strips source accent; produces speaker-agnostic content features from 16 kHz audio. |
| **Pitch extraction** | **RMVPE** (Robust Model for Vocal Pitch Estimation) | Extracts F0; preserved and fed to the generator to keep intonation. |
| **Generator** | Trained mapping (content + F0 → mel) | Produces target-speaker mel spectrogram. |
| **Vocoder** | **HiFi-GAN v2** | Mel → waveform at 22.05 kHz. |

Pipeline flow: **WAV (source) → resample to 16k → ContentVec → content; RMVPE → F0 → Generator(content, F0) → mel → HiFi-GAN → WAV (target).**

---

## 2. Data Sources

- **Source (Indian English):** **L2-ARCTIC**  
  Speakers: **ASI, RRBI, or TNI** (or other L2-ARCTIC speakers).  
  Used as input at inference; for training, used in parallel with target data (see INDIAN_TO_AMERICAN.md).

- **Target (US English):**  
  - **CMU-ARCTIC (Speaker BDL)** — primary target; same prompts as L2-ARCTIC for parallel training.  
  - **LJSpeech** — alternative target; use the same training pipeline with LJSpeech WAVs and (if needed) aligned prompts.

---

## 3. Inference Script (WAV → WAV)

**Single entry point:** one Python script takes a WAV file and writes a converted WAV file (target speaker, preserved intonation, accent reduced).

### Option A: Minimal CLI (`run_inference.py`)

```bash
python run_inference.py -i input.wav -o output.wav [--model_dir models] [--model_name bdl_accent]
```

### Option B: Full options (`convert_accent.py`)

```bash
python convert_accent.py --input input.wav --output output.wav --model_dir models [--model_name bdl_accent] [--streaming 0]
```

- **Input:** Any WAV (e.g. L2-ARCTIC Indian English).
- **Output:** Converted WAV at 22.05 kHz (or `--out_sr 44100`), same duration, target US English voice (BDL or LJSpeech depending on trained model).

---

## 4. Model Training (for reproducibility)

- **Target-voice (same speaker):** `training/train_rvc.py` on CMU-ARCTIC BDL (or LJSpeech) only → use for BDL→BDL or as baseline.
- **Accent conversion (Indian → US):** `training/train_accent.py` on **parallel** L2-ARCTIC + CMU-ARCTIC BDL (same utterance IDs) → use with `--model_name bdl_accent` at inference. See **INDIAN_TO_AMERICAN.md**.

Weights and config: **ContentVec** in `assets/contentvec/`, **RMVPE** in `assets/rmvpe/`, **HiFi-GAN** in `models/` or `assets/hifigan/`. See `assets/README.md`.

---

## 5. Requirements Checklist

| Requirement | Implementation |
|-------------|----------------|
| Content encoder: ContentVec (or HuBERT-Soft) | `rvc/content_encoder.py` (ContentVec, 16k input) |
| Pitch: RMVPE | `rvc/pitch_rmvpe.py` |
| Vocoder: HiFi-GAN v2 | `rvc/vocoder_hifigan.py` (22k) |
| Source: L2-ARCTIC (ASI, RRBI, TNI) | Supported; use L2 WAVs as input and/or in paired training |
| Target: CMU-ARCTIC BDL or LJSpeech | BDL: full path in docs; LJSpeech: same training flow, different data dir |
| One script: WAV in → WAV out | `run_inference.py` / `convert_accent.py` |
| Preserve intonation, remove accent | F0 from source (RMVPE); content from source (ContentVec); generator trained to output target mel |
