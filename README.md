# accent_rvc_mvp

**Indian English → US English accent conversion (MVP)** using the RVC v2 / Soft-VC stack: ContentVec content encoder, RMVPE pitch, HiFi-GAN v2 vocoder. This repo is **CPU-only** (no CUDA, no MPS), runs on macOS with Python 3.10+ via venv and pip, and is structured for a later Windows port.

**What it does:** Takes a WAV (e.g. L2-ARCTIC Indian English) and a trained target-voice model (e.g. CMU-ARCTIC BDL), and outputs a converted WAV that sounds like the target speaker while preserving intonation and reducing accent. Inference is streaming-shaped (160 ms windows + overlap-add) to avoid future leak.

**What it doesn’t:** No automatic dataset downloads (you place CMU-ARCTIC and L2-ARCTIC manually). No GUI/WebUI; CLI only. Training is CPU-friendly but slower than GPU; use short clips for iteration.

---

## Quick start

### 1. Create venv and install

```bash
cd accent_rvc_mvp
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data

- **CMU-ARCTIC BDL:** Download from [CMU ARCTIC](https://festvox.org/cmu_arctic/) and place WAVs in `data/cmu_arctic_bdl/raw/`.
- **L2-ARCTIC:** Download (ASI, RRBI, TNI, etc.) and place test WAVs in e.g. `data/l2_arctic/` or any folder you prefer.

### 3. Preprocess BDL

```bash
python -m training.preprocess \
  --raw_dir data/cmu_arctic_bdl/raw \
  --out_dir data/cmu_arctic_bdl/processed \
  --sr 40000 \
  --min_dur 2.0 --max_dur 6.0
```

This creates `data/cmu_arctic_bdl/processed/` and a manifest (e.g. `manifest.json`).

### 4. Feature extraction

```bash
python -m training.feature_extract \
  --processed_dir data/cmu_arctic_bdl/processed \
  --out_dir data/cmu_arctic_bdl/features \
  --contentvec_dir assets/contentvec
```

Requires ContentVec weights in `assets/contentvec/` and RMVPE in `assets/rmvpe/` (see `assets/README.md`). Writes per-segment `.npz` and a feature index.

### 5. Train target-voice model

```bash
python -m training.train_rvc \
  --feature_dir data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl \
  --batch_size 4 --accum 4 \
  --steps 2000 --save_every 500
```

Produces `models/bdl_rvc.pth` and `models/config.json`.

### 6. (Optional) Build retrieval index

```bash
python -m training.build_index \
  --feature_dir data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl
```

Produces `models/bdl.index` (and metadata) for `--index_rate` at inference.

### 7. Convert L2-ARCTIC samples

```bash
python convert_accent.py \
  --input data/l2_arctic/sample.wav \
  --output samples/sample_bdl.wav \
  --model_dir models \
  --device cpu
```

Use `--silence_db -45 --silence_hang 3` to tune the silence gate. See `samples/README.md` for naming conventions.

---

## CPU expectations and clip length

- **Inference:** On a modern Mac (M1/M2 or Intel i7), expect roughly 1–3× realtime for 160 ms chunked processing (e.g. 1 min of audio in ~1–3 min). Older CPUs will be slower.
- **Training:** Use small batches and gradient accumulation; 2–6 s segments are recommended. Full BDL training to convergence can take hours on CPU; shorten `--steps` for quick sanity runs.
- **Recommendation:** Test with 5–15 s L2-ARCTIC clips first.

---

## Three pitfalls and how this repo addresses them

### 1. Future leak / streaming shape

- **Risk:** Using full-file context in the model or features causes future leak and breaks real-time/streaming.
- **Approach:** Inference is designed around **160 ms windows** with **80 ms hop** (configurable). `StreamingChunker` yields overlapped chunks; `OverlapAdd` crossfades outputs. The pipeline never looks ahead beyond the current chunk; it’s producer/consumer compatible and safe for streaming.

### 2. Normalization mismatch

- **Risk:** InstanceNorm (or similar) trained on batch statistics can behave differently at inference (single chunk, different scale).
- **Approach:** We avoid InstanceNorm where possible in the minimal RVC compat layer. If you load external RVC weights that use it, behavior is documented in `rvc/vc_model.py`: inference uses running stats or eval mode; we recommend LayerNorm or no norm in custom modules.

### 3. Silence hallucinations

- **Risk:** Model generates noise or “mumbling” on silent or near-silent input.
- **Approach:** `rvc/silence_gate.py` implements an **energy-based gate** (configurable threshold in dBFS, e.g. -45 dB, and hangover frames). When a chunk is below threshold, we bypass conversion and output near-silence (or optional passthrough). Optional noise-floor logic can be enabled for very quiet segments.

---

## Write-up (for Jakob)

- **ContentVec and accent:** ContentVec (or HuBERT-Soft) gives a **content representation** that encodes phonetic/linguistic information and suppresses speaker identity and accent. We use these features as input to the target-voice model so that the conversion preserves “what is said” while the generator and vocoder produce “how it sounds” in the target voice (e.g. BDL). That’s how we strip accent while keeping linguistic content.

- **RMVPE and intonation:** RMVPE extracts **f0** with high temporal resolution and robustness to noise. We feed f0 (and optionally scaled f0 for pitch shift) into the model so that the output preserves the **intonation** of the source (e.g. Indian English) while the timbre is replaced by the target (e.g. US English BDL). This keeps prosody and avoids a flat or wrong pitch contour.

- **Future-leak avoidance:** The pipeline is **streaming-shaped**: we never pass the full file through the model. We use **160 ms windows** and **overlap-add** so that each step only sees the current chunk (and possibly a small past context inside the encoder if needed, but no future frames). So there is no dependence on future samples, and the same code path is suitable for real-time streaming later.

- **Silence hallucinations:** To avoid generating sound on silence, we use an **energy-based silence gate** (short-time RMS per chunk, threshold in dBFS, hangover). Chunks classified as silent bypass the model and output near-silence (or passthrough). Optional **noise-floor** logic can be added so that very low-level noise is not amplified by the vocoder.

---

## Layout

- `convert_accent.py` — CLI entrypoint: WAV in → converted WAV out.
- `rvc/` — Inference: config, audio I/O, streaming chunker, overlap-add, silence gate, content encoder, RMVPE pitch, HiFi-GAN vocoder, VC model compat, pipeline, utils.
- `training/` — Preprocess, feature extraction, train_rvc, build_index.
- `assets/` — Place pretrained weights (ContentVec, RMVPE, HiFi-GAN); see `assets/README.md`.
- `models/` — Trained BDL model and config; see `models/README.md`.
- `data/` — Raw and processed datasets; see `data/README.md`.
- `samples/` — Input/output WAVs; see `samples/README.md`.
