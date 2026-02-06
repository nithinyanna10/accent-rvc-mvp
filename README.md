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



