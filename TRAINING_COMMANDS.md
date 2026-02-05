# Training Commands for Accent RVC MVP

## ⚠️ IMPORTANT: Python 3.12 Compatibility Issue

**Current Issue:** The `checkpoint_best_500.pt` ContentVec model requires fairseq, which has Python 3.12 compatibility issues. 

**Solutions:**
1. **Use Python 3.11 or earlier** (recommended)
2. **Download standalone ContentVec model** (doesn't require fairseq):
   ```bash
   # Download contentvec_256.pt from:
   # https://github.com/auspicious3000/contentvec/releases
   # Place it in assets/contentvec/contentvec_256.pt
   ```

## Setup (if not already done)

```bash
cd /Users/nithinyanna/Downloads/accent_rvc_mvp
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Preprocess Raw Audio Data

Preprocesses the CMU-ARCTIC BDL audio files (converts to mono, resamples to 40kHz, segments into 2-6 second clips):

```bash
python -m training.preprocess \
  --raw_dir data/cmu_arctic_bdl/raw \
  --out_dir data/cmu_arctic_bdl/processed \
  --sr 40000 \
  --min_dur 2.0 \
  --max_dur 6.0
```

This creates:
- Processed WAV files in `data/cmu_arctic_bdl/processed/`
- A `manifest.json` file listing all segments

## Step 2: Extract Features (ContentVec + RMVPE)

Extracts content features using ContentVec and pitch features using RMVPE:

```bash
python -m training.feature_extract \
  --processed_dir data/cmu_arctic_bdl/processed \
  --out_dir data/cmu_arctic_bdl/features \
  --contentvec_dir assets/contentvec \
  --device cpu
```

**Note:** Make sure you have:
- ContentVec model: `assets/contentvec/checkpoint_best_500.pt` ✅ (already present)
- RMVPE weights: `assets/rmvpe/rmvpe.pt` (check if needed)
- HiFi-GAN weights: `assets/hifigan/hifigan.pth` (check if needed)

This creates:
- `.npz` feature files in `data/cmu_arctic_bdl/features/`
- A `feature_index.json` file

## Step 3: Train the RVC Model

Trains the voice conversion model:

```bash
python -m training.train_rvc \
  --feature_dir data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl \
  --batch_size 4 \
  --accum 4 \
  --steps 2000 \
  --save_every 500
```

**Parameters:**
- `--batch_size 4`: Batch size (adjust based on your CPU memory)
- `--accum 4`: Gradient accumulation steps (effective batch size = 4 × 4 = 16)
- `--steps 2000`: Total training steps
- `--save_every 500`: Save checkpoint every 500 steps

This creates:
- `models/bdl_rvc.pth`: Trained model weights
- `models/config.json`: Model configuration

## Step 4: (Optional) Build Retrieval Index

Builds a retrieval index for better voice similarity:

```bash
python -m training.build_index \
  --feature_dir data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl
```

This creates:
- `models/bdl.index`: Retrieval index file

## Step 5: Test Conversion

Convert a sample audio file:

```bash
python convert_accent.py \
  --input data/l2_arctic_flat/sample.wav \
  --output samples/sample_bdl.wav \
  --model_dir models \
  --device cpu
```

---

## Quick Start (All Steps Combined)

If you want to run everything in sequence:

```bash
cd /Users/nithinyanna/Downloads/accent_rvc_mvp
source venv/bin/activate

# Step 1: Preprocess
python -m training.preprocess \
  --raw_dir data/cmu_arctic_bdl/raw \
  --out_dir data/cmu_arctic_bdl/processed \
  --sr 40000 \
  --min_dur 2.0 \
  --max_dur 6.0

# Step 2: Extract features
python -m training.feature_extract \
  --processed_dir data/cmu_arctic_bdl/processed \
  --out_dir data/cmu_arctic_bdl/features \
  --contentvec_dir assets/contentvec \
  --device cpu

# Step 3: Train model
python -m training.train_rvc \
  --feature_dir data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl \
  --batch_size 4 \
  --accum 4 \
  --steps 2000 \
  --save_every 500

# Step 4: Build index (optional)
python -m training.build_index \
  --feature_dir data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl
```

---

## Notes

- **CPU Training:** Expect 1-3× realtime for inference, but training can take hours on CPU
- **Memory:** Adjust `--batch_size` if you run out of memory (try 2 or 1)
- **Training Time:** Full training may take several hours on CPU; use fewer `--steps` for quick tests
- **Checkpoints:** Model saves every `--save_every` steps, so you can resume if interrupted
