# Big quality push — what changed

## 1. **TemporalGeneratorV2** (new model)

- **512 hidden**, 4 residual conv blocks, kernel 9.
- Train with: `--temporal_v2` (and optionally `--steps 80000`).
- Config: `"generator": "temporal_v2"`, `"hidden": 512`. Loaded automatically at inference.

## 2. **Mel smoothing** before vocoder

- Light 1D smoothing along time (default 5 frames) on predicted mel before vocoder.
- Reduces frame-to-frame jitter and often cleans up artifacts.
- Disable with `--no-mel-smooth` in `run_inference.py`.

## 3. **Stronger F0 smoothing**

- F0 median filter size increased from 7 to **11** frames for smoother pitch and less robotic sound.

## 4. **Inference defaults** (quality-first)

- **Full-file** by default (`--streaming 0`) to avoid chunk-boundary artifacts.
- **Low-pass 11 kHz** by default to reduce hiss (`--lowpass 0` to disable).
- **Mel smoothing** on (5 frames); `--no-mel-smooth` to disable.
- **Post gate** still on (-60 dBFS); `--no-post-gate` to disable.

## 5. **Training**

- **--temporal_v2**: use the big residual temporal model (recommended).
- **--hidden 512**: default for temporal/temporal_v2 (smaller: 384 if too slow).
- **--steps 80000** (or 100k): recommended for better accent.

---

## What to run

**Retrain (best quality):**
```bash
cd /Users/nithinyanna/Downloads/accent_rvc_mvp
source venv/bin/activate

python -m training.train_accent \
  --paired_manifest data/l2_arctic_flat/paired_manifest.json \
  --l2_features data/l2_arctic_flat/features \
  --bdl_features data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl_accent \
  --steps 80000 \
  --normalize_mel \
  --multiscale_mel \
  --temporal_v2
```

**Inference (defaults are already quality-focused):**
```bash
python run_inference.py -i input.wav -o output.wav --model_dir models --model_name bdl_accent
```

To turn off low-pass or mel smoothing: add `--lowpass 0` and/or `--no-mel-smooth`.
