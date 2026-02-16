# Improving accent and reducing disturbance

## 1. Retrain with temporal_v2 (best quality)

For best accent and stability, retrain with **TemporalGeneratorV2** (512 hidden, 4 residual blocks):

```bash
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

- **--temporal_v2**: bigger model (512 hidden, residual conv blocks). Saves as `"generator": "temporal_v2"`.
- **--steps 80000**: more steps often improve accent.
- Optional **--hidden 384** if 512 is too slow; default 512.

## 2. Inference defaults (quality-focused)

**run_inference.py** now defaults to:
- **Full-file** (`--streaming 0`) to avoid chunk seams
- **Low-pass 11 kHz** to reduce hiss
- **Mel temporal smoothing** (5 frames) before vocoder to reduce jitter

So a minimal run is already tuned for quality:

```bash
python run_inference.py -i in.wav -o out.wav --model_dir models --model_name bdl_accent
```

To disable gate or mel smoothing: `--no-post-gate` and/or `--no-mel-smooth`. To turn off low-pass: `--lowpass 0`.

## 3. Train longer

If accent is still not strong enough, try 100k steps with temporal_v2:

```bash
python -m training.train_accent ... --steps 100000 --temporal_v2
```

## Summary

| Goal | Action |
|------|--------|
| Best accent + stability | Retrain with `--temporal_v2` (80k+ steps) |
| Less hiss | Default lowpass 11k; or `--lowpass 0` to disable |
| No gate clicks | `--no-post-gate` |
| No chunk seams | Default is full-file (`--streaming 0`) |
| Less mel jitter | Default mel smoothing 5 frames; `--no-mel-smooth` to disable |
