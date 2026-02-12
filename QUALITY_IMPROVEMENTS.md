# Improving accent and reducing disturbance

## 1. Retrain with temporal model (better accent + less jitter)

The default generator is **frame-wise** (no context). For better accent and more stable output, retrain with **temporal** (1D conv over time):

```bash
python -m training.train_accent \
  --paired_manifest data/l2_arctic_flat/paired_manifest.json \
  --l2_features data/l2_arctic_flat/features \
  --bdl_features data/cmu_arctic_bdl/features \
  --model_dir models \
  --name bdl_accent \
  --steps 50000 \
  --normalize_mel \
  --multiscale_mel \
  --temporal
```

This produces a model that uses **TemporalGenerator** (saved with `"generator": "temporal"` in config). Inference will load it automatically. You can keep the same `--model_name bdl_accent`; the new checkpoint overwrites the old.

## 2. Reduce disturbance at inference (no retrain)

- **Full-file** (avoids chunk seams): `--streaming 0`
- **Disable post gate** (avoids gate clicks): `--no-post-gate`
- **Low-pass** (reduces hiss): `--lowpass 10000` or `--lowpass 12000`

Example:

```bash
python run_inference.py \
  -i data/l2_arctic_flat/ASI_arctic_a0001.wav \
  -o samples/clean.wav \
  --model_dir models \
  --model_name bdl_accent \
  --streaming 0 \
  --no-post-gate \
  --lowpass 10000
```

## 3. Train longer

If accent is still not strong enough, try more steps (e.g. 80k–100k) with the temporal model:

```bash
python -m training.train_accent ... --steps 80000 --temporal
```

## Summary

| Goal | Action |
|------|--------|
| Better accent + more stable mel | Retrain with `--temporal` |
| Less hiss | `--lowpass 10000` at inference |
| No gate clicks | `--no-post-gate` |
| No chunk seams | `--streaming 0` |
