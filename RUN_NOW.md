# What to run now

You have **L2-ARCTIC** (Indian) in `data/l2_arctic_flat/`. You do **not** have a trained accent model yet (`models/` has no `.pth`). So you need to **train** first, then **run inference**.

---

## If you don’t have BDL data yet

1. Download **CMU ARCTIC** (BDL speaker): https://festvox.org/cmu_arctic/
2. Put WAVs in: `data/cmu_arctic_bdl/raw/`  
   Filenames should match L2 (e.g. `bdl_arctic_a0001.wav`).

---

## 1. Manifests (once per dataset)

```bash
cd /Users/nithinyanna/Downloads/accent_rvc_mvp
source venv/bin/activate   # or: . venv/bin/activate

# L2 (you already have WAVs here)
python -m training.build_manifest --wav_dir data/l2_arctic_flat

# BDL (after you add WAVs)
python -m training.build_manifest --wav_dir data/cmu_arctic_bdl/raw
```

---

## 2. Extract features (L2 and BDL)

```bash
# Indian
python -m training.feature_extract \
  --processed_dir data/l2_arctic_flat \
  --out_dir data/l2_arctic_flat/features

# American (BDL)
python -m training.feature_extract \
  --processed_dir data/cmu_arctic_bdl/raw \
  --out_dir data/cmu_arctic_bdl/features
```

---

## 3. Build paired manifest (Indian ↔ American same sentence)

```bash
python -m training.build_paired_manifest \
  --l2_dir data/l2_arctic_flat \
  --bdl_dir data/cmu_arctic_bdl/raw \
  --out data/l2_arctic_flat/paired_manifest.json
```

Check the log: it should say how many pairs were found (e.g. 1000+). If 0, fix BDL filenames.

---

## 4. Train Indian → American model

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
  --save_every 2000
```

This creates `models/bdl_accent_rvc.pth` and `models/config.json`. Training is CPU-friendly but slow; 50k steps can take hours.

---

## 5. Run inference (WAV → WAV)

```bash
python run_inference.py \
  -i data/l2_arctic_flat/ASI_arctic_a0001.wav \
  -o samples/indian_to_american.wav \
  --model_dir models \
  --model_name bdl_accent
```

Or with full-file mode (no chunking) for debugging:

```bash
python run_inference.py -i data/l2_arctic_flat/ASI_arctic_a0001.wav -o samples/out.wav --model_dir models --model_name bdl_accent --streaming 0
```

**If the output has disturbance (clicks, hiss, pumping):** use full-file and/or disable the post gate:
`--streaming 0 --no-post-gate`

---

## Short summary

| Step | Command |
|------|--------|
| Manifests | `build_manifest` for L2 and BDL dirs |
| Features | `feature_extract` for L2 and BDL |
| Pairs | `build_paired_manifest` |
| Train | `train_accent` (50k steps) |
| Infer | `run_inference.py -i in.wav -o out.wav --model_name bdl_accent` |

**Right now:** If BDL data is already in `data/cmu_arctic_bdl/raw/`, run steps 1 → 2 → 3 → 4 → 5 in order. If BDL is missing, add it first, then run 1–5.
