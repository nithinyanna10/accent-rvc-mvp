# Indian → American Accent Conversion (Pipeline)

**Goal:** Input = Indian English speech (e.g. L2-ARCTIC). Output = same content in American (BDL) accent/voice.

This is done by **training on parallel data**: same sentences in Indian and American. L2-ARCTIC and CMU ARCTIC share the same prompts, so we can pair e.g. `ASI_arctic_a0001.wav` (Indian) with `bdl_arctic_a0001.wav` (American).

---

## 1. Data you need

- **L2-ARCTIC** (Indian): e.g. `data/l2_arctic_flat/` with WAVs like `ASI_arctic_a0001.wav`, and a `manifest.json` (from `build_manifest.py`).
- **CMU ARCTIC BDL** (American): e.g. `data/cmu_arctic_bdl/raw/` with WAVs like `bdl_arctic_a0001.wav`. Create a manifest there too.

BDL filenames must match the L2 utterance ID. Typical forms:
- L2: `{SPEAKER}_arctic_{a|b}{NNNN}.wav` → utterance ID = `arctic_a0001`
- BDL: `bdl_arctic_a0001.wav` or `bdl_arctic0001.wav` (script tries to match both)

---

## 2. Step-by-step commands

### 2.1 Manifests (if not already present)

```bash
# L2 (Indian)
python -m training.build_manifest --wav_dir data/l2_arctic_flat

# BDL (American)
python -m training.build_manifest --wav_dir data/cmu_arctic_bdl/raw
```

### 2.2 Extract features (Indian and American)

```bash
# Indian: content + f0 from L2 WAVs
python -m training.feature_extract \
  --processed_dir data/l2_arctic_flat \
  --out_dir data/l2_arctic_flat/features

# American: content + f0 + mel from BDL WAVs (we use mel as target)
python -m training.feature_extract \
  --processed_dir data/cmu_arctic_bdl/raw \
  --out_dir data/cmu_arctic_bdl/features
```

### 2.3 Build paired manifest (match by utterance ID)

```bash
python -m training.build_paired_manifest \
  --l2_dir data/l2_arctic_flat \
  --bdl_dir data/cmu_arctic_bdl/raw \
  --out data/l2_arctic_flat/paired_manifest.json
```

Check the log: it should report how many pairs were found. If 0, fix BDL/L2 filenames so IDs align.

### 2.4 Train accent conversion model

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

Output: `models/bdl_accent_rvc.pth` and `models/config.json`. This model is trained to map **Indian content + f0** → **BDL mel** (American).

### 2.5 Convert Indian WAV → American

Use the same `convert_accent.py`; pass the accent checkpoint with `--model_name bdl_accent`:

```bash
python convert_accent.py \
  -i data/l2_arctic_flat/ASI_arctic_a0001.wav \
  -o samples/indian_to_american.wav \
  --model_dir models \
  --model_name bdl_accent \
  --streaming 0
```

This loads `models/bdl_accent_rvc.pth` and `models/config.json` (the one produced by `train_accent`, with `"accent_conversion": true`). For BDL-only models you can omit `--model_name` (script auto-detects `bdl_rvc.pth`).

---

## 3. What this fixes

- **Before:** Model was trained only on BDL (BDL content → BDL mel). At inference we fed Indian content, so the model had never seen Indian input → mumbling.
- **After:** Model is trained on **pairs**: (Indian content, Indian f0) → BDL mel. So at inference, Indian input is in-distribution and the model is trained to produce American (BDL) mel.

You still use the same pipeline (ContentVec + RMVPE → generator → mel → vocoder); only the training data and the checkpoint change.

---

## 4. If you don’t have BDL data yet

1. Download **CMU ARCTIC** (BDL speaker) and put WAVs in e.g. `data/cmu_arctic_bdl/raw/` with names like `bdl_arctic_a0001.wav` (or ensure `build_paired_manifest.py` can match your naming).
2. Run the steps above from 2.1.

---

## 5. Optional: multiple Indian speakers

L2-ARCTIC has several Indian speakers (e.g. ASI, RRBI, TNI). The paired manifest can include all of them; each L2 file is matched to the BDL file with the same utterance ID. So you get more training pairs and the model sees more Indian variation.
