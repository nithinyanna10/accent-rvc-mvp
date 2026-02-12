# Data (datasets)

Datasets are organized and ready for use.

## CMU-ARCTIC BDL (target voice - US English)

- **Location:** `data/cmu_arctic_bdl/raw/`  
- **Files:** 906 WAV files from BDL speaker  
- **Source:** [CMU ARCTIC](https://festvox.org/cmu_arctic/)  
- **Usage:**  
  ```bash
  python -m training.preprocess \
    --raw_dir data/cmu_arctic_bdl/raw \
    --out_dir data/cmu_arctic_bdl/processed \
    --sr 40000 --min_dur 2.0 --max_dur 6.0
  ```

## L2-ARCTIC (source - Indian English)

- **Location:** `data/l2_arctic_flat/`  
- **Speakers:** ASI, RRBI, TNI (Indian English)  
- **Files:** 3,392 WAV files (prefixed with speaker ID)  
- **Source:** [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic/)  
- **Usage:**  
  ```bash
  python convert_accent.py \
    --input data/l2_arctic_flat/ASI_arctic_a0001.wav \
    --output samples/asi_converted.wav \
    --model_dir models
  ```

## Optional target: LJSpeech

For **LJSpeech** as US English target (instead of BDL): place WAVs in e.g. `data/ljspeech/raw/`, run `build_manifest.py` and `feature_extract.py`, then train with `train_rvc.py` (same-speaker) or build parallel pairs with L2-ARCTIC if you have sentence alignment. The inference pipeline is unchanged; point `--model_dir` to the LJSpeech-trained model.

## Current structure

```
data/
  cmu_arctic_bdl/
    raw/           <- 906 BDL WAVs (ready for preprocessing)
    processed/     <- (created by preprocess.py)
    features/      <- (created by feature_extract.py)
  l2_arctic_flat/  <- 3,392 L2-ARCTIC WAVs (ASI, RRBI, TNI)
  ljspeech/        <- (optional) LJSpeech WAVs as alternative US target
```
