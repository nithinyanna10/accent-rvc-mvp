# Data (datasets)

You **manually** download and place datasets here. No scraping or auto-download is implemented.

## CMU-ARCTIC BDL (target voice)

- **Source:** [CMU ARCTIC](https://festvox.org/cmu_arctic/)  
- **Place:** `data/cmu_arctic_bdl/raw/`  
- Put BDL speaker WAVs in `raw/`.  
- Then run:  
  `python -m training.preprocess --raw_dir data/cmu_arctic_bdl/raw --out_dir data/cmu_arctic_bdl/processed ...`  
  to get `data/cmu_arctic_bdl/processed/` and a manifest.

## L2-ARCTIC (test inputs)

- **Source:** [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic/)  
- **Place:** e.g. `data/l2_arctic/` or any folder.  
- Use ASI, RRBI, TNI (or other) WAVs as input to `convert_accent.py --input <path> --output samples/out.wav`.

## Layout example

```
data/
  cmu_arctic_bdl/
    raw/           <- BDL WAVs
    processed/     <- from preprocess
    features/      <- from feature_extract
  l2_arctic/       <- L2-ARCTIC WAVs for conversion
```
