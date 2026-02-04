# Samples (before/after WAVs)

- **Input:** L2-ARCTIC (or other) WAVs used as `--input`.  
- **Output:** Converted WAVs from `convert_accent.py --output`.

## Naming convention

- **Before:** Keep original filenames or use e.g. `*_input.wav`, `*_l2.wav`.  
- **After:** Use e.g. `*_bdl.wav`, `*_converted.wav` to indicate target speaker (BDL).

Example:

- Input: `samples/asi_utt1.wav`  
- Output: `samples/asi_utt1_bdl.wav`

Generated WAVs are gitignored; only this README is tracked.
