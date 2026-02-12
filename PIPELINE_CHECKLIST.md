# Pipeline checklist (22k, index, silence, training)

## 1) One consistent SR: 22,050 Hz ✅

- **SR_MODEL = 22050** in `rvc/mel_config.py` and `rvc/config.py`.
- Mel extraction uses that SR (no time-downsample).
- Vocoder (jik876 HiFi-GAN) runs at 22.05 kHz; output is 22050 (optional `--out_sr 44100` for playback).
- No 40k↔22k mel hack.

## 2) Training: L1 + multiscale mel ✅; full RVC v2 optional

- **Done:** L1 mel, L2 mel, multi-scale mel loss (see `training/train_rvc.py --loss_l1 --loss_l2 --multiscale_mel`).
- **Not done:** Multi-resolution STFT on waveform, adversarial discriminator, feature matching (would require RVC v2 training pipeline or vocoder-in-loop + target wav in features).
- For “commercial-grade” quality, consider integrating standard RVC v2 training (generator + discriminator + MR-STFT + feature matching). Current setup is an improvement over mel-MSE only.

## 3) Target voice: BDL only ✅

- Train on **CMU-ARCTIC BDL** features only (`data/cmu_arctic_bdl/features`).
- Indian (L2-ARCTIC) = inference inputs only, not training target.

## 4) Retrieval index (accent reduction) ✅

- **Build index:**  
  `python training/build_index.py --feature_dir data/cmu_arctic_bdl/features --model_dir models [--k 16]`
- **Inference:**  
  `--index_rate 0.6` (try 0.7–0.9), `--protect 0.2`, `--index_k 16`.
- Index is loaded from `model_dir` (or `--index_path`) when `index_rate > 0`; content is blended with kNN-retrieved BDL content.

## 5) Streaming vs full-file ✅

- **Full-file** first for debugging: `--streaming 0`.
- **Streaming:** 160 ms window, 80 ms hop, overlap-add; enable once full-file is intelligible.

## 6) Silence gate ✅

- **Chunk gate:** RMS below threshold (e.g. -45 to -55 dBFS), hangover (default 5 frames), optional passthrough (`--silence_passthrough`).
- **Post-process gate:** Optional RMS gate on output (`--no-post-gate` to disable).
- Bypass conversion on silent chunks (output clean silence or passthrough).

---

## Quick commands

**Build BDL index (once):**
```bash
python training/build_index.py --feature_dir data/cmu_arctic_bdl/features --model_dir models --k 16
```

**Convert with accent reduction (start low; high index_rate can mumble):**
```bash
# If 0.6 sounds like mumbling, the model wasn't trained on blended content — try 0.2–0.3
python convert_accent.py -i samples/original_input.wav -o samples/out_index.wav --model_dir models --streaming 0 --index_rate 0.3 --protect 0.2 --index_k 16
```

**Stricter silence gate (-50 dBFS, passthrough):**
```bash
python convert_accent.py -i in.wav -o out.wav --model_dir models --silence_db -50 --silence_hang 5 --silence_passthrough
```
