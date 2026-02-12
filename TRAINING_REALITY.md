# Why You Get Mumbling — Training vs Use Case

## What we're actually training

- **Data:** BDL only. Each sample: `(content_BDL, f0_BDL) → mel_BDL`.
- **Model:** `MinimalGenerator`: frame-wise MLP. For each time step `t` it does  
  `(content[t], f0[t]) → mel[t]` **with no temporal context** (no conv, no attention over time).
- **Loss:** L1 mel + L2 + multi-scale mel. All good as far as they go.

So we're training: **“when I see BDL content + BDL f0, output BDL mel.”**

---

## What we're doing at inference

- **Input:** Indian (or other L2) WAV → Indian content + Indian f0.
- **Model:** Same net: `(content, f0) → mel`.
- So we're asking: **“given Indian content + Indian f0, output mel.”**

The model has **never** seen Indian content during training. It only ever saw BDL content. So at inference we're **out of distribution**: the network is extrapolating to a different speaker’s content space with a very small, frame-wise MLP. That’s a fundamental mismatch and a main reason for mumbling.

---

## Two main issues

### 1. Train vs infer mismatch (cross-speaker)

- **Trained on:** BDL content → BDL mel.
- **Used on:** Indian content → ??? mel.

So we are **not** training “Indian → BDL.” We’re training “BDL → BDL” and then using the same model for “Indian → BDL.” The model was never taught what to do with Indian content, so output quality (e.g. mumbling) is expected to be bad.

### 2. No temporal context

- The generator is **frame-independent**: each `(content[t], f0[t])` is mapped to `mel[t]` with no look at other frames.
- Real VC systems usually use temporal context (conv, attention, etc.) so each frame is predicted with surrounding context. Without that, frame-by-frame predictions can be inconsistent and sound like mumbling even when content is in-distribution.

---

## Sanity check: same-speaker (BDL → BDL)

To check that training itself is reasonable, run conversion on **BDL** (same speaker as training), not Indian. Use any WAV that was used for training (same speaker as the model).

```bash
# Use a BDL file as input (same content domain as training).
# If your BDL data is elsewhere, point -i to any BDL WAV.
python convert_accent.py -i /path/to/bdl_file.wav -o samples/bdl_to_bdl.wav --model_dir models --streaming 0
```

- If **BDL → BDL** is clear(ish) and intelligible → training and vocoder are roughly correct; the problem is **cross-speaker** (Indian content OOD).
- If **BDL → BDL** is also mumbling → then we have a training/architecture/vocoder issue on top (e.g. frame-wise MLP too weak, or mel/vocoder pipeline).

---

## What “training correctly” would mean for Indian → BDL

To get good Indian → BDL conversion you’d need at least one of:

1. **Paired data:** Same utterance (or close) in Indian and BDL: train  
   `(content_Indian, f0_Indian) → mel_BDL` (or similar). Then the model sees “Indian content → BDL mel” in training.

2. **Multi-speaker training:** Train on several speakers (including Indian-like content) so the model learns to map diverse content to target mel. Then Indian at inference is less OOD.

3. **Stronger model:** Add temporal context (e.g. 1D conv or transformer over time) so each mel frame is predicted using context. That can help coherence and reduce mumbling even before solving (1) or (2).

4. **Different paradigm:** e.g. ASR → text → TTS(US), or an explicit accent-conversion model in content space, instead of “VC trained only on BDL.”

---

## Summary

- **We are training “correctly” for BDL → BDL** (same speaker): same content domain in train and test.
- **We are not training for Indian → BDL:** we never use Indian content (or multi-speaker data) in training, and the generator is frame-wise with no temporal context. So mumbling on Indian input is expected.
- **Next step:** Run the BDL → BDL sanity check above. If that’s clear, the fix is data/paradigm for cross-speaker (and optionally a better architecture). If that’s also mumbling, the fix is model size, temporal context, and/or loss/vocoder.
