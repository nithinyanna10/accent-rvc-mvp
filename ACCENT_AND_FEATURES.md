# Why the accent stays Indian – features and pipeline

## What features we generate

From **each WAV** we extract three things (see `training/feature_extract.py`):

| Feature   | From what                    | What it encodes                          | Who uses it        |
|----------|------------------------------|------------------------------------------|--------------------|
| **content** | ContentVec( wav @ 16k )     | Phonetic/content – **what is said + how it’s pronounced (accent)** | Generator input    |
| **f0**      | RMVPE( wav @ 16k )        | Pitch contour (voiced/unvoiced, Hz)      | Generator input    |
| **mel**     | Mel spectrogram( wav @ 22k ) | Target spectrum (timbre, quality)        | Generator target   |

- **Content** is the main carrier of **accent**: same sentence in Indian vs US English gives different ContentVec trajectories (different pronunciation).
- **Mel** is what we train the model to predict; it mainly reflects **voice (timbre)**, not accent.

So: **accent lives in `content`; we never modify `content` at inference.**

---

## What happens in code

### Training (on BDL – US English)

- **Data:** `data/cmu_arctic_bdl/raw/` → manifest → `feature_extract` → `data/cmu_arctic_bdl/features/*.npz`.
- Each `.npz` has: `content` (from BDL WAV), `f0` (from BDL WAV), `mel` (from BDL WAV).
- **Model learns:** `(content_BDL, f0_BDL) → mel_BDL`.
- So it only ever sees **US (BDL) content** and learns to produce BDL-style mel for that.

### Inference (convert Indian → ?)

- **Input:** e.g. `samples/original_input.wav` or `data/l2_arctic_flat/ASI_arctic_a0001.wav` (Indian).
- **Pipeline** (`rvc/pipeline.py`):
  1. Load Indian WAV, resample to 16k.
  2. **content = ContentEncoder.encode(Indian_wav_16k)** → **Indian content** (Indian pronunciation).
  3. **f0 = RMVPE(Indian_wav_16k)** → Indian pitch.
  4. **mel_pred = model(content, f0)** → predicted mel.
  5. **wav_out = vocoder(mel_pred)** → output audio.

So at inference we feed **Indian content + Indian f0** into the model. The model was trained on **BDL content + BDL f0**. It has never been trained to map “Indian content” to “US-accent mel”. The output mel is still driven by **Indian content**, so the decoded speech keeps **Indian pronunciation (accent)**. The only thing that can shift toward BDL is **timbre/quality** (mel), not **accent** (content).

---

## What is *not* happening (and why accent doesn’t change)

1. **We do not transform content.**  
   Indian WAV → Indian content → that same content goes straight into the generator. There is no “Indian content → US content” step.

2. **We do not use BDL content at inference.**  
   We only use BDL at **training** time. At inference we only have the **input** (Indian) WAV, so we only have Indian content.

3. **The generator does not do accent conversion.**  
   It was trained to do **(content, f0) → mel**. It never learned “take Indian content and output US-accent mel.” Accent is in content; we don’t change content, so accent doesn’t change.

So: **the code is doing voice conversion (same content, different mel), not accent conversion (different content/accent).**

---

## Data and features summary

| Data source              | Role        | Content in features | Mel in features |
|--------------------------|------------|---------------------|-----------------|
| **BDL** (`cmu_arctic_bdl`) | **Target**  | US English          | BDL voice       |
| **L2-ARCTIC** (`l2_arctic_flat`) | **Input** only (inference) | Indian English (when we convert Indian WAVs) | Not used for training target |

- **Training:** only BDL features; model sees only US (BDL) content.
- **Inference:** we build features from the **input** WAV (e.g. Indian) and use that **input content** unchanged. So output accent follows input accent.

---

## What would be needed to change accent

To get **US accent** from Indian input, we’d need one of these (all are beyond the current pipeline):

1. **Accent conversion in content space**  
   A model that maps **Indian content → US content** (same words, US pronunciation). Then: Indian WAV → Indian content → accent converter → US content → (current generator) → BDL mel → vocoder → US-accent BDL voice.  
   Needs: accent converter (e.g. trained on Indian–US paired or accent-labeled content).

2. **ASR + TTS**  
   Indian WAV → ASR → text → TTS (US accent) → US WAV.  
   No ContentVec; different pipeline.

3. **Train on “Indian content → US mel”**  
   Would require **paired data**: same sentences in Indian (for content) and US (for mel). Hard to get; not what we have with BDL + L2-ARCTIC.

So: **with the current design (content from input WAV only, no content transform), output accent will always follow input accent.** The model only changes the “voice” (mel), not the “pronunciation” (content).
