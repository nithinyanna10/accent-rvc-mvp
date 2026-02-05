#!/usr/bin/env python3
"""
Test: bypass the model and feed GROUND TRUTH mel (from original wav) through the vocoder.
If output is clear speech -> vocoder path is correct, problem is the model.
If output is noise -> our mel->vocoder conversion is wrong.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import librosa

# Repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rvc.audio import load_wav, resample
from rvc.vocoder_hifigan import HiFiGANVocoder
from rvc.hifigan_jik876 import load_jik876_checkpoint


def main():
    # Same mel params as training (feature_extract.py)
    MEL_SR = 40000
    MEL_N_FFT = 1024
    MEL_HOP = 256
    MEL_N_MELS = 80

    input_wav = Path("samples/original_input.wav")
    if not input_wav.exists():
        input_wav = Path("data/l2_arctic_flat/RRBI_arctic_a0190.wav")
    if not input_wav.exists():
        print("No input wav found. Place a wav at samples/original_input.wav or use data/")
        return 1

    out_wav = Path("samples/vocoder_test_ground_truth.wav")
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_wav}")
    wav = load_wav(input_wav, MEL_SR)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Ground truth mel exactly as in training
    mel_power = librosa.feature.melspectrogram(
        y=wav.astype(np.float64),
        sr=MEL_SR,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        fmin=0,
        fmax=8000,
    )
    mel_db = librosa.power_to_db(mel_power, ref=1.0).astype(np.float32)  # [80, T]

    # Load jik876 vocoder
    ckpt = Path("assets/hifigan/generator_v2.pth")
    if not ckpt.exists():
        print("Run: python download_jik876_hifigan.py")
        return 1
    model = load_jik876_checkpoint(ckpt, "cpu")

    # Convert dB mel -> log mel (same as in vocoder_hifigan._decode_jik876_hifigan)
    mel_power_gt = librosa.db_to_power(mel_db, ref=1.0)
    mel_log = np.log(np.clip(mel_power_gt, 1e-5, None)).astype(np.float32)
    n_mels, T = mel_log.shape

    # Downsample time for 22k vocoder
    T_22k = max(1, int(T * 22050 / 40000))
    mel_22k = np.zeros((n_mels, T_22k), dtype=np.float32)
    for i in range(n_mels):
        mel_22k[i] = np.interp(
            np.linspace(0, T - 1, T_22k),
            np.arange(T),
            mel_log[i],
        )

    # Vocode
    x = torch.from_numpy(mel_22k).unsqueeze(0)
    with torch.no_grad():
        wav_22k = model(x).squeeze().cpu().numpy()

    # Resample to 40k
    wav_40k = librosa.resample(
        wav_22k.astype(np.float64),
        orig_sr=22050,
        target_sr=40000,
        res_type="kaiser_best",
    )

    # Normalize
    mx = np.abs(wav_40k).max()
    if mx > 0:
        wav_40k = (wav_40k / mx * 0.9).astype(np.float32)
    else:
        wav_40k = wav_40k.astype(np.float32)

    import soundfile as sf
    sf.write(out_wav, wav_40k, 40000)
    print(f"Saved: {out_wav}")
    print("Listen to it: if CLEAR SPEECH -> vocoder is fine, problem is the model output.")
    print("If NOISE -> our mel->vocoder conversion is wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
