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

from rvc.audio import load_wav
from rvc.mel_config import DEFAULT_MEL_CONFIG
from rvc.hifigan_jik876 import load_jik876_checkpoint


def main():
    cfg = DEFAULT_MEL_CONFIG

    input_wav = Path("samples/original_input.wav")
    if not input_wav.exists():
        input_wav = Path("data/l2_arctic_flat/RRBI_arctic_a0190.wav")
    if not input_wav.exists():
        print("No input wav found. Place a wav at samples/original_input.wav or use data/")
        return 1

    out_wav = Path("samples/vocoder_test_ground_truth.wav")
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_wav}")
    wav = load_wav(input_wav, cfg.sr)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Ground truth mel exactly as in training (single MelConfig)
    mel_power = librosa.feature.melspectrogram(
        y=wav.astype(np.float64),
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    mel_db = librosa.power_to_db(mel_power, ref=1.0).astype(np.float32)  # [80, T]

    # Load jik876 vocoder
    ckpt = Path("assets/hifigan/generator_v2.pth")
    if not ckpt.exists():
        print("Run: python download_jik876_hifigan.py")
        return 1
    model = load_jik876_checkpoint(ckpt, "cpu")

    # Convert dB mel -> log mel. Mel is already at 22.05k (cfg.sr), no time-downsample.
    mel_power_gt = librosa.db_to_power(mel_db, ref=1.0)
    mel_log = np.log(np.clip(mel_power_gt, 1e-5, None)).astype(np.float32)

    # Vocode at 22.05 kHz (no hack)
    x = torch.from_numpy(mel_log).unsqueeze(0)
    with torch.no_grad():
        wav_out = model(x).squeeze().cpu().numpy()

    # Normalize to -1 dBFS
    peak = 10.0 ** (-1.0 / 20.0)
    mx = np.abs(wav_out).max()
    if mx > 1e-8:
        wav_out = (wav_out / mx * peak).astype(np.float32)

    import soundfile as sf
    sf.write(out_wav, wav_out, cfg.sr)
    print(f"Saved: {out_wav} ({cfg.sr} Hz)")
    print("Listen at 22.05 kHz: if CLEAR SPEECH -> vocoder is fine; if NOISE -> mel/vocoder mismatch.")
    print("If NOISE -> our mel->vocoder conversion is wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
