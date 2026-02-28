"""
Wet/dry blending utilities.

blend(original, converted, alpha) mixes the two waveforms linearly:
  output = alpha * converted + (1 - alpha) * original

alpha=1.0  → full conversion
alpha=0.0  → original passthrough
alpha=0.5  → equal mix (useful for soft accent-reduction without full replacement)

Also provides spectral blending (blend in the mel/frequency domain rather than
waveform domain), which can sound smoother than linear waveform mixing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def blend_waveforms(
    original: np.ndarray,
    converted: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Linear wet/dry blend of two waveforms.

    Args:
        original: original (source accent) waveform, float32
        converted: converted (target accent) waveform, float32
        alpha: blend weight 0..1. 1.0 = full converted, 0.0 = original.

    Returns:
        blended waveform, float32, same length as the longer of the two.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return converted.astype(np.float32)
    if alpha <= 0.0:
        return original.astype(np.float32)

    # Pad shorter to match longer
    max_len = max(len(original), len(converted))
    orig = np.zeros(max_len, dtype=np.float64)
    conv = np.zeros(max_len, dtype=np.float64)
    orig[:len(original)] = original.astype(np.float64)
    conv[:len(converted)] = converted.astype(np.float64)

    out = alpha * conv + (1.0 - alpha) * orig
    return out.astype(np.float32)


def blend_spectral(
    original: np.ndarray,
    converted: np.ndarray,
    sr: int,
    alpha: float = 1.0,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Spectral (magnitude) blend: interpolate STFT magnitude between original and
    converted, keep converted phase.  Smoother than waveform blending.

    Args:
        original: original waveform, float32, at `sr`
        converted: converted waveform, float32, at `sr`
        sr: sample rate
        alpha: 0..1. 1.0 = full converted.
        n_fft, hop_length: STFT parameters.

    Returns:
        blended waveform, float32.
    """
    try:
        import librosa
    except ImportError:
        return blend_waveforms(original, converted, alpha)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return converted.astype(np.float32)
    if alpha <= 0.0:
        return original.astype(np.float32)

    max_len = max(len(original), len(converted))
    orig = np.zeros(max_len, dtype=np.float32)
    conv = np.zeros(max_len, dtype=np.float32)
    orig[:len(original)] = original
    conv[:len(converted)] = converted

    D_orig = librosa.stft(orig, n_fft=n_fft, hop_length=hop_length)
    D_conv = librosa.stft(conv, n_fft=n_fft, hop_length=hop_length)

    mag_orig = np.abs(D_orig)
    mag_conv = np.abs(D_conv)
    phase_conv = np.angle(D_conv)

    # Blend magnitude, use converted phase
    mag_blend = alpha * mag_conv + (1.0 - alpha) * mag_orig
    D_blend = mag_blend * np.exp(1j * phase_conv)

    out = librosa.istft(D_blend, hop_length=hop_length, length=max_len)
    return out.astype(np.float32)


def crossfade(
    wav_a: np.ndarray,
    wav_b: np.ndarray,
    fade_samples: int = 1024,
) -> np.ndarray:
    """
    Concatenate two waveforms with a crossfade at the boundary.
    wav_a's tail fades out, wav_b's head fades in.

    Useful for stitching together streaming chunks without clicks.
    """
    fade_samples = min(fade_samples, len(wav_a), len(wav_b))
    if fade_samples <= 0:
        return np.concatenate([wav_a, wav_b]).astype(np.float32)

    ramp_down = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
    ramp_up = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)

    head_a = wav_a[:-fade_samples].astype(np.float64)
    tail_a = wav_a[-fade_samples:].astype(np.float64) * ramp_down
    head_b = wav_b[:fade_samples].astype(np.float64) * ramp_up
    tail_b = wav_b[fade_samples:].astype(np.float64)

    xfade = tail_a + head_b
    return np.concatenate([head_a, xfade, tail_b]).astype(np.float32)


def apply_accent_strength(
    original: np.ndarray,
    converted: np.ndarray,
    strength: float = 1.0,
    mode: str = "waveform",
    sr: int = 22050,
) -> np.ndarray:
    """
    High-level 'accent strength' control.

    strength=1.0 → full conversion (no trace of original accent)
    strength=0.5 → halfway between original and converted
    strength=0.0 → original unchanged

    mode: 'waveform' (fast, default) or 'spectral' (smoother, needs librosa).
    """
    if mode == "spectral":
        return blend_spectral(original, converted, sr=sr, alpha=strength)
    return blend_waveforms(original, converted, alpha=strength)
