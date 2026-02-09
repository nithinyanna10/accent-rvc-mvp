"""
Single source of truth for mel spectrogram parameters.
Must match: preprocessing, training feature extraction, vocoder (HiFi-GAN expects the format it was trained on).
Mismatch → noise output.
"""

from __future__ import annotations

from dataclasses import dataclass


# jik876 HiFi-GAN is trained at 22.05 kHz; run entire pipeline at 22050 (no mel time-downsample hack).
MODEL_SR = 22050

@dataclass(frozen=True)
class MelConfig:
    """Mel params used everywhere. Must match vocoder (jik876: 22.05k, hop 256, 80 mels)."""
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0


# Single global config; use this everywhere.
DEFAULT_MEL_CONFIG = MelConfig()
