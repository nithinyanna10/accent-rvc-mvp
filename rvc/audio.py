"""Audio I/O: load/save WAV, resample, normalize (peak-safe)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa


def load_wav(path: str | Path, sr: int, mono: bool = True) -> np.ndarray:
    """
    Load WAV as float32 mono at target sample rate.
    Resamples if needed; normalizes peak safely (avoid div-by-zero).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    wav, file_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1 and mono:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr, res_type="kaiser_best")
    wav = normalize_peak(wav)
    return wav.astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Write float32 audio to WAV. Clips to [-1, 1] if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr)


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Normalize so max absolute value is `peak`. Safe for silence (no div-by-zero)."""
    mx = np.abs(audio).max()
    if mx < 1e-8:
        return audio
    return (audio / mx * peak).astype(np.float32)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample to target_sr using librosa."""
    if orig_sr == target_sr:
        return wav
    return librosa.resample(
        wav.astype(np.float64),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    ).astype(np.float32)
