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
    """Write float32 audio to WAV at sample rate sr. Caller should normalize/limit before this."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr)


def normalize_to_dbfs(audio: np.ndarray, dbfs: float = -1.0) -> np.ndarray:
    """Normalize so peak is at dbfs (e.g. -1 dBFS). Prevents clipping distortion."""
    peak = 10.0 ** (dbfs / 20.0)  # -1 dBFS -> ~0.891
    mx = np.abs(audio).max()
    if mx < 1e-8:
        return audio
    return (audio / mx * peak).astype(np.float32)


def soft_limit(audio: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] to catch any remaining overs (e.g. after OLA)."""
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


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


def silence_gate_rms(
    wav: np.ndarray,
    sr: int,
    window_sec: float = 0.02,
    threshold_dbfs: float = -45.0,
) -> np.ndarray:
    """Zero out regions where RMS is below threshold (reduces gargling in pauses)."""
    if len(wav) < 10 or sr <= 0:
        return wav
    window = max(1, int(sr * window_sec))
    out = wav.astype(np.float64)
    threshold_linear = 10.0 ** (threshold_dbfs / 20.0)
    for i in range(0, len(out) - window, window):
        chunk = out[i : i + window]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        if rms < threshold_linear:
            out[i : i + window] = 0.0
    # last partial window
    if len(out) % window:
        i = (len(out) // window) * window
        if i < len(out):
            chunk = out[i:]
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
            if rms < threshold_linear:
                out[i:] = 0.0
    return out.astype(np.float32)


def silence_gate_rms_smooth(
    wav: np.ndarray,
    sr: int,
    window_sec: float = 0.02,
    threshold_dbfs: float = -60.0,
    ramp_sec: float = 0.01,
) -> np.ndarray:
    """
    Apply RMS-based gate with smooth gain (soft knee + smoothing) to reduce clicks.
    Gain is smoothed over ramp_sec so gate open/close is not abrupt.
    """
    if len(wav) < 10 or sr <= 0:
        return wav
    window = max(1, int(sr * window_sec))
    ramp_samples = max(1, int(sr * ramp_sec))
    threshold_linear = 10.0 ** (threshold_dbfs / 20.0)
    n = len(wav)
    # RMS per small window, then interpolate to per-sample
    n_win = (n + window - 1) // window
    rms_per_win = np.zeros(n_win, dtype=np.float64)
    for i in range(n_win):
        start = i * window
        end = min(start + window, n)
        chunk = wav[start:end].astype(np.float64)
        rms_per_win[i] = np.sqrt(np.mean(chunk ** 2) + 1e-12)
    # Per-sample: linear interpolate RMS then soft knee gain = min(1, rms/threshold)
    x_win = np.arange(n_win) * window + window // 2
    x_sample = np.arange(n, dtype=np.float64)
    rms_samp = np.interp(x_sample, x_win, rms_per_win).astype(np.float64)
    gain = np.clip(rms_samp / (threshold_linear + 1e-12), 0.0, 1.0).astype(np.float64)
    # Smooth gain to avoid sharp edges (moving average over ramp_samples)
    from scipy.ndimage import uniform_filter1d
    k = min(2 * ramp_samples + 1, len(gain))
    if k > 1:
        gain = uniform_filter1d(gain, size=k, mode="nearest")
    out = (wav.astype(np.float64) * gain).astype(np.float32)
    return out
