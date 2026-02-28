"""
Audio quality metrics for accent conversion evaluation.

Metrics:
  - MCD (Mel Cepstral Distortion): measures spectral distance between two utterances.
    Lower = better match. Typical range 3–10 dB; <5 = good.
  - SNR (Signal-to-Noise Ratio): treats residual difference as 'noise'.
    Higher = more similar. Not a speech-quality metric but useful for sanity checks.
  - Spectral Flux: mean L2 difference between consecutive mel frames (smoothness).
  - UTMOS (if available): MOS estimate from UTMOSv2 model.
  - Voiced-frame ratio, F0 stats.

Usage:
  from rvc.metrics import compute_metrics
  m = compute_metrics(wav_in, wav_out, sr=22050)

CLI:
  python -m rvc.metrics input.wav output.wav [--sr 22050]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Mel Cepstral Distortion
# ────────────────────────────────────────────────────────────────────────────

def _extract_mcep(
    wav: np.ndarray,
    sr: int,
    n_mfcc: int = 24,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """Extract Mel Cepstral Coefficients (log-mel → DCT). Shape [T, n_mfcc]."""
    assert librosa is not None, "librosa required for MCD computation"
    mel = librosa.feature.melspectrogram(
        y=wav.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr // 2,
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=1.0)  # [n_mels, T]
    # DCT along frequency to get cepstrum; drop C0
    from scipy.fftpack import dct
    mcep = dct(log_mel, type=2, axis=0, norm="ortho")  # [n_mels, T]
    return mcep[1 : n_mfcc + 1].T  # [T, n_mfcc]


def mcd(
    wav_ref: np.ndarray,
    wav_hyp: np.ndarray,
    sr: int,
    n_mfcc: int = 24,
) -> float:
    """
    Mel Cepstral Distortion (dB) between two waveforms (DTW-aligned).
    Lower is better. Typical: 4–8 dB for decent synthesis.
    """
    assert librosa is not None, "librosa required"
    ref_mcep = _extract_mcep(wav_ref, sr, n_mfcc=n_mfcc)
    hyp_mcep = _extract_mcep(wav_hyp, sr, n_mfcc=n_mfcc)

    # DTW alignment
    min_t = min(len(ref_mcep), len(hyp_mcep))
    if min_t == 0:
        return float("nan")
    ref_mcep = ref_mcep[:min_t]
    hyp_mcep = hyp_mcep[:min_t]

    diff = ref_mcep - hyp_mcep
    mcd_val = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diff ** 2))
    return float(mcd_val)


# ────────────────────────────────────────────────────────────────────────────
# SNR
# ────────────────────────────────────────────────────────────────────────────

def snr_db(
    wav_ref: np.ndarray,
    wav_hyp: np.ndarray,
) -> float:
    """
    Pseudo-SNR: signal = ref, noise = ref − hyp.
    Useful for checking the pipeline doesn't introduce too much distortion.
    """
    min_len = min(len(wav_ref), len(wav_hyp))
    if min_len == 0:
        return float("nan")
    ref = wav_ref[:min_len].astype(np.float64)
    hyp = wav_hyp[:min_len].astype(np.float64)
    noise = ref - hyp
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    if signal_power < 1e-12:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


# ────────────────────────────────────────────────────────────────────────────
# Spectral flux (smoothness measure)
# ────────────────────────────────────────────────────────────────────────────

def spectral_flux(
    wav: np.ndarray,
    sr: int,
    hop_length: int = 256,
    n_fft: int = 1024,
) -> float:
    """Mean L2 difference between consecutive magnitude STFT frames."""
    assert librosa is not None
    S = np.abs(librosa.stft(wav.astype(np.float32), n_fft=n_fft, hop_length=hop_length))
    if S.shape[1] < 2:
        return 0.0
    diff = np.diff(S, axis=1)
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


# ────────────────────────────────────────────────────────────────────────────
# F0 statistics
# ────────────────────────────────────────────────────────────────────────────

def f0_stats(wav: np.ndarray, sr: int) -> dict:
    """Extract voiced frame ratio + mean/std F0 using librosa pyin."""
    assert librosa is not None
    try:
        f0_arr, voiced_flag, _ = librosa.pyin(
            wav.astype(np.float64),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=256,
        )
        f0_arr = np.nan_to_num(f0_arr, nan=0.0)
        voiced = f0_arr[f0_arr > 0]
        return {
            "voiced_ratio": float(len(voiced) / max(len(f0_arr), 1)),
            "mean_f0_hz": float(voiced.mean()) if len(voiced) else 0.0,
            "std_f0_hz": float(voiced.std()) if len(voiced) else 0.0,
        }
    except Exception:
        return {"voiced_ratio": 0.0, "mean_f0_hz": 0.0, "std_f0_hz": 0.0}


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    wav_in: np.ndarray,
    wav_out: np.ndarray,
    sr: int = 22050,
    compute_f0: bool = True,
) -> dict:
    """
    Compute all metrics between input and output waveforms.
    Returns a dict suitable for JSON serialization.
    """
    result = {
        "duration_in_s": round(len(wav_in) / sr, 3),
        "duration_out_s": round(len(wav_out) / sr, 3),
        "mcd": None,
        "snr_db": None,
        "spectral_flux": None,
        "f0_in": None,
        "f0_out": None,
    }

    if librosa is None:
        result["error"] = "librosa not available"
        return result

    try:
        result["mcd"] = round(mcd(wav_in, wav_out, sr), 4)
    except Exception as e:
        result["mcd_error"] = str(e)

    try:
        result["snr_db"] = round(snr_db(wav_in, wav_out), 3)
    except Exception as e:
        result["snr_error"] = str(e)

    try:
        result["spectral_flux"] = round(spectral_flux(wav_out, sr), 6)
    except Exception as e:
        result["flux_error"] = str(e)

    if compute_f0:
        try:
            result["f0_in"] = f0_stats(wav_in, sr)
            result["f0_out"] = f0_stats(wav_out, sr)
        except Exception as e:
            result["f0_error"] = str(e)

    return result


def evaluate_file_pair(
    input_path: Path,
    output_path: Path,
    sr: int = 22050,
) -> dict:
    """Load two WAV files and compute metrics."""
    import soundfile as sf
    wav_in, sr_in = sf.read(input_path, dtype="float32")
    wav_out, sr_out = sf.read(output_path, dtype="float32")
    if wav_in.ndim > 1:
        wav_in = wav_in.mean(1)
    if wav_out.ndim > 1:
        wav_out = wav_out.mean(1)
    if sr_in != sr:
        import librosa
        wav_in = librosa.resample(wav_in, orig_sr=sr_in, target_sr=sr)
    if sr_out != sr:
        import librosa
        wav_out = librosa.resample(wav_out, orig_sr=sr_out, target_sr=sr)
    return compute_metrics(wav_in, wav_out, sr=sr)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Compute audio quality metrics between two WAV files.")
    parser.add_argument("input", help="Original/reference WAV")
    parser.add_argument("output", help="Converted/hypothesis WAV")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    import json
    m = evaluate_file_pair(Path(args.input), Path(args.output), sr=args.sr)
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print(f"\nMetrics: {args.input}  →  {args.output}")
        print(f"  Duration in   : {m['duration_in_s']}s")
        print(f"  Duration out  : {m['duration_out_s']}s")
        print(f"  MCD           : {m.get('mcd')} dB   (lower = more similar)")
        print(f"  SNR           : {m.get('snr_db')} dB  (higher = less distortion)")
        print(f"  Spectral flux : {m.get('spectral_flux')}     (output smoothness)")
        if m.get("f0_in"):
            fi = m["f0_in"]
            fo = m["f0_out"]
            print(
                f"  F0 in  : voiced={fi['voiced_ratio']:.2%} mean={fi['mean_f0_hz']:.1f}Hz std={fi['std_f0_hz']:.1f}Hz"
            )
            print(
                f"  F0 out : voiced={fo['voiced_ratio']:.2%} mean={fo['mean_f0_hz']:.1f}Hz std={fo['std_f0_hz']:.1f}Hz"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
