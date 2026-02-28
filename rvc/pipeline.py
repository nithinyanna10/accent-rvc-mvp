"""
Full conversion pipeline: chunk -> gate -> content -> f0 -> model -> vocoder -> overlap-add.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from .config import InferenceParams
from .audio import (
    load_wav,
    save_wav,
    resample,
    normalize_to_dbfs,
    soft_limit,
    silence_gate_rms,
    silence_gate_rms_smooth,
)
from .streaming import StreamingChunker, OverlapAdd
from .silence_gate import SilenceGate
from .content_encoder import ContentEncoder
from .pitch_rmvpe import RMVPExtractor, DEFAULT_SR_F0
from .vocoder_hifigan import HiFiGANVocoder
from .vc_model import VCModel
from .retrieval_index import load_index
from .blend import apply_accent_strength


# Content encoder frame rate (~50 Hz); vocoder expects sr/256 fps
CONTENT_FPS = 50.0

# F0 smoothing: reduce jitter that can cause "disturbance" / robotic sound
F0_SMOOTH_KERNEL = 11  # frames (odd; larger = smoother, less jitter)
F0_MIN_HZ = 50.0
F0_MAX_HZ = 500.0
# Mel smoothing along time before vocoder (reduces frame jitter)
MEL_SMOOTH_FRAMES = 5  # 0 = disabled

# Post-process gate: -60 dBFS so we don't nuke speech (was -45)
GATE_THRESHOLD_DB = -60.0


def _smooth_f0(f0: np.ndarray) -> np.ndarray:
    """Median filter + clamp to realistic range."""
    out = median_filter(f0.astype(np.float64), size=F0_SMOOTH_KERNEL, mode="nearest")
    out = np.clip(out, F0_MIN_HZ, F0_MAX_HZ).astype(np.float32)
    # Preserve unvoiced (zeros): don't fill with smoothed value where original was 0
    voiced = f0 > 0
    out = np.where(voiced, out, f0)
    return out


def _smooth_mel_time(mel: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Light 1D smooth along time (last axis) to reduce mel jitter before vocoder."""
    if kernel < 2 or mel.size == 0:
        return mel
    from scipy.ndimage import uniform_filter1d
    out = mel.astype(np.float64)
    for b in range(out.shape[0]):
        for m in range(out.shape[1]):
            out[b, m, :] = uniform_filter1d(out[b, m, :], size=min(kernel, out.shape[2]), mode="nearest")
    return out.astype(np.float32)


def _interp_mel_to_vocoder_rate(mel: torch.Tensor, sr: int, mel_smooth_frames: int = 0) -> torch.Tensor:
    """Interpolate mel from content fps (~50) to vocoder fps (sr/256). Optionally smooth mel along time first."""
    if mel.dim() != 3 or mel.shape[1] != 80:
        return mel
    B, _, T_c = mel.shape
    mel_np = mel.cpu().numpy()
    if mel_smooth_frames > 0:
        mel_np = _smooth_mel_time(mel_np, kernel=mel_smooth_frames)
    T_v = max(1, int(T_c * (sr / 256.0) / CONTENT_FPS))
    out = np.zeros((B, 80, T_v), dtype=np.float32)
    for b in range(B):
        for m in range(80):
            out[b, m, :] = np.interp(
                np.linspace(0, T_c - 1, T_v),
                np.arange(T_c),
                mel_np[b, m, :].astype(np.float64),
            ).astype(np.float32)
    return torch.from_numpy(out).to(mel.device)


def _post_process_audio(
    wav: np.ndarray,
    sr: int,
    highpass_hz: float = 80.0,
    gate_dbfs: Optional[float] = -60.0,
    lowpass_hz: Optional[int] = None,
) -> np.ndarray:
    """Light high-pass + optional RMS gate + optional low-pass (reduces hiss). gate_dbfs None or <= -100 = gate disabled."""
    if sr <= 0 or len(wav) < 100:
        return wav
    out = wav.astype(np.float32)
    if highpass_hz > 0:
        nyq = 0.5 * sr
        cut = highpass_hz / nyq
        if cut < 1.0:
            b, a = butter(2, cut, btype="high")
            out = filtfilt(b, a, out.astype(np.float64)).astype(np.float32)
    if lowpass_hz is not None and lowpass_hz > 0 and lowpass_hz < sr // 2:
        nyq = 0.5 * sr
        cut = lowpass_hz / nyq
        if cut < 1.0:
            b, a = butter(2, cut, btype="low")
            out = filtfilt(b, a, out.astype(np.float64)).astype(np.float32)
    if gate_dbfs is not None and gate_dbfs > -100:
        out = silence_gate_rms_smooth(out, sr, window_sec=0.02, threshold_dbfs=gate_dbfs, ramp_sec=0.01)
    return out


def _finalize_output(out: np.ndarray, sr: int, out_sr: Optional[int]) -> tuple[np.ndarray, int]:
    """Normalize to -1 dBFS, soft limit, optionally resample. Returns (wav, save_sr)."""
    out = normalize_to_dbfs(out, dbfs=-1.0)
    out = soft_limit(out)
    save_sr = int(out_sr) if out_sr is not None and out_sr > 0 else sr
    if save_sr != sr:
        out = resample(out, sr, save_sr)
    return out, save_sr


def _debug_print(
    chunk_index: int,
    content: torch.Tensor,
    f0: np.ndarray,
    mel: torch.Tensor,
) -> None:
    """Print content/f0/mel stats once per file (chunk 0 or only chunk) for debugging."""
    # 1) Content feature shape + mean/std
    c = content.cpu().numpy() if content.dim() >= 2 else content.unsqueeze(0).cpu().numpy()
    if c.size > 0:
        print(f"[debug chunk {chunk_index}] content: shape={content.shape} mean={float(c.mean()):.4f} std={float(c.std()):.4f}")
    # 2) F0 voiced% + mean f0 (voiced only)
    voiced = f0 > 0
    n_voiced = int(np.sum(voiced))
    pct = 100.0 * n_voiced / len(f0) if len(f0) else 0
    mean_f0 = float(np.mean(f0[voiced])) if n_voiced else 0.0
    print(f"[debug chunk {chunk_index}] f0: voiced%={pct:.1f} mean_f0(voiced)={mean_f0:.1f} Hz")
    # 3) Mel stats before vocoder (min/max/mean)
    if mel.dim() >= 2:
        m = mel.detach().cpu().numpy()
        print(f"[debug chunk {chunk_index}] mel: shape={mel.shape} min={float(m.min()):.4f} max={float(m.max()):.4f} mean={float(m.mean()):.4f}")


def convert_file(
    input_wav: str | Path,
    output_wav: str | Path,
    model_dir: Path,
    params: InferenceParams,
) -> None:
    """
    Convert one WAV: load -> (chunk or full-file) -> encode -> f0 -> model -> vocoder -> save.
    If streaming: 160 ms window, 80 ms hop, OLA. If not streaming: full-file (no chunking).
    """
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    # Single SR end-to-end (22,050 Hz); no mel time-downsample hack
    sr = params.sample_rate  # 22050
    wav = load_wav(input_wav, sr)
    total_samples = len(wav)

    # Resample to 16k for content + f0
    wav_16k = resample(wav, sr, DEFAULT_SR_F0)

    content_encoder = ContentEncoder(
        weights_dir=params.contentvec_dir,
        device=params.device,
        cache_dir=None,
    )
    rmvpe = RMVPExtractor(
        weights_dir=params.rmvpe_dir,
        device=params.device,
    )
    vc = VCModel(
        model_dir=model_dir,
        device=params.device,
        model_name=params.model_name,
    )
    vocoder = HiFiGANVocoder(
        weights_dir=params.vocoder_dir or model_dir,
        device=params.device,
        sr=sr,
    )
    retrieval = load_index(model_dir, params.index_path, k=params.index_k) if params.index_rate > 0 else None

    if not params.streaming:
        # Full-file: one big chunk (no OLA, better for debugging quality)
        if len(wav_16k) < 320:
            out = np.zeros(total_samples, dtype=np.float32)
        else:
            # content = phonetic/content from INPUT wav (accent lives here; we do not transform it)
            content = content_encoder.encode(wav_16k, cache=False)
            f0 = rmvpe.extract_f0(wav_16k, DEFAULT_SR_F0)
            n_content = content.shape[0]
            if len(f0) != n_content:
                f0 = np.interp(
                    np.linspace(0, len(f0) - 1, n_content),
                    np.arange(len(f0)),
                    f0,
                ).astype(np.float32)
            f0 = _smooth_f0(f0)
            f0_t = torch.from_numpy(f0).to(params.device).unsqueeze(0).unsqueeze(-1)
            content_t = content.unsqueeze(0)
            index_feat = None
            if retrieval is not None and params.index_rate > 0:
                index_arr = retrieval.get_index_feat(content.cpu().numpy(), k=params.index_k)
                index_feat = torch.from_numpy(index_arr).to(params.device).unsqueeze(0)
            mel_or_wav = vc.forward(
                content_t,
                f0_t,
                index_feat=index_feat,
                index_rate=params.index_rate,
            )
            cfg = vc.config
            if cfg.get("mel_normalize") and mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
                mel_mean = cfg.get("mel_mean", -20.0)
                mel_std = cfg.get("mel_std", 10.0)
                mel_or_wav = mel_or_wav * mel_std + mel_mean
            # Debug: content shape+mean/std, f0 voiced%+mean, mel min/max/mean
            mel_for_debug = mel_or_wav.squeeze(0) if mel_or_wav.dim() == 3 else mel_or_wav
            _debug_print(0, content, f0, mel_for_debug)
            if mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
                mel_voc = _interp_mel_to_vocoder_rate(mel_or_wav, sr, mel_smooth_frames=params.mel_smooth_frames)
                wav_out = vocoder.decode(mel_voc.squeeze(0))
            else:
                wav_out = mel_or_wav.squeeze().cpu().numpy()
                if wav_out.ndim > 1:
                    wav_out = wav_out.mean(axis=0)
            # Trim/pad to input length
            out = np.zeros(total_samples, dtype=np.float32)
            copy_len = min(len(wav_out), total_samples)
            if copy_len > 0:
                out[:copy_len] = wav_out[:copy_len].astype(np.float32)
            out = _post_process_audio(out, sr, gate_dbfs=params.post_gate_dbfs, lowpass_hz=params.lowpass_hz)
        # Wet/dry blend with original
        blend = getattr(params, "blend", 1.0)
        blend_mode = getattr(params, "blend_mode", "waveform")
        if 0.0 <= blend < 1.0:
            # Align original to same length
            orig_aligned = np.zeros(len(out), dtype=np.float32)
            orig_len = min(len(wav), len(out))
            orig_aligned[:orig_len] = wav[:orig_len]
            out = apply_accent_strength(orig_aligned, out, strength=blend, mode=blend_mode, sr=sr)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        out, save_sr = _finalize_output(out, sr, params.out_sr)
        save_wav(output_wav, out, save_sr)
        return

    # Streaming path
    chunker = StreamingChunker(sr, params.window_ms, params.hop_ms)
    ola = OverlapAdd(sr, params.window_ms, params.hop_ms)
    gate = SilenceGate(
        threshold_db=params.silence_db,
        hangover_frames=params.silence_hang_frames,
        passthrough=params.silence_passthrough,
    )
    out = ola.allocate_output(total_samples)
    debug_done = False

    for chunk_index, (chunk, start, end) in enumerate(chunker.chunk(wav)):
        if gate.is_silent(chunk):
            silent_out = gate.silent_output(chunk)
            ola.add_chunk(out, silent_out, start)
            continue

        start_16k = int(start * DEFAULT_SR_F0 / sr)
        end_16k = int(end * DEFAULT_SR_F0 / sr)
        chunk_16k = wav_16k[start_16k:end_16k]
        if len(chunk_16k) < 320:
            ola.add_chunk(out, np.zeros_like(chunk), start)
            continue

        content = content_encoder.encode_chunk(chunk_16k)
        f0 = rmvpe.extract_f0(chunk_16k, DEFAULT_SR_F0)
        n_content = content.shape[0]
        if len(f0) != n_content:
            f0 = np.interp(
                np.linspace(0, len(f0) - 1, n_content),
                np.arange(len(f0)),
                f0,
            ).astype(np.float32)
        f0 = _smooth_f0(f0)
        f0_t = torch.from_numpy(f0).to(params.device).unsqueeze(0).unsqueeze(-1)
        content_t = content.unsqueeze(0)
        index_feat = None
        if retrieval is not None and params.index_rate > 0:
            index_arr = retrieval.get_index_feat(content.cpu().numpy(), k=params.index_k)
            index_feat = torch.from_numpy(index_arr).to(params.device).unsqueeze(0)
        mel_or_wav = vc.forward(
            content_t,
            f0_t,
            index_feat=index_feat,
            index_rate=params.index_rate,
        )
        cfg = vc.config
        if cfg.get("mel_normalize") and mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
            mel_mean = cfg.get("mel_mean", -20.0)
            mel_std = cfg.get("mel_std", 10.0)
            mel_or_wav = mel_or_wav * mel_std + mel_mean
        if not debug_done:
            _debug_print(chunk_index, content, f0, mel_or_wav.squeeze(0) if mel_or_wav.dim() == 3 else mel_or_wav)
            debug_done = True
        if mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
            mel_voc = _interp_mel_to_vocoder_rate(mel_or_wav, sr, mel_smooth_frames=params.mel_smooth_frames)
            wav_chunk = vocoder.decode(mel_voc.squeeze(0))
        else:
            wav_chunk = mel_or_wav.squeeze().cpu().numpy()
            if wav_chunk.ndim > 1:
                wav_chunk = wav_chunk.mean(axis=0)

        need = min(end - start, len(wav_chunk), chunker.window_samples)
        pad = np.zeros(chunker.window_samples, dtype=np.float32)
        pad[:need] = wav_chunk[:need]
        ola.add_chunk(out, pad, start)

    out = _post_process_audio(out, sr, gate_dbfs=params.post_gate_dbfs, lowpass_hz=params.lowpass_hz)
    # Wet/dry blend with original (streaming path)
    blend = getattr(params, "blend", 1.0)
    blend_mode = getattr(params, "blend_mode", "waveform")
    if 0.0 <= blend < 1.0:
        orig_aligned = np.zeros(len(out), dtype=np.float32)
        orig_len = min(len(wav), len(out))
        orig_aligned[:orig_len] = wav[:orig_len]
        out = apply_accent_strength(orig_aligned, out, strength=blend, mode=blend_mode, sr=sr)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    out, save_sr = _finalize_output(out, sr, params.out_sr)
    save_wav(output_wav, out, save_sr)
