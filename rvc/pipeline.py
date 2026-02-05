"""
Full conversion pipeline: chunk -> gate -> content -> f0 -> model -> vocoder -> overlap-add.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import InferenceParams
from .audio import load_wav, save_wav, resample
from .streaming import StreamingChunker, OverlapAdd
from .silence_gate import SilenceGate
from .content_encoder import ContentEncoder
from .pitch_rmvpe import RMVPExtractor, DEFAULT_SR_F0
from .vocoder_hifigan import HiFiGANVocoder
from .vc_model import VCModel


def convert_file(
    input_wav: str | Path,
    output_wav: str | Path,
    model_dir: Path,
    params: InferenceParams,
) -> None:
    """
    Convert one WAV: load -> chunk (160 ms, 80 ms hop) -> gate -> encode -> f0 -> model -> vocoder -> OLA -> save.
    """
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    sr = params.sample_rate
    wav = load_wav(input_wav, sr)
    total_samples = len(wav)

    # Resample to 16k for content + f0
    wav_16k = resample(wav, sr, DEFAULT_SR_F0)

    chunker = StreamingChunker(sr, params.window_ms, params.hop_ms)
    ola = OverlapAdd(sr, params.window_ms, params.hop_ms)
    gate = SilenceGate(
        threshold_db=params.silence_db,
        hangover_frames=params.silence_hang_frames,
        passthrough=params.silence_passthrough,
    )

    content_encoder = ContentEncoder(
        weights_dir=params.contentvec_dir,
        device=params.device,
        cache_dir=None,
    )
    rmvpe = RMVPExtractor(
        weights_dir=params.rmvpe_dir,
        device=params.device,
    )
    vc = VCModel(model_dir=model_dir, device=params.device)
    vocoder = HiFiGANVocoder(
        weights_dir=params.vocoder_dir or model_dir,
        device=params.device,
        sr=sr,
    )

    out = ola.allocate_output(total_samples)
    index_feat = None  # TODO: load index and query per chunk if index_rate > 0

    for chunk, start, end in chunker.chunk(wav):
        if gate.is_silent(chunk):
            silent_out = gate.silent_output(chunk)
            ola.add_chunk(out, silent_out, start)
            continue

        # Chunk at 16k for encoder + f0
        start_16k = int(start * DEFAULT_SR_F0 / sr)
        end_16k = int(end * DEFAULT_SR_F0 / sr)
        chunk_16k = wav_16k[start_16k:end_16k]
        if len(chunk_16k) < 320:
            ola.add_chunk(out, np.zeros_like(chunk), start)
            continue

        content = content_encoder.encode_chunk(chunk_16k)
        f0 = rmvpe.extract_f0(chunk_16k, DEFAULT_SR_F0)
        # Align f0 length to content frames (e.g. 50 Hz)
        n_content = content.shape[0]
        if len(f0) != n_content:
            f0 = np.interp(
                np.linspace(0, len(f0) - 1, n_content),
                np.arange(len(f0)),
                f0,
            ).astype(np.float32)
        f0_t = torch.from_numpy(f0).to(params.device).unsqueeze(0).unsqueeze(-1)
        content_t = content.unsqueeze(0)

        mel_or_wav = vc.forward(
            content_t,
            f0_t,
            index_feat=index_feat,
            index_rate=params.index_rate,
        )
        # Denormalize mel if model was trained with --normalize_mel
        cfg = vc.config
        if cfg.get("mel_normalize") and mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
            mel_mean = cfg.get("mel_mean", -20.0)
            mel_std = cfg.get("mel_std", 10.0)
            mel_or_wav = mel_or_wav * mel_std + mel_mean
        if mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
            # mel [B, n_mel, T] -> vocoder
            wav_chunk = vocoder.decode(mel_or_wav.squeeze(0))
        else:
            wav_chunk = mel_or_wav.squeeze().cpu().numpy()
            if wav_chunk.ndim > 1:
                wav_chunk = wav_chunk.mean(axis=0)

        # Match chunk length for OLA (same as window_samples)
        need = min(end - start, len(wav_chunk), chunker.window_samples)
        pad = np.zeros(chunker.window_samples, dtype=np.float32)
        pad[:need] = wav_chunk[:need]
        ola.add_chunk(out, pad, start)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    save_wav(output_wav, out, sr)
