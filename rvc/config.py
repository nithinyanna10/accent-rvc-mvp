"""Inference and pipeline configuration (CPU-only defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class InferenceParams:
    """Parameters for convert_accent pipeline."""

    device: str = "cpu"
    sample_rate: int = 22050  # match jik876 HiFi-GAN (no mel time-downsample)
    # Streaming: 160 ms window, 80 ms hop (no future leak). If False, process whole file at once (better for debugging).
    streaming: bool = True
    window_ms: float = 160.0
    hop_ms: float = 80.0
    # Silence gate (RMS below threshold -> bypass conversion; hangover avoids chatter)
    silence_db: float = -45.0
    silence_hang_frames: int = 5
    silence_passthrough: bool = False  # if True, output original chunk when silent
    # RVC-style
    f0_up_key: int = 0  # semitones
    index_rate: float = 0.0  # 0 = no retrieval; 0.6–0.9 for accent reduction (blend with BDL content)
    protect: float = 0.2  # protect unvoiced; 0–0.2 so retrieval can shift articulation
    index_k: int = 16  # kNN neighbors (8–16)
    # Paths (filled by CLI)
    model_dir: Optional[Path] = None
    model_name: Optional[str] = None  # e.g. "bdl_accent" to load bdl_accent_rvc.pth
    contentvec_dir: Optional[Path] = None
    rmvpe_dir: Optional[Path] = None
    vocoder_dir: Optional[Path] = None
    index_path: Optional[Path] = None
    # Output: save at model SR (sample_rate) unless out_sr is set (e.g. 44100 for playback)
    out_sr: Optional[int] = None
    # Post-process RMS gate (quiet regions zeroed). None or <= -100 = disabled (less disturbance, may leave hiss).
    post_gate_dbfs: Optional[float] = -60.0
    # Low-pass filter (Hz) to reduce hiss; None = disabled. Try 10000–12000 if output is harsh.
    lowpass_hz: Optional[int] = None
    # Optional
    cpu_threads: int = 0  # 0 = default

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_ms / 1000.0)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000.0)

    def validate(self) -> None:
        if self.model_dir is None or not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Place trained model (bdl_rvc.pth + config.json) in models/ and pass --model_dir."
            )
        if self.window_ms <= 0 or self.hop_ms <= 0 or self.hop_ms > self.window_ms:
            raise ValueError("Require 0 < hop_ms <= window_ms.")
