"""Inference and pipeline configuration (CPU-only defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class InferenceParams:
    """Parameters for convert_accent pipeline."""

    device: str = "cpu"
    sample_rate: int = 40000
    # Streaming: 160 ms window, 80 ms hop (no future leak)
    window_ms: float = 160.0
    hop_ms: float = 80.0
    # Silence gate
    silence_db: float = -45.0
    silence_hang_frames: int = 3
    silence_passthrough: bool = False  # if True, output original chunk when silent
    # RVC-style
    f0_up_key: int = 0  # semitones
    index_rate: float = 0.0  # 0 = no retrieval blend
    protect: float = 0.5  # protect unvoiced; 0 = none, 1 = max
    # Paths (filled by CLI)
    model_dir: Optional[Path] = None
    contentvec_dir: Optional[Path] = None
    rmvpe_dir: Optional[Path] = None
    vocoder_dir: Optional[Path] = None
    index_path: Optional[Path] = None
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
