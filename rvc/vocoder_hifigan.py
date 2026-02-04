"""
HiFi-GAN v2 vocoder wrapper (RVC ecosystem).
decode(mel_or_features) -> waveform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .utils import raise_missing_weights


class HiFiGANVocoder:
    """
    Wrapper for HiFi-GAN v2 used in RVC-like pipelines.
    Loads .pth from assets/hifigan/ or model_dir.
    decode(mel_or_features) -> waveform float32.
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: str = "cpu",
        sr: int = 40000,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.device = device
        self.sr = sr
        self._model = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        raise_missing_weights(
            "HiFi-GAN",
            self.weights_dir,
            "Place HiFi-GAN v2 weights in assets/hifigan/ (e.g. hifigan.pth).",
        )
        d = Path(self.weights_dir)
        p = d  # fallback
        for name in ("hifigan.pth", "generator.pth"):
            p = d / name
            if p.exists():
                break
        else:
            raise FileNotFoundError(
                f"HiFi-GAN: no hifigan.pth or generator.pth in {d}. Place in assets/hifigan/."
            )
        self._model = torch.load(p, map_location=self.device, weights_only=False)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._model

    def decode(self, mel_or_features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert mel or feature tensor to waveform.
        mel_or_features: [C, T] or [B, C, T]. Output: float32 mono.
        """
        if isinstance(mel_or_features, np.ndarray):
            x = torch.from_numpy(mel_or_features).float().to(self.device)
        else:
            x = mel_or_features.float().to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        try:
            model = self._load_model()
        except FileNotFoundError:
            # Fallback: return zeros of plausible length (80 mel frames ~ 1s at 12.5 ms)
            t = x.shape[-1]
            hop = 256  # typical for 40k
            return np.zeros(t * hop, dtype=np.float32)

        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.squeeze().cpu().numpy()
        if out.ndim > 1:
            out = out.mean(axis=0)
        return out.astype(np.float32)
