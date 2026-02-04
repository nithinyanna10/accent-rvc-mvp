"""
RMVPE f0 extraction.
extract_f0(wav, sr) -> f0 array. Unvoiced -> 0; optional median smoothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .utils import raise_missing_weights

# Default hop for f0 (e.g. 160 samples at 16k = 10 ms)
DEFAULT_HOP = 160
DEFAULT_SR_F0 = 16000


class RMVPExtractor:
    """
    Wrapper for RMVPE f0 extraction.
    If weights aren't present, raises with instructions for assets/rmvpe/.
    Safe: unvoiced frames -> 0; optional median smoothing.
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: str = "cpu",
        hop_length: int = DEFAULT_HOP,
        median_smooth: bool = True,
        radius: int = 3,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.device = device
        self.hop_length = hop_length
        self.median_smooth = median_smooth
        self.radius = radius
        self._model = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        raise_missing_weights(
            "RMVPE",
            self.weights_dir,
            "Place RMVPE weights in assets/rmvpe/ (e.g. rmvpe.pt).",
        )
        d = Path(self.weights_dir)
        for name in ("rmvpe.pt", "rmvpe.onnx"):
            p = d / name
            if p.exists() and p.suffix == ".pt":
                self._model = torch.load(p, map_location=self.device, weights_only=False)
                if hasattr(self._model, "eval"):
                    self._model.eval()
                return self._model
        raise FileNotFoundError(
            f"RMVPE: no rmvpe.pt in {d}. Place RMVPE weights in assets/rmvpe/."
        )

    def extract_f0(
        self,
        wav: np.ndarray,
        sr: int,
        pad: bool = True,
    ) -> np.ndarray:
        """
        Extract f0 (Hz) per frame. Unvoiced frames are 0.
        wav: float32 mono. Resampled to 16k internally if sr != 16000.
        """
        import librosa
        if sr != DEFAULT_SR_F0:
            wav = librosa.resample(
                wav.astype(np.float64),
                orig_sr=sr,
                target_sr=DEFAULT_SR_F0,
                res_type="kaiser_best",
            ).astype(np.float32)
        n_frames = (len(wav) + self.hop_length - 1) // self.hop_length
        try:
            model = self._load_model()
        except FileNotFoundError:
            return np.zeros(n_frames, dtype=np.float32)
        if model is None:  # type guard
            return np.zeros(n_frames, dtype=np.float32)

        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if hasattr(model, "infer_from_audio"):
                f0 = model.infer_from_audio(x, self.hop_length)
            elif hasattr(model, "forward"):
                f0 = model(x, hop_length=self.hop_length)
            else:
                f0 = model(x)
            if isinstance(f0, torch.Tensor):
                f0 = f0.squeeze().cpu().numpy()
            else:
                f0 = np.asarray(f0).flatten()
        # Ensure length
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)), constant_values=0)
        f0 = f0[:n_frames].astype(np.float32)
        # Unvoiced -> 0 (RMVPE usually already does this)
        f0[f0 < 50] = 0
        f0[f0 > 500] = 0
        if self.median_smooth and self.radius > 0:
            from scipy.ndimage import median_filter
            voiced = f0 > 0
            if np.any(voiced):
                f0_smooth = median_filter(f0, size=2 * self.radius + 1, mode="constant", cval=0)
                f0 = np.where(voiced, f0_smooth, 0).astype(np.float32)
        return f0


def _f0_placeholder(wav: np.ndarray, sr: int, hop: int = DEFAULT_HOP) -> np.ndarray:
    """Placeholder f0 (all unvoiced) when RMVPE not available."""
    n = (len(wav) + hop - 1) // hop
    return np.zeros(n, dtype=np.float32)
