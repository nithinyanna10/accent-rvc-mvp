"""
Content encoder: ContentVec / HuBERT-Soft features.
encode(wav_16k) -> [T, C]. CPU-only; optional caching to disk (npz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .audio import resample
from .utils import get_cache_path, raise_missing_weights

CONTENTVEC_SR = 16000


class ContentEncoder:
    """
    Wrapper for ContentVec/HuBERT-Soft feature extraction.
    Expects 16 kHz input; use resample if needed.
    If weights aren't present, raises with instructions to place under assets/contentvec/.
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        raise_missing_weights(
            "ContentEncoder",
            self.weights_dir,
            "Download ContentVec/HuBERT-Soft weights and place in assets/contentvec/ (e.g. contentvec_256.pt).",
        )
        d = Path(self.weights_dir)
        for name in ("contentvec_256.pt", "contentvec.pt", "hubert_soft.pt"):
            p = d / name
            if p.exists():
                self._model = torch.load(p, map_location=self.device, weights_only=False)
                if hasattr(self._model, "eval"):
                    self._model.eval()
                return self._model
        raise FileNotFoundError(
            f"ContentEncoder: no known weight file in {d}. "
            "Place contentvec_256.pt (or contentvec.pt / hubert_soft.pt) in assets/contentvec/."
        )

    def encode(
        self,
        wav_16k: np.ndarray,
        cache: bool = True,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode to content features [T, C].
        wav_16k: float32 mono at 16 kHz.
        If cache_dir and cache=True, read/write npz by cache_key (default: hash of wav).
        """
        if self.cache_dir and cache:
            key = cache_key or (str(wav_16k.shape) + str(wav_16k[:100].tobytes()))
            path = get_cache_path(self.cache_dir, key, ".npz")
            if path.exists():
                data = np.load(path)
                return torch.from_numpy(data["feat"]).to(self.device)

        # Try to use loaded model; else return placeholder so pipeline can run without weights for testing
        try:
            model = self._load_model()
        except FileNotFoundError:
            raise

        if wav_16k.dtype != np.float32:
            wav_16k = wav_16k.astype(np.float32)
        # Ensure 16 kHz
        if len(wav_16k) == 0:
            return torch.zeros(0, 256, device=self.device, dtype=torch.float32)

        x = torch.from_numpy(wav_16k).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Common ContentVec interface: (B, T) -> (B, T', C)
            if hasattr(model, "extract_features"):
                feat = model.extract_features(x)
            elif callable(model):
                feat = model(x)
            else:
                feat = model(x) if hasattr(model, "forward") else model(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feat = feat.squeeze(0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            feat = feat.float()

        out = feat.cpu().numpy()
        if self.cache_dir and cache:
            key = cache_key or (str(wav_16k.shape) + str(wav_16k[:100].tobytes()))
            path = get_cache_path(self.cache_dir, key, ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, feat=out)

        return torch.from_numpy(out).to(self.device)

    def encode_chunk(self, wav_16k_chunk: np.ndarray) -> torch.Tensor:
        """Encode a single chunk (no cache key). Same output shape [T, C]."""
        return self.encode(wav_16k_chunk, cache=False)


