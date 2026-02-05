"""
Content encoder: ContentVec / HuBERT-Soft features.
encode(wav_16k) -> [T, C]. CPU-only; optional caching to disk (npz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

# Patch fairseq before any imports
try:
    import patch_fairseq  # noqa: F401
except ImportError:
    # If patch_fairseq.py is not in path, try to import from parent
    parent_dir = Path(__file__).resolve().parent.parent
    if (parent_dir / "patch_fairseq.py").exists():
        sys.path.insert(0, str(parent_dir))
        import patch_fairseq  # noqa: F401

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
        for name in ("contentvec_256.pt", "contentvec_500.pt", "contentvec.pt", "hubert_soft.pt", "checkpoint_best_100.pt", "checkpoint_best_500.pt"):
            p = d / name
            if p.exists():
                checkpoint = torch.load(p, map_location=self.device, weights_only=False)
                
                # Check if it's a fairseq checkpoint dict with state_dict
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    # This is a fairseq checkpoint - modify cfg and reload
                    try:
                        from fairseq import checkpoint_utils
                        from omegaconf import OmegaConf
                        import tempfile
                        import os
                        
                        # fairseq only has "hubert" in MODEL_REGISTRY, not "contentvec"
                        # Modify checkpoint cfg: task=hubert_pretraining, model=hubert
                        cfg_dict = checkpoint.get("cfg", {})
                        cfg = OmegaConf.create(cfg_dict)
                        if "task" in cfg:
                            cfg.task._name = "hubert_pretraining"
                        if "model" in cfg:
                            cfg.model._name = "hubert"
                        checkpoint["cfg"] = OmegaConf.to_container(cfg, resolve=True)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                            torch.save(checkpoint, tmp_file.name)
                            tmp_path = tmp_file.name
                        try:
                            models, args, task = checkpoint_utils.load_model_ensemble_and_task(
                                [tmp_path],
                                arg_overrides={"data": "/tmp"},
                                strict=False,  # ContentVec 500 has extra layers/keys
                            )
                            model = models[0]
                            self._model = model
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    except (ImportError, ValueError, AttributeError, KeyError) as e:
                        raise ImportError(
                            f"Failed to load ContentVec checkpoint: {e}\n"
                            "The checkpoint_best_*.pt format requires fairseq, which has Python 3.12 compatibility issues.\n"
                            "SOLUTION: Download contentvec_256.pt (standalone model) from:\n"
                            "  https://github.com/auspicious3000/contentvec/releases\n"
                            "Place it in assets/contentvec/contentvec_256.pt"
                        )
                elif isinstance(checkpoint, dict):
                    # Try to use as-is
                    self._model = checkpoint
                else:
                    # Direct model instance
                    self._model = checkpoint
                
                # Set to eval mode if it's a model
                if hasattr(self._model, "eval") and not isinstance(self._model, dict):
                    self._model.eval()
                return self._model
        raise FileNotFoundError(
            f"ContentEncoder: no known weight file in {d}. "
            "Place contentvec_256.pt, contentvec.pt, hubert_soft.pt, or checkpoint_best_*.pt in assets/contentvec/."
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
            # Handle fairseq HubertModel
            if hasattr(model, "extract_features"):
                feat = model.extract_features(x, padding_mask=None, mask=False)[0]
            elif hasattr(model, "forward"):
                feat = model(x, padding_mask=None, mask=False)
                if isinstance(feat, (list, tuple)):
                    feat = feat[0]
            else:
                feat = model(x)
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


