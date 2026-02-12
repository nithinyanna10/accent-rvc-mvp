"""
Load kNN index built from BDL content and return retrieved (or blended) content for inference.
Used for accent reduction: blend input content with BDL content via index_rate.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class RetrievalIndex:
    """kNN index over BDL content vectors. Query by mean(content) -> k neighbors -> mean -> broadcast to [T, C]."""

    def __init__(self, index_dir: Path, k: int = 16):
        self.index_dir = Path(index_dir)
        self.k = k
        self._data: Optional[np.ndarray] = None
        self._nn = None
        self._load()

    def _load(self) -> None:
        npz_path = self.index_dir / "bdl_index.npz"
        pkl_path = self.index_dir / "bdl_index_nn.pkl"
        if not npz_path.exists() or not pkl_path.exists():
            raise FileNotFoundError(
                f"Index not found in {self.index_dir}. Run: python training/build_index.py --feature_dir data/cmu_arctic_bdl/features --model_dir {self.index_dir}"
            )
        with np.load(npz_path) as z:
            self._data = z["data"].astype(np.float32)
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self._nn = obj["nn"]
        self.k = min(self.k, obj.get("k", self.k))

    def get_index_feat(self, content: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        content: [T, C]. Query with mean(content), get k nearest BDL vectors, average, broadcast to [T, C].
        Returns float32 [T, C] for blending: (1 - index_rate) * content + index_rate * this.
        """
        k = k or self.k
        q = content.mean(axis=0, keepdims=True).astype(np.float32)
        dists, indices = self._nn.kneighbors(q, n_neighbors=min(k, self._data.shape[0]))
        retrieved = self._data[indices[0]].mean(axis=0).astype(np.float32)
        return np.broadcast_to(retrieved, content.shape).copy()


def load_index(model_dir: Path, index_path: Optional[Path] = None, k: int = 16) -> Optional[RetrievalIndex]:
    """Load index from index_path or model_dir. Returns None if no index found."""
    root = index_path if index_path is not None else model_dir
    for name in ("bdl_index.npz", "index.npz"):
        if (root / name).exists():
            try:
                return RetrievalIndex(root, k=k)
            except Exception:
                pass
    return None
