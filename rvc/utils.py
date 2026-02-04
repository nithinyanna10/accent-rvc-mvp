"""Shared utilities: paths, device, cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


def get_cache_path(base_dir: Path, key: str, suffix: str = ".npz") -> Path:
    """Deterministic cache path from key (e.g. file path + sr)."""
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{h}{suffix}"


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def raise_missing_weights(
    name: str,
    dir_path: Optional[Path],
    hint: str,
) -> None:
    if dir_path is None:
        raise FileNotFoundError(
            f"{name}: weights directory not set. {hint}"
        )
    d = Path(dir_path)
    if not d.exists():
        raise FileNotFoundError(
            f"{name}: weights directory not found: {d}. {hint}"
        )
