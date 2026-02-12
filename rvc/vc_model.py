"""
Minimal RVC v2-like generator wrapper for inference (and training).
Compat layer: load exported RVC model (pth + config) and run forward with
content features + f0 + optional index retrieval features.
Avoid InstanceNorm where possible; document behavior if present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .utils import raise_missing_weights


class VCModel:
    """
    Wrapper for RVC v2-style generator.
    Loads models/bdl_rvc.pth + models/config.json.
    Forward: (content_feat, f0, [index_feat]) -> mel or waveform.
    CPU-only; no InstanceNorm in our minimal net (document if loading external weights).
    """

    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
        config_path: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.config_path = config_path or (self.model_dir / "config.json")
        self.model_name = model_name  # e.g. "bdl_accent" -> bdl_accent_rvc.pth
        self._model = None
        self._config: Dict[str, Any] = {}

    def _find_weights(self) -> Path:
        if self.model_name:
            p = self.model_dir / f"{self.model_name}_rvc.pth"
            if p.exists():
                return p
        for name in ("bdl_rvc.pth", "bdl_accent_rvc.pth", "rvc.pth", "model.pth", "generator.pth"):
            p = self.model_dir / name
            if p.exists():
                return p
        raise_missing_weights(
            "VCModel",
            self.model_dir,
            "Place trained model (e.g. bdl_rvc.pth) and config.json in model_dir.",
        )

    def load(self) -> nn.Module:
        if self._model is not None:
            return self._model
        p = self._find_weights()
        raw = torch.load(p, map_location=self.device, weights_only=False)
        if isinstance(raw, dict) and "model" in raw:
            raw = raw["model"]
        if isinstance(raw, dict) and "weight" in raw:
            raw = raw["weight"]
        if isinstance(raw, nn.Module):
            self._model = raw
        else:
            # state_dict from our MinimalGenerator or TemporalGenerator
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self._config = json.load(f)
            content_dim = self._config.get("content_dim", 256)
            mel_dim = self._config.get("mel_dim", 80)
            use_temporal = self._config.get("generator") == "temporal"
            if use_temporal:
                self._model = TemporalGenerator(content_dim=content_dim, f0_dim=1, mel_dim=mel_dim)
            else:
                self._model = MinimalGenerator(content_dim=content_dim, f0_dim=1, mel_dim=mel_dim)
            self._model.load_state_dict(raw, strict=False)
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = json.load(f)
        if hasattr(self._model, "eval"):
            self._model.eval()
        self._model = self._model.to(self.device)
        return self._model

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        index_feat: Optional[torch.Tensor] = None,
        index_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        content: [T, C], f0: [T]. index_feat optional [T, C].
        Returns mel or waveform depending on saved model.
        """
        model = self.load()
        content = content.to(self.device)
        f0 = f0.to(self.device)
        if content.dim() == 2:
            content = content.unsqueeze(0)
        if f0.dim() == 1:
            f0 = f0.unsqueeze(0).unsqueeze(-1)
        if index_feat is not None and index_rate > 0:
            index_feat = index_feat.to(self.device)
            if index_feat.dim() == 2:
                index_feat = index_feat.unsqueeze(0)
            # Interpolate to match content length if needed
            if index_feat.shape[1] != content.shape[1]:
                index_feat = torch.nn.functional.interpolate(
                    index_feat.transpose(1, 2),
                    size=content.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            content = (1 - index_rate) * content + index_rate * index_feat

        with torch.no_grad():
            if hasattr(model, "infer"):
                out = model.infer(content, f0)
            elif hasattr(model, "forward"):
                out = model(content, f0)
            else:
                out = model(content, f0)
            if isinstance(out, (list, tuple)):
                out = out[0]
        return out

    @property
    def config(self) -> Dict[str, Any]:
        if not self._config and self.config_path.exists():
            with open(self.config_path) as f:
                self._config = json.load(f)
        return self._config


class MinimalGenerator(nn.Module):
    """
    Minimal generator for training when not loading external RVC.
    content [B,T,C] + f0 [B,T,1] -> mel [B, n_mel, T].
    No InstanceNorm to avoid train/infer mismatch.
    """

    def __init__(
        self,
        content_dim: int = 256,
        f0_dim: int = 1,
        mel_dim: int = 80,
        hidden: int = 256,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(content_dim + f0_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, mel_dim),
        )

    def forward(self, content: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        # content [B,T,C], f0 [B,T,1]
        x = torch.cat([content, f0], dim=-1)
        return self.proj(x).transpose(1, 2)


class TemporalGenerator(nn.Module):
    """
    Generator with 1D temporal context: each mel frame sees neighboring content/f0.
    Reduces frame-to-frame jitter and often improves accent clarity.
    content [B,T,C] + f0 [B,T,1] -> mel [B, n_mel, T].
    """

    def __init__(
        self,
        content_dim: int = 256,
        f0_dim: int = 1,
        mel_dim: int = 80,
        hidden: int = 256,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.in_dim = content_dim + f0_dim
        self.hidden = hidden
        self.mel_dim = mel_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2  # same length
        # GroupNorm(1, C) on [B, C, T] normalizes over channels (Conv1d output)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.in_dim, hidden, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, mel_dim, 1),
        )

    def forward(self, content: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        # content [B,T,C], f0 [B,T,1] -> x [B,T,C+1]
        x = torch.cat([content, f0], dim=-1)
        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)
        return self.temporal(x)
