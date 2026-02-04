"""
Energy-based silence gate.
If chunk is below threshold (dBFS), bypass conversion and output near-silence
(or optional passthrough). Hangover frames avoid flicker at boundaries.
"""

from __future__ import annotations

import numpy as np


def rms_dbfs(chunk: np.ndarray) -> float:
    """RMS of chunk in dBFS (ref 1.0). Safe for zeros."""
    rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12)
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms + 1e-12)


class SilenceGate:
    def __init__(
        self,
        threshold_db: float = -45.0,
        hangover_frames: int = 3,
        passthrough: bool = False,
    ):
        self.threshold_db = threshold_db
        self.hangover_frames = hangover_frames
        self.passthrough = passthrough
        self._hangover_count = 0

    def is_silent(self, chunk: np.ndarray) -> bool:
        """True if chunk is below threshold or in hangover."""
        db = rms_dbfs(chunk)
        if db >= self.threshold_db:
            self._hangover_count = 0
            return False
        if self._hangover_count > 0:
            self._hangover_count -= 1
            return True
        self._hangover_count = self.hangover_frames
        return True

    def silent_output(self, chunk: np.ndarray) -> np.ndarray:
        """
        Return output for a silent chunk: near-silence or passthrough.
        Same length and dtype as chunk.
        """
        if self.passthrough:
            return chunk.copy()
        return np.zeros_like(chunk)

    def reset(self) -> None:
        self._hangover_count = 0
