"""
Streaming-shaped chunking and overlap-add.
160 ms windows + 80 ms hop; no future leak; producer/consumer compatible.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class StreamingChunker:
    """
    Yields overlapped chunks of fixed window_ms and hop_ms.
    Designed for real-time shape: each chunk is independent (no look-ahead).
    """

    def __init__(
        self,
        sr: int,
        window_ms: float = 160.0,
        hop_ms: float = 80.0,
    ):
        self.sr = sr
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.window_samples = int(sr * window_ms / 1000.0)
        self.hop_samples = int(sr * hop_ms / 1000.0)

    def chunk(self, wav: np.ndarray) -> Iterator[tuple[np.ndarray, int, int]]:
        """
        Yield (chunk, start_sample, end_sample) for overlap-add indexing.
        chunk is window_samples long; start/end are in full wav indices.
        """
        n = len(wav)
        start = 0
        while start < n:
            end = min(start + self.window_samples, n)
            chunk = np.zeros(self.window_samples, dtype=wav.dtype)
            chunk[: end - start] = wav[start:end]
            yield chunk, start, end
            start += self.hop_samples
            if start >= n:
                break


class OverlapAdd:
    """
    Crossfade overlapped chunks into a single waveform.
    Uses linear (triangular) window for overlap-add to avoid boundary clicks.
    """

    def __init__(
        self,
        sr: int,
        window_ms: float = 160.0,
        hop_ms: float = 80.0,
    ):
        self.sr = sr
        self.window_samples = int(sr * window_ms / 1000.0)
        self.hop_samples = int(sr * hop_ms / 1000.0)
        # Build OLA weight: triangular so that sum of overlapping windows = 1
        self.weight = np.zeros(self.window_samples, dtype=np.float32)
        for i in range(self.window_samples):
            # Linear ramp up then down over window
            self.weight[i] = 1.0 - abs(2.0 * i / (self.window_samples - 1) - 1.0) if self.window_samples > 1 else 1.0
        # Normalize so that overlapping windows sum to ~1 at steady state
        num_overlaps = max(1, self.window_samples // self.hop_samples)
        self.weight /= num_overlaps

    def add_chunk(
        self,
        out: np.ndarray,
        chunk: np.ndarray,
        start_sample: int,
    ) -> None:
        """Add one chunk into out at start_sample with OLA weight."""
        end = min(start_sample + len(chunk), len(out))
        n = end - start_sample
        if n <= 0:
            return
        w = self.weight[:n]
        out[start_sample:end] += chunk[:n] * w

    def allocate_output(self, total_samples: int) -> np.ndarray:
        """Allocate float32 buffer for final output."""
        return np.zeros(total_samples, dtype=np.float32)
