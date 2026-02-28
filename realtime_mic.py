#!/usr/bin/env python3
"""
Real-time microphone accent conversion.
Captures audio from mic → converts accent → plays back through speakers.
Uses sounddevice for cross-platform audio I/O.
Runs CPU-only; latency ~320-480 ms (2–3 pipeline chunks).

Usage:
  python realtime_mic.py --model_dir models [options]
  python realtime_mic.py --model_dir models --list_devices
  python realtime_mic.py --model_dir models --input_device 1 --output_device 2
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not found. Install with: pip install sounddevice", file=sys.stderr)
    sys.exit(1)

from rvc.config import InferenceParams
from rvc.content_encoder import ContentEncoder
from rvc.pitch_rmvpe import RMVPExtractor, DEFAULT_SR_F0
from rvc.vocoder_hifigan import HiFiGANVocoder
from rvc.vc_model import VCModel
from rvc.retrieval_index import load_index
from rvc.audio import resample
from rvc.silence_gate import SilenceGate
from rvc.pipeline import _smooth_f0, _interp_mel_to_vocoder_rate, _post_process_audio
import torch

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

PIPELINE_SR = 22050       # internal pipeline sample rate
MIC_BLOCKSIZE = int(PIPELINE_SR * 0.160)  # 160 ms input block
LATENCY_HINT = "low"      # sounddevice latency hint


# ────────────────────────────────────────────────────────────────────────────
# Pipeline state (loaded once, shared across threads)
# ────────────────────────────────────────────────────────────────────────────

class RealtimePipeline:
    """Holds loaded models for real-time conversion."""

    def __init__(self, params: InferenceParams, model_dir: Path):
        self.params = params
        self.sr = params.sample_rate
        self._content_encoder: Optional[ContentEncoder] = None
        self._rmvpe: Optional[RMVPExtractor] = None
        self._vc: Optional[VCModel] = None
        self._vocoder: Optional[HiFiGANVocoder] = None
        self._retrieval = None
        self._gate = SilenceGate(
            threshold_db=params.silence_db,
            hangover_frames=params.silence_hang_frames,
            passthrough=params.silence_passthrough,
        )
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self.frames_processed = 0
        self.total_latency_ms = 0.0

    def _ensure_loaded(self):
        """Lazy-load all models (thread-safe)."""
        if self._content_encoder is not None:
            return
        with self._lock:
            if self._content_encoder is not None:
                return
            p = self.params
            print("  Loading ContentEncoder...", flush=True)
            self._content_encoder = ContentEncoder(
                weights_dir=p.contentvec_dir, device=p.device
            )
            print("  Loading RMVPE...", flush=True)
            self._rmvpe = RMVPExtractor(weights_dir=p.rmvpe_dir, device=p.device)
            print("  Loading VCModel...", flush=True)
            self._vc = VCModel(
                model_dir=self._model_dir,
                device=p.device,
                model_name=p.model_name,
            )
            print("  Loading HiFi-GAN vocoder...", flush=True)
            self._vocoder = HiFiGANVocoder(
                weights_dir=p.vocoder_dir or self._model_dir,
                device=p.device,
                sr=self.sr,
            )
            if p.index_rate > 0:
                self._retrieval = load_index(
                    self._model_dir, p.index_path, k=p.index_k
                )
            print("  All models loaded.", flush=True)

    def process_chunk(self, chunk_22k: np.ndarray) -> np.ndarray:
        """
        Process one chunk at pipeline SR (22050 Hz).
        Returns converted chunk at same SR and same length.
        """
        t0 = time.perf_counter()
        self._ensure_loaded()

        # Silence gate: skip heavy processing for silent frames
        if self._gate.is_silent(chunk_22k):
            return self._gate.silent_output(chunk_22k)

        chunk_16k = resample(chunk_22k, self.sr, DEFAULT_SR_F0)
        if len(chunk_16k) < 320:
            return np.zeros_like(chunk_22k)

        p = self.params
        content = self._content_encoder.encode_chunk(chunk_16k)
        f0 = self._rmvpe.extract_f0(chunk_16k, DEFAULT_SR_F0)

        n_content = content.shape[0]
        if len(f0) != n_content:
            f0 = np.interp(
                np.linspace(0, len(f0) - 1, n_content),
                np.arange(len(f0)),
                f0,
            ).astype(np.float32)
        f0 = _smooth_f0(f0)

        f0_t = torch.from_numpy(f0).to(p.device).unsqueeze(0).unsqueeze(-1)
        content_t = content.unsqueeze(0)

        index_feat = None
        if self._retrieval is not None and p.index_rate > 0:
            idx_arr = self._retrieval.get_index_feat(content.cpu().numpy(), k=p.index_k)
            index_feat = torch.from_numpy(idx_arr).to(p.device).unsqueeze(0)

        mel_or_wav = self._vc.forward(
            content_t, f0_t, index_feat=index_feat, index_rate=p.index_rate
        )

        cfg = self._vc.config
        if cfg.get("mel_normalize") and mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
            mel_mean = cfg.get("mel_mean", -20.0)
            mel_std = cfg.get("mel_std", 10.0)
            mel_or_wav = mel_or_wav * mel_std + mel_mean

        if mel_or_wav.dim() == 3 and mel_or_wav.shape[1] > 1:
            mel_voc = _interp_mel_to_vocoder_rate(
                mel_or_wav, self.sr, mel_smooth_frames=p.mel_smooth_frames
            )
            wav_out = self._vocoder.decode(mel_voc.squeeze(0))
        else:
            wav_out = mel_or_wav.squeeze().cpu().numpy()
            if wav_out.ndim > 1:
                wav_out = wav_out.mean(axis=0)

        # Trim / pad to input chunk length
        target_len = len(chunk_22k)
        out = np.zeros(target_len, dtype=np.float32)
        copy_len = min(len(wav_out), target_len)
        if copy_len > 0:
            out[:copy_len] = wav_out[:copy_len].astype(np.float32)

        # Blend with original (wet/dry) if requested
        blend = getattr(p, "blend", 1.0)
        if 0.0 < blend < 1.0:
            out = blend * out + (1.0 - blend) * chunk_22k

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.frames_processed += 1
        self.total_latency_ms += elapsed_ms

        return out.astype(np.float32)

    @property
    def avg_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_latency_ms / self.frames_processed


# ────────────────────────────────────────────────────────────────────────────
# Audio I/O with sounddevice
# ────────────────────────────────────────────────────────────────────────────

class RealtimeSession:
    """Manages the sounddevice stream + processing queue."""

    def __init__(
        self,
        pipeline: RealtimePipeline,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        output_file: Optional[Path] = None,
        passthrough: bool = False,
    ):
        self.pipeline = pipeline
        self.input_device = input_device
        self.output_device = output_device
        self.output_file = output_file
        self.passthrough = passthrough
        self.sr = pipeline.sr

        self._input_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._output_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._running = False
        self._recorded_chunks: list[np.ndarray] = []
        self._worker_thread: Optional[threading.Thread] = None

    def _mic_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ):
        if status:
            print(f"[mic] {status}", flush=True)
        chunk = indata[:, 0].copy().astype(np.float32)
        # Resample from mic SR to pipeline SR if necessary
        try:
            self._input_q.put_nowait(chunk)
        except queue.Full:
            pass  # drop frame under overload

    def _spk_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ):
        if status:
            print(f"[spk] {status}", flush=True)
        try:
            chunk = self._output_q.get_nowait()
        except queue.Empty:
            chunk = np.zeros(frames, dtype=np.float32)
        n = min(len(chunk), frames)
        outdata[:n, 0] = chunk[:n]
        if n < frames:
            outdata[n:, 0] = 0.0

    def _worker(self):
        """Processing thread: reads from input_q, converts, writes to output_q."""
        while self._running:
            try:
                chunk = self._input_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.passthrough:
                out = chunk
            else:
                out = self.pipeline.process_chunk(chunk)
            if self.output_file is not None:
                self._recorded_chunks.append(out.copy())
            try:
                self._output_q.put_nowait(out)
            except queue.Full:
                pass

    def run(self):
        """Start streams and block until user presses Ctrl+C."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        blocksize = int(self.sr * 0.160)  # 160 ms blocks

        with sd.InputStream(
            samplerate=self.sr,
            blocksize=blocksize,
            device=self.input_device,
            channels=1,
            dtype="float32",
            callback=self._mic_callback,
            latency=LATENCY_HINT,
        ), sd.OutputStream(
            samplerate=self.sr,
            blocksize=blocksize,
            device=self.output_device,
            channels=1,
            dtype="float32",
            callback=self._spk_callback,
            latency=LATENCY_HINT,
        ):
            print("\nStreaming... Press Ctrl+C to stop.\n", flush=True)
            self._print_header()
            try:
                while True:
                    time.sleep(2.0)
                    self._print_stats()
            except KeyboardInterrupt:
                print("\nStopping...", flush=True)

        self._running = False
        self._worker_thread.join(timeout=2.0)

        if self.output_file and self._recorded_chunks:
            self._save_recording()

    def _print_header(self):
        print(
            f"{'Frame':>6}  {'Avg Latency':>12}  {'Queue In':>8}  {'Queue Out':>9}"
        )
        print("-" * 44)

    def _print_stats(self):
        p = self.pipeline
        print(
            f"{p.frames_processed:>6}  "
            f"{p.avg_latency_ms:>10.1f}ms  "
            f"{self._input_q.qsize():>8}  "
            f"{self._output_q.qsize():>9}",
            flush=True,
        )

    def _save_recording(self):
        import soundfile as sf
        audio = np.concatenate(self._recorded_chunks)
        sf.write(self.output_file, audio, self.sr)
        print(f"Saved recording to {self.output_file} ({len(audio)/self.sr:.1f}s)")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def list_devices():
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        tag = ""
        if i == sd.default.device[0]:
            tag += " [default-in]"
        if i == sd.default.device[1]:
            tag += " [default-out]"
        print(
            f"  [{i:2d}] {dev['name']:<36} "
            f"in={dev['max_input_channels']} out={dev['max_output_channels']}{tag}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time accent conversion: mic → convert → speakers."
    )
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--list_devices", action="store_true", help="Print audio devices and exit")
    parser.add_argument("--input_device", type=int, default=None, help="Mic device index (default: system default)")
    parser.add_argument("--output_device", type=int, default=None, help="Speaker device index (default: system default)")
    parser.add_argument("--passthrough", action="store_true", help="Pass mic audio unchanged (for testing latency)")
    parser.add_argument("--save", type=str, default=None, metavar="PATH", help="Save converted output to WAV")
    parser.add_argument("--index_rate", type=float, default=0.0)
    parser.add_argument("--index_k", type=int, default=16)
    parser.add_argument("--blend", type=float, default=1.0, metavar="0..1", help="Wet/dry mix: 1.0=full conversion, 0.0=original (default: 1.0)")
    parser.add_argument("--protect", type=float, default=0.2)
    parser.add_argument("--silence_db", type=float, default=-45.0)
    parser.add_argument("--silence_hang", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--contentvec_dir", type=str, default="assets/contentvec")
    parser.add_argument("--rmvpe_dir", type=str, default="assets/rmvpe")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    params = InferenceParams(
        device=args.device,
        streaming=True,
        silence_db=args.silence_db,
        silence_hang_frames=args.silence_hang,
        index_rate=args.index_rate,
        index_k=args.index_k,
        protect=args.protect,
        model_dir=model_dir,
        model_name=args.model_name,
        contentvec_dir=Path(args.contentvec_dir),
        rmvpe_dir=Path(args.rmvpe_dir),
        blend=args.blend,
    )

    print("Loading models (first chunk may be slow)...")
    rt_pipeline = RealtimePipeline(params, model_dir)
    rt_pipeline._ensure_loaded()

    session = RealtimeSession(
        pipeline=rt_pipeline,
        input_device=args.input_device,
        output_device=args.output_device,
        output_file=Path(args.save) if args.save else None,
        passthrough=args.passthrough,
    )

    print(
        f"\nRealtime accent conversion"
        f"\n  Model:   {model_dir}"
        f"\n  Blend:   {args.blend:.2f} (1.0=full convert, 0.0=bypass)"
        f"\n  Device:  {args.device}"
    )

    session.run()

    print(
        f"\nStats: {rt_pipeline.frames_processed} frames, "
        f"avg latency {rt_pipeline.avg_latency_ms:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
