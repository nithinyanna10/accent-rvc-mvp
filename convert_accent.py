#!/usr/bin/env python3
"""
Convert WAV: Indian English (or other L2) -> US English accent using trained BDL model.
CPU-only; streaming-shaped (160 ms windows + overlap-add).
Usage:
  python convert_accent.py --input in.wav --output out.wav --model_dir models [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rvc.config import InferenceParams
from rvc.pipeline import convert_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert accent: WAV input -> converted WAV (target speaker, preserved intonation)."
    )
    parser.add_argument("--input", "-i", required=True, help="Input WAV path")
    parser.add_argument("--output", "-o", required=True, help="Output WAV path")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory with checkpoint + config.json")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for checkpoint (e.g. bdl_accent → bdl_accent_rvc.pth; default: auto-detect)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    parser.add_argument("--streaming", type=int, default=1, choices=(0, 1), help="1=chunked 160ms+OLA (default), 0=full-file (use for debugging quality)")
    parser.add_argument("--window_ms", type=float, default=160.0, help="Chunk window ms (default: 160)")
    parser.add_argument("--hop_ms", type=float, default=80.0, help="Hop ms (default: 80)")
    parser.add_argument("--index_rate", type=float, default=0.0, help="Retrieval blend 0..1; start with 0.2–0.3 (high values can mumble, model not trained on blend) (default: 0)")
    parser.add_argument("--index_k", type=int, default=16, help="kNN neighbors (8–16, default: 16)")
    parser.add_argument("--index_path", type=str, default=None, help="Index dir (default: model_dir)")
    parser.add_argument("--protect", type=float, default=0.2, help="Protect unvoiced 0..1; 0–0.2 for accent (default: 0.2)")
    parser.add_argument("--f0_up_key", type=int, default=0, help="Pitch shift semitones (default: 0)")
    parser.add_argument("--silence_db", type=float, default=-45.0, help="Silence gate threshold dBFS (default: -45; -50 to -55 for stricter)")
    parser.add_argument("--silence_hang", type=int, default=5, help="Silence gate hangover frames to avoid chatter (default: 5)")
    parser.add_argument("--silence_passthrough", action="store_true", help="Pass through original on silent chunks")
    parser.add_argument("--cpu_threads", type=int, default=0, help="CPU threads (0 = default)")
    parser.add_argument("--contentvec_dir", type=str, default="assets/contentvec", help="ContentVec weights dir")
    parser.add_argument("--rmvpe_dir", type=str, default="assets/rmvpe", help="RMVPE weights dir")
    parser.add_argument("--vocoder_dir", type=str, default=None, help="HiFi-GAN weights dir (default: model_dir)")
    parser.add_argument("--out_sr", type=int, default=None, help="Output sample rate (default: 22050). Use 44100 for playback.")
    parser.add_argument("--no-post-gate", action="store_true", help="Disable post-process silence gate (can reduce disturbance)")
    parser.add_argument("--lowpass", type=int, default=None, metavar="HZ", help="Low-pass cutoff Hz to reduce hiss (e.g. 10000)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model_dir not found: {model_dir}", file=sys.stderr)
        print("Place trained model (bdl_rvc.pth + config.json) in models/ or set --model_dir.", file=sys.stderr)
        return 1

    params = InferenceParams(
        device=args.device,
        streaming=bool(args.streaming),
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        silence_db=args.silence_db,
        silence_hang_frames=args.silence_hang,
        silence_passthrough=args.silence_passthrough,
        f0_up_key=args.f0_up_key,
        index_rate=args.index_rate,
        index_k=args.index_k,
        protect=args.protect,
        model_dir=model_dir,
        model_name=args.model_name,
        index_path=Path(args.index_path) if args.index_path else None,
        contentvec_dir=Path(args.contentvec_dir) if args.contentvec_dir else None,
        rmvpe_dir=Path(args.rmvpe_dir) if args.rmvpe_dir else None,
        vocoder_dir=Path(args.vocoder_dir) if args.vocoder_dir else None,
        cpu_threads=args.cpu_threads,
        out_sr=args.out_sr,
        post_gate_dbfs=None if args.no_post_gate else -60.0,
        lowpass_hz=args.lowpass,
    )
    try:
        params.validate()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "streaming" if params.streaming else "full-file"
    print(f"Converting: {input_path} -> {output_path} (device={params.device}, mode={mode})")
    try:
        convert_file(input_path, output_path, model_dir, params)
        out_rate = params.out_sr or params.sample_rate
        print(f"Done: {output_path} ({out_rate} Hz)")
    except FileNotFoundError as e:
        print(f"Error (missing weights): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
