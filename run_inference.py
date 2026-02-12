#!/usr/bin/env python3
"""
RVC v2 (Soft-VC) inference: WAV in → WAV out.
Indian English (e.g. L2-ARCTIC) → US English (target speaker, preserved intonation, accent reduced).

This is the single entry-point script for the pipeline specified in PIPELINE_SPEC.md.
Architecture: ContentVec (content) + RMVPE (pitch) + trained generator + HiFi-GAN v2 (vocoder).

Usage:
  python run_inference.py -i input.wav -o output.wav
  python run_inference.py -i input.wav -o output.wav --model_dir models --model_name bdl_accent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rvc.config import InferenceParams
from rvc.pipeline import convert_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RVC v2 Soft-VC inference: convert WAV (Indian English) → WAV (US English accent)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input WAV path (e.g. L2-ARCTIC)")
    parser.add_argument("-o", "--output", required=True, help="Output WAV path")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory with checkpoint + config.json")
    parser.add_argument("--model_name", type=str, default=None, help="Checkpoint name, e.g. bdl_accent → bdl_accent_rvc.pth")
    parser.add_argument("--streaming", type=int, default=1, choices=(0, 1), help="1=streaming (default), 0=full-file (often cleaner)")
    parser.add_argument("--no-post-gate", action="store_true", help="Disable post-process silence gate (try if output has clicks)")
    parser.add_argument("--lowpass", type=int, default=None, metavar="HZ", help="Low-pass cutoff in Hz to reduce hiss (e.g. 10000 or 12000)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_sr", type=int, default=None, help="Output sample rate (default: 22050)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    params = InferenceParams(
        device=args.device,
        streaming=bool(args.streaming),
        model_dir=model_dir,
        model_name=args.model_name,
        contentvec_dir=Path("assets/contentvec"),
        rmvpe_dir=Path("assets/rmvpe"),
        vocoder_dir=None,  # pipeline uses model_dir when None
        index_rate=0.0,
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

    try:
        convert_file(input_path, output_path, model_dir, params)
        sr = params.out_sr or params.sample_rate
        print(f"Done: {output_path} ({sr} Hz)")
    except FileNotFoundError as e:
        print(f"Error (missing weights): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
