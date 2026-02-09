#!/usr/bin/env python3
"""
Create manifest.json in a directory of WAVs so feature_extract.py can run.
Usage: python training/build_manifest.py --wav_dir data/l2_arctic_flat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest.json from a directory of WAV files.")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing .wav files")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {wav_dir}")

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files in {wav_dir}")

    manifest = []
    for p in wavs:
        try:
            info = sf.info(p)
            duration = float(info.frames) / info.samplerate
        except Exception as e:
            print(f"Warning: skip {p.name}: {e}")
            continue
        manifest.append({
            "path": p.name,
            "duration": round(duration, 3),
            "source": p.stem,
        })

    out = wav_dir / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out} with {len(manifest)} entries.")


if __name__ == "__main__":
    main()
