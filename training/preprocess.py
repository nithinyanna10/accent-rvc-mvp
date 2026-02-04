"""
Preprocess raw dataset: mono, fixed sr (40k), light trim, segment into 2–6 s with manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 30) -> np.ndarray:
    """Trim leading/trailing silence (light)."""
    wav_trim, _ = librosa.effects.trim(wav, top_db=top_db)
    return wav_trim


def segment_audio(
    wav: np.ndarray,
    sr: int,
    min_dur: float,
    max_dur: float,
) -> list[tuple[int, int]]:
    """Return list of (start_sample, end_sample) for segments of length [min_dur, max_dur]."""
    min_s = int(min_dur * sr)
    max_s = int(max_dur * sr)
    segments = []
    start = 0
    n = len(wav)
    while start < n:
        end = min(start + max_s, n)
        if end - start >= min_s:
            segments.append((start, end))
        start = end
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw BDL (or other) WAVs into segments.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Raw WAV directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output processed directory")
    parser.add_argument("--sr", type=int, default=40000, help="Target sample rate")
    parser.add_argument("--min_dur", type=float, default=2.0, help="Min segment duration (s)")
    parser.add_argument("--max_dur", type=float, default=6.0, help="Max segment duration (s)")
    parser.add_argument("--trim_db", type=float, default=30.0, help="Trim silence below this dB")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = args.sr
    min_dur = args.min_dur
    max_dur = args.max_dur

    wav_files = list(raw_dir.glob("*.wav")) + list(raw_dir.glob("*.flac"))
    manifest = []
    for f in sorted(wav_files):
        wav, file_sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = librosa.resample(wav.astype(np.float64), orig_sr=file_sr, target_sr=sr, res_type="kaiser_best").astype(np.float32)
        wav = trim_silence(wav, sr, top_db=args.trim_db)
        segments = segment_audio(wav, sr, min_dur, max_dur)
        for i, (start, end) in enumerate(segments):
            seg = wav[start:end]
            out_name = f"{f.stem}_seg{i:03d}.wav"
            out_path = out_dir / out_name
            sf.write(out_path, seg, sr)
            dur = (end - start) / sr
            manifest.append({"path": out_name, "duration": dur, "source": f.name})
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Wrote {len(manifest)} segments to {out_dir}, manifest: {manifest_path}")


if __name__ == "__main__":
    main()
