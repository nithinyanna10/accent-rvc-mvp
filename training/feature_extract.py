"""
Extract and save content (ContentVec @ 16k) + f0 (RMVPE) per segment as .npz.
Write index manifest mapping segment -> feature paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import librosa

# Import from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rvc.content_encoder import ContentEncoder
from rvc.pitch_rmvpe import RMVPExtractor, DEFAULT_SR_F0
from rvc.audio import load_wav, resample

MEL_SR = 40000
MEL_N_FFT = 1024
MEL_HOP = 256
MEL_N_MELS = 80


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ContentVec + RMVPE features for segments.")
    parser.add_argument("--processed_dir", type=str, required=True, help="Processed WAV dir (with manifest.json)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output features directory")
    parser.add_argument("--contentvec_dir", type=str, default="assets/contentvec", help="ContentVec weights dir")
    parser.add_argument("--rmvpe_dir", type=str, default="assets/rmvpe", help="RMVPE weights dir")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {processed_dir}. Run preprocess first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    encoder = ContentEncoder(weights_dir=Path(args.contentvec_dir), device=args.device, cache_dir=out_dir / "content_cache")
    rmvpe = RMVPExtractor(weights_dir=Path(args.rmvpe_dir), device=args.device)

    index = []
    for i, entry in enumerate(manifest):
        path = Path(entry["path"])
        if not path.is_absolute():
            path = processed_dir / path.name
        if not path.exists():
            path = processed_dir / Path(entry["path"]).name
        wav = load_wav(path, 40000)
        wav_16k = resample(wav, 40000, DEFAULT_SR_F0)
        content = encoder.encode(wav_16k, cache=True, cache_key=path.name)
        content_np = content.cpu().numpy()
        f0 = rmvpe.extract_f0(wav_16k, DEFAULT_SR_F0)
        mel = librosa.feature.melspectrogram(
            y=wav.astype(np.float64),
            sr=MEL_SR,
            n_fft=MEL_N_FFT,
            hop_length=MEL_HOP,
            n_mels=MEL_N_MELS,
            fmin=0,
            fmax=8000,
        )
        mel = librosa.power_to_db(mel, ref=1.0).astype(np.float32)
        out_name = path.stem + ".npz"
        out_path = out_dir / out_name
        np.savez_compressed(out_path, content=content_np, f0=f0, mel=mel)
        index.append({"path": str(out_path), "duration": entry["duration"], "source": entry.get("source", path.name)})
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(manifest)}")

    index_path = out_dir / "feature_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} feature files to {out_dir}, index: {index_path}")


if __name__ == "__main__":
    main()
