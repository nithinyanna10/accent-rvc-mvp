#!/usr/bin/env python3
"""
Build kNN index from BDL (target) content features for retrieval at inference.
Blending with this index (index_rate 0.6–0.9) pushes pronunciation toward target (accent reduction).
Usage:
  python training/build_index.py --feature_dir data/cmu_arctic_bdl/features --model_dir models [--k 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

import pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval index from BDL content features.")
    parser.add_argument("--feature_dir", type=str, required=True, help="BDL feature dir (with .npz + feature_index.json)")
    parser.add_argument("--model_dir", type=str, default="models", help="Save index to this dir (bdl_index.npz + bdl_index_nn.joblib)")
    parser.add_argument("--k", type=int, default=16, help="Default k for kNN (8–16 typical)")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    index_path = feature_dir / "feature_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        paths = [Path(e["path"]) for e in index]
    else:
        paths = list(feature_dir.glob("*.npz"))

    rows = []
    for p in paths:
        if not p.exists():
            p = feature_dir / Path(p).name
        if not p.exists():
            continue
        d = np.load(p)
        c = d["content"]  # [T, C]
        rows.append(c)
    if not rows:
        raise FileNotFoundError(f"No .npz content in {feature_dir}")

    data = np.vstack(rows).astype(np.float32)
    n_neighbors = min(args.k, len(data))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(data)

    np.savez_compressed(model_dir / "bdl_index.npz", data=data)
    with open(model_dir / "bdl_index_nn.pkl", "wb") as f:
        pickle.dump({"nn": nn, "k": args.k}, f)

    print(f"Built index: {data.shape[0]} vectors, dim {data.shape[1]}, k={args.k}")
    print(f"Saved: {model_dir / 'bdl_index.npz'}, {model_dir / 'bdl_index_nn.pkl'}")
    print("Use: convert_accent.py ... --index_rate 0.6 --protect 0.2")


if __name__ == "__main__":
    main()
