"""
Optional retrieval index (RVC-style): build embeddings from content features.
faiss-cpu if available; else sklearn NearestNeighbors.
Output models/bdl.index + metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
from sklearn.neighbors import NearestNeighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval index from content features.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Feature directory (with .npz + index)")
    parser.add_argument("--model_dir", type=str, default="models", help="Output directory for index")
    parser.add_argument("--name", type=str, default="bdl", help="Index name (output: {name}.index)")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    index_path = feature_dir / "feature_index.json"
    if not index_path.exists():
        paths = list(feature_dir.glob("*.npz"))
    else:
        with open(index_path) as f:
            index = json.load(f)
        paths = [Path(e["path"]) for e in index]
        paths = [p if p.exists() else feature_dir / p.name for p in paths]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"No feature .npz in {feature_dir}")

    embeddings = []
    for p in paths:
        d = np.load(p)
        c = d["content"]
        embeddings.append(c)
    X = np.vstack(embeddings).astype(np.float32)
    print(f"Index shape: {X.shape}")

    if HAS_FAISS:
        d = X.shape[1]
        indexer = faiss.IndexFlatL2(d)
        indexer.add(X)
        out_path = model_dir / f"{args.name}.index"
        faiss.write_index(indexer, str(out_path))
        meta = {"n_vectors": X.shape[0], "dim": d, "type": "faiss"}
    else:
        nbrs = NearestNeighbors(n_neighbors=min(8, len(X)), algorithm="auto", metric="euclidean")
        nbrs.fit(X)
        out_path = model_dir / f"{args.name}_sklearn.npz"
        # Save X for later knn query
        np.savez_compressed(out_path, embeddings=X, paths=[str(p) for p in paths])
        meta = {"n_vectors": X.shape[0], "dim": X.shape[1], "type": "sklearn", "path": str(out_path)}
    meta_path = model_dir / f"{args.name}_index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote index to {model_dir}, meta: {meta_path}")


if __name__ == "__main__":
    main()
