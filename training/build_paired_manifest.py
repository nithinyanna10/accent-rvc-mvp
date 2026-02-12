#!/usr/bin/env python3
"""
Build a manifest of parallel (Indian, American BDL) pairs for accent conversion.
L2-ARCTIC and CMU ARCTIC share the same prompts: e.g. ASI_arctic_a0001.wav (Indian)
and bdl_arctic_a0001.wav (BDL) are the same sentence. We match by utterance ID.
Output: paired_manifest.json with { "utt_id", "l2_path", "bdl_path" }.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def utt_id_from_l2(stem: str) -> str | None:
    """L2: ASI_arctic_a0001 -> arctic_a0001; RRBI_arctic_b0023 -> arctic_b0023."""
    if "_" not in stem:
        return None
    return stem.split("_", 1)[1]


def utt_id_from_bdl(stem: str) -> str | None:
    """BDL: bdl_arctic_a0001 -> arctic_a0001; arctic_a0019.wav (no prefix) -> arctic_a0019."""
    if stem.startswith("bdl_"):
        return stem.replace("bdl_", "", 1)
    if stem.startswith("arctic_"):
        return stem
    return None


def normalize_utt_id(uid: str) -> str:
    """Allow matching arctic_a0001 with arctic0001."""
    # arctic_a0001 -> arctic0001; arctic_b0023 -> arctic0023 (or keep b for variety)
    m = re.match(r"arctic_([ab])(\d+)", uid, re.I)
    if m:
        return f"arctic{m[2]}"  # arctic0001
    return uid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired manifest: L2 (Indian) WAVs matched to BDL (American) WAVs by utterance ID."
    )
    parser.add_argument("--l2_dir", type=str, required=True, help="L2-ARCTIC flat dir (e.g. data/l2_arctic_flat)")
    parser.add_argument("--bdl_dir", type=str, required=True, help="CMU ARCTIC BDL raw dir (e.g. data/cmu_arctic_bdl/raw)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: l2_dir/paired_manifest.json)")
    parser.add_argument("--l2_suffix", type=str, default=".wav", help="L2 file suffix")
    parser.add_argument("--bdl_suffix", type=str, default=".wav", help="BDL file suffix")
    args = parser.parse_args()

    l2_dir = Path(args.l2_dir)
    bdl_dir = Path(args.bdl_dir)
    out_path = Path(args.out) if args.out else l2_dir / "paired_manifest.json"

    if not l2_dir.is_dir():
        raise FileNotFoundError(f"L2 dir not found: {l2_dir}")
    if not bdl_dir.is_dir():
        raise FileNotFoundError(f"BDL dir not found: {bdl_dir}")

    # Index BDL by normalized utt_id (and raw)
    bdl_by_uid: dict[str, Path] = {}
    for p in bdl_dir.glob(f"*{args.bdl_suffix}"):
        uid = utt_id_from_bdl(p.stem)
        if uid is None:
            continue
        bdl_by_uid[uid] = p
        norm = normalize_utt_id(uid)
        if norm != uid:
            bdl_by_uid.setdefault(norm, p)

    # Scan L2 and match
    pairs = []
    for p in sorted(l2_dir.glob(f"*{args.l2_suffix}")):
        uid = utt_id_from_l2(p.stem)
        if uid is None:
            continue
        bdl_path = bdl_by_uid.get(uid) or bdl_by_uid.get(normalize_utt_id(uid))
        if bdl_path is None:
            continue
        pairs.append({
            "utt_id": uid,
            "l2_path": str(p),
            "l2_name": p.name,
            "bdl_path": str(bdl_path),
            "bdl_name": bdl_path.name,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"pairs": pairs, "l2_dir": str(l2_dir), "bdl_dir": str(bdl_dir)}, f, indent=2)
    print(f"Wrote {out_path} with {len(pairs)} pairs (L2 dir: {l2_dir}, BDL dir: {bdl_dir}).")
    if not pairs:
        print("No pairs found. BDL filenames: bdl_arctic_a0001.wav or arctic_a0001.wav; L2: ASI_arctic_a0001.wav.")


if __name__ == "__main__":
    main()
