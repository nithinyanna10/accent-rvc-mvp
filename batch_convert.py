#!/usr/bin/env python3
"""
Batch accent conversion: process a folder of WAV files.
Writes converted WAVs to output directory with a JSON summary.

Usage:
  python batch_convert.py --input_dir data/l2_arctic --output_dir samples/batch [options]
  python batch_convert.py --input_dir data/l2_arctic --output_dir samples/batch --pattern "*.wav" --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from rvc.config import InferenceParams
from rvc.pipeline import convert_file


# ────────────────────────────────────────────────────────────────────────────
# Per-file conversion with timing + error capture
# ────────────────────────────────────────────────────────────────────────────

def _convert_one(
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    params: InferenceParams,
) -> dict:
    """Convert a single file; return a result dict with timing and status."""
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "status": "ok",
        "error": None,
        "duration_s": None,
        "elapsed_ms": None,
    }
    t0 = time.perf_counter()
    try:
        import soundfile as sf
        info = sf.info(input_path)
        result["duration_s"] = round(info.duration, 3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_file(input_path, output_path, model_dir, params)
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except FileNotFoundError as e:
        result["status"] = "missing_weights"
        result["error"] = str(e)
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Batch runner
# ────────────────────────────────────────────────────────────────────────────

def run_batch(
    input_dir: Path,
    output_dir: Path,
    model_dir: Path,
    params: InferenceParams,
    pattern: str = "*.wav",
    workers: int = 1,
    skip_existing: bool = False,
    flat: bool = False,
    summary_path: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """
    Convert all files matching `pattern` under `input_dir`.
    Returns summary dict.
    """
    input_files = sorted(input_dir.rglob(pattern))
    if not input_files:
        print(f"No files matching '{pattern}' found in {input_dir}", file=sys.stderr)
        return {"total": 0, "ok": 0, "skipped": 0, "errors": 0, "results": []}

    # Build output paths
    pairs: list[tuple[Path, Path]] = []
    for src in input_files:
        if flat:
            dest = output_dir / src.name
        else:
            rel = src.relative_to(input_dir)
            dest = output_dir / rel
        if skip_existing and dest.exists():
            continue
        pairs.append((src, dest))

    skipped_count = len(input_files) - len(pairs)
    print(
        f"Batch convert: {len(input_files)} files found, "
        f"{skipped_count} skipped (existing), "
        f"{len(pairs)} to process."
    )
    if dry_run:
        print("[dry-run] Would process:")
        for src, dest in pairs:
            print(f"  {src} -> {dest}")
        return {"total": len(pairs), "ok": 0, "skipped": skipped_count, "errors": 0, "dry_run": True, "results": []}

    results: list[dict] = []
    start_time = time.perf_counter()

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(pairs), desc="Converting", unit="file")

    if workers <= 1:
        for src, dest in pairs:
            r = _convert_one(src, dest, model_dir, params)
            results.append(r)
            _log_result(r, pbar)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_convert_one, src, dest, model_dir, params): (src, dest)
                for src, dest in pairs
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                _log_result(r, pbar)

    if pbar is not None:
        pbar.close()

    total_elapsed = time.perf_counter() - start_time
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = sum(1 for r in results if r["status"] not in ("ok", "skipped"))
    total_audio_s = sum(r["duration_s"] or 0 for r in results)
    avg_rtf = (
        (sum(r["elapsed_ms"] or 0 for r in results) / 1000.0 / total_audio_s)
        if total_audio_s > 0
        else None
    )

    summary = {
        "total_files": len(input_files),
        "processed": len(pairs),
        "ok": ok,
        "skipped": skipped_count,
        "errors": errors,
        "total_audio_s": round(total_audio_s, 2),
        "wall_time_s": round(total_elapsed, 2),
        "avg_rtf": round(avg_rtf, 3) if avg_rtf else None,
        "results": results,
    }

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")

    _print_summary(summary)
    return summary


def _log_result(r: dict, pbar) -> None:
    status_icon = "✓" if r["status"] == "ok" else "✗"
    line = (
        f"  {status_icon} {Path(r['input']).name:<40} "
        f"{'→ ' + Path(r['output']).name:<40} "
        f"[{r['elapsed_ms']:.0f}ms]"
        if r["elapsed_ms"]
        else f"  {status_icon} {r['input']}"
    )
    if r["status"] != "ok":
        line += f" ERROR: {r['error']}"
    if pbar is not None:
        pbar.set_postfix_str(Path(r["input"]).name[:30])
        pbar.update(1)
    else:
        print(line, flush=True)


def _print_summary(s: dict) -> None:
    rtf = f"{s['avg_rtf']:.3f}x" if s["avg_rtf"] else "N/A"
    print(
        f"\n{'─'*50}\n"
        f"  Total files : {s['total_files']}\n"
        f"  Processed   : {s['processed']}\n"
        f"  OK          : {s['ok']}\n"
        f"  Skipped     : {s['skipped']}\n"
        f"  Errors      : {s['errors']}\n"
        f"  Audio total : {s['total_audio_s']:.1f}s\n"
        f"  Wall time   : {s['wall_time_s']:.1f}s\n"
        f"  Avg RTF     : {rtf}\n"
        f"{'─'*50}"
    )


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch accent conversion: folder → folder, with progress + JSON summary."
    )
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory (searched recursively)")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory (mirrors input structure)")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.wav", help="Glob pattern to match audio files (default: *.wav)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1; >1 uses threads)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files where output already exists")
    parser.add_argument("--flat", action="store_true", help="Write all outputs to output_dir (no sub-dirs)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done; don't convert")
    parser.add_argument("--summary", type=str, default=None, metavar="PATH", help="Write JSON summary to path (default: output_dir/summary.json)")
    # InferenceParams
    parser.add_argument("--streaming", type=int, default=0, choices=(0, 1))
    parser.add_argument("--index_rate", type=float, default=0.0)
    parser.add_argument("--blend", type=float, default=1.0, metavar="0..1", help="Wet/dry mix: 1.0=full conversion")
    parser.add_argument("--lowpass", type=int, default=11000, metavar="HZ")
    parser.add_argument("--out_sr", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--contentvec_dir", type=str, default="assets/contentvec")
    parser.add_argument("--rmvpe_dir", type=str, default="assets/rmvpe")
    parser.add_argument("--no_post_gate", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if not input_dir.exists():
        print(f"Error: input_dir not found: {input_dir}", file=sys.stderr)
        return 1
    if not model_dir.exists():
        print(f"Error: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    params = InferenceParams(
        device=args.device,
        streaming=bool(args.streaming),
        model_dir=model_dir,
        model_name=args.model_name,
        contentvec_dir=Path(args.contentvec_dir),
        rmvpe_dir=Path(args.rmvpe_dir),
        index_rate=args.index_rate,
        out_sr=args.out_sr,
        post_gate_dbfs=None if args.no_post_gate else -60.0,
        lowpass_hz=args.lowpass if args.lowpass and args.lowpass > 0 else None,
        blend=args.blend,
    )
    try:
        params.validate()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    summary_path = Path(args.summary) if args.summary else (output_dir / "summary.json")

    run_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        params=params,
        pattern=args.pattern,
        workers=args.workers,
        skip_existing=args.skip_existing,
        flat=args.flat,
        summary_path=summary_path,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
