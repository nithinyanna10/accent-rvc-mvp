"""
Export trained RVC generator to ONNX for faster CPU inference.

ONNX inference is typically 2-5x faster than PyTorch on CPU for these small models
and eliminates the PyTorch startup overhead.

Usage:
  python -m rvc.export_onnx --model_dir models --name bdl --output models/bdl_rvc.onnx
  python -m rvc.export_onnx --model_dir models --validate  # validate after export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def export_model(
    model_dir: Path,
    output_path: Path,
    model_name: str | None = None,
    opset: int = 17,
    validate: bool = True,
    verbose: bool = False,
) -> None:
    """
    Load a trained VCModel and export it to ONNX.

    Exported model signature:
      inputs:  content [B, T, content_dim], f0 [B, T, 1]
      outputs: mel     [B, mel_dim, T]
    """
    try:
        import onnx
    except ImportError:
        print(
            "onnx not installed. Run: pip install onnx onnxruntime",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load model ──────────────────────────────────────────────────────────
    from rvc.vc_model import VCModel
    vc = VCModel(model_dir=model_dir, device="cpu", model_name=model_name)
    net = vc.load()
    net.eval()
    cfg = vc.config
    content_dim = cfg.get("content_dim", 256)
    mel_dim = cfg.get("mel_dim", 80)

    # ── Dummy inputs for tracing ─────────────────────────────────────────────
    T = 50  # typical sequence length for tracing
    dummy_content = torch.randn(1, T, content_dim, dtype=torch.float32)
    dummy_f0 = torch.randn(1, T, 1, dtype=torch.float32).abs()

    # ── Export ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {output_path} (opset={opset})...")

    torch.onnx.export(
        net,
        (dummy_content, dummy_f0),
        str(output_path),
        opset_version=opset,
        input_names=["content", "f0"],
        output_names=["mel"],
        dynamic_axes={
            "content": {0: "batch", 1: "time"},
            "f0":      {0: "batch", 1: "time"},
            "mel":     {0: "batch", 2: "time"},
        },
        verbose=verbose,
    )

    # ── Validate ONNX model ──────────────────────────────────────────────────
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    print(f"ONNX model is valid.  ({output_path.stat().st_size / 1024:.1f} KB)")

    if validate:
        _validate_with_onnxruntime(
            output_path,
            dummy_content.numpy(),
            dummy_f0.numpy(),
            net,
        )

    # ── Save companion metadata ───────────────────────────────────────────────
    meta_path = output_path.with_suffix(".onnx.json")
    meta = {
        "content_dim": content_dim,
        "mel_dim": mel_dim,
        "opset": opset,
        "source": str(model_dir),
        "model_name": model_name,
        "dynamic_axes": ["batch", "time"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def _validate_with_onnxruntime(
    onnx_path: Path,
    content_np: np.ndarray,
    f0_np: np.ndarray,
    torch_net: torch.nn.Module,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Run both PyTorch and ONNX Runtime, compare outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping validation. pip install onnxruntime")
        return

    print("Validating with onnxruntime...")
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

    # ONNX output
    ort_out = sess.run(
        None,
        {"content": content_np, "f0": f0_np},
    )[0]

    # PyTorch output
    with torch.no_grad():
        content_t = torch.from_numpy(content_np)
        f0_t = torch.from_numpy(f0_np)
        pt_out = torch_net(content_t, f0_t).numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    match = np.allclose(ort_out, pt_out, rtol=rtol, atol=atol)
    status = "✅ MATCH" if match else f"⚠️  MISMATCH (max_diff={max_diff:.6f})"
    print(f"ORT vs PyTorch output: {status}")


# ────────────────────────────────────────────────────────────────────────────
# ONNX inference helper (drop-in for VCModel)
# ────────────────────────────────────────────────────────────────────────────

class OnnxVCModel:
    """
    Drop-in replacement for VCModel that uses an exported ONNX model.
    Typically 2-5x faster on CPU for small generators.

    Usage:
        vc = OnnxVCModel("models/bdl_rvc.onnx")
        mel = vc.forward(content, f0)
    """

    def __init__(self, onnx_path: str | Path, device: str = "cpu"):
        self.onnx_path = Path(onnx_path)
        self.device = device
        self._sess = None
        self._config: dict = {}

        meta_path = self.onnx_path.with_suffix(".onnx.json")
        if meta_path.exists():
            with open(meta_path) as f:
                self._config = json.load(f)

    def _ensure_session(self):
        if self._sess is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self._sess = ort.InferenceSession(str(self.onnx_path), opts)

    def forward(
        self,
        content: "torch.Tensor",
        f0: "torch.Tensor",
        index_feat=None,
        index_rate: float = 0.0,
    ) -> "torch.Tensor":
        self._ensure_session()
        c = content.cpu().numpy()
        f = f0.cpu().numpy()
        # Reshape: expect [1, T, C] and [1, T, 1]
        if c.ndim == 2:
            c = c[np.newaxis]
        if f.ndim == 1:
            f = f[np.newaxis, :, np.newaxis]
        elif f.ndim == 2:
            f = f[np.newaxis]
        mel_np = self._sess.run(None, {"content": c, "f0": f})[0]
        return torch.from_numpy(mel_np)

    @property
    def config(self) -> dict:
        return self._config


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Export RVC generator to ONNX.")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path (default: model_dir/<name>_rvc.onnx)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no_validate", action="store_true", help="Skip onnxruntime validation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    name = args.model_name or "bdl"
    output_path = Path(args.output) if args.output else (model_dir / f"{name}_rvc.onnx")

    export_model(
        model_dir=model_dir,
        output_path=output_path,
        model_name=args.model_name,
        opset=args.opset,
        validate=not args.no_validate,
        verbose=args.verbose,
    )
    print(f"\nExported: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
