"""
CPU-friendly RVC training: small batches, gradient accumulation, checkpoint every N steps.
Outputs models/bdl_rvc.pth + models/config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys_path = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path))
from rvc.vc_model import MinimalGenerator


def load_features(feature_dir: Path, index_path: Optional[Path] = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load (content, f0, mel) from .npz files listed in feature_index.json or from directory."""
    feature_dir = Path(feature_dir)
    if index_path is None:
        index_path = feature_dir / "feature_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        paths = [Path(e["path"]) for e in index]
    else:
        paths = list(feature_dir.glob("*.npz"))
    data = []
    for p in paths:
        if not p.exists():
            p = feature_dir / Path(p).name
        if not p.exists():
            continue
        d = np.load(p)
        content = d["content"]
        f0 = d["f0"]
        mel = d["mel"]  # [n_mel, T]
        if len(f0) != content.shape[0]:
            f0 = np.interp(np.linspace(0, len(f0) - 1, content.shape[0]), np.arange(len(f0)), f0).astype(np.float32)
        # Align mel time to content frames (content ~50 Hz, mel ~156 Hz at 40k/256)
        mel_t = mel.shape[1]
        content_t = content.shape[0]
        if mel_t != content_t:
            new_mel = np.zeros((mel.shape[0], content_t), dtype=np.float32)
            for row in range(mel.shape[0]):
                new_mel[row] = np.interp(
                    np.linspace(0, mel_t - 1, content_t),
                    np.arange(mel_t),
                    mel[row].astype(np.float64),
                ).astype(np.float32)
            mel = new_mel
        else:
            mel = mel.astype(np.float32)
        data.append((content, f0, mel))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RVC target-voice model (CPU-friendly).")
    parser.add_argument("--feature_dir", type=str, required=True, help="Feature directory (with .npz + index)")
    parser.add_argument("--model_dir", type=str, default="models", help="Output model directory")
    parser.add_argument("--name", type=str, default="bdl", help="Model name (output: {name}_rvc.pth)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--steps", type=int, default=30000, help="Training steps (30k+ recommended for intelligible voice)")
    parser.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps (listen to samples)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--content_dim", type=int, default=None, help="Content feature dim (default: infer from data; 256 for contentvec_256, 768 for contentvec_500)")
    parser.add_argument("--mel_dim", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize_mel", action="store_true", help="Normalize mel to ~0 mean, 1 std so loss is in reasonable range (not hundreds)")
    parser.add_argument("--mel_mean", type=float, default=-20.0, help="Mel mean for normalization (typical dB mel ~-20)")
    parser.add_argument("--mel_std", type=float, default=10.0, help="Mel std for normalization")
    # Loss: L1 (better for mel) + optional L2 + multi-scale mel (like MR-STFT but on mel)
    parser.add_argument("--loss_l1", type=float, default=1.0, help="Weight for L1 mel loss (recommended 1.0)")
    parser.add_argument("--loss_l2", type=float, default=0.3, help="Weight for L2 mel loss (0 to disable)")
    parser.add_argument("--multiscale_mel", action="store_true", help="Add multi-scale mel loss (2x and 4x time-downsampled L1)")
    parser.add_argument("--multiscale_weight", type=float, default=0.5, help="Weight for multi-scale mel loss terms")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(feature_dir)
    if not data:
        raise FileNotFoundError(f"No .npz features in {feature_dir}. Run feature_extract first.")
    # Infer content_dim from data if not set (ContentVec 256 -> 256, ContentVec 500 -> 768)
    if args.content_dim is None:
        args.content_dim = int(data[0][0].shape[1])
        print(f"Inferred content_dim={args.content_dim} from features")
    print(f"Loaded {len(data)} segments")
    print(f"Loss: L1={args.loss_l1}, L2={args.loss_l2}, multiscale_mel={args.multiscale_mel} (weight={args.multiscale_weight})")

    device = torch.device(args.device)
    net = MinimalGenerator(
        content_dim=args.content_dim,
        f0_dim=1,
        mel_dim=args.mel_dim,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def get_batch():
        indices = np.random.choice(len(data), size=args.batch_size, replace=len(data) < args.batch_size)
        contents, f0s, mels = [], [], []
        max_t = 0
        for i in indices:
            c, f, m = data[i]
            contents.append(c)
            f0s.append(f)
            mels.append(m)
            max_t = max(max_t, c.shape[0])
        B, C = args.batch_size, args.content_dim
        content_t = torch.zeros(B, max_t, C, device=device)
        f0_t = torch.zeros(B, max_t, 1, device=device)
        mel_dim = args.mel_dim
        mel_target = torch.zeros(B, mel_dim, max_t, device=device)
        for i in range(args.batch_size):
            c, f, m = contents[i], f0s[i], mels[i]
            t = c.shape[0]
            content_t[i, :t] = torch.from_numpy(c)
            f0_t[i, :t, 0] = torch.from_numpy(f)
            if m.ndim == 1:
                m = m.reshape(1, -1)
            dm, tm = m.shape[0], m.shape[1]
            mel_target[i, : min(dm, mel_dim), : min(tm, t)] = torch.from_numpy(
                m[:mel_dim, : min(tm, t)].astype(np.float32)
            )
        if args.normalize_mel:
            mel_target = (mel_target - args.mel_mean) / (args.mel_std + 1e-8)
        return content_t, f0_t, mel_target

    def mel_loss(mel_pred: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
        """L1 + optional L2 + optional multi-scale mel loss."""
        loss = torch.tensor(0.0, device=mel_pred.device, dtype=mel_pred.dtype)
        if args.loss_l1 > 0:
            loss = loss + args.loss_l1 * nn.functional.l1_loss(mel_pred, mel_target)
        if args.loss_l2 > 0:
            loss = loss + args.loss_l2 * nn.functional.mse_loss(mel_pred, mel_target)
        if args.multiscale_mel and args.multiscale_weight > 0:
            # Multi-scale: L1 at 2x and 4x time-downsampled (like MR-STFT but on mel)
            for k, stride in [(2, 2), (4, 4)]:
                if mel_pred.shape[-1] < k:
                    continue
                p = nn.functional.avg_pool1d(mel_pred, kernel_size=k, stride=stride)
                t = nn.functional.avg_pool1d(mel_target, kernel_size=k, stride=stride)
                loss = loss + (args.multiscale_weight / stride) * nn.functional.l1_loss(p, t)
        return loss

    step = 0
    pbar = tqdm(total=args.steps, desc="Train")
    while step < args.steps:
        net.zero_grad()
        loss_acc = 0.0
        for _ in range(args.accum):
            content_t, f0_t, mel_target = get_batch()
            mel_pred = net(content_t, f0_t)
            loss = mel_loss(mel_pred, mel_target)
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=loss_acc / args.accum)
        if step % args.save_every == 0:
            ckpt_path = model_dir / f"{args.name}_rvc.pth"
            torch.save(net.state_dict(), ckpt_path)
            config = {
                "content_dim": args.content_dim,
                "mel_dim": args.mel_dim,
                "sample_rate": 22050,
                "preprocess": "contentvec_16k, rmvpe_f0",
                "mel_normalize": args.normalize_mel,
                "mel_mean": args.mel_mean,
                "mel_std": args.mel_std,
            }
            with open(model_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved {ckpt_path} + config.json")
    pbar.close()
    torch.save(net.state_dict(), model_dir / f"{args.name}_rvc.pth")
    config = {
        "content_dim": args.content_dim,
        "mel_dim": args.mel_dim,
        "sample_rate": 22050,
        "preprocess": "contentvec_16k, rmvpe_f0",
        "mel_normalize": args.normalize_mel,
        "mel_mean": args.mel_mean,
        "mel_std": args.mel_std,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done: {model_dir / f'{args.name}_rvc.pth'}, config.json")


if __name__ == "__main__":
    main()
