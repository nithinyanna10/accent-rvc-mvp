"""
CPU-friendly RVC training: small batches, gradient accumulation, checkpoint every N steps.
Outputs models/bdl_rvc.pth + models/config.json.

New features:
  --temporal       : Use TemporalGenerator (Conv1d, reduces jitter)
  --temporal_v2    : Use TemporalGeneratorV2 (deeper, better quality)
  --cosine_lr      : Cosine-annealing LR schedule (smoother convergence)
  --warmup_steps N : Linear warmup before cosine decay
  --ema_decay D    : Exponential moving average of weights for smoother output
  --eval_every N   : Compute validation loss every N steps on held-out split
  --val_split F    : Fraction of data held out for validation (default 0.1)
  --checkpoint_dir : Directory for intermediate checkpoints (default: model_dir)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys_path = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path))
from rvc.vc_model import MinimalGenerator, TemporalGenerator, TemporalGeneratorV2


# ────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average of model weights
# ────────────────────────────────────────────────────────────────────────────

class EMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.detach()
                )

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into model (use for inference/saving)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            self.shadow[k] = v.clone()


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
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory for intermediate checkpoints (default: model_dir)")
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
    # Generator architecture
    parser.add_argument("--temporal", action="store_true", help="Use TemporalGenerator (Conv1d; reduces jitter; slightly slower)")
    parser.add_argument("--temporal_v2", action="store_true", help="Use TemporalGeneratorV2 (deeper; best quality; slowest)")
    # LR scheduling
    parser.add_argument("--cosine_lr", action="store_true", help="Use cosine annealing LR schedule (smoother convergence)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps before cosine decay (default: 500)")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for cosine schedule (default: 1e-6)")
    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay for weights (0=off; 0.999 recommended for stability)")
    # Validation
    parser.add_argument("--eval_every", type=int, default=0, help="Evaluate validation loss every N steps (0=off)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction held out for validation (default: 0.1)")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else model_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(feature_dir)
    if not data:
        raise FileNotFoundError(f"No .npz features in {feature_dir}. Run feature_extract first.")
    # Infer content_dim from data if not set (ContentVec 256 -> 256, ContentVec 500 -> 768)
    if args.content_dim is None:
        args.content_dim = int(data[0][0].shape[1])
        print(f"Inferred content_dim={args.content_dim} from features")

    # Validation split
    val_data, train_data = [], data
    if args.eval_every > 0 and args.val_split > 0:
        n_val = max(1, int(len(data) * args.val_split))
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        val_data = [data[i] for i in indices[:n_val]]
        train_data = [data[i] for i in indices[n_val:]]
        print(f"Train: {len(train_data)} segments, Val: {len(val_data)} segments")
    else:
        print(f"Loaded {len(data)} segments")

    # Determine generator type
    gen_type = "minimal"
    if args.temporal_v2:
        gen_type = "temporal_v2"
    elif args.temporal:
        gen_type = "temporal"

    print(f"Generator: {gen_type}")
    print(f"Loss: L1={args.loss_l1}, L2={args.loss_l2}, multiscale_mel={args.multiscale_mel} (weight={args.multiscale_weight})")
    if args.cosine_lr:
        print(f"LR schedule: cosine (warmup={args.warmup_steps}, lr_min={args.lr_min})")
    if args.ema_decay > 0:
        print(f"EMA decay: {args.ema_decay}")

    device = torch.device(args.device)
    if gen_type == "temporal_v2":
        net = TemporalGeneratorV2(
            content_dim=args.content_dim, f0_dim=1, mel_dim=args.mel_dim
        ).to(device)
    elif gen_type == "temporal":
        net = TemporalGenerator(
            content_dim=args.content_dim, f0_dim=1, mel_dim=args.mel_dim
        ).to(device)
    else:
        net = MinimalGenerator(
            content_dim=args.content_dim, f0_dim=1, mel_dim=args.mel_dim
        ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    # EMA
    ema: Optional[EMA] = EMA(net, decay=args.ema_decay) if args.ema_decay > 0 else None

    # LR scheduler
    def get_lr(step: int) -> float:
        if not args.cosine_lr:
            return args.lr
        if step < args.warmup_steps:
            return args.lr * max(1e-6, step / max(args.warmup_steps, 1))
        # Cosine decay from warmup end to lr_min
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr_min + (args.lr - args.lr_min) * cosine

    def get_batch(source=None):
        src = source if source is not None else train_data
        indices = np.random.choice(len(src), size=args.batch_size, replace=len(src) < args.batch_size)
        contents, f0s, mels = [], [], []
        max_t = 0
        for i in indices:
            c, f, m = src[i]
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

    def save_checkpoint(step: int, tag: str = "") -> None:
        """Save model (and EMA snapshot if enabled)."""
        suffix = f"_step{step}" if tag == "intermediate" else ""
        ckpt_path = checkpoint_dir / f"{args.name}_rvc{suffix}.pth"

        # Use EMA weights for the main checkpoint if EMA is enabled
        if ema is not None:
            ema_net = copy.deepcopy(net)
            ema.apply_to(ema_net)
            torch.save(ema_net.state_dict(), ckpt_path)
            # Also save regular weights alongside
            torch.save(net.state_dict(), ckpt_path.with_suffix("") .parent / f"{args.name}_rvc_raw{suffix}.pth")
        else:
            torch.save(net.state_dict(), ckpt_path)

        config = {
            "content_dim": args.content_dim,
            "mel_dim": args.mel_dim,
            "sample_rate": 22050,
            "preprocess": "contentvec_16k, rmvpe_f0",
            "generator": gen_type,
            "mel_normalize": args.normalize_mel,
            "mel_mean": args.mel_mean,
            "mel_std": args.mel_std,
            "hidden": 256 if gen_type != "temporal_v2" else 512,
            "step": step,
        }
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Keep latest at canonical path (model_dir/{name}_rvc.pth)
        canonical = model_dir / f"{args.name}_rvc.pth"
        if canonical != ckpt_path:
            import shutil
            if ema is not None:
                shutil.copy(ckpt_path, canonical)
            else:
                torch.save(net.state_dict(), canonical)
        print(f"  [step {step}] Saved {ckpt_path.name} + config.json")

    def compute_val_loss() -> Optional[float]:
        if not val_data:
            return None
        net.eval()
        total = 0.0
        n_batches = max(1, min(8, len(val_data) // max(args.batch_size, 1)))
        with torch.no_grad():
            for _ in range(n_batches):
                content_t, f0_t, mel_target = get_batch(val_data)
                mel_pred = net(content_t, f0_t)
                total += mel_loss(mel_pred, mel_target).item()
        net.train()
        return total / n_batches

    # ── Training loop ─────────────────────────────────────────────────────────
    step = 0
    best_val_loss = float("inf")
    pbar = tqdm(total=args.steps, desc="Train")
    while step < args.steps:
        # Update LR
        if args.cosine_lr:
            new_lr = get_lr(step)
            for pg in opt.param_groups:
                pg["lr"] = new_lr

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

        # EMA update
        if ema is not None:
            ema.update(net)

        step += 1
        pbar.update(1)
        postfix = {"loss": f"{loss_acc / args.accum:.4f}"}
        if args.cosine_lr:
            postfix["lr"] = f"{get_lr(step):.2e}"

        # Validation
        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = compute_val_loss()
            if val_loss is not None:
                postfix["val_loss"] = f"{val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best checkpoint
                    best_path = checkpoint_dir / f"{args.name}_rvc_best.pth"
                    if ema is not None:
                        ema_net = copy.deepcopy(net)
                        ema.apply_to(ema_net)
                        torch.save(ema_net.state_dict(), best_path)
                    else:
                        torch.save(net.state_dict(), best_path)
                    postfix["best"] = "✓"

        pbar.set_postfix(postfix)

        if step % args.save_every == 0:
            save_checkpoint(step, tag="intermediate")

    pbar.close()
    save_checkpoint(step, tag="final")
    print(f"Done: {model_dir / f'{args.name}_rvc.pth'}, config.json")


if __name__ == "__main__":
    main()
