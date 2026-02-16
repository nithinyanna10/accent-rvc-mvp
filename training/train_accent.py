"""
Train Indian → American (BDL) accent conversion using parallel L2-ARCTIC / CMU ARCTIC data.
Input: Indian content + Indian f0 (aligned to BDL length). Target: BDL mel.
So the model learns: (Indian content, f0) → American mel.
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
from rvc.vc_model import MinimalGenerator, TemporalGenerator, TemporalGeneratorV2


def load_paired_features(
    paired_manifest_path: Path,
    l2_features_dir: Path,
    bdl_features_dir: Path,
    mel_dim: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load (content_Indian_aligned, f0_Indian_aligned, mel_BDL) for each pair.
    Indian content/f0 are interpolated to BDL mel length so we learn Indian → BDL mel.
    """
    with open(paired_manifest_path) as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]
    data = []
    for entry in pairs:
        l2_name = Path(entry["l2_name"]).stem  # ASI_arctic_a0001
        bdl_name = Path(entry["bdl_name"]).stem  # bdl_arctic_a0001
        l2_npz = l2_features_dir / f"{l2_name}.npz"
        bdl_npz = bdl_features_dir / f"{bdl_name}.npz"
        if not l2_npz.exists() or not bdl_npz.exists():
            continue
        d_l2 = np.load(l2_npz)
        d_bdl = np.load(bdl_npz)
        content = d_l2["content"]  # [T_l2, C]
        f0 = d_l2["f0"]
        mel = d_bdl["mel"]  # [n_mel, T_bdl]
        T_bdl = mel.shape[1]
        T_l2 = content.shape[0]
        if T_bdl < 10 or T_l2 < 10:
            continue
        # Align f0 to content length if needed
        if len(f0) != T_l2:
            f0 = np.interp(
                np.linspace(0, len(f0) - 1, T_l2),
                np.arange(len(f0)),
                f0.astype(np.float64),
            ).astype(np.float32)
        # Warp Indian content and f0 to BDL length (same sentence, different speaking rate)
        x_old = np.linspace(0, T_l2 - 1, T_l2)
        x_new = np.linspace(0, T_l2 - 1, T_bdl)
        content_aligned = np.zeros((T_bdl, content.shape[1]), dtype=np.float32)
        for c in range(content.shape[1]):
            content_aligned[:, c] = np.interp(x_new, x_old, content[:, c].astype(np.float64)).astype(np.float32)
        f0_aligned = np.interp(x_new, x_old, f0.astype(np.float64)).astype(np.float32)
        mel = mel.astype(np.float32)
        if mel.shape[0] > mel_dim:
            mel = mel[:mel_dim]
        data.append((content_aligned, f0_aligned, mel))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Indian → American (BDL) accent conversion with parallel L2/BDL data."
    )
    parser.add_argument("--paired_manifest", type=str, required=True, help="JSON from build_paired_manifest.py")
    parser.add_argument("--l2_features", type=str, required=True, help="L2-ARCTIC feature dir (Indian .npz)")
    parser.add_argument("--bdl_features", type=str, required=True, help="BDL feature dir (American .npz)")
    parser.add_argument("--model_dir", type=str, default="models", help="Output model directory")
    parser.add_argument("--name", type=str, default="bdl_accent", help="Model name (output: {name}_rvc.pth)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50000, help="Steps (more than BDL-only; cross-speaker is harder)")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--content_dim", type=int, default=None)
    parser.add_argument("--mel_dim", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize_mel", action="store_true")
    parser.add_argument("--mel_mean", type=float, default=-20.0)
    parser.add_argument("--mel_std", type=float, default=10.0)
    parser.add_argument("--loss_l1", type=float, default=1.0)
    parser.add_argument("--loss_l2", type=float, default=0.3)
    parser.add_argument("--multiscale_mel", action="store_true")
    parser.add_argument("--multiscale_weight", type=float, default=0.5)
    parser.add_argument("--temporal", action="store_true", help="Use TemporalGenerator (1D conv)")
    parser.add_argument("--temporal_v2", action="store_true", help="Use TemporalGeneratorV2 (512 hidden, 4 residual blocks, best quality)")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size for temporal models (default 512 for v2)")
    args = parser.parse_args()

    paired_path = Path(args.paired_manifest)
    l2_feat = Path(args.l2_features)
    bdl_feat = Path(args.bdl_features)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = load_paired_features(
        paired_path,
        l2_feat,
        bdl_feat,
        mel_dim=args.mel_dim,
    )
    if not data:
        raise FileNotFoundError(
            f"No pairs loaded. Check --paired_manifest, --l2_features, --bdl_features and that .npz exist for each pair."
        )
    if args.content_dim is None:
        args.content_dim = int(data[0][0].shape[1])
        print(f"Inferred content_dim={args.content_dim}")
    print(f"Loaded {len(data)} parallel (Indian → BDL) pairs")
    print(f"Loss: L1={args.loss_l1}, L2={args.loss_l2}, multiscale_mel={args.multiscale_mel}, temporal={args.temporal}, temporal_v2={args.temporal_v2}")

    device = torch.device(args.device)
    if args.temporal_v2:
        net = TemporalGeneratorV2(
            content_dim=args.content_dim,
            f0_dim=1,
            mel_dim=args.mel_dim,
            hidden=args.hidden,
        ).to(device)
    elif args.temporal:
        net = TemporalGenerator(
            content_dim=args.content_dim,
            f0_dim=1,
            mel_dim=args.mel_dim,
            hidden=args.hidden,
        ).to(device)
    else:
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
        mel_target = torch.zeros(B, args.mel_dim, max_t, device=device)
        for i in range(args.batch_size):
            c, f, m = contents[i], f0s[i], mels[i]
            t = c.shape[0]
            content_t[i, :t] = torch.from_numpy(c)
            f0_t[i, :t, 0] = torch.from_numpy(f)
            dm, tm = m.shape[0], m.shape[1]
            mel_target[i, : min(dm, args.mel_dim), : min(tm, t)] = torch.from_numpy(
                m[: args.mel_dim, : min(tm, t)].astype(np.float32)
            )
        if args.normalize_mel:
            mel_target = (mel_target - args.mel_mean) / (args.mel_std + 1e-8)
        return content_t, f0_t, mel_target

    def mel_loss(mel_pred: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=mel_pred.device, dtype=mel_pred.dtype)
        if args.loss_l1 > 0:
            loss = loss + args.loss_l1 * nn.functional.l1_loss(mel_pred, mel_target)
        if args.loss_l2 > 0:
            loss = loss + args.loss_l2 * nn.functional.mse_loss(mel_pred, mel_target)
        if args.multiscale_mel and args.multiscale_weight > 0:
            for k, stride in [(2, 2), (4, 4)]:
                if mel_pred.shape[-1] < k:
                    continue
                p = nn.functional.avg_pool1d(mel_pred, kernel_size=k, stride=stride)
                t = nn.functional.avg_pool1d(mel_target, kernel_size=k, stride=stride)
                loss = loss + (args.multiscale_weight / stride) * nn.functional.l1_loss(p, t)
        return loss

    step = 0
    pbar = tqdm(total=args.steps, desc="Train accent")
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
                "accent_conversion": True,
                "trained_on": "Indian (L2) content → BDL mel",
                "generator": "temporal_v2" if args.temporal_v2 else ("temporal" if args.temporal else "minimal"),
                "hidden": args.hidden if (args.temporal_v2 or args.temporal) else None,
            }
            if config.get("hidden") is None:
                config.pop("hidden", None)
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
        "accent_conversion": True,
        "trained_on": "Indian (L2) content → BDL mel",
        "generator": "temporal_v2" if args.temporal_v2 else ("temporal" if args.temporal else "minimal"),
        "hidden": args.hidden if (args.temporal_v2 or args.temporal) else None,
    }
    if config.get("hidden") is None:
        config.pop("hidden", None)
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done: {model_dir / f'{args.name}_rvc.pth'}, config.json (Indian → American accent model)")


if __name__ == "__main__":
    main()
