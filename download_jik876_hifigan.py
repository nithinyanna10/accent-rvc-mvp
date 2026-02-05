#!/usr/bin/env python3
"""
Download jik876 HiFi-GAN pretrained checkpoint (22.05 kHz) for use as vocoder.
Saves to assets/hifigan/generator_v2.pth + config.json.
Run: python download_jik876_hifigan.py
"""

import json
import sys
from pathlib import Path

def main():
    assets = Path("assets/hifigan")
    assets.mkdir(parents=True, exist_ok=True)
    out_ckpt = assets / "generator_v2.pth"
    out_config = assets / "config.json"
    
    # jik876 v2 config (22.05 kHz, 80 mels, hop 256)
    config_v2 = {
        "resblock": "1",
        "num_gpus": 0,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "upsample_rates": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_initial_channel": 128,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "segment_size": 8192,
        "num_mels": 80,
        "num_freq": 1025,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 1024,
        "sampling_rate": 22050,
        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,
        "num_workers": 4,
        "dist_config": {"dist_backend": "nccl", "dist_url": "tcp://localhost:54321", "world_size": 1},
    }
    
    try:
        from huggingface_hub import hf_hub_download
        import torch
        # jaketae/hifigan-lj-v1 on Hugging Face (LJ Speech, 22k) - has pytorch_model.bin
        path_bin = hf_hub_download(repo_id="jaketae/hifigan-lj-v1", filename="pytorch_model.bin", local_dir=str(assets))
        path_cfg = hf_hub_download(repo_id="jaketae/hifigan-lj-v1", filename="config.json", local_dir=str(assets))
        state = torch.load(path_bin, map_location="cpu", weights_only=False)
        # Save as generator_v2.pth with "generator" key for our loader
        if isinstance(state, dict) and "generator" not in state:
            state = {"generator": state}
        torch.save(state, out_ckpt)
        # Use repo config if compatible, else our v2 config
        try:
            with open(path_cfg) as f:
                repo_cfg = json.load(f)
            if "sampling_rate" in repo_cfg and repo_cfg.get("num_mels") == 80:
                config_v2 = repo_cfg
        except Exception:
            pass
        with open(out_config, "w") as f:
            json.dump(config_v2, f, indent=2)
        print(f"Downloaded HiFi-GAN to {out_ckpt} and {out_config}")
        return 0
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("\nManual option: download from https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y", file=sys.stderr)
        print("  - Get LJ_V2 or UNIVERSAL_V1 folder, copy the generator checkpoint to assets/hifigan/generator_v2.pth", file=sys.stderr)
        print("  - Copy config_v2.json to assets/hifigan/config.json", file=sys.stderr)
    
    # Write config anyway so user can drop generator_v2.pth manually
    with open(out_config, "w") as f:
        json.dump(config_v2, f, indent=2)
    print(f"Wrote {out_config}. Place generator_v2.pth (jik876 HiFi-GAN checkpoint) in assets/hifigan/ and run conversion again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
