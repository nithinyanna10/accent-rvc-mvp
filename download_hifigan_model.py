#!/usr/bin/env python3
"""
Download a compatible HiFi-GAN vocoder model for RVC.
Tries to download from common sources or use a pre-trained model.
"""

import torch
from pathlib import Path

def download_hifigan():
    """Try to download/load a HiFi-GAN model compatible with RVC mel spectrograms."""
    assets_dir = Path("assets/hifigan")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    print("Attempting to load HiFi-GAN from PyTorch Hub...")
    try:
        # Try NVIDIA's HiFi-GAN
        hifigan = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_hifigan',
            pretrained=True,
            verbose=False
        )
        generator = hifigan.generator
        
        # Save the model
        model_path = assets_dir / "hifigan_pytorch_hub.pth"
        torch.save(generator.state_dict(), model_path)
        print(f"✓ Saved HiFi-GAN generator to {model_path}")
        
        # Also save full model for reference
        torch.save(generator, assets_dir / "hifigan_full_model.pth")
        print(f"✓ Saved full HiFi-GAN model to {assets_dir / 'hifigan_full_model.pth'}")
        
        return generator
    except Exception as e:
        print(f"✗ Failed to load from PyTorch Hub: {e}")
        return None

if __name__ == "__main__":
    model = download_hifigan()
    if model:
        print("\n✓ HiFi-GAN model downloaded successfully!")
        print("You can now use it in the vocoder by updating vocoder_hifigan.py")
    else:
        print("\n✗ Could not download HiFi-GAN model")
        print("You may need to manually download a compatible HiFi-GAN checkpoint")
