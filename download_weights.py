#!/usr/bin/env python3
"""
Download RVC assets needed for accent_rvc_mvp:
  - assets/rmvpe/rmvpe.pt
  - assets/hifigan/hifigan.pth (from RVC pretrained G40k.pth)
Uses same Hugging Face source as RVC-Project.
"""
from pathlib import Path
import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
BASE_DIR = Path(__file__).resolve().parent


def dl_model(link: str, model_name: str, dest_dir: Path, dest_name: str = None):
    dest_dir = Path(dest_dir)
    dest_name = dest_name or model_name
    dest_path = dest_dir / dest_name
    url = f"{link}{model_name}" if not model_name.startswith("http") else model_name
    print(f"  -> {dest_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    print("Downloading rmvpe.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets/rmvpe")

    print("Downloading HiFi-GAN 40k (G40k.pth) as assets/hifigan/hifigan.pth...")
    dl_model(
        RVC_DOWNLOAD_LINK + "pretrained/",
        "G40k.pth",
        BASE_DIR / "assets/hifigan",
        dest_name="hifigan.pth",
    )

    print("Done. You have:")
    print("  assets/rmvpe/rmvpe.pt")
    print("  assets/hifigan/hifigan.pth")
    print("Run convert_accent.py again to get audible output.")
