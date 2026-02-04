# Assets (pretrained weights)

Place pretrained weights here. **Do not commit** large `.pth` / `.onnx` files.

## Directories

- **`contentvec/`** — ContentVec or HuBERT-Soft  
  - Put one of: `contentvec_256.pt`, `contentvec.pt`, `hubert_soft.pt`  
  - Download from the official ContentVec/HuBERT-Soft releases (e.g. fairseq or RVC ecosystem).

- **`rmvpe/`** — RMVPE f0 extractor  
  - Put: `rmvpe.pt`  
  - Available from RVC repos (e.g. RVC-Project).

- **`hifigan/`** — HiFi-GAN v2 vocoder  
  - Put: `hifigan.pth` or `generator.pth`  
  - Same as used in RVC (e.g. Universal or 40k checkpoint).

If any of these are missing, `convert_accent.py` and `training/feature_extract.py` will raise clear errors telling you which path to fill.
