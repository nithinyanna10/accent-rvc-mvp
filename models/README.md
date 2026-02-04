# Models (trained outputs)

After training you should have:

- **`bdl_rvc.pth`** — Trained target-voice model (BDL).  
  - Either full `nn.Module` state or state_dict from `MinimalGenerator` (see `rvc/vc_model.py`).

- **`config.json`** — Architecture and preprocessing settings.  
  - Example: `content_dim`, `mel_dim`, `sample_rate`, `preprocess`.

Optional:

- **`bdl.index`** (and `bdl_index_meta.json`)** — Retrieval index for `--index_rate` at inference.  
  - Built by `python -m training.build_index ...`.

`convert_accent.py` expects `--model_dir` to point here and will error clearly if `bdl_rvc.pth` or `config.json` is missing.
