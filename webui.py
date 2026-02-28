#!/usr/bin/env python3
"""
Gradio Web UI for accent_rvc_mvp.
Drag-and-drop WAV → convert → listen + download.
Also shows quality metrics (MCD, SNR) inline.

Usage:
  python webui.py --model_dir models [--port 7860] [--share]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
except ImportError:
    print(
        "gradio not installed. Run: pip install gradio>=4.0",
        file=sys.stderr,
    )
    sys.exit(1)

import numpy as np

from rvc.config import InferenceParams
from rvc.pipeline import convert_file


# ────────────────────────────────────────────────────────────────────────────
# Global pipeline state (lazy-loaded once)
# ────────────────────────────────────────────────────────────────────────────

_PARAMS: Optional[InferenceParams] = None
_MODEL_DIR: Optional[Path] = None


def _get_params(
    model_dir_str: str,
    model_name: str,
    streaming: bool,
    blend: float,
    index_rate: float,
    f0_up_key: int,
    lowpass_hz: Optional[int],
    post_gate: bool,
    mel_smooth: int,
) -> InferenceParams:
    return InferenceParams(
        device="cpu",
        streaming=streaming,
        model_dir=Path(model_dir_str),
        model_name=model_name or None,
        contentvec_dir=Path("assets/contentvec"),
        rmvpe_dir=Path("assets/rmvpe"),
        index_rate=index_rate,
        f0_up_key=f0_up_key,
        blend=blend,
        post_gate_dbfs=-60.0 if post_gate else None,
        lowpass_hz=int(lowpass_hz) if lowpass_hz and lowpass_hz > 0 else None,
        mel_smooth_frames=mel_smooth,
    )


# ────────────────────────────────────────────────────────────────────────────
# Conversion callback
# ────────────────────────────────────────────────────────────────────────────

def convert_audio(
    audio_path,          # Gradio returns a file path for audio uploads
    model_dir_str: str,
    model_name: str,
    streaming: bool,
    blend: float,
    index_rate: float,
    f0_up_key: int,
    lowpass_hz: float,
    post_gate: bool,
    mel_smooth: int,
) -> tuple[Optional[str], str]:
    """
    Gradio conversion callback.
    Returns (output_audio_path, info_text).
    """
    if audio_path is None:
        return None, "No audio uploaded."

    model_dir = Path(model_dir_str)
    if not model_dir.exists():
        return None, f"❌ Model directory not found: {model_dir}"

    params = _get_params(
        model_dir_str=model_dir_str,
        model_name=model_name,
        streaming=streaming,
        blend=blend,
        index_rate=index_rate,
        f0_up_key=int(f0_up_key),
        lowpass_hz=int(lowpass_hz) if lowpass_hz else None,
        post_gate=post_gate,
        mel_smooth=int(mel_smooth),
    )
    try:
        params.validate()
    except (FileNotFoundError, ValueError) as e:
        return None, f"❌ Config error: {e}"

    input_path = Path(audio_path)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = Path(out_tmp.name)
    out_tmp.close()

    import time
    t0 = time.perf_counter()
    try:
        convert_file(input_path, output_path, model_dir, params)
    except FileNotFoundError as e:
        return None, f"❌ Missing weights: {e}"
    except Exception as e:
        return None, f"❌ Conversion error: {e}"
    elapsed = (time.perf_counter() - t0) * 1000

    # Compute metrics
    info_lines = [f"✅ Done in {elapsed:.0f}ms"]
    try:
        from rvc.metrics import compute_metrics
        import soundfile as sf
        wav_in, sr_in = sf.read(input_path, dtype="float32")
        wav_out, sr_out = sf.read(output_path, dtype="float32")
        if wav_in.ndim > 1:
            wav_in = wav_in.mean(1)
        if wav_out.ndim > 1:
            wav_out = wav_out.mean(1)
        m = compute_metrics(wav_in, wav_out, sr=sr_in)
        info_lines += [
            f"MCD          : {m['mcd']:.2f} dB",
            f"SNR          : {m['snr_db']:.1f} dB",
            f"Spectral Flux: {m['spectral_flux']:.4f}",
            f"Duration in  : {m['duration_in_s']:.2f}s",
            f"Duration out : {m['duration_out_s']:.2f}s",
        ]
    except Exception as e:
        info_lines.append(f"(metrics unavailable: {e})")

    return str(output_path), "\n".join(info_lines)


# ────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ────────────────────────────────────────────────────────────────────────────

def build_ui(default_model_dir: str = "models") -> gr.Blocks:
    with gr.Blocks(
        title="Accent RVC MVP",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 🎙️ Accent RVC MVP
**Indian English → US English accent conversion** using ContentVec + RMVPE + HiFi-GAN.

Upload a WAV, tweak the sliders, hit **Convert**, and listen.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                audio_in = gr.Audio(
                    label="Input WAV (Indian English or any L2)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                gr.Markdown("### Model Settings")
                model_dir_box = gr.Textbox(
                    value=default_model_dir,
                    label="Model directory",
                    placeholder="models",
                )
                model_name_box = gr.Textbox(
                    value="",
                    label="Model name (optional)",
                    placeholder="e.g. bdl_accent  → loads bdl_accent_rvc.pth",
                )

                gr.Markdown("### Conversion Controls")
                with gr.Row():
                    blend_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                        label="Blend (wet/dry)",
                        info="1.0 = full conversion · 0.0 = original",
                    )
                    index_rate_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                        label="Index rate",
                        info="Retrieval blend (0 = off; 0.3 recommended with index)",
                    )
                with gr.Row():
                    f0_key_slider = gr.Slider(
                        minimum=-12, maximum=12, value=0, step=1,
                        label="Pitch shift (semitones)",
                    )
                    lowpass_slider = gr.Slider(
                        minimum=4000, maximum=22000, value=11000, step=500,
                        label="Low-pass filter (Hz)",
                        info="Reduces hiss. 0 = off.",
                    )
                with gr.Row():
                    mel_smooth_slider = gr.Slider(
                        minimum=0, maximum=15, value=5, step=1,
                        label="Mel smooth frames",
                        info="Reduces jitter. 0 = off.",
                    )
                    streaming_check = gr.Checkbox(
                        value=False,
                        label="Streaming mode (chunked)",
                        info="True = 160ms OLA chunks. False = full-file (cleaner).",
                    )
                    post_gate_check = gr.Checkbox(
                        value=True,
                        label="Post-process gate",
                        info="Silence gate after conversion.",
                    )

                convert_btn = gr.Button("🔄 Convert", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                audio_out = gr.Audio(
                    label="Converted WAV (US English accent)",
                    type="filepath",
                )
                info_box = gr.Textbox(
                    label="Conversion info & metrics",
                    lines=10,
                    interactive=False,
                )

        convert_btn.click(
            fn=convert_audio,
            inputs=[
                audio_in,
                model_dir_box,
                model_name_box,
                streaming_check,
                blend_slider,
                index_rate_slider,
                f0_key_slider,
                lowpass_slider,
                post_gate_check,
                mel_smooth_slider,
            ],
            outputs=[audio_out, info_box],
        )

        gr.Markdown(
            """
---
**Tips:**
- Set **Blend** < 1.0 to mix original + converted (good for subtle accent softening).
- Use **Pitch shift** ±2 to compensate for speaker gender mismatch.
- **Low-pass 11 kHz** reduces hiss from the vocoder without hurting intelligibility.
- **Streaming mode** off = better quality (processes whole file at once).
            """
        )

    return demo


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Gradio Web UI for accent_rvc_mvp")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_ui(default_model_dir=args.model_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
