"""
HiFi-GAN v2 vocoder wrapper (RVC ecosystem).
decode(mel_or_features) -> waveform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .utils import raise_missing_weights


class HiFiGANVocoder:
    """
    Wrapper for HiFi-GAN v2 used in RVC-like pipelines.
    Loads .pth from assets/hifigan/ or model_dir.
    decode(mel_or_features) -> waveform float32.
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: str = "cpu",
        sr: int = 40000,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.device = device
        self.sr = sr
        self._model = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        
        # Check multiple locations: weights_dir, assets/hifigan, model_dir
        search_dirs = []
        if self.weights_dir:
            search_dirs.append(Path(self.weights_dir))
        # Always check assets/hifigan as fallback
        search_dirs.append(Path("assets/hifigan"))
        
        p = None
        for d in search_dirs:
            for name in ("hifigan.pth", "generator.pth"):
                candidate = d / name
                if candidate.exists():
                    p = candidate
                    break
            if p is not None:
                break
        
        if p is None:
            raise FileNotFoundError(
                f"HiFi-GAN: no hifigan.pth or generator.pth found. "
                f"Searched in: {', '.join(str(d) for d in search_dirs)}. "
                f"Place HiFi-GAN v2 weights in assets/hifigan/ (e.g. hifigan.pth)."
            )
        
        return self._load_from_path(p, search_dirs)
    
    def _try_load_jik876_hifigan(self, search_dirs: list) -> Optional[torch.nn.Module]:
        """Try to load jik876 HiFi-GAN from generator_v2.pth (or g_*.pth with 'generator' key) in assets/hifigan."""
        for d in search_dirs:
            for name in ("generator_v2.pth", "generator_v1.pth"):
                c = d / name
                if c.exists():
                    try:
                        from .hifigan_jik876 import load_jik876_checkpoint
                        return load_jik876_checkpoint(c, self.device)
                    except Exception:
                        pass
            for f in sorted(d.glob("g_*.pth")):
                try:
                    ckpt = torch.load(f, map_location=self.device, weights_only=False)
                    if isinstance(ckpt, dict) and "generator" in ckpt:
                        from .hifigan_jik876 import load_jik876_checkpoint
                        return load_jik876_checkpoint(f, self.device)
                except Exception:
                    pass
        return None
    
    def _load_from_path(self, p: Path, search_dirs: list) -> torch.nn.Module:
        loaded_data = torch.load(p, map_location=self.device, weights_only=False)
        
        # Handle checkpoint format: {'model': state_dict, ...} or {'generator': ...} (jik876)
        if isinstance(loaded_data, dict):
            if 'model' in loaded_data:
                # RVC checkpoint format - try jik876 HiFi-GAN from separate file, else Griffin-Lim
                self._model = self._try_load_jik876_hifigan(search_dirs)
                if self._model is not None:
                    return self._model
                self._model = "state_dict_only"
                import warnings
                warnings.warn(
                    "HiFi-GAN checkpoint is RVC format (state_dict only). "
                    "No jik876 vocoder found. Using Griffin-Lim fallback. Quality will be reduced.",
                    UserWarning,
                    stacklevel=2,
                )
                return self._model
            if 'generator' in loaded_data and 'enc_p' not in str(loaded_data.get('model', '')):
                # jik876-style checkpoint
                try:
                    from .hifigan_jik876 import load_jik876_checkpoint
                    self._model = load_jik876_checkpoint(p, self.device)
                    return self._model
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load jik876 HiFi-GAN: {e}. Using Griffin-Lim.", UserWarning)
                    self._model = "state_dict_only"
                    return self._model
            elif hasattr(loaded_data, '__call__') or (not isinstance(loaded_data, dict)):
                # Direct model object (unlikely but possible)
                self._model = loaded_data
            else:
                # Unknown dict format - try PyTorch Hub HiFi-GAN
                try:
                    import torch.hub as hub_module
                    hifigan = hub_module.load(
                        'NVIDIA/DeepLearningExamples:torchhub',
                        'nvidia_hifigan',
                        pretrained=True,
                        verbose=False
                    )
                    self._model = hifigan.generator
                    self._model.eval()
                    return self._model
                except Exception:
                    self._model = "state_dict_only"
                    import warnings
                    warnings.warn(
                        "HiFi-GAN checkpoint format not recognized. "
                        "Using Griffin-Lim fallback vocoder.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return self._model
        else:
            # Direct model object
            if hasattr(loaded_data, '__call__') or hasattr(loaded_data, 'forward'):
                self._model = loaded_data
            else:
                # Unknown type, try PyTorch Hub
                try:
                    import torch.hub as hub_module
                    hifigan = hub_module.load(
                        'NVIDIA/DeepLearningExamples:torchhub',
                        'nvidia_hifigan',
                        pretrained=True,
                        verbose=False
                    )
                    self._model = hifigan.generator
                    self._model.eval()
                    return self._model
                except Exception:
                    self._model = "state_dict_only"
                    return self._model
        
        if isinstance(self._model, str):
            # state_dict_only marker - will use fallback
            return self._model
        
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._model

    def decode(self, mel_or_features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert mel or feature tensor to waveform.
        mel_or_features: [C, T] or [B, C, T]. Output: float32 mono.
        """
        if isinstance(mel_or_features, np.ndarray):
            x = torch.from_numpy(mel_or_features).float().to(self.device)
        else:
            x = mel_or_features.float().to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        try:
            model = self._load_model()
        except FileNotFoundError as e:
            # Fallback: return zeros so pipeline doesn't crash, but warn
            import warnings
            warnings.warn(
                "HiFi-GAN vocoder weights not found. Output will be SILENT. "
                "Download hifigan.pth (e.g. from RVC-Project or contentvec) and place in "
                f"assets/hifigan/ or {self.weights_dir}. Error: {e}",
                UserWarning,
                stacklevel=2,
            )
            t = x.shape[-1]
            hop = 256  # typical for 40k
            return np.zeros(t * hop, dtype=np.float32)

        # Handle state_dict_only case - use fallback vocoder
        if model == "state_dict_only":
            import warnings
            warnings.warn(
                "HiFi-GAN model architecture not available. Using Griffin-Lim as fallback vocoder. "
                "Quality will be reduced.",
                UserWarning,
                stacklevel=2,
            )
            return self._decode_fallback_griffin_lim(x)
        
        # jik876 HiFi-GAN: expects log-mel [B, 80, T] at 22k frame rate; we have dB mel at 40k
        if type(model).__name__ == "HiFiGANGeneratorJik876":
            return self._decode_jik876_hifigan(x, model)
        
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.squeeze().cpu().numpy()
        if out.ndim > 1:
            out = out.mean(axis=0)
        return out.astype(np.float32)
    
    def _decode_jik876_hifigan(self, mel_or_features: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
        """Decode using jik876 HiFi-GAN: convert our dB mel -> log mel, downsample time for 22k, vocode, resample to 40k."""
        import librosa
        # mel: [B, 80, T] in dB (our training). jik876 expects log(magnitude), 22.05 kHz (hop 256).
        # Convert dB -> log: log(magnitude) = dB * ln(10)/20
        mel = mel_or_features.float().to(self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel_np = mel.squeeze(0).cpu().numpy()  # [80, T]
        # dB to power then to log(magnitude): log(sqrt(power)) = 0.5*log(power)
        mel_power = librosa.db_to_power(mel_np, ref=1.0)
        mel_log = np.log(np.clip(mel_power, 1e-5, None)).astype(np.float32)  # natural log
        # Downsample time: 40k has 40000/256 frames/sec, 22k has 22050/256. T' = T * 22050/40000
        n_mels, T = mel_log.shape
        T_22k = max(1, int(T * 22050 / 40000))
        mel_22k = np.zeros((n_mels, T_22k), dtype=np.float32)
        for i in range(n_mels):
            mel_22k[i] = np.interp(
                np.linspace(0, T - 1, T_22k),
                np.arange(T),
                mel_log[i],
            )
        x = torch.from_numpy(mel_22k).unsqueeze(0).to(self.device)
        with torch.no_grad():
            wav_22k = model(x).squeeze().cpu().numpy()
        # Resample 22050 -> 40000 so duration matches our expected T*256/40000
        wav_40k = librosa.resample(
            wav_22k.astype(np.float64),
            orig_sr=22050,
            target_sr=self.sr,
            res_type="kaiser_best",
        )
        # Our chunk expects length T*256 (at 40k). wav_40k length = len(wav_22k)*40000/22050. Match by trimming/padding
        target_len = T * 256  # frames at 40k hop
        if len(wav_40k) > target_len:
            wav_40k = wav_40k[:target_len]
        elif len(wav_40k) < target_len:
            wav_40k = np.pad(wav_40k, (0, target_len - len(wav_40k)), mode="constant", constant_values=0)
        return wav_40k.astype(np.float32)
    
    def _decode_fallback_griffin_lim(self, mel_or_features: torch.Tensor) -> np.ndarray:
        """Fallback vocoder using librosa's mel_to_audio (simpler and more reliable)."""
        import librosa
        
        # mel: [B, C, T] or [C, T] -> [C, T] (n_mels, time_frames)
        if mel_or_features.dim() == 3:
            mel = mel_or_features.squeeze(0).cpu().numpy()
        else:
            mel = mel_or_features.cpu().numpy()
        
        # Ensure mel is [n_mels, time_frames]
        if mel.ndim == 1:
            mel = mel.reshape(-1, 1)
        if mel.shape[0] < mel.shape[1]:
            # If time is first dimension, transpose
            mel = mel.T
        
        n_mels, n_frames = mel.shape
        
        # Parameters MUST match training exactly
        sr = self.sr
        n_fft = 1024  # Match training MEL_N_FFT
        hop_length = 256  # Match training MEL_HOP
        
        # RVC mel spectrograms are in dB scale (from librosa.power_to_db during training)
        # Convert from dB to power: librosa.db_to_power(mel_db, ref=1.0)
        mel_power = librosa.db_to_power(mel, ref=1.0)
        
        # Use librosa's built-in mel_to_audio which handles the conversion properly
        # Note: mel_to_audio expects mel spectrogram in power scale (which we have)
        # But it may expect linear scale, not power - let's try both
        try:
            # Try with power scale first (reduced iterations for speed)
            audio = librosa.feature.inverse.mel_to_audio(
                M=mel_power,  # Power mel spectrogram
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_iter=20,  # Reduced for speed (was 60)
                fmin=0,
                fmax=8000,  # Match training fmax
            )
        except AttributeError:
            # librosa.feature.inverse might not exist in older versions
            # Fall back to manual conversion
            import warnings
            warnings.warn("librosa.feature.inverse.mel_to_audio not available. Using manual conversion.", UserWarning)
            
            # Manual conversion: mel -> magnitude -> griffin-lim
            mel_basis = librosa.filters.mel(
                sr=sr,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=0,
                fmax=8000,
                norm=None,
                htk=False,
            )
            
            # Pseudo-inverse to convert mel to magnitude
            mel_basis_pinv = np.linalg.pinv(mel_basis)
            magnitude = mel_basis_pinv @ mel_power
            magnitude = np.sqrt(np.maximum(magnitude, 0))
            
            # Griffin-Lim (reduced iterations for speed)
            audio = librosa.griffinlim(
                magnitude,
                n_iter=30,  # Reduced for speed
                hop_length=hop_length,
                n_fft=n_fft,
            )
        
        # Normalize audio
        audio_max = np.abs(audio).max()
        if audio_max > 0:
            audio = audio / audio_max * 0.8
        
        return audio.astype(np.float32)
