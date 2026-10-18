"""CLAP (Contrastive Language-Audio Pretraining) feature extractor.

Extracts semantic 512-d audio embeddings from video audio tracks using
LAION's CLAP model ('laion/larger_clap_general' via HuggingFace transformers).

Unlike AST (which was trained purely for categorical sound classification),
CLAP is trained contrastively on audio-text pairs, giving it natural semantic
alignment with ViCLIP (both share a CLIP-like multimodal latent space).
"""

import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel


def _extract_audio_wav_48k(video_path: str, out_dir: str) -> str:
    """Extract mono 48 kHz WAV from video using ffmpeg (matching CLAP native sample rate)."""
    stem = Path(video_path).stem
    out = os.path.join(out_dir, f"{stem}_48k.wav")
    if os.path.exists(out):
        return out

    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", "48000", "-vn", out
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Fallback: create 3s of silence (144,000 samples @ 48 kHz)
        silence = np.zeros(144000, dtype=np.int16)
        wavfile.write(out, 48000, silence)

    return out


class CLAPExtractor:
    """Extracts 512-d audio embeddings using LAION's CLAP."""

    def __init__(
        self,
        model_name: str = "laion/larger_clap_general",
        device: str = "cuda",
        audio_cache_dir: str = "./cache/wav_clap",
    ):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.audio_cache_dir = audio_cache_dir
        os.makedirs(self.audio_cache_dir, exist_ok=True)

        print(f"Loading CLAP from '{model_name}' on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()

    def extract_audio(self, video_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract L2-normalized 512-d audio embeddings for each video."""
        all_embeddings = []

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            audio_signals = []

            for vp in batch_paths:
                wav_path = _extract_audio_wav_48k(vp, self.audio_cache_dir)
                try:
                    sr, data = wavfile.read(wav_path)
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype != np.float32:
                        data = data.astype(np.float32)

                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    # Target ~3 seconds @ 48 kHz = 144,000 samples
                    target_len = 144000
                    if len(data) < target_len:
                        data = np.pad(data, (0, target_len - len(data)))
                    else:
                        data = data[:target_len]
                except Exception:
                    data = np.zeros(144000, dtype=np.float32)

                audio_signals.append(data)

            inputs = self.processor(
                audios=audio_signals,
                sampling_rate=48000,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                audio_embed = self.model.get_audio_features(**inputs)  # (B, 512)
                audio_embed = F.normalize(audio_embed, p=2, dim=-1)

            all_embeddings.append(audio_embed.cpu().numpy())

            if (i + batch_size) % 128 == 0 or (i + len(batch_paths)) == len(video_paths):
                print(f"  CLAP extracted {min(i + batch_size, len(video_paths)):4d}/{len(video_paths)} audio tracks")

        return np.concatenate(all_embeddings, axis=0)
