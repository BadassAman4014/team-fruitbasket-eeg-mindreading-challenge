"""AST (Audio Spectrogram Transformer) feature extractor — Phase 1D.

Extracts rich 768-d audio embeddings from video audio tracks using
MIT's Audio Spectrogram Transformer pretrained on AudioSet.

Model: 'MIT/ast-finetuned-audioset-10-10-0.4593' (downloaded via HuggingFace).
No extra repositories needed (pure HuggingFace transformers + scipy/ffmpeg).
"""

import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, ASTModel


def _extract_audio_wav(video_path: str, out_dir: str) -> str:
    """Extract mono 16 kHz WAV from a video file using ffmpeg.

    If video has no audio track, creates a silent 3s WAV file.
    """
    stem = Path(video_path).stem
    out = os.path.join(out_dir, f"{stem}.wav")
    if os.path.exists(out):
        return out

    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", "16000", "-vn", out
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Fallback: create 3s of silence (48,000 samples @ 16 kHz)
        silence = np.zeros(48000, dtype=np.int16)
        wavfile.write(out, 16000, silence)

    return out


class ASTExtractor:
    """Extracts 768-d audio embeddings using Audio Spectrogram Transformer."""

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: str = "cuda",
        audio_cache_dir: str = "./cache/wav",
    ):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.audio_cache_dir = audio_cache_dir
        os.makedirs(self.audio_cache_dir, exist_ok=True)

        print(f"Loading AST from '{model_name}' on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(self.device).eval()

    def extract_audio(self, video_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract L2-normalized 768-d audio embeddings for each video.

        Parameters
        ----------
        video_paths : list of str
        batch_size  : int

        Returns
        -------
        feats : (N, 768) float32 ndarray
        """
        all_embeddings = []

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            audio_signals = []

            for vp in batch_paths:
                wav_path = _extract_audio_wav(vp, self.audio_cache_dir)
                try:
                    sr, data = wavfile.read(wav_path)
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype != np.float32:
                        data = data.astype(np.float32)

                    # Ensure 1D mono
                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    # Pad or trim to ~3 seconds (48000 samples)
                    target_len = 48000
                    if len(data) < target_len:
                        data = np.pad(data, (0, target_len - len(data)))
                    elif len(data) > target_len:
                        data = data[:target_len]

                except Exception:
                    data = np.zeros(48000, dtype=np.float32)

                audio_signals.append(data)

            inputs = self.processor(
                audio_signals,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS token / pooled representation
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    emb = outputs.last_hidden_state[:, 0, :]

                emb = F.normalize(emb, p=2, dim=-1)
                all_embeddings.append(emb.cpu().numpy())

            if (i + batch_size) % 32 == 0 or (i + len(batch_paths)) == len(video_paths):
                print(f"  AST {min(i + batch_size, len(video_paths)):5d}/{len(video_paths)} audio tracks processed")

        return np.concatenate(all_embeddings, axis=0)
