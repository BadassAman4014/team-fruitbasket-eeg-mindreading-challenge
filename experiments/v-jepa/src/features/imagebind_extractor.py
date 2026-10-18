"""ImageBind feature extractor — Phase 1B / Phase 1C.

ImageBind aligns video and audio into the same 1024-d embedding space,
making it natural for multimodal fusion without needing a separate model
per modality.

Installation (cluster)
----------------------
    pip install imagebind
    # or from source:
    pip install git+https://github.com/facebookresearch/ImageBind

    # Audio extraction also requires ffmpeg:
    sudo apt-get install ffmpeg   # or load the cluster module

Weights
-------
    Downloaded automatically on first use to ~/.cache/torch/hub/
    (ImageBind self-fetches from the Meta CDN).

Output
------
    1024-d L2-normalised embedding per video (visual pathway).
    1024-d L2-normalised embedding per audio clip.
    Both live in the same metric space — can be concatenated or averaged.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


# ── Audio extraction helper ──────────────────────────────────────────────────

def _extract_audio_wav(video_path: str, out_dir: str) -> str:
    """Extract mono 16 kHz WAV from a video file using ffmpeg.

    Returns
    -------
    str — path to the extracted .wav file
    """
    stem   = Path(video_path).stem
    out    = os.path.join(out_dir, f"{stem}.wav")
    if not os.path.exists(out):
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", "16000", "-vn", out],
            check=True, capture_output=True,
        )
    return out


# ── ImageBind model wrapper ──────────────────────────────────────────────────

class ImageBindExtractor:
    """Wraps Meta's ImageBind model for video + audio embedding.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.
    audio_cache_dir : str, optional
        Directory to cache extracted WAV files.
        Defaults to a temp directory (deleted on exit if not set).
    """

    def __init__(self, device: str = "cuda", audio_cache_dir: str | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._tmp_dir = None
        if audio_cache_dir:
            os.makedirs(audio_cache_dir, exist_ok=True)
            self.audio_dir = audio_cache_dir
        else:
            self._tmp_dir  = tempfile.mkdtemp()
            self.audio_dir = self._tmp_dir

        self.model, self.load_and_transform = self._load_model()

    def _load_model(self):
        # ------------------------------------------------------------------ #
        # ImageBind loading via the official facebookresearch package.        #
        # Compatibility bridge for environments with older timm (e.g., net2brain)
        # where timm.layers was located at timm.models.layers:
        try:
            import timm.layers  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            import sys
            import timm
            import timm.models.layers as _timm_layers
            timm.layers = _timm_layers
            sys.modules["timm.layers"] = _timm_layers

        from imagebind import data as ib_data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType

        model = imagebind_model.imagebind_huge(pretrained=True)
        model = model.eval().to(self.device)
        return model, ib_data

    @torch.no_grad()
    def extract_video(self, video_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract ImageBind visual embeddings.

        Returns
        -------
        (N, 1024) float32 ndarray — L2-normalised video embeddings
        """
        from imagebind.models.imagebind_model import ModalityType

        all_embs = []
        for start in range(0, len(video_paths), batch_size):
            batch = video_paths[start : start + batch_size]
            inputs = {
                ModalityType.VISION: self.load_and_transform.load_and_transform_video_data(
                    batch, self.device
                )
            }
            emb = self.model(inputs)[ModalityType.VISION]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
            print(f"  ImageBind video  {start + len(batch):4d}/{len(video_paths)}")

        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def extract_audio(self, video_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract ImageBind audio embeddings (requires ffmpeg).

        Returns
        -------
        (N, 1024) float32 ndarray — L2-normalised audio embeddings
        """
        from imagebind.models.imagebind_model import ModalityType

        wav_paths = [_extract_audio_wav(vp, self.audio_dir) for vp in video_paths]

        all_embs = []
        for start in range(0, len(wav_paths), batch_size):
            batch = wav_paths[start : start + batch_size]
            inputs = {
                ModalityType.AUDIO: self.load_and_transform.load_and_transform_audio_data(
                    batch, self.device
                )
            }
            emb = self.model(inputs)[ModalityType.AUDIO]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
            print(f"  ImageBind audio  {start + len(batch):4d}/{len(wav_paths)}")

        return np.concatenate(all_embs, axis=0)

    def extract_video_audio(
        self,
        video_paths: List[str],
        batch_size: int = 8,
        fusion: str = "concat",
    ) -> np.ndarray:
        """Extract and fuse video + audio embeddings.

        Parameters
        ----------
        fusion : 'concat' (2048-d) | 'mean' (1024-d)

        Returns
        -------
        (N, D) float32 ndarray
        """
        V = self.extract_video(video_paths, batch_size)
        A = self.extract_audio(video_paths, batch_size)
        if fusion == "concat":
            return np.concatenate([V, A], axis=1)      # (N, 2048)
        elif fusion == "mean":
            combined = F.normalize(
                torch.tensor(V + A), dim=-1
            ).numpy()
            return combined                             # (N, 1024)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion!r}")

    def __del__(self):
        if self._tmp_dir and os.path.isdir(self._tmp_dir):
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
