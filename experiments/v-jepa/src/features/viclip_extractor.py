"""ViCLIP video feature extractor — Phase 1A.

ViCLIP (InternVideo2) is a spatiotemporal CLIP model with full temporal
attention, substantially stronger than ImageBind for video-only retrieval.

Installation (cluster)
----------------------
    pip install timm decord transformers

Weights
-------
    Downloaded and cached automatically from Hugging Face
    ('OpenGVLab/ViCLIP-L-14-hf') on the first run.

Output
------
    768-d L2-normalised embedding per 3-second video clip.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List

# ── Video loading ────────────────────────────────────────────────────────────
try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD = True
except ImportError:
    import cv2
    _DECORD = False

N_FRAMES = 8   # uniformly sampled frames per clip (matches ViCLIP default)


def _load_frames(video_path: str, n_frames: int = N_FRAMES) -> np.ndarray:
    """Sample n_frames uniformly from a video.

    Returns
    -------
    (n_frames, H, W, 3) uint8 RGB array
    """
    if _DECORD:
        vr   = VideoReader(video_path, ctx=decord_cpu(0))
        idxs = np.linspace(0, len(vr) - 1, n_frames).astype(int)
        return vr.get_batch(idxs).asnumpy()
    else:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs  = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames)


DEFAULT_MODEL_ID = "OpenGVLab/ViCLIP-L-14-hf"


class ViCLIPExtractor:
    """Wraps ViCLIP for bulk video embedding.

    Defaults to 'OpenGVLab/ViCLIP-L-14-hf' from Hugging Face, which downloads
    weights and model code automatically on the first run.

    Parameters
    ----------
    ckpt_path : str, optional
        HuggingFace model ID or local directory (default: 'OpenGVLab/ViCLIP-L-14-hf').
        Falls back to VICLIP_CKPT environment variable.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, ckpt_path: str | None = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = ckpt_path or os.environ.get("VICLIP_CKPT") or DEFAULT_MODEL_ID
        print(f"Loading ViCLIP from '{ckpt}' on {self.device} ...")
        self.model = self._load_model(ckpt)
        self.transform = self._build_transform()

    def _load_model(self, ckpt: str):
        """Load ViCLIP via Hugging Face AutoModel (auto-downloads on first run)."""
        # OpenGVLab's viclip.py hardcodes tokenizer_path='./bpe_simple_vocab_16e6.txt.gz' in cwd
        bpe_name = "bpe_simple_vocab_16e6.txt.gz"
        if not os.path.exists(bpe_name):
            candidates = [
                os.path.join(ckpt, bpe_name) if os.path.isdir(ckpt) else "",
                os.path.join("weights", "viclip", bpe_name),
                os.path.join("weights", bpe_name),
            ]
            found = next((p for p in candidates if p and os.path.exists(p)), None)
            if found:
                import shutil
                shutil.copy2(found, bpe_name)
                print(f"Copied {bpe_name} from {found} to .")
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    repo = ckpt if "/" in ckpt and not os.path.isdir(ckpt) else DEFAULT_MODEL_ID
                    print(f"Fetching {bpe_name} from {repo} ...")
                    hf_hub_download(repo_id=repo, filename=bpe_name, local_dir=".")
                except Exception as e:
                    print(f"Notice: could not auto-fetch {bpe_name}: {e}")

        from transformers import AutoModel
        model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        return model.eval().to(self.device)

    def _build_transform(self):
        """Standard ViCLIP / CLIP video pre-processing."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275,  0.40821073],
                std =[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def _preprocess_clip(self, frames: np.ndarray) -> torch.Tensor:
        """frames: (T, H, W, 3) uint8  →  (1, T, 3, H, W) float tensor"""
        tensors = torch.stack([self.transform(f) for f in frames])   # (T, 3, H, W)
        return tensors.unsqueeze(0).to(self.device)                 # (1, T, C, H, W)

    @torch.no_grad()
    def extract(self, video_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract ViCLIP embeddings for a list of video files.

        Parameters
        ----------
        video_paths : list of str
        batch_size  : number of videos processed per GPU batch

        Returns
        -------
        (N, 768) float32 ndarray — L2-normalised video embeddings
        """
        all_embs = []
        for start in range(0, len(video_paths), batch_size):
            batch_paths  = video_paths[start : start + batch_size]
            batch_tensors = []
            for vp in batch_paths:
                frames = _load_frames(vp, N_FRAMES)
                batch_tensors.append(self._preprocess_clip(frames))

            video_input = torch.cat(batch_tensors, dim=0)  # (B, T, C, H, W)

            # Robust feature extraction supporting both HF AutoModel & native InternVideo2
            if hasattr(self.model, "get_vid_features"):
                emb = self.model.get_vid_features(video_input)
            elif hasattr(self.model, "encode_video"):
                try:
                    emb = self.model.encode_video(video_input)
                except Exception:
                    # Some versions expect (B, C, T, H, W)
                    emb = self.model.encode_video(video_input.permute(0, 2, 1, 3, 4))
            else:
                out = self.model(video_input)
                emb = getattr(out, "video_embeds", out)

            if hasattr(emb, "video_embeds"):
                emb = emb.video_embeds
            elif isinstance(emb, (tuple, list)):
                emb = emb[0]

            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())

            print(f"  ViCLIP  {start + len(batch_paths):4d}/{len(video_paths)} videos")
        return np.concatenate(all_embs, axis=0)

