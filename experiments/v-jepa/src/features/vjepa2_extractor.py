"""V-JEPA 2 video feature extractor — PER-SUBCLIP (8 frames) variant.

To mirror the exp4 DINOv2 pipeline (which caches per-frame features (N, 8, D)
and mean-pools inside the model), this extractor splits each 3-second clip into
8 equal temporal SUB-CLIPS and embeds each sub-clip with V-JEPA 2. Because
V-JEPA 2 is a motion/video model, embedding short sub-clips (option b) respects
its temporal modeling, unlike embedding isolated single frames.

Output: (N, 8, 1024) float32, L2-normalised per sub-clip. The downstream recipe
model pools the 8 sub-clip embeddings (DinoPool mean), matching exp4.

Model: 'facebook/vjepa2-vitl-fpc64-256' (ViT-L, 1024-d), HuggingFace transformers.
Requires transformers>=4.52. Install: pip install "transformers>=4.52" decord
"""

import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD = True
except ImportError:
    import cv2
    _DECORD = False

DEFAULT_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
N_SUBCLIPS = 8                 # default number of temporal sub-clips (tunable)
FRAMES_PER_SUBCLIP = 8         # frames fed to V-JEPA2 per sub-clip
EMBED_DIM = 1024


def _load_all_frames(video_path: str, total_needed: int) -> np.ndarray:
    """Uniformly sample `total_needed` frames spanning the whole clip → (T,H,W,3) uint8."""
    if _DECORD:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        idxs = np.linspace(0, len(vr) - 1, total_needed).astype(int)
        return vr.get_batch(idxs).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), total_needed).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return np.zeros((total_needed, 256, 256, 3), dtype=np.uint8)
        while len(frames) < total_needed:
            frames.append(frames[-1])
        return np.stack(frames[:total_needed])


class VJEPA2Extractor:
    """V-JEPA 2 per-sub-clip embeddings: (N, 8, 1024), one vector per temporal segment."""

    def __init__(self, model_id: str | None = None, device: str = "cuda",
                 n_subclips: int = N_SUBCLIPS, frames_per_subclip: int = FRAMES_PER_SUBCLIP):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_id = model_id or os.environ.get("VJEPA2_CKPT") or DEFAULT_MODEL_ID
        self.n_subclips = n_subclips
        self.frames_per_subclip = frames_per_subclip
        print(f"Loading V-JEPA 2 from '{self.model_id}' on {self.device} "
              f"({n_subclips} sub-clips x {frames_per_subclip} frames) ...")

        import transformers
        from transformers import AutoModel
        processor_cls = None
        try:
            from transformers import AutoVideoProcessor as processor_cls  # type: ignore
        except Exception:
            try:
                from transformers.models.vjepa2 import VJEPA2VideoProcessor as processor_cls  # type: ignore
            except Exception:
                processor_cls = None
        if processor_cls is None:
            raise ImportError(
                f"transformers {transformers.__version__} lacks a V-JEPA2 video processor; "
                "upgrade with: pip install -U 'transformers>=4.52'"
            )
        self.processor = processor_cls.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(
            self.model_id, attn_implementation="sdpa"
        ).eval().to(self.device)

    @torch.no_grad()
    def _embed_subclips(self, frames_all: np.ndarray) -> np.ndarray:
        """frames_all: (n_subclips*frames_per_subclip, H, W, 3) → (n_subclips, 1024)."""
        ns, fps = self.n_subclips, self.frames_per_subclip
        subclips = frames_all.reshape(ns, fps, *frames_all.shape[1:])
        clip_list = [subclips[i] for i in range(ns)]               # list of (fps, H, W, 3)
        inputs = self.processor(clip_list, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)                # (n_subclips, 1024)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().float().numpy()

    @torch.no_grad()
    def extract(self, video_paths: List[str]) -> np.ndarray:
        """Returns (N_videos, n_subclips, 1024) float32 L2-normalised sub-clip embeddings."""
        total_frames = self.n_subclips * self.frames_per_subclip
        all_vids = []
        for k, vp in enumerate(video_paths, 1):
            frames_all = _load_all_frames(vp, total_frames)
            all_vids.append(self._embed_subclips(frames_all))      # (n_subclips, 1024)
            if k % 25 == 0 or k == len(video_paths):
                print(f"  V-JEPA2 {k:4d}/{len(video_paths)} videos ({self.n_subclips} sub-clips each)")
        return np.stack(all_vids).astype(np.float32)                # (N, n_subclips, 1024)
