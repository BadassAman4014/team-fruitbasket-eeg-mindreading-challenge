"""DINOv2 per-frame feature extractor (Component-A spec), frame-level cache.

Produces (N_videos, 8, 1536) so it flows through the SAME recipe pipeline as the
frame-level V-JEPA2 cache (train_recipe.py pools frames with DinoPool mean, then
concat-fuses with CLAP). This keeps DINOv2 and V-JEPA2 on an identical backbone,
loss, audio, and protocol — the ONLY difference is the video encoder, which is
exactly what the ensemble is meant to combine.

Component A: DINOv2 ViT-B/14 (hidden 768), 8 frames per clip, per-frame feature =
CLS token concatenated with the mean of patch tokens → 1536-d. Here we keep the
8 per-frame vectors (N, 8, 1536) and let the model pool them (matching how the
V-JEPA2 recipe pools its 8 sub-clips), rather than mean-pooling at extraction.

Model: 'facebook/dinov2-base' via HuggingFace (AutoModel + AutoImageProcessor).
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

DEFAULT_MODEL_ID = "facebook/dinov2-base"   # ViT-B/14, hidden 768 → 1536 (CLS+patch)
N_FRAMES = 8
EMBED_DIM = 1536


def _load_frames(video_path: str, n_frames: int = N_FRAMES) -> np.ndarray:
    if _DECORD:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        idxs = np.linspace(0, len(vr) - 1, n_frames).astype(int)
        return vr.get_batch(idxs).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return np.zeros((n_frames, 224, 224, 3), dtype=np.uint8)
        while len(frames) < n_frames:
            frames.append(frames[-1])
        return np.stack(frames[:n_frames])


class DINOv2Extractor:
    """DINOv2 per-frame CLS+patch features → (N, n_frames, 1536), L2-normalised."""

    def __init__(self, model_id: str | None = None, device: str = "cuda",
                 n_frames: int = N_FRAMES, layer: int = 11):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_id = model_id or os.environ.get("DINOV2_CKPT") or DEFAULT_MODEL_ID
        self.n_frames = n_frames
        self.layer = layer   # transformer block index to read (exp4 uses 11)
        print(f"Loading DINOv2 from '{self.model_id}' on {self.device} "
              f"({n_frames} frames, layer {layer}) ...")
        from transformers import AutoModel, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    @torch.no_grad()
    def extract(self, video_paths: List[str]) -> np.ndarray:
        """Returns (N_videos, n_frames, 1536) float32 per-frame CLS+patch features.

        Uses the output of transformer block index `self.layer` (default 11, matching
        exp4's `encoder.layer.11` cache) — i.e. the hidden state BEFORE the model's
        final layernorm, not `last_hidden_state`. No L2-normalize at extraction (exp4
        caches raw features and z-score-normalizes later in the trainer).
        """
        all_vids = []
        for k, vp in enumerate(video_paths, 1):
            frames = _load_frames(vp, self.n_frames)              # (n_frames, H, W, 3)
            inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
            out = self.model(**inputs, output_hidden_states=True)
            # hidden_states: tuple len 13 → [emb, layer0_out, ..., layer11_out].
            # exp4 uses encoder.layer.11 output = hidden_states[12] (pre final-norm).
            hs = out.hidden_states[self.layer + 1]                # (n_frames, 1+P, 768)
            cls = hs[:, 0, :]                                     # (n_frames, 768)
            patch = hs[:, 1:, :].mean(dim=1)                     # (n_frames, 768)
            frame_emb = torch.cat([cls, patch], dim=-1)          # (n_frames, 1536)
            all_vids.append(frame_emb.cpu().float().numpy())
            if k % 25 == 0 or k == len(video_paths):
                print(f"  DINOv2 {k:4d}/{len(video_paths)} videos "
                      f"({self.n_frames} frames, layer {self.layer})")
        return np.stack(all_vids).astype(np.float32)              # (N, n_frames, 1536)
