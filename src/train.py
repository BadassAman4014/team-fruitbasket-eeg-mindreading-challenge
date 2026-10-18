"""
src/train.py - Contrastive Model Training Loop with EMA and Cosine Warmup
========================================================================
Implements WP3 of the Contrastive Retrieval Plan:
- AdamW (lr 3e-4, weight decay 0.01).
- Linear warmup (2 epochs) + Cosine Annealing learning rate schedule.
- Exponential Moving Average (EMA, decay 0.99) for stable inference.
- InfoNCE loss + SubjectLayer identity penalty.
"""

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.models.contrastive_eeg import info_nce


class ModelEMA:
    """Maintains moving average of model parameters."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_contrastive_epoch(model, dataloader, optimizer, scheduler, ema=None, device='cpu'):
    """Trains model for one epoch."""
    model.train()
    total_loss = 0.0
    total_infonce = 0.0
    total_penalty = 0.0
    num_batches = 0

    for eeg_batch, sub_batch, vid_batch, stim_batch in dataloader:
        eeg_batch = eeg_batch.to(device)
        sub_batch = sub_batch.to(device)
        vid_batch = vid_batch.to(device)
        stim_batch = {k: v.to(device) for k, v in stim_batch.items()}

        optimizer.zero_grad()
        e_emb, s_emb = model(eeg_batch, sub_batch, stim_batch)

        loss_nce = info_nce(e_emb, s_emb, model.log_scale, vid_ids=vid_batch)
        pen = model.subject_layer.penalty()
        loss = loss_nce + pen

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_infonce += loss_nce.item()
        total_penalty += pen.item()
        num_batches += 1

    return {
        'loss': total_loss / max(1, num_batches),
        'infonce': total_infonce / max(1, num_batches),
        'penalty': total_penalty / max(1, num_batches),
        'temperature': float(1.0 / model.log_scale.exp().clamp(max=100.0).item()),
        'fusion_weights': model.stim_projector.get_fusion_weights(),
    }
