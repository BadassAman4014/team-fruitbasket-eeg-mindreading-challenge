"""Phase 2 Deep Learning Models for EEG-Stimulus Alignment.
"""
from src.models.contrastive_eeg import (
    EEGEncoder,
    StimulusProjector,
    MultimodalContrastiveModel,
)

__all__ = ["EEGEncoder", "StimulusProjector", "MultimodalContrastiveModel"]
