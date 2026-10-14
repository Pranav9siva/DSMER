"""Loss functions for HMEM-Net."""

from .emotion_loss import EmotionClassificationLoss
from .memory_losses import (
    ContrastiveMemoryLoss,
    MemoryConsistencyLoss,
    MemoryRegularizationLoss,
    ModalityAlignmentLoss,
    TemporalSmoothnessLoss,
    MultiObjectiveLoss,
)

__all__ = [
    "EmotionClassificationLoss",
    "MemoryConsistencyLoss",
    "ContrastiveMemoryLoss",
    "TemporalSmoothnessLoss",
    "ModalityAlignmentLoss",
    "MemoryRegularizationLoss",
    "MultiObjectiveLoss",
]
