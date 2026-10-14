from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionClassificationLoss(nn.Module):
    """Cross-entropy loss for emotion class prediction."""

    def __init__(self, reduction: str = "mean", class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.reduction = reduction
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy over the final dimension."""
        if logits.dim() != 3:
            raise ValueError("Expected [B, T, C], got " + str(logits.shape))

        if labels.dim() == 1:
            if labels.numel() != logits.size(0):
                raise ValueError(f"Expected {logits.size(0)} labels, got {labels.numel()}")
            logits_for_loss = logits.mean(dim=1)
            label_targets = labels.long()
        elif labels.dim() == 2:
            logits_for_loss = logits.reshape(-1, logits.size(-1))
            label_targets = labels.reshape(-1).long()
        else:
            raise ValueError("Unsupported label shape " + str(labels.shape))

        return F.cross_entropy(
            logits_for_loss,
            label_targets,
            weight=self.class_weights,
            reduction=self.reduction,
        )
