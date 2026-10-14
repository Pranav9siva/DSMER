from __future__ import annotations

from typing import Dict

import torch


def _normalize_predictions(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize prediction and target shapes for metric computation."""
    if preds.dim() == 3 and targets.dim() == 1:
        preds = preds.mean(dim=1)
    if preds.dim() == 3 and targets.dim() == 2:
        preds = preds.reshape(-1, preds.size(-1))
        targets = targets.reshape(-1)
    if preds.dim() == 2 and targets.dim() == 2:
        targets = targets.reshape(-1)
    return preds, targets


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute simple accuracy over flattened predictions."""
    preds, targets = _normalize_predictions(preds, targets)
    preds = preds.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def classification_report(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    accuracy_value = accuracy(preds, targets)
    return {"accuracy": accuracy_value}
