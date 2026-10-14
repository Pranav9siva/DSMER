from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import BaseEncoder


class VisionEncoder(BaseEncoder):
    """Vision encoder backed by a Swin-style backbone when available."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256) -> None:
        super().__init__(input_dim, hidden_dim)
        self.proj = nn.Linear(64, hidden_dim)
        self._use_timm = False
        self.timm_model = None

    def use_timm_backbone(self, model_name: str = "swin_tiny_patch4_window7_224", fine_tune: bool = False) -> None:
        try:
            import timm  # type: ignore

            self._use_timm = True
            self.timm_model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
            if not fine_tune:
                for p in self.timm_model.parameters():
                    p.requires_grad = False
            self.proj = nn.Linear(self.timm_model.num_features, self.hidden_dim)
        except Exception:
            self._use_timm = False
            self.timm_model = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 5:
            batch_size = inputs.size(0)
            images = inputs.flatten(0, 1)
        elif inputs.dim() == 4:
            batch_size = inputs.size(0)
            images = inputs
        else:
            raise ValueError("Expected [B, C, H, W] or [B, T, C, H, W], got " + str(inputs.shape))

        if getattr(self, "_use_timm", False) and self.timm_model is not None:
            feats = self.timm_model(images)
            projected = self.proj(feats)
            return projected.view(batch_size, 1, self.hidden_dim)

        pooled = F.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (8, 8)) if images.dim() == 4 else F.adaptive_avg_pool2d(images, (8, 8))
        flattened = pooled.flatten(1)
        projected = self.proj(flattened)
        return projected.view(batch_size, 1, self.hidden_dim)
