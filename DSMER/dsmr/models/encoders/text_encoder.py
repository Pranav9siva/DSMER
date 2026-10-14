from __future__ import annotations

import torch
import torch.nn as nn

from .base_encoder import BaseEncoder


class TextEncoder(BaseEncoder):
    """Text encoder backed by RoBERTa when available.

    The encoder preserves the existing interface of [B, T, D] -> [B, T, H], but
    it now uses a HuggingFace RoBERTa-style backbone whenever a pretrained model
    name is supplied. The fallback remains a learned projection for environments
    where the dependency is unavailable.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, pretrained_model: str | None = None, fine_tune: bool = False) -> None:
        super().__init__(input_dim, hidden_dim)
        self.pretrained_model = pretrained_model
        self.fine_tune = fine_tune
        self._use_hf = False
        self.transformer = None
        self.proj = nn.Linear(input_dim, hidden_dim)
        if pretrained_model is not None:
            try:
                from transformers import AutoModel  # type: ignore

                self._use_hf = True
                self.transformer = AutoModel.from_pretrained(pretrained_model)
                self.proj = nn.Linear(self.transformer.config.hidden_size, hidden_dim)
                if not fine_tune:
                    for p in self.transformer.parameters():
                        p.requires_grad = False
            except Exception:
                self._use_hf = False
                self.transformer = None
                self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError("Expected [B, T, D], got " + str(inputs.shape))
        if self._use_hf and self.transformer is not None:
            outputs = self.transformer(inputs)
            feats = outputs.last_hidden_state
            return torch.tanh(self.proj(feats))
        return torch.tanh(self.proj(inputs))
