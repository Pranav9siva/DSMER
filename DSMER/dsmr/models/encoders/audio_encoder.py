from __future__ import annotations

import torch
import torch.nn as nn

from .base_encoder import BaseEncoder


class AudioEncoder(BaseEncoder):
    """Audio encoder backed by Wav2Vec2 when available.

    The encoder preserves the existing interface of [B, T, F] -> [B, T, H], but
    it now uses a HuggingFace Wav2Vec2 model by default when a pretrained model
    name is supplied. The fallback remains a learned projection for environments
    where the dependency is unavailable.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, pretrained_model: str | None = None, fine_tune: bool = False) -> None:
        super().__init__(input_dim, hidden_dim)
        self.pretrained_model = pretrained_model
        self.fine_tune = fine_tune
        self._use_hf = False
        self.wav2vec = None
        self.proj = nn.Linear(input_dim, hidden_dim)
        if pretrained_model is not None:
            try:
                from transformers import Wav2Vec2Model  # type: ignore

                self._use_hf = True
                self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
                self.proj = nn.Linear(self.wav2vec.config.hidden_size, hidden_dim)
                if not fine_tune:
                    for p in self.wav2vec.parameters():
                        p.requires_grad = False
            except Exception:
                self._use_hf = False
                self.wav2vec = None
                self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError("Expected [B, T, F], got " + str(inputs.shape))
        if self._use_hf and self.wav2vec is not None:
            hidden_states = self.wav2vec(inputs).last_hidden_state
            return torch.tanh(self.proj(hidden_states))
        return torch.tanh(self.proj(inputs))
