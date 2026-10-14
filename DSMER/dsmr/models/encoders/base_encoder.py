from __future__ import annotations

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Abstract encoder interface for unimodal features."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
