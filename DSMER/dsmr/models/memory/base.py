from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """Base interface for all HMEM-Net memory modules."""

    def __init__(self, memory_dim: int) -> None:
        super().__init__()
        self.memory_dim = memory_dim

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist module state to disk for checkpointing."""
        torch.save(self.state_dict(), str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load module state from a checkpoint."""
        state = torch.load(str(path), map_location="cpu")
        self.load_state_dict(state)

