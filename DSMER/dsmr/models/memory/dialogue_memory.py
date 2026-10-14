from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import MemoryModule


class DialogueMemory(MemoryModule):
    """Short-term memory for recent conversational context.

    Mathematical formulation
    ------------------------
    A rolling buffer stores the most recent turns. For a turn embedding $e_t$,
    the dialogue state is updated by appending the current turn and dropping the
    oldest turn when the buffer is full.

    Tensor shapes
    -------------
    - embeddings: [B, T, D]
    - attention_mask: [B, T]
    - buffer: [1, max_turns, D]
    - history: [B, T, D]
    """

    def __init__(self, memory_dim: int, max_turns: int = 64) -> None:
        super().__init__(memory_dim)
        self.max_turns = max_turns
        self.batch_size = 1
        self.proj = nn.Linear(memory_dim, memory_dim)
        self.gate = nn.Linear(memory_dim, 1)
        self.register_buffer("buffer", torch.zeros(1, max_turns, memory_dim))

    def enqueue(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Append a batch of turn embeddings to the dialogue buffer."""
        if embeddings.dim() != 3:
            raise ValueError("Expected [B, T, D], got " + str(embeddings.shape))
        batch_size, seq_len, _ = embeddings.shape
        if seq_len > self.max_turns:
            seq_len = self.max_turns
            embeddings = embeddings[:, -self.max_turns:, :]

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.size(0) != batch_size or attention_mask.size(1) != seq_len:
                raise ValueError("Expected attention_mask [B, T], got " + str(attention_mask.shape))
            valid_lengths = attention_mask.sum(dim=1)
            if (valid_lengths == 0).any():
                raise ValueError("Each batch item must contain at least one valid turn")

        pooled = embeddings.mean(dim=1, keepdim=True)
        pooled_flat = pooled.squeeze(1)
        projected = torch.tanh(self.proj(pooled_flat)).unsqueeze(1)
        gate_input = pooled_flat
        gate_value = torch.sigmoid(self.gate(gate_input)).squeeze(-1).unsqueeze(-1)
        candidate = (1.0 - gate_value) * pooled + gate_value * projected
        buffer = self.buffer.expand(batch_size, -1, -1)
        if self.max_turns > 1:
            if buffer.size(1) == self.max_turns:
                new_buffer = torch.cat([buffer[:, 1:, :], candidate], dim=1)
            else:
                new_buffer = torch.cat([buffer, candidate], dim=1)
            new_buffer = new_buffer[:, -self.max_turns:, :]
        else:
            new_buffer = candidate
        self.buffer.copy_(new_buffer[:1].detach())
        self.batch_size = batch_size
        return new_buffer[:, -seq_len:, :]

    def dequeue(self) -> torch.Tensor:
        """Return the current buffer state without modifying it."""
        return self.buffer

    def retrieve(self) -> torch.Tensor:
        """Return the current dialogue memory buffer."""
        return self.buffer.expand(self.batch_size, -1, -1)

    def update(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Alias for enqueue for compatibility with the memory API."""
        return self.enqueue(embeddings, attention_mask=attention_mask)

    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Append current embeddings and return the current history state."""
        return self.enqueue(embeddings, attention_mask=attention_mask)
