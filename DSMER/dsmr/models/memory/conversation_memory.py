from __future__ import annotations

import torch
import torch.nn as nn

from .base import MemoryModule


class ConversationMemory(MemoryModule):
    """Global conversation memory for long-range semantic and emotional history.

    Mathematical formulation
    ------------------------
    The global state is maintained as a running aggregate over the conversation:

    $c_t = \beta c_{t-1} + (1 - \beta) \bar{e}_t$

    where $\bar{e}_t$ is the average embedding for the current turn and $\beta$
    is derived from the conversation scale. This keeps the memory efficient while
    preserving long-range context.

    Tensor shapes
    -------------
    - embeddings: [B, T, D]
    - state: [1, D]
    - retrieved_state: [B, D]
    """

    def __init__(self, memory_dim: int, max_steps: int = 64) -> None:
        super().__init__(memory_dim)
        self.max_steps = max_steps
        self.proj = nn.Linear(memory_dim, memory_dim)
        self.gate = nn.Linear(memory_dim, 1)
        self.register_buffer("state", torch.zeros(1, memory_dim))

    def update(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Update the conversational state from new embeddings."""
        if embeddings.dim() != 3:
            raise ValueError("Expected [B, T, D], got " + str(embeddings.shape))
        pooled = embeddings.mean(dim=1)
        current_state = pooled.mean(dim=0, keepdim=True)
        if self.state.shape[1] != self.memory_dim:
            self.state = torch.zeros(1, self.memory_dim, device=current_state.device)
        projected_state = torch.tanh(self.proj(current_state))
        gate = torch.sigmoid(self.gate(current_state)).squeeze(-1)
        candidate_state = (1.0 - gate.unsqueeze(-1)) * current_state + gate.unsqueeze(-1) * projected_state
        damping = min(0.9, 1.0 - 1.0 / max(self.max_steps, 2))
        updated_state = damping * self.state + (1.0 - damping) * candidate_state
        self.state.copy_(updated_state.detach())
        return updated_state.expand(embeddings.size(0), -1)

    def retrieve(self) -> torch.Tensor:
        """Return the current conversation state."""
        return self.state

    def reset_memory(self) -> torch.Tensor:
        """Reset the conversational state to zeros."""
        self.state.zero_()
        return self.state

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate a sequence into a global conversation state."""
        return self.update(embeddings)
