from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveMemoryUpdate(nn.Module):
    """Adaptive update engine for the three memory banks.

    Person memory uses a slow update rate, dialogue memory uses a fast rate,
    and conversation memory uses a dynamic gate-based update.
    """

    def __init__(self, memory_dim: int, person_rate: float = 0.05, dialogue_rate: float = 0.25) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.person_rate = person_rate
        self.dialogue_rate = dialogue_rate
        self.gate = nn.Linear(memory_dim, 1)

    def forward(self, current_context: torch.Tensor, person_memory: torch.Tensor, dialogue_memory: torch.Tensor, conversation_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute updated memory states."""
        if current_context.dim() != 3:
            raise ValueError("Expected [B, T, D], got " + str(current_context.shape))
        pooled = current_context.mean(dim=1)
        gate = torch.sigmoid(self.gate(pooled)).squeeze(-1)

        if person_memory.dim() == 3:
            pooled_broadcast = pooled.unsqueeze(1)
        else:
            pooled_broadcast = pooled

        updated_person = (1 - self.person_rate) * person_memory + self.person_rate * pooled_broadcast
        updated_dialogue = (1 - self.dialogue_rate) * dialogue_memory + self.dialogue_rate * pooled_broadcast

        if conversation_memory.dim() == 3:
            convo_broadcast = gate.unsqueeze(-1).unsqueeze(1)
            pooled_convo = pooled.unsqueeze(1)
        else:
            convo_broadcast = gate.unsqueeze(-1)
            pooled_convo = pooled

        updated_conversation = (1 - convo_broadcast) * conversation_memory + convo_broadcast * pooled_convo
        return updated_person, updated_dialogue, updated_conversation
