from __future__ import annotations

import torch
import torch.nn as nn


class MemoryGuidedFusion(nn.Module):
    """Fuse multimodal context with person, dialogue, and conversation memories.

    Mathematical formulation
    ------------------------
    The fused representation is computed as:

    $h_t^{fuse} = \mathrm{LayerNorm}(h_t + W_f [h_t; m_p; m_d; m_c])$

    where $m_p, m_d, m_c$ are the weighted memory streams.

    Tensor shapes
    -------------
    - context: [B, T, D]
    - person_memory: [B, T, D]
    - dialogue_memory: [B, T, D]
    - conversation_memory: [B, T, D]
    - memory_weights: [B, 3]
    - fused_features: [B, T, D]
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context: torch.Tensor,
        person_memory: torch.Tensor,
        dialogue_memory: torch.Tensor,
        conversation_memory: torch.Tensor,
        memory_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate and project memory-guided features."""
        if context.dim() != 3:
            raise ValueError("Expected [B, T, D], got " + str(context.shape))

        batch_size, seq_len, hidden_dim = context.shape
        person_memory = person_memory.reshape(batch_size, seq_len, -1)
        dialogue_memory = dialogue_memory.reshape(batch_size, seq_len, -1)
        conversation_memory = conversation_memory.reshape(batch_size, seq_len, -1)

        if memory_weights is not None:
            if memory_weights.dim() != 2 or memory_weights.size(1) != 3:
                raise ValueError("Expected memory_weights [B, 3], got " + str(memory_weights.shape))
            weights = memory_weights.unsqueeze(1)
            person_memory = person_memory * weights[:, :, 0:1]
            dialogue_memory = dialogue_memory * weights[:, :, 1:2]
            conversation_memory = conversation_memory * weights[:, :, 2:3]

        combined = torch.cat([context, person_memory, dialogue_memory, conversation_memory], dim=-1)
        projected = self.proj(combined)
        return self.norm(context + projected)
