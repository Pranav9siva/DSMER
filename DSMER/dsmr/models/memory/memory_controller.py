from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdaptiveMemoryController(nn.Module):
    """Attention-based weights over hierarchical memory sources.

    Mathematical formulation
    ------------------------
    The controller computes query vectors from the current context and compares
    them against memory key vectors. For a query $q_b$ and memory key $k_j$,
    a compatibility score is computed as:

    $s_{b,j} = \frac{q_b^T k_j}{\sqrt{d}}$

    followed by a softmax over the three memory sources.

    Tensor shapes
    -------------
    - context: [B, T, D]
    - person_memory: [B, D]
    - dialogue_memory: [B, T, D]
    - conversation_memory: [B, T, D]
    - weights: [B, 3]
    """

    def __init__(self, hidden_dim: int, num_memories: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_memories = num_memories
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim * 2, num_memories)

    def forward(
        self,
        context: torch.Tensor,
        person_memory: torch.Tensor,
        dialogue_memory: torch.Tensor,
        conversation_memory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute memory weights."""
        if context.dim() != 3:
            raise ValueError("Expected context [B, T, D], got " + str(context.shape))

        batch_size, seq_len, dim = context.shape
        if dim != self.hidden_dim:
            raise ValueError(f"Context dim {dim} does not match controller dim {self.hidden_dim}")

        query = context.mean(dim=1)
        memory_keys = torch.stack(
            [person_memory, dialogue_memory.mean(dim=1), conversation_memory.mean(dim=1)],
            dim=1,
        )

        q = torch.tanh(self.query_proj(query))
        k = torch.tanh(self.key_proj(memory_keys))
        base_scores = torch.sum(q.unsqueeze(1) * k, dim=-1) / math.sqrt(self.hidden_dim)
        context_signature = torch.cat([q, memory_keys.mean(dim=1)], dim=-1)
        gate_scores = self.output(context_signature)
        scores = base_scores + gate_scores
        weights = torch.softmax(scores, dim=-1)
        return weights
