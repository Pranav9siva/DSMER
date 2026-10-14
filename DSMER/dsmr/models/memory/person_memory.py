from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .base import MemoryModule


class PersonMemory(MemoryModule):
    """Per-speaker emotional memory with slow update dynamics.

    Mathematical formulation
    ------------------------
    For a speaker $s$ and embedding $e_t$, the long-term memory is updated by
    an exponential moving average:

    $m_s^{(t+1)} = (1 - \alpha) m_s^{(t)} + \alpha e_t$

    Retrieval uses a soft gate that balances the stored memory and the current
    embedding.

    Tensor shapes
    -------------
    - embeddings: [B, D] or [B, T, D]
    - speaker_ids: [B] or [B, T]
    - memory_bank: [S, D]
    - retrieved_memory: [B, D] or [B, T, D]
    - attention_weights: [B, 1] or [B, T, 1]
    """

    def __init__(self, memory_dim: int, num_speakers: int, update_rate: float = 0.05, temperature: float = 0.5) -> None:
        super().__init__(memory_dim)
        self.num_speakers = num_speakers
        self.update_rate = update_rate
        self.temperature = temperature
        self.memory_bank = nn.Parameter(torch.zeros(num_speakers, memory_dim))
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)

    def reset_memory(self) -> None:
        """Reset the memory bank to zeros."""
        with torch.no_grad():
            self.memory_bank.zero_()

    def _prepare_inputs(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        if embeddings.dim() == 2:
            batch_embeddings = embeddings.unsqueeze(1)
            squeeze_output = True
        elif embeddings.dim() == 3:
            batch_embeddings = embeddings
            squeeze_output = False
        else:
            raise ValueError("Expected embeddings of shape [B, D] or [B, T, D], got " + str(embeddings.shape))

        if speaker_ids.dim() != 1:
            speaker_ids = speaker_ids.view(-1)
        flat_embeddings = batch_embeddings.reshape(-1, self.memory_dim)
        return batch_embeddings, flat_embeddings, speaker_ids, squeeze_output

    def forward(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor, update: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve and optionally update per-speaker memory."""
        batch_embeddings, flat_embeddings, speaker_ids, squeeze_output = self._prepare_inputs(embeddings, speaker_ids)
        memory_vectors = self.memory_bank[speaker_ids]

        similarity = torch.sum(flat_embeddings * memory_vectors, dim=-1, keepdim=True) / max(self.temperature, 1e-8)
        attention_weights = torch.sigmoid(similarity)

        retrieved_memory = attention_weights * memory_vectors + (1.0 - attention_weights) * flat_embeddings
        retrieved_memory = retrieved_memory.reshape(batch_embeddings.shape[0], batch_embeddings.shape[1], self.memory_dim)
        attention_weights = attention_weights.reshape(batch_embeddings.shape[0], batch_embeddings.shape[1], 1)

        if update:
            with torch.no_grad():
                candidate = flat_embeddings.detach()
                self.memory_bank.data[speaker_ids] = (1.0 - self.update_rate) * self.memory_bank.data[speaker_ids] + self.update_rate * candidate

        if squeeze_output and retrieved_memory.dim() == 3:
            retrieved_memory = retrieved_memory.squeeze(1)
            attention_weights = attention_weights.squeeze(1)

        return retrieved_memory, attention_weights, self.memory_bank
