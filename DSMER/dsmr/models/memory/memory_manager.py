from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MemoryModule
from .conversation_memory import ConversationMemory
from .dialogue_memory import DialogueMemory
from .person_memory import PersonMemory


class HierarchicalMemoryManager(MemoryModule):
    """Hierarchical memory manager for HMEM-Net.

    Mathematical formulation
    ------------------------
    The manager exposes a single API that retrieves three memory views:

    - person memory: long-term speaker memory $m_p$
    - dialogue memory: recent turn history $m_d$
    - conversation memory: global conversation state $m_c$

    These are then combined by the fusion module with learned weights.

    Tensor shapes
    -------------
    - context: [B, T, D]
    - speaker_ids: [B] or [B, T]
    - person_memory: [B, D]
    - dialogue_memory: [B, T, D]
    - conversation_memory: [B, T, D]
    - memory_bank: [S, D]
    """

    def __init__(
        self,
        memory_dim: int,
        num_speakers: int,
        dialogue_turns: int = 64,
        conversation_steps: int = 64,
        person_update_rate: float = 0.05,
        dialogue_update_rate: float = 0.25,
        temperature: float = 0.5,
    ) -> None:
        super().__init__(memory_dim)
        self.person_memory = PersonMemory(memory_dim=memory_dim, num_speakers=num_speakers, update_rate=person_update_rate, temperature=temperature)
        self.dialogue_memory = DialogueMemory(memory_dim=memory_dim, max_turns=dialogue_turns)
        self.conversation_memory = ConversationMemory(memory_dim=memory_dim, max_steps=conversation_steps)
        self.memory_dim = memory_dim
        self.num_speakers = num_speakers
        self.register_buffer("speaker_usage", torch.zeros(num_speakers, dtype=torch.float32))
        self.rank_projection = nn.Linear(memory_dim, memory_dim)
        self.forgetting_gate = nn.Linear(memory_dim * 2, 1)

    def forward(self, context: torch.Tensor, speaker_ids: torch.Tensor, update: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve hierarchical memories for the current context."""
        if context.dim() != 3:
            raise ValueError("Expected context [B, T, D], got " + str(context.shape))

        batch_size, seq_len, hidden_dim = context.shape
        if hidden_dim != self.memory_dim:
            raise ValueError(f"Context dim {hidden_dim} does not match memory dim {self.memory_dim}")

        query = context.mean(dim=1)
        person_memory, person_attention, _ = self.person_memory(query, speaker_ids, update=update)
        dialogue_memory = self.dialogue_memory(context)
        conversation_state = self.conversation_memory(context)
        conversation_memory = conversation_state.unsqueeze(1).expand(batch_size, seq_len, self.memory_dim)
        dialogue_memory = self._align_memory_sequence(dialogue_memory, seq_len)
        conversation_memory = self._align_memory_sequence(conversation_memory, seq_len)

        if update:
            self._register_speaker_usage(speaker_ids)

        return person_memory, dialogue_memory, conversation_memory, person_attention

    def retrieve_person_ranking(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rank person memory slots by similarity to the query."""
        if query.dim() == 3:
            query = query.squeeze(1)
        if query.dim() != 2:
            raise ValueError("Expected query [B, D] or [B, 1, D], got " + str(query.shape))

        projected_query = self.rank_projection(query)
        keys = self.person_memory.memory_bank
        scores = torch.matmul(projected_query, keys.transpose(0, 1)) / max(self.person_memory.temperature, 1e-8)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, keys.size(0)), dim=-1)
        return top_scores, top_indices

    def merge_memories(self, person_memory: torch.Tensor, dialogue_memory: torch.Tensor, conversation_memory: torch.Tensor) -> torch.Tensor:
        """Produce a compact fused memory representation of shape [B, D]."""
        if person_memory.dim() != 2 or dialogue_memory.dim() != 3 or conversation_memory.dim() != 3:
            raise ValueError("Expected [B, D], [B, T, D], [B, T, D]")
        pooled_dialogue = dialogue_memory.mean(dim=1)
        pooled_conversation = conversation_memory.mean(dim=1)
        gate = torch.sigmoid(self.forgetting_gate(torch.cat([person_memory, pooled_dialogue], dim=-1)).squeeze(-1))
        return gate.unsqueeze(-1) * person_memory + (1.0 - gate).unsqueeze(-1) * pooled_conversation

    def compress_memories(self, decay_rate: float = 0.95, threshold: float = 0.1) -> None:
        """Compress low-usage speaker memories to free capacity over long conversations."""
        usage = self.speaker_usage.clamp(max=1.0)
        mask = usage < threshold
        if mask.any():
            with torch.no_grad():
                self.person_memory.memory_bank.data[mask] *= decay_rate

    def remove_stale_memories(self, min_usage: float = 0.01, decay: float = 0.99) -> None:
        """Slowly forget speaker memory slots that are not used often."""
        mask = self.speaker_usage < min_usage
        if mask.any():
            with torch.no_grad():
                self.person_memory.memory_bank.data[mask] *= decay

    def consistency_loss(self, context: torch.Tensor, retrieved_memory: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between context and person memory."""
        if context.dim() == 3:
            context = context.mean(dim=1)
        if context.shape != retrieved_memory.shape:
            raise ValueError(f"Context shape {context.shape} and memory shape {retrieved_memory.shape} must match")
        return F.mse_loss(context, retrieved_memory)

    def rank_memories(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rank person memory slots by similarity to the query."""
        if query.dim() == 3:
            query = query.squeeze(1)
        if query.dim() != 2:
            raise ValueError("Expected query [B, D] or [B, 1, D], got " + str(query.shape))

        projected_query = self.rank_projection(query)
        keys = self.person_memory.memory_bank
        scores = torch.matmul(projected_query, keys.transpose(0, 1)) / max(self.person_memory.temperature, 1e-8)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, keys.size(0)), dim=-1)
        return top_scores, top_indices

    def _align_memory_sequence(self, memory: torch.Tensor, target_len: int) -> torch.Tensor:
        if memory.dim() != 3:
            return memory
        if memory.size(1) == target_len:
            return memory
        if memory.size(1) < target_len:
            pad = memory[:, -1:, :].expand(-1, target_len - memory.size(1), -1)
            return torch.cat([memory, pad], dim=1)
        return memory[:, -target_len:, :]

    def _register_speaker_usage(self, speaker_ids: torch.Tensor) -> None:
        speaker_ids = speaker_ids.view(-1)
        with torch.no_grad():
            increments = torch.ones_like(speaker_ids, dtype=torch.float32)
            self.speaker_usage[speaker_ids] += increments
            self.speaker_usage.clamp_(max=1.0)
