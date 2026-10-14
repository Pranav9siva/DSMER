from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryGuidedSpeechAttention(nn.Module):
    """Memory-guided speech attention for emotionally salient frame-level speech features.

    The module performs memory-conditioned attention over speech frames using the
    current speech evidence and the retrieved hierarchical memory context.

    Mathematical formulation
    ------------------------
    Let x_t be frame-level features for time t, and m_t = [p, d_t, c_t] be the
    concatenated memory context built from person memory p, dialogue memory d_t,
    and conversation memory c_t. The module computes

        a_t = softmax((W_q x_t)ᵀ (W_k m_t) / sqrt(d))
        z_t = a_t · (W_v x_t)
        y_t = W_o z_t

    where a_t is the frame attention weight and y_t is the emotion-aware speech
    embedding for the current frame. Variable-length utterances are supported with
    a padding mask so that attention is computed only over valid frames.

    Tensor shapes
    -------------
    - speech_features: [B, T, F]
    - person_memory: [B, D]
    - dialogue_memory: [B, T, D]
    - conversation_memory: [B, T, D]
    - attention_weights: [B, T]
    - attended_speech: [B, T, F]
    - memory_context: [B, T, 3D]
    """

    def __init__(self, feature_dim: int, memory_dim: int, attention_dim: int = 256) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if memory_dim <= 0:
            raise ValueError("memory_dim must be positive")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")

        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.attention_dim = attention_dim
        self.context_dim = memory_dim * 3

        self.query_proj = nn.Linear(feature_dim, attention_dim)
        self.key_proj = nn.Linear(self.context_dim, attention_dim)
        self.value_proj = nn.Linear(feature_dim, attention_dim)
        self.output_proj = nn.Linear(attention_dim, feature_dim)
        self.memory_gate = nn.Linear(self.context_dim, attention_dim)
        self.norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        speech_features: torch.Tensor,
        person_memory: torch.Tensor,
        dialogue_memory: torch.Tensor,
        conversation_memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute memory-guided attention over speech frames.

        Parameters
        ----------
        speech_features:
            Frame-level speech features shaped [B, T, F].
        person_memory:
            Person memory shaped [B, D].
        dialogue_memory:
            Dialogue memory shaped [B, T, D].
        conversation_memory:
            Conversation memory shaped [B, T, D].
        attention_mask:
            Optional boolean or float mask shaped [B, T] where valid frames are
            True/1 and padded frames are False/0.

        Returns
        -------
        attended_speech:
            Emotion-aware frame representations shaped [B, T, F].
        attention_weights:
            Attention weights shaped [B, T].
        """
        if speech_features.dim() != 3:
            raise ValueError("Expected speech_features [B, T, F], got " + str(speech_features.shape))
        if person_memory.dim() != 2:
            raise ValueError("Expected person_memory [B, D], got " + str(person_memory.shape))
        if dialogue_memory.dim() != 3:
            raise ValueError("Expected dialogue_memory [B, T, D], got " + str(dialogue_memory.shape))
        if conversation_memory.dim() != 3:
            raise ValueError("Expected conversation_memory [B, T, D], got " + str(conversation_memory.shape))

        batch_size, seq_len, feature_dim = speech_features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Speech feature dim {feature_dim} does not match configured {self.feature_dim}")
        if person_memory.size(0) != batch_size:
            raise ValueError("Batch mismatch: person_memory " + str(person_memory.shape) + " vs speech_features " + str(speech_features.shape))
        if dialogue_memory.size(0) != batch_size or conversation_memory.size(0) != batch_size:
            raise ValueError("Batch size mismatch across memory tensors")
        dialogue_memory = self._align_memory_sequence(dialogue_memory, seq_len)
        conversation_memory = self._align_memory_sequence(conversation_memory, seq_len)

        person_expanded = person_memory.unsqueeze(1).expand(-1, seq_len, -1)
        memory_context = torch.cat([person_expanded, dialogue_memory, conversation_memory], dim=-1)

        query = self.query_proj(speech_features)
        key = self.key_proj(memory_context)
        value = self.value_proj(speech_features)

        gate = torch.sigmoid(self.memory_gate(memory_context))
        query = self.norm(query + gate * query)

        scores = torch.einsum("btd,btd->bt", query, key) / math.sqrt(self.attention_dim)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=speech_features.device, dtype=torch.bool)
        else:
            if attention_mask.dim() != 2:
                raise ValueError("Expected attention_mask [B, T], got " + str(attention_mask.shape))
            if attention_mask.size(0) != batch_size or attention_mask.size(1) != seq_len:
                raise ValueError("attention_mask must match [B, T]")
            attention_mask = attention_mask.to(device=speech_features.device, dtype=torch.bool)

        scores = scores.masked_fill(~attention_mask, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = attention_weights * attention_mask.to(dtype=speech_features.dtype)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        attended_values = attention_weights.unsqueeze(-1) * value
        attended_speech = self.output_proj(attended_values)
        return attended_speech, attention_weights

    def _align_memory_sequence(self, memory: torch.Tensor, target_len: int) -> torch.Tensor:
        if memory.dim() != 3:
            return memory
        if memory.size(1) == target_len:
            return memory
        if memory.size(1) < target_len:
            pad = memory[:, -1:, :].expand(-1, target_len - memory.size(1), -1)
            return torch.cat([memory, pad], dim=1)
        return memory[:, -target_len:, :]
