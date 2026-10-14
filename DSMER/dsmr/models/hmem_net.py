from __future__ import annotations

import logging
import torch
import torch.nn as nn

from .encoders.audio_encoder import AudioEncoder
from .encoders.text_encoder import TextEncoder
from .encoders.vision_encoder import VisionEncoder
from .memory import (
    AdaptiveMemoryController,
    AdaptiveMemoryUpdate,
    HierarchicalMemoryManager,
    MemoryGuidedFusion,
    MemoryGuidedSpeechAttention,
)


class HMEMNet(nn.Module):
    """Hierarchical Multi-scale Emotional Memory Network.

    This model provides the research backbone for the proposed HMEM-Net. It uses
    modality-specific encoders, cross-modal fusion, and a hierarchical memory system
    that combines person, dialogue, and conversation memories.
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        text_dim: int = 768,
        vision_dim: int = 2048,
        hidden_dim: int = 256,
        num_speakers: int = 100,
        pretrained_audio: str | None = None,
        pretrained_text: str | None = None,
        pretrained_vision: str | None = None,
        fine_tune: bool = False,
    ) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, hidden_dim=hidden_dim, pretrained_model=pretrained_audio, fine_tune=fine_tune)
        self.text_encoder = TextEncoder(input_dim=text_dim, hidden_dim=hidden_dim, pretrained_model=pretrained_text, fine_tune=fine_tune)
        self.vision_encoder = VisionEncoder(input_dim=vision_dim, hidden_dim=hidden_dim)
        if pretrained_vision is not None:
            try:
                self.vision_encoder.use_timm_backbone(pretrained_vision, fine_tune=fine_tune)
            except Exception:
                pass
        self.memory_manager = HierarchicalMemoryManager(memory_dim=hidden_dim, num_speakers=num_speakers)
        self.memory_controller = AdaptiveMemoryController(hidden_dim=hidden_dim)
        self.speech_attention = MemoryGuidedSpeechAttention(feature_dim=audio_dim, memory_dim=hidden_dim, attention_dim=hidden_dim)
        self.memory_fusion = MemoryGuidedFusion(hidden_dim=hidden_dim)
        self.fusion_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.memory_update = AdaptiveMemoryUpdate(memory_dim=hidden_dim)
        self.temporal_transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True)
        self.emotion_head = nn.Linear(hidden_dim, 7)
        # Learnable temporal pooling to map variable-length audio reprs to
        # the reference text token length when needed.
        self.temporal_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

    @staticmethod
    def _pool_temporal(features: torch.Tensor, target_length: int) -> torch.Tensor:
        """Downsample temporal features without the CUDA pooling backward kernel."""
        source_length = features.size(1)
        if source_length == target_length:
            return features
        if source_length % target_length == 0:
            factor = source_length // target_length
            return features.reshape(features.size(0), target_length, factor, features.size(2)).mean(dim=2)
        return torch.nn.functional.interpolate(
            features.transpose(1, 2), size=target_length, mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        speaker_ids: torch.Tensor,
        update: bool = True,
        use_memories: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run a full forward pass through the multimodal memory pipeline.

        Args:
            audio_features: [B, T, F]
            text_features: [B, T, D]
            vision_features: [B, C, H, W]
            speaker_ids: [B] or [B, T]

        Returns:
            logits: [B, T, C]
            memory_weights: [B, 3]
            updated_memories: tuple of tensors
            fused_features: [B, T, D]
        """
        audio_repr = self.audio_encoder(audio_features)  # [B, T, H]
        text_repr = self.text_encoder(text_features)  # [B, T, H]
        vision_repr = self.vision_encoder(vision_features)  # [B, 1, H]

        # Align time dimensions across modalities. Text tokens are treated as
        # the reference temporal length; adaptively pool audio/vision to match
        # the text time dimension if they differ.
        target_T = text_repr.size(1)

        if audio_repr.size(1) != target_T:
            # audio_repr: [B, T_audio, H] -> pool time dim to target_T
            pooled = self._pool_temporal(audio_repr, target_T)
            # apply a small learnable projection per time-step
            audio_repr = self.temporal_pool(pooled)

        if vision_repr.size(1) != target_T:
            vision_repr = vision_repr.repeat(1, target_T, 1)

        initial_fused = torch.cat([audio_repr, text_repr, vision_repr], dim=-1)
        initial_fused = self.fusion_projection(initial_fused)
        initial_context = self.temporal_transformer(initial_fused)

        if use_memories:
            person_memory, dialogue_memory, conversation_memory, person_attention = self.memory_manager(initial_context, speaker_ids, update=update)
            audio_attended, speech_attention_weights = self.speech_attention(
                audio_features,
                person_memory,
                dialogue_memory,
                conversation_memory,
            )

            audio_repr = self.audio_encoder(audio_attended)
            # Ensure re-encoded audio matches text temporal length
            if audio_repr.size(1) != target_T:
                pooled = self._pool_temporal(audio_repr, target_T)
                audio_repr = self.temporal_pool(pooled)
            if vision_repr.size(1) != target_T:
                vision_repr = vision_repr.repeat(1, target_T, 1)
            fused_modality = torch.cat([audio_repr, text_repr, vision_repr], dim=-1)
            fused_modality = self.fusion_projection(fused_modality)
            context = self.temporal_transformer(fused_modality)
            memory_weights = self.memory_controller(context, person_memory, dialogue_memory, conversation_memory)

            fused_features = self.memory_fusion(
                context,
                person_memory.unsqueeze(1).repeat(1, context.size(1), 1),
                dialogue_memory,
                conversation_memory,
                memory_weights=memory_weights,
            )

            temporal_features = self.temporal_transformer(fused_features)
            logits = self.emotion_head(temporal_features)

            updated_person, updated_dialogue, updated_conversation = self.memory_update(
                context,
                person_memory.unsqueeze(1).repeat(1, context.size(1), 1),
                dialogue_memory,
                conversation_memory,
            )
            return logits, memory_weights, speech_attention_weights, (updated_person, updated_dialogue, updated_conversation), temporal_features

        context = initial_context
        memory_weights = torch.full((context.size(0), 3), 1.0 / 3.0, device=context.device)
        speech_attention_weights = torch.ones(context.size(0), context.size(1), device=context.device) / max(context.size(1), 1)
        fused_features = self.memory_fusion(
            context,
            torch.zeros_like(context),
            torch.zeros_like(context),
            torch.zeros_like(context),
            memory_weights=memory_weights,
        )
        temporal_features = self.temporal_transformer(fused_features)
        logits = self.emotion_head(temporal_features)
        return logits, memory_weights, speech_attention_weights, (torch.zeros_like(context[:, 0, :]), torch.zeros_like(context[:, 0, :]), torch.zeros_like(context[:, 0, :])), temporal_features
