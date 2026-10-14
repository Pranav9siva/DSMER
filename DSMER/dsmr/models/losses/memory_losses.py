from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_loss import EmotionClassificationLoss


class MemoryConsistencyLoss(nn.Module):
    """Encourage the current context to stay close to the retrieved memory."""

    def forward(self, context: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(context, memory)


class ContrastiveMemoryLoss(nn.Module):
    """Pull together matching context-memory pairs and push apart mismatches."""

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if anchor.dim() == 2 and positive.dim() == 3:
            positive = positive.mean(dim=1)
        if anchor.dim() == 2 and negative.dim() == 3:
            negative = negative.mean(dim=1)
        if anchor.dim() == 3 and positive.dim() == 2:
            anchor = anchor.mean(dim=1)
        if anchor.dim() == 3 and negative.dim() == 2:
            anchor = anchor.mean(dim=1)

        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        negative_norm = F.normalize(negative, dim=-1)
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=-1)
        neg_sim = torch.sum(anchor_norm * negative_norm, dim=-1)
        return F.relu(1.0 - pos_sim + neg_sim).mean()


class TemporalSmoothnessLoss(nn.Module):
    """Encourage smooth temporal evolution across adjacent utterances."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError("Expected [B, T, D], got " + str(features.shape))
        if features.size(1) < 2:
            return features.sum() * 0.0
        diff = features[:, 1:, :] - features[:, :-1, :]
        return diff.pow(2).mean()


class ModalityAlignmentLoss(nn.Module):
    """Align audio, text, and vision representations by minimizing pairwise MSE."""

    def forward(self, audio_repr: torch.Tensor, text_repr: torch.Tensor, vision_repr: torch.Tensor) -> torch.Tensor:
        audio_flat = audio_repr.reshape(audio_repr.size(0), -1)
        text_flat = text_repr.reshape(text_repr.size(0), -1)
        vision_flat = vision_repr.reshape(vision_repr.size(0), -1)

        target_dim = min(audio_flat.size(1), text_flat.size(1), vision_flat.size(1))
        audio_flat = audio_flat[:, :target_dim]
        text_flat = text_flat[:, :target_dim]
        vision_flat = vision_flat[:, :target_dim]
        return F.mse_loss(audio_flat, text_flat) + F.mse_loss(audio_flat, vision_flat) + F.mse_loss(text_flat, vision_flat)


class SpeechAttentionRegularizationLoss(nn.Module):
    """Encourage sharp, memory-guided speech attention distributions."""

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        if attention_weights.dim() != 2:
            raise ValueError("Expected [B, T], got " + str(attention_weights.shape))
        return -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()


class MemoryRegularizationLoss(nn.Module):
    """Prevent memory banks from exploding in norm."""

    def forward(self, memory_bank: torch.Tensor) -> torch.Tensor:
        return memory_bank.pow(2).mean()


class MultiObjectiveLoss(nn.Module):
    """Combine the major loss terms used by HMEM-Net."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.1, delta: float = 0.1, epsilon: float = 0.01, zeta: float = 0.01, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.classification = EmotionClassificationLoss(class_weights=class_weights)
        self.consistency = MemoryConsistencyLoss()
        self.contrastive = ContrastiveMemoryLoss()
        self.temporal = TemporalSmoothnessLoss()
        self.alignment = ModalityAlignmentLoss()
        self.speech_attention = SpeechAttentionRegularizationLoss()
        self.regularization = MemoryRegularizationLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        context: torch.Tensor,
        memory: torch.Tensor,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        temporal_features: torch.Tensor,
        audio_repr: torch.Tensor,
        text_repr: torch.Tensor,
        vision_repr: torch.Tensor,
        memory_bank: torch.Tensor,
        speech_attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        cls_loss = self.classification(logits, labels)
        consistency_loss = self.consistency(context, memory)
        contrastive_loss = self.contrastive(anchor, positive, negative)
        temporal_loss = self.temporal(temporal_features)
        alignment_loss = self.alignment(audio_repr, text_repr, vision_repr)
        attention_loss = self.speech_attention(speech_attention_weights)
        regularization_loss = self.regularization(memory_bank)
        return (
            self.alpha * cls_loss
            + self.beta * consistency_loss
            + self.gamma * contrastive_loss
            + self.delta * temporal_loss
            + self.epsilon * alignment_loss
            + self.zeta * attention_loss
            + 0.001 * regularization_loss
        )
