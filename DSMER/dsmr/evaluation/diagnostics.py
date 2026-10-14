from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dsmr.data.meld_dataset import MELDConversationDataset
from dsmr.evaluation.ablation import AblationConfig, AblationStudy
from dsmr.models import HMEMNet
from dsmr.models.losses import MultiObjectiveLoss
from dsmr.utils.config import ExperimentConfig


class DiagnosticTrainer:
    def __init__(self, model: HMEMNet, config: ExperimentConfig, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(2, self.config.epochs))
        self.criterion = MultiObjectiveLoss()
        self.logs: List[Dict[str, Any]] = []

    def _batch_to_device(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_features = batch["audio_features"].to(self.device)
        text_features = batch["text_features"].to(self.device)
        vision_features = batch["vision_features"].to(self.device)
        speaker_ids = batch["speaker_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        if speaker_ids.dim() == 0:
            speaker_ids = speaker_ids.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        return audio_features, text_features, vision_features, speaker_ids, labels

    def _module_grad_norm(self, name: str, module: nn.Module) -> float:
        total = 0.0
        for p in module.parameters(recurse=True):
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return math.sqrt(total) if total > 0.0 else 0.0

    def _collect_grad_norms(self) -> Dict[str, float]:
        return {
            "audio_encoder": self._module_grad_norm("audio_encoder", self.model.audio_encoder),
            "text_encoder": self._module_grad_norm("text_encoder", self.model.text_encoder),
            "vision_encoder": self._module_grad_norm("vision_encoder", self.model.vision_encoder),
            "mgsa": self._module_grad_norm("mgsa", self.model.speech_attention),
            "person_memory": self._module_grad_norm("person_memory", self.model.memory_manager.person_memory),
            "dialogue_memory": self._module_grad_norm("dialogue_memory", self.model.memory_manager.dialogue_memory),
            "conversation_memory": self._module_grad_norm("conversation_memory", self.model.memory_manager.conversation_memory),
            "memory_manager": self._module_grad_norm("memory_manager", self.model.memory_manager),
            "memory_controller": self._module_grad_norm("memory_controller", self.model.memory_controller),
            "memory_fusion": self._module_grad_norm("memory_fusion", self.model.memory_fusion),
            "emotion_classifier": self._module_grad_norm("emotion_classifier", self.model.emotion_head),
        }

    def _tensor_summary(self, tensor: torch.Tensor) -> Dict[str, float]:
        tensor = tensor.detach().float()
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std(unbiased=False).item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }

    def _memory_stats(self, before: torch.Tensor, after: torch.Tensor) -> Dict[str, float]:
        diff = (after - before).abs().mean().item()
        flat = after.detach().flatten()
        if flat.numel() == 0:
            entropy = 0.0
            utilization = 0.0
            saturation = 0.0
        else:
            probs = torch.softmax(flat, dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            utilization = (after.abs() > 1e-6).float().mean().item()
            saturation = float((after.abs().max().item() > 1e-3))
        return {
            "average_change": diff,
            "utilization": utilization,
            "entropy": entropy,
            "saturation": saturation,
        }

    def _check_graph_connectivity(self, logits: torch.Tensor, memory_weights: torch.Tensor, speech_attention_weights: torch.Tensor, temporal_features: torch.Tensor) -> Dict[str, Any]:
        issues: List[str] = []
        for name, tensor in [("logits", logits), ("memory_weights", memory_weights), ("speech_attention_weights", speech_attention_weights), ("temporal_features", temporal_features)]:
            if tensor.requires_grad and tensor.grad_fn is None:
                issues.append(f"{name} has no grad_fn")
        return {"issues": issues}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        self.model.train()
        losses: List[float] = []
        accuracies: List[float] = []
        grad_norms: List[Dict[str, float]] = []
        controller_stats: List[Dict[str, Any]] = []
        mgsa_stats: List[Dict[str, Any]] = []
        memory_stats: List[Dict[str, Any]] = []
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad(set_to_none=True)
            audio_features, text_features, vision_features, speaker_ids, labels = self._batch_to_device(batch)
            pre_dialogue = self.model.memory_manager.dialogue_memory.buffer.detach().clone()
            pre_conversation = self.model.memory_manager.conversation_memory.state.detach().clone()
            logits, memory_weights, speech_attention_weights, memories, temporal_features = self.model(audio_features, text_features, vision_features, speaker_ids, update=False, use_memories=True)
            person_memory, dialogue_memory, conversation_memory = memories
            loss = self.criterion(
                logits=logits,
                labels=labels,
                context=temporal_features,
                memory=person_memory,
                anchor=temporal_features.mean(dim=1),
                positive=person_memory.mean(dim=1) if person_memory.dim() == 3 else person_memory,
                negative=conversation_memory,
                temporal_features=temporal_features,
                audio_repr=audio_features.mean(dim=1),
                text_repr=text_features.mean(dim=1),
                vision_repr=vision_features.mean(dim=1),
                memory_bank=self.model.memory_manager.person_memory.memory_bank,
                speech_attention_weights=speech_attention_weights,
            )
            loss.backward()
            grads = self._collect_grad_norms()
            graph_status = self._check_graph_connectivity(logits, memory_weights, speech_attention_weights, temporal_features)
            grad_norms.append(grads)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            with torch.no_grad():
                self.model.eval()
                _ = self.model(audio_features, text_features, vision_features, speaker_ids, update=True)
                self.model.train()
            post_dialogue = self.model.memory_manager.dialogue_memory.buffer.detach().clone()
            post_conversation = self.model.memory_manager.conversation_memory.state.detach().clone()
            with torch.no_grad():
                self.model.eval()
                logits_no_mem, _, _, _, _ = self.model(audio_features, text_features, vision_features, speaker_ids, update=False, use_memories=False)
                self.model.train()
            if logits_no_mem.shape == logits.shape:
                hidden_delta = (logits - logits_no_mem).abs().mean().item()
            else:
                hidden_delta = float('nan')
            graph_status = self._check_graph_connectivity(logits, memory_weights, speech_attention_weights, temporal_features)
            controller_stats.append({
                "memory_weights": memory_weights.detach().cpu().tolist(),
                "summary": {
                    "mean": memory_weights.mean().item(),
                    "std": memory_weights.std(unbiased=False).item(),
                    "min": memory_weights.min().item(),
                    "max": memory_weights.max().item(),
                },
            })
            mgsa_stats.append({
                "attention": speech_attention_weights.detach().cpu().tolist(),
                "summary": {
                    "mean": speech_attention_weights.mean().item(),
                    "std": speech_attention_weights.std(unbiased=False).item(),
                    "min": speech_attention_weights.min().item(),
                    "max": speech_attention_weights.max().item(),
                },
            })
            memory_stats.append({
                "dialogue": self._memory_stats(pre_dialogue, post_dialogue),
                "conversation": self._memory_stats(pre_conversation, post_conversation),
            })
            losses.append(loss.item())
            pred = logits.mean(dim=1).argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
            accuracies.append(acc)
            if batch_idx >= 2:
                break
        self.scheduler.step()
        epoch_loss = sum(losses) / max(len(losses), 1)
        epoch_acc = sum(accuracies) / max(len(accuracies), 1)
        metrics = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norms": grad_norms[-1],
            "controller_stats": controller_stats[-1] if controller_stats else {},
            "mgsa_stats": mgsa_stats[-1] if mgsa_stats else {},
            "memory_stats": memory_stats[-1] if memory_stats else {},
            "graph_status": graph_status if 'graph_status' in locals() else {"issues": []},
            "hidden_delta": hidden_delta if 'hidden_delta' in locals() else float('nan'),
        }
        self.logs.append(metrics)
        return metrics


def build_dataset(config: ExperimentConfig) -> Tuple[DataLoader, List[Dict[str, str]]]:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    csv_path = data_dir / "MELD" / "train_sent_emo.csv"
    dataset = MELDConversationDataset(csv_path=csv_path, max_len=32)
    subset_size = config.debug_subset_size or 64
    if subset_size > 0 and subset_size < len(dataset):
        dataset = Subset(dataset, list(range(subset_size)))
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return DataLoader(dataset, batch_size=min(config.batch_size, 4), shuffle=True), rows


def collect_dataset_stats(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {}
    emotion_counter = Counter(str(row.get("Emotion", "")).strip() for row in rows)
    speaker_counter = Counter(str(row.get("Speaker", "")).strip() for row in rows)
    utterance_lengths = [len(str(row.get("Utterance", ""))) for row in rows]
    return {
        "num_rows": len(rows),
        "emotion_distribution": dict(emotion_counter),
        "speaker_distribution": dict(speaker_counter),
        "utterance_length_mean": sum(utterance_lengths) / max(len(utterance_lengths), 1),
        "utterance_length_max": max(utterance_lengths, default=0),
    }


def collect_feature_stats(dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return {}
    return {
        "audio": {"mean": batch["audio_features"].mean().item(), "std": batch["audio_features"].std(unbiased=False).item()},
        "text": {"mean": batch["text_features"].mean().item(), "std": batch["text_features"].std(unbiased=False).item()},
        "vision": {"mean": batch["vision_features"].mean().item(), "std": batch["vision_features"].std(unbiased=False).item()},
    }


def run_diagnostics(config_path: str | Path | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HMEMNet(
        audio_dim=config.audio_dim,
        text_dim=config.text_dim,
        vision_dim=config.vision_dim,
        hidden_dim=config.hidden_dim,
        num_speakers=config.num_speakers,
        pretrained_audio=config.pretrained_audio,
        pretrained_text=config.pretrained_text,
        pretrained_vision=config.pretrained_vision,
        fine_tune=config.fine_tune,
    )
    trainer = DiagnosticTrainer(model, config, device=device)
    dataloader, rows = build_dataset(config)
    dataset_stats = collect_dataset_stats(rows)
    feature_stats = collect_feature_stats(dataloader)
    logs: List[Dict[str, Any]] = []
    for epoch in range(1, max(2, config.epochs) + 1):
        metrics = trainer.train_epoch(dataloader, epoch)
        logs.append(metrics)
    checkpoint_path = Path(__file__).resolve().parents[2] / "outputs" / f"checkpoint_epoch_{max(2, config.epochs)}.pth"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    loaded = HMEMNet(
        audio_dim=config.audio_dim,
        text_dim=config.text_dim,
        vision_dim=config.vision_dim,
        hidden_dim=config.hidden_dim,
        num_speakers=config.num_speakers,
        pretrained_audio=config.pretrained_audio,
        pretrained_text=config.pretrained_text,
        pretrained_vision=config.pretrained_vision,
        fine_tune=config.fine_tune,
    )
    loaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"])
    checkpoint_ok = all(torch.equal(model.state_dict()[k], loaded.state_dict()[k]) for k in model.state_dict())
    output_path = Path(__file__).resolve().parents[2] / "outputs" / "diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.__dict__,
        "dataset_stats": dataset_stats,
        "feature_stats": feature_stats,
        "checkpoint_ok": checkpoint_ok,
        "logs": logs,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = output_path.with_suffix(".md")
    loss_progression = ", ".join(f"{entry['epoch']}: {entry['loss']:.4f}" for entry in logs)
    accuracy_progression = ", ".join(f"{entry['epoch']}: {entry['accuracy']:.4f}" for entry in logs)
    gradient_lines = "\n".join(f"- {name}: {logs[-1]['grad_norms'][name]:.4f}" for name in sorted(logs[-1]['grad_norms']))
    report_path.write_text(
        "# HMEM-Net diagnostics report\n\n"
        + f"- Loss progression: {loss_progression}\n"
        + f"- Accuracy progression: {accuracy_progression}\n"
        + f"- Checkpoint reload: {'passed' if checkpoint_ok else 'failed'}\n\n"
        + "## Gradient norms\n"
        + gradient_lines + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    run_diagnostics()
