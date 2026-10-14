from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from dsmr.models import HMEMNet
from dsmr.models.losses import MultiObjectiveLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional
    SummaryWriter = None

try:  # optional wandb integration
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional
    wandb = None


class HMEMTrainer:
    """Simple trainer wrapper for HMEM-Net research experiments."""

    def __init__(
        self,
        model: HMEMNet,
        learning_rate: float = 1e-4,
        device: str | torch.device | None = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        resolved_device = device or "cpu"
        if isinstance(resolved_device, str) and resolved_device.lower() == "cuda":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)
        self.model = model.to(self.device)
        self.logger = logging.getLogger(__name__)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = MultiObjectiveLoss(class_weights=class_weights).to(self.device)
        self.log_dir = log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir) if self.log_dir and SummaryWriter is not None else None
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb and wandb is not None:
            wandb.init(project="hmem-net", reinit=True)
        # tracking best checkpoint by loss (lower is better)
        self.best_metric: float | None = None
        self.best_ckpt_path: str | None = None
        self.training_history: list[Dict[str, Any]] = []

    def record_epoch_result(self, epoch: int, loss: float, **metrics: Any) -> Dict[str, Any]:
        """Store a single-epoch metric record in memory and return it."""
        entry: Dict[str, Any] = {"epoch": int(epoch), "loss": float(loss)}
        entry.update(metrics)
        self.training_history.append(entry)
        return entry

    def save_training_summary(
        self,
        path: str | os.PathLike[str],
        metadata: Dict[str, Any] | None = None,
    ) -> Path:
        """Persist all epoch metrics to one JSON summary file.

        This intentionally avoids checkpoint artifacts and instead keeps a
        compact record of every epoch result in a single file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        best_loss = min((float(item["loss"]) for item in self.training_history if "loss" in item), default=None)
        best_epoch = None
        if best_loss is not None:
            for item in self.training_history:
                if float(item.get("loss", float("inf"))) == best_loss:
                    best_epoch = int(item.get("epoch", 0))
                    break

        dev_entries = [item for item in self.training_history if "dev_macro_f1" in item]
        best_dev_entry = max(dev_entries, key=lambda item: float(item["dev_macro_f1"]), default=None)

        summary = {
            "metadata": metadata or {},
            "total_epochs": len(self.training_history),
            "epochs": self.training_history,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_dev_epoch": int(best_dev_entry["epoch"]) if best_dev_entry else None,
            "best_dev_macro_f1": float(best_dev_entry["dev_macro_f1"]) if best_dev_entry else None,
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        return output_path

    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        audio_features = batch["audio_features"].to(self.device)
        text_features = batch["text_features"].to(self.device)
        vision_features = batch["vision_features"].to(self.device)
        speaker_ids = batch["speaker_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        if speaker_ids.dim() == 0:
            speaker_ids = speaker_ids.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        # Log shapes for debugging
        self.logger.debug("Batch shapes - audio: %s, text: %s, vision: %s, speakers: %s, labels: %s", tuple(audio_features.shape), tuple(text_features.shape), tuple(vision_features.shape), tuple(speaker_ids.shape), tuple(labels.shape))

        logits, memory_weights, speech_attention_weights, memories, temporal_features = self.model(audio_features, text_features, vision_features, speaker_ids)
        self.logger.debug("Model outputs - logits: %s, memory_weights: %s, speech_attn: %s, temporal: %s", tuple(logits.shape), tuple(memory_weights.shape), tuple(speech_attention_weights.shape), tuple(temporal_features.shape))
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
        self.optimizer.step()
        return loss

    def fit(self, dataloader: DataLoader, epochs: int = 5) -> None:
        for epoch in range(epochs):
            losses = []
            for batch_idx, batch in enumerate(dataloader):
                try:
                    loss = self.train_step(batch)
                    losses.append(loss.item())
                except RuntimeError as exc:
                    print(f"Batch {batch_idx} failed: {exc}")
                    raise
            avg_loss = sum(losses) / max(len(losses), 1)
            self.record_epoch_result(len(self.training_history) + 1, avg_loss, batches=len(losses))
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
            if self.writer is not None:
                self.writer.add_scalar("train/loss", avg_loss, epoch)
            if self.use_wandb and wandb is not None:
                wandb.log({"train/loss": avg_loss, "epoch": epoch})

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save a checkpoint and write metadata JSON. If `metrics` contains a 'loss', track best checkpoint."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(state, path)

        # write metadata alongside checkpoint
        try:
            import json
            from datetime import datetime

            meta = {
                "epoch": epoch,
                "path": str(path),
                "metrics": metrics or {},
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            meta_path = f"{path}.metadata.json"
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

            # update best checkpoint if loss improved
            if metrics and "loss" in metrics:
                loss = float(metrics["loss"])
                if self.best_metric is None or loss < self.best_metric:
                    self.best_metric = loss
                    self.best_ckpt_path = str(path)
                    best_path = os.path.join(os.path.dirname(path), "best_checkpoint.json")
                    with open(best_path, "w", encoding="utf-8") as fh:
                        json.dump({"best_path": self.best_ckpt_path, "best_loss": self.best_metric}, fh, indent=2)
        except Exception:
            # metadata should not break training
            pass

    def load_checkpoint(self, path: str, map_location: Optional[Any] = None) -> Dict[str, Any]:
        data = torch.load(path, map_location=map_location)
        if "model_state" in data:
            self.model.load_state_dict(data["model_state"])
        if "optimizer_state" in data:
            try:
                self.optimizer.load_state_dict(data["optimizer_state"])
            except Exception:
                # optimizer state may be incompatible; ignore
                pass
        return data

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step if step is not None else 0)
        if self.use_wandb and wandb is not None:
            wandb.log(metrics, step=step)
