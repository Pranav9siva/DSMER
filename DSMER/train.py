from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score

from dsmr.utils.config import ExperimentConfig
from dsmr.data import MELDConversationDataset
from dsmr.models.hmem_net import HMEMNet
from dsmr.trainers.trainer import HMEMTrainer


def collate_batch(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    # simple stacking collate - return raw batched tensors
    batch = {}
    batch["text_input_ids"] = torch.stack([s["text_input_ids"] for s in samples])
    batch["text_attention_mask"] = torch.stack([s["text_attention_mask"] for s in samples])
    batch["audio"] = torch.stack([s["audio"] for s in samples])
    batch["vision"] = torch.stack([s["vision"] for s in samples])
    batch["labels"] = torch.stack([s["label"] for s in samples])
    batch["speaker_ids"] = torch.stack([s.get("speaker_id", torch.tensor(0, dtype=torch.long)) for s in samples])
    return batch


def _unwrap_dataset(dataset: Any) -> Any:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _sampling_weights(dataset: Any) -> tuple[torch.Tensor, Dict[str, int]]:
    sample_indices = list(range(len(dataset)))
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        sample_indices = [base_dataset.indices[index] for index in sample_indices]
        base_dataset = base_dataset.dataset

    label_column = next((name for name in ("Emotion", "emotion", "label", "labels") if name in base_dataset.dataset.column_names), None)
    if label_column is None:
        raise ValueError("Training split has no emotion label column")

    label_map = MELDConversationDataset.EMOTION_TO_ID
    sample_labels: list[int] = []
    for index in sample_indices:
        label = base_dataset.dataset[label_column][index]
        if isinstance(label, str):
            class_id = label_map.get(label.lower())
        else:
            class_id = int(label) if label is not None else None
        if class_id is not None and 0 <= class_id < len(label_map):
            sample_labels.append(class_id)
        else:
            raise ValueError(f"Invalid emotion label at training row {index}: {label!r}")

    counts: Counter[int] = Counter(sample_labels)
    if not counts:
        raise ValueError("No valid emotion labels found in the training split")
    per_class_weight = {class_id: 1.0 / (count ** 0.5) for class_id, count in counts.items()}
    weights = torch.tensor([per_class_weight[class_id] for class_id in sample_labels], dtype=torch.double)
    names_by_id = {class_id: name for name, class_id in label_map.items()}
    return weights, {names_by_id[class_id]: counts.get(class_id, 0) for class_id in range(len(label_map))}


def _encode_text(text_backbone: Any, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    input_ids = batch["text_input_ids"].to(device)
    attention_mask = batch["text_attention_mask"].to(device)
    return text_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def _evaluate(model: HMEMNet, text_backbone: Any, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    text_backbone.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            text_features = _encode_text(text_backbone, raw_batch, device)
            audio = raw_batch["audio"].to(device)
            if audio.dim() == 2:
                audio = audio.unsqueeze(-1)
            vision = raw_batch["vision"].to(device)
            labels = raw_batch["labels"].to(device)
            speaker_ids = raw_batch["speaker_ids"].to(device)
            logits, _, _, _, _ = model(audio, text_features, vision, speaker_ids, update=False)
            utterance_logits = logits.mean(dim=1)
            losses.append(F.cross_entropy(utterance_logits, labels).item())
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(utterance_logits.argmax(dim=-1).cpu().tolist())

    class_names = [name for name, _ in sorted(MELDConversationDataset.EMOTION_TO_ID.items(), key=lambda item: item[1])]
    class_f1 = f1_score(all_targets, all_predictions, labels=list(range(len(class_names))), average=None, zero_division=0)
    return {
        "dev_loss": sum(losses) / max(len(losses), 1),
        "dev_accuracy": accuracy_score(all_targets, all_predictions),
        "dev_macro_f1": f1_score(all_targets, all_predictions, labels=list(range(len(class_names))), average="macro", zero_division=0),
        "dev_class_f1": {name: float(class_f1[index]) for index, name in enumerate(class_names)},
        "dev_samples": len(all_targets),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="dsmr/configs/experiment.yaml")
    parser.add_argument("--dataset", type=str, default="meld")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--media_root", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--text_unfreeze_epoch", type=int, default=2)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "dsmr" / "configs" / "experiment.yaml"

    cfg = ExperimentConfig.from_yaml(cfg_path)

    device = torch.device(cfg.device if isinstance(cfg.device, str) else str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    default_media_root = Path(__file__).resolve().parent.parent / "dataset" / "MELD" / "media"
    media_root = Path(args.media_root) if args.media_root else (default_media_root if default_media_root.exists() else None)
    print("Loading dataset:", args.dataset, "split:", args.split)
    print("Media root:", media_root if media_root else "not configured")
    tokenizer_name = cfg.pretrained_text or "bert-base-uncased"
    processed_dir = args.processed_dir
    if processed_dir is None and media_root:
        # Prevent older cached .npz files from silently replacing real media.
        processed_dir = str(media_root / "_processed_disabled")
    dataset = MELDConversationDataset(
        dataset_name=args.dataset,
        split=args.split,
        tokenizer_name=tokenizer_name,
        media_root=str(media_root) if media_root else None,
        processed_dir=processed_dir,
    )

    if args.max_samples and args.max_samples > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    # optional debug subset
    if cfg.debug_subset_size and cfg.debug_subset_size > 0:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(min(len(dataset), int(cfg.debug_subset_size)))))

    batch_size = args.batch_size or cfg.batch_size
    sample_weights, class_counts = _sampling_weights(dataset)
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_batch)
    dev_dataset = MELDConversationDataset(
        dataset_name=args.dataset,
        split="dev",
        tokenizer_name=tokenizer_name,
        media_root=str(media_root) if media_root else None,
        processed_dir=str(media_root / "_processed_disabled") if media_root else None,
    )
    if args.max_samples and args.max_samples > 0:
        from torch.utils.data import Subset

        dev_dataset = Subset(dev_dataset, list(range(min(args.max_samples, len(dev_dataset)))))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # prepare text model for embeddings
    try:
        from transformers import AutoModel

        text_backbone = AutoModel.from_pretrained(cfg.pretrained_text or "bert-base-uncased")
        text_hidden_size = text_backbone.config.hidden_size
    except Exception as exc:
        raise RuntimeError("Could not load the configured text backbone; refusing to train on random text features") from exc

    speaker_map = getattr(dataset, "speaker_map", {}) or {}
    if not speaker_map and hasattr(dataset, "dataset"):
        speaker_map = getattr(dataset.dataset, "speaker_map", {}) or {}
    speaker_count = len(speaker_map)
    if speaker_map:
        speaker_count = max(speaker_count, max(speaker_map.values(), default=-1) + 1)
    num_speakers = max(cfg.num_speakers, speaker_count, 1)

    model = HMEMNet(
        audio_dim=1,
        text_dim=text_hidden_size,
        vision_dim=cfg.vision_dim,
        hidden_dim=cfg.hidden_dim,
        num_speakers=num_speakers,
        pretrained_audio=cfg.pretrained_audio,
        pretrained_text=None,
        pretrained_vision=cfg.pretrained_vision,
        fine_tune=cfg.fine_tune,
    )

    trainer = HMEMTrainer(
        model=model,
        learning_rate=cfg.learning_rate,
        device=device,
        log_dir=cfg.log_dir,
        use_wandb=cfg.use_wandb,
        class_weights=None,
    )

    # move backbone to device if present
    if text_backbone is not None:
        text_backbone = text_backbone.to(trainer.device)
        if cfg.fine_tune:
            text_backbone.requires_grad_(False)
            trainer.optimizer.add_param_group({"params": text_backbone.parameters(), "lr": cfg.learning_rate * 0.1})
        else:
            text_backbone.eval()

    print("Starting training loop")
    epochs = args.epochs or cfg.epochs
    training_metadata = {
        "dataset": args.dataset,
        "train_split": args.split,
        "dev_split": "dev",
        "train_samples": len(dataset),
        "dev_samples": len(dev_dataset),
        "class_counts": class_counts,
        "class_balancing": "inverse_sqrt_frequency_weighted_sampling",
        "media_root": str(media_root) if media_root else None,
        "text_backbone": cfg.pretrained_text,
        "text_backbone_fine_tune_enabled": bool(cfg.fine_tune),
        "learning_rate": cfg.learning_rate,
        "epochs_requested": int(args.epochs or cfg.epochs),
        "batch_size": batch_size,
        "text_unfreeze_epoch": args.text_unfreeze_epoch if cfg.fine_tune else None,
        "seed": cfg.seed,
        "device": str(trainer.device),
    }
    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir or cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir = out_dir.resolve()
    training_summary_path = out_dir / "training_results.json"
    best_checkpoint_path = out_dir / "checkpoint_best.pth"
    export_dir = out_dir / "exports"
    training_metadata["best_checkpoint"] = str(best_checkpoint_path)
    training_metadata["torchscript_export"] = str(export_dir / "hmemnet_ts.pt")
    training_metadata["text_backbone_export"] = str(export_dir / "text_backbone")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(int(epochs)):
        epoch_losses = []
        model.train()
        if cfg.fine_tune:
            text_is_unfrozen = epoch + 1 >= max(1, args.text_unfreeze_epoch)
            text_backbone.requires_grad_(text_is_unfrozen)
            text_backbone.train(text_is_unfrozen)
        for batch_idx, raw_batch in enumerate(loader):
            # prepare model-ready batch
            # text -> embeddings
            if text_backbone is not None:
                if cfg.fine_tune:
                    text_feats = _encode_text(text_backbone, raw_batch, trainer.device)
                else:
                    with torch.no_grad():
                        text_feats = _encode_text(text_backbone, raw_batch, trainer.device)

            # audio -> ensure [B, T, 1]
            audio = raw_batch["audio"]
            if audio.dim() == 2:
                audio = audio.unsqueeze(-1)
            audio = audio.to(trainer.device)

            # vision -> [B, C, H, W]
            vision = raw_batch["vision"].to(trainer.device)

            labels = raw_batch["labels"].to(trainer.device)
            if "speaker_ids" in raw_batch:
                speaker_ids = raw_batch["speaker_ids"].to(trainer.device)
            else:
                speaker_ids = torch.zeros(labels.size(0), dtype=torch.long, device=trainer.device)

            batch = {
                "audio_features": audio,
                "text_features": text_feats,
                "vision_features": vision,
                "speaker_ids": speaker_ids,
                "labels": labels,
            }

            loss = trainer.train_step(batch)
            epoch_losses.append(loss.item())
            if batch_idx % max(1, cfg.log_every_n_steps) == 0:
                print(f"Epoch {epoch+1} batch {batch_idx} loss {loss.item():.4f}")
        # compute average loss for the epoch and record it in the single JSON summary
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        trainer.record_epoch_result(epoch + 1, avg_loss, batches=len(epoch_losses), log_every_n_steps=cfg.log_every_n_steps)
        dev_metrics = _evaluate(model, text_backbone, dev_loader, trainer.device)
        trainer.training_history[-1].update(dev_metrics)
        trainer.training_history[-1]["log_every_n_steps"] = cfg.log_every_n_steps
        trainer.training_history[-1]["text_backbone_unfrozen"] = bool(cfg.fine_tune and epoch + 1 >= max(1, args.text_unfreeze_epoch))
        trainer.save_training_summary(training_summary_path, metadata=training_metadata)
        print(
            f"Epoch {epoch + 1}/{int(epochs)} train_loss={avg_loss:.4f} "
            f"dev_accuracy={dev_metrics['dev_accuracy']:.4f} dev_macro_f1={dev_metrics['dev_macro_f1']:.4f}"
        )
        current_dev_macro_f1 = float(dev_metrics["dev_macro_f1"])
        if current_dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = current_dev_macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "dev_macro_f1": best_dev_macro_f1,
                    "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    "text_backbone_state": {name: value.detach().cpu().clone() for name, value in text_backbone.state_dict().items()},
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
            print("Early stopping: dev macro-F1 did not improve within the patience window")
            break

    if not best_checkpoint_path.exists():
        raise RuntimeError("Training completed without producing a best-model checkpoint")

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state"])
    text_backbone.load_state_dict(best_checkpoint["text_backbone_state"])
    text_backbone.save_pretrained(export_dir / "text_backbone")
    export_script = Path(__file__).resolve().parent / "tools" / "export_model.py"
    subprocess.run(
        [
            sys.executable,
            str(export_script),
            "--checkpoint",
            str(best_checkpoint_path),
            "--outdir",
            str(export_dir),
        ],
        check=True,
        cwd=project_root,
    )
    if not (export_dir / "hmemnet_ts.pt").exists():
        raise RuntimeError("TorchScript export did not produce hmemnet_ts.pt")

    trainer.save_training_summary(training_summary_path, metadata=training_metadata)
    print(f"Saved best checkpoint: {best_checkpoint_path} (epoch {best_checkpoint['epoch']})")
    print(f"Saved TorchScript export: {export_dir / 'hmemnet_ts.pt'}")
    print(f"Saved training summary: {training_summary_path}")


if __name__ == "__main__":
    main()
