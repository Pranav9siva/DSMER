"""Evaluate a checkpoint on a dataset split and print basic metrics.

Usage:
    .venv\\Scripts\\python DSMER\\tools\\evaluate.py --ckpt outputs/checkpoint_epoch_1.pth --split train
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# ensure DSMER is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsmr.data.meld_dataset import MELDConversationDataset
from dsmr.models.hmem_net import HMEMNet
from dsmr.utils.config import ExperimentConfig
from transformers import AutoModel

try:
    import yaml
except Exception:
    yaml = None


def load_checkpoint(model: HMEMNet, path: str, device: str = "cpu"):
    data = torch.load(path, map_location=device)
    if "model_state" in data:
        model.load_state_dict(data["model_state"])
    else:
        model.load_state_dict(data)


def evaluate(model: HMEMNet, dataloader: DataLoader, device: str = "cpu", text_backbone=None):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            # prepare features similar to train.py
            input_ids = batch["text_input_ids"].to(device)
            attention_mask = batch["text_attention_mask"].to(device)
            if text_backbone is not None:
                outputs = text_backbone(input_ids=input_ids, attention_mask=attention_mask)
                text_feats = outputs.last_hidden_state
            else:
                bs, seq = input_ids.shape
                # simple random projection for text if no backbone
                # construct random text features with the expected encoder input dim
                text_input_dim = getattr(model.text_encoder.proj, "in_features", None)
                if text_input_dim is None:
                    text_input_dim = model.emotion_head.in_features
                text_feats = torch.randn(bs, seq, text_input_dim, device=device)
            audio = batch["audio"].to(device)
            if audio.dim() == 2:
                audio = audio.unsqueeze(-1)
            vision = batch["vision"].to(device)
            speaker_ids = batch["speaker_id"].to(device) if "speaker_id" in batch else torch.zeros(bs, dtype=torch.long, device=device)

            logits, *_ = model(audio, text_feats, vision, speaker_ids)
            # logits: [B, T, C] -> take mean over time and argmax
            out = logits.mean(dim=1)
            pred = out.argmax(dim=1)
            lab = batch["label"].to(device)
            preds.extend(pred.cpu().tolist())
            labels.extend(lab.cpu().tolist())
            correct += (pred == lab).sum().item()
            total += lab.size(0)
    acc = correct / max(total, 1)
    class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    class_ids = list(range(len(class_names)))
    detailed_report = classification_report(
        labels,
        preds,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, preds) if labels else 0.0,
        "precision": precision_score(labels, preds, average="weighted", zero_division=0) if labels else 0.0,
        "recall": recall_score(labels, preds, average="weighted", zero_division=0) if labels else 0.0,
        "f1": f1_score(labels, preds, average="weighted", zero_division=0) if labels else 0.0,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0) if labels else 0.0,
        "per_class": {
            name: {metric: float(detailed_report[name][metric]) for metric in ("precision", "recall", "f1-score")}
            for name in class_names
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=class_ids).tolist(),
        "class_names": class_names,
        "total": total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="dsmr/configs/experiment.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--pretrained_text", type=str, default=None)
    p.add_argument("--text_backbone_path", type=str, default=None)
    p.add_argument("--media_root", type=str, default=None)
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent.parent / "dsmr" / "configs" / "experiment.yaml"

    cfg = ExperimentConfig.from_yaml(cfg_path)
    device = torch.device(cfg.device if isinstance(cfg.device, str) else str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")
    pretrained_text = args.text_backbone_path or args.pretrained_text or cfg.pretrained_text
    tokenizer_name = args.tokenizer_name or cfg.pretrained_text or "bert-base-uncased"

    ds = MELDConversationDataset(
        dataset_name="meld",
        split=args.split,
        tokenizer_name=tokenizer_name,
        media_root=args.media_root,
        processed_dir=args.processed_dir,
    )
    if args.max_samples > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(args.max_samples, len(ds))))

    def collate_batch(samples):
        import torch

        batch = {}
        batch["text_input_ids"] = torch.stack([s["text_input_ids"] for s in samples])
        batch["text_attention_mask"] = torch.stack([s["text_attention_mask"] for s in samples])
        batch["audio"] = torch.stack([s["audio"] for s in samples])
        batch["vision"] = torch.stack([s["vision"] for s in samples])
        batch["label"] = torch.stack([s["label"] for s in samples])
        batch["speaker_id"] = torch.stack([s.get("speaker_id", torch.tensor(0, dtype=torch.long)) for s in samples])
        return batch

    batch_size = args.batch_size or cfg.batch_size
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_batch)

    # load checkpoint first to infer model input sizes if needed
    ckpt_path = Path(args.ckpt)
    data = torch.load(str(ckpt_path), map_location="cpu")
    state = data.get("model_state", data)

    # infer dims from state dict when possible
    audio_dim = 1
    hidden_dim = 256
    text_dim = 768
    vision_dim = 2048
    if "audio_encoder.proj.weight" in state:
        w = state["audio_encoder.proj.weight"]
        # shape [hidden_dim, audio_dim]
        audio_dim = int(w.shape[1])
        hidden_dim = int(w.shape[0])
    # fusion_projection weight shape: [hidden_dim, hidden_dim*3]
    if "fusion_projection.weight" in state:
        w = state["fusion_projection.weight"]
        hidden_dim = int(w.shape[0])
    if "vision_encoder.proj.weight" in state:
        vision_dim = int(state["vision_encoder.proj.weight"].shape[1])
    if "text_encoder.proj.weight" in state:
        text_dim = int(state["text_encoder.proj.weight"].shape[1])
    num_speakers = int(state["memory_manager.person_memory.memory_bank"].shape[0]) if "memory_manager.person_memory.memory_bank" in state else cfg.num_speakers

    model = HMEMNet(audio_dim=audio_dim, text_dim=text_dim, vision_dim=vision_dim, hidden_dim=hidden_dim, num_speakers=num_speakers)
    compatible = {k: v for k, v in state.items() if k in model.state_dict() and hasattr(v, "shape") and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(compatible, strict=False)

    text_backbone = None
    if pretrained_text is not None:
        try:
            text_backbone = AutoModel.from_pretrained(pretrained_text)
            text_backbone = text_backbone.to(device)
            text_backbone.eval()
        except Exception as exc:
            print(f"Warning: failed to load pretrained text model '{pretrained_text}': {exc}")
            text_backbone = None

    print('Running evaluation on', len(ds), 'examples using media root', args.media_root)
    metrics = evaluate(model, loader, device=device, text_backbone=text_backbone)
    print('Evaluation complete')
    print(json.dumps(metrics, indent=2))
    workspace_root = ROOT.parent
    output_path = Path(args.output) if args.output else workspace_root / "outputs" / f"evaluation_{args.split}.json"
    if not output_path.is_absolute():
        output_path = workspace_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
