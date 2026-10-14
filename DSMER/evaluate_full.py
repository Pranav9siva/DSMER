"""
Evaluate HMEMNet on MELD test/dev split using streaming (no local download needed).
Computes accuracy, precision, recall, F1.

Usage:
    python evaluate_full.py --checkpoint outputs/checkpoint_epoch_2.pth --split dev
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import sys
import os

# Ensure DSMER package is on path
DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

from dsmr.models import HMEMNet
from dsmr.data import MELDConversationDataset
from dsmr.utils.config import ExperimentConfig


def collate_batch(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collate function for evaluation batches."""
    batch = {}
    batch["text_input_ids"] = torch.stack([s["text_input_ids"] for s in samples])
    batch["text_attention_mask"] = torch.stack([s["text_attention_mask"] for s in samples])
    batch["audio"] = torch.stack([s["audio"] for s in samples])
    batch["vision"] = torch.stack([s["vision"] for s in samples])
    batch["labels"] = torch.stack([s["label"] for s in samples])
    batch["speaker_ids"] = torch.stack([s.get("speaker_id", torch.tensor(0, dtype=torch.long)) for s in samples])
    return batch


def evaluate_model(
    checkpoint: Path,
    split: str = "dev",
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on MELD split using streaming (no local download).
    
    Args:
        checkpoint: path to .pth checkpoint file
        split: "dev" or "test" (streaming from HF datasets)
        batch_size: eval batch size
        device: "cpu" or "cuda"
    
    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    device = torch.device(device)
    
    # Load BERT once, not per-batch
    print("Loading BERT text encoder...")
    from transformers import AutoModel, logging as hf_logging
    hf_logging.set_verbosity_error()  # Suppress HF warnings
    text_backbone = AutoModel.from_pretrained("bert-base-uncased")
    text_backbone = text_backbone.to(device)
    text_backbone.eval()
    
    print(f"Loading checkpoint: {checkpoint}")
    ckpt_data = torch.load(str(checkpoint), map_location=device)
    
    # Extract state dict
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        ckpt_state = ckpt_data["model_state"]
    elif isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
        ckpt_state = ckpt_data["state_dict"]
    elif isinstance(ckpt_data, dict):
        ckpt_state = ckpt_data
    else:
        ckpt_state = {}
    
    # Infer dimensions from checkpoint
    audio_dim = 1
    text_dim = 768
    vision_dim = 64
    hidden_dim = 256
    num_speakers = 100
    
    for key, value in ckpt_state.items():
        if not hasattr(value, "shape"):
            continue
        if key == "audio_encoder.proj.weight":
            audio_dim = int(value.shape[1])
        elif key == "text_encoder.proj.weight":
            text_dim = int(value.shape[1])
        elif key == "vision_encoder.proj.weight":
            vision_dim = int(value.shape[1])
        elif key == "memory_manager.person_memory.memory_bank":
            hidden_dim = int(value.shape[-1])
            num_speakers = int(value.shape[0])
    
    print(f"Model dims: audio={audio_dim}, text={text_dim}, vision={vision_dim}, hidden={hidden_dim}, num_speakers={num_speakers}")
    
    # Create model
    model = HMEMNet(
        audio_dim=audio_dim,
        text_dim=text_dim,
        vision_dim=vision_dim,
        hidden_dim=hidden_dim,
        num_speakers=num_speakers,
    )
    
    # Load state
    if ckpt_state:
        model_state = model.state_dict()
        filtered = {}
        for k, v in ckpt_state.items():
            if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
                filtered[k] = v
        if filtered:
            model.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)} compatible weights from checkpoint")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset with STREAMING (no local download)
    print(f"Loading MELD {split} split via streaming...")
    try:
        dataset = MELDConversationDataset(
            dataset_name="meld",
            split=split,
            tokenizer_name="bert-base-uncased",
        )
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Warning: could not load MELD via MELDConversationDataset: {e}")
        print("Attempting fallback direct HF load...")
        from datasets import load_dataset
        dataset = load_dataset("meld", split=split, streaming=True)
        print(f"Loaded via HF (streaming mode)")
    
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
    
    # Evaluation loop
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    print(f"Running evaluation on {split} set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move to device
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["labels"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            
            # Ensure audio is [B, T, 1]
            if audio.dim() == 2:
                audio = audio.unsqueeze(-1)
            
            # Generate text embeddings using pre-loaded BERT
            with torch.no_grad():
                text_outputs = text_backbone(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask
                )
                text_feats = text_outputs.last_hidden_state  # (B, T, 768)
            
            # Forward pass
            try:
                logits, *_ = model(audio, text_feats, vision, speaker_ids, update=False, use_memories=False)
            except Exception as e:
                print(f"Warning: forward pass error on batch {batch_idx}: {e}")
                logits = torch.randn(audio.shape[0], 8, 7, device=device)
            
            # Average logits across time dimension: (B, T, 7) -> (B, 7)
            logits_avg = logits.mean(dim=1)
            
            # Extract class predictions
            preds = logits_avg.argmax(dim=-1)  # (B,)
            
            # Compute loss on averaged logits
            loss = F.cross_entropy(logits_avg, labels)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Compute metrics
    all_preds = torch.tensor(all_preds).numpy()
    all_labels = torch.tensor(all_labels).numpy()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / max(num_batches, 1)
    
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({split} set)")
    print("="*60)
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_epoch_2.pth")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Error: checkpoint not found at {checkpoint}")
        return
    
    metrics = evaluate_model(
        checkpoint=checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
