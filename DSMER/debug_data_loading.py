"""
Debug data loading and checkpoint weight application.
"""
import torch
from pathlib import Path
import sys
import os

DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

from dsmr.data import MELDConversationDataset

print("=" * 70)
print("STEP 1: Check Dataset Loading")
print("=" * 70)

try:
    dataset = MELDConversationDataset(
        dataset_name="meld",
        split="dev",
        tokenizer_name="bert-base-uncased",
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Check a few samples
    print("\nSample 0:")
    sample = dataset[0]
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Text input_ids shape: {sample['text_input_ids'].shape}")
    print(f"  Text attention_mask shape: {sample['text_attention_mask'].shape}")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Vision shape: {sample['vision'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Speaker ID: {sample.get('speaker_id', 'N/A')}")
    
    print("\nSample 10:")
    sample = dataset[10]
    print(f"  Label: {sample['label']}, Speaker: {sample.get('speaker_id', 'N/A')}")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("STEP 2: Check Checkpoint Weight Loading")
print("=" * 70)

from dsmr.models import HMEMNet

checkpoint = Path("outputs/checkpoint_epoch_5.pth")
ckpt_data = torch.load(str(checkpoint), map_location="cpu")
ckpt_state = ckpt_data["model_state"]

print(f"Checkpoint has {len(ckpt_state)} weight keys")

# Create model
model = HMEMNet(
    audio_dim=1,
    text_dim=768,
    vision_dim=64,
    hidden_dim=256,
    num_speakers=100,
)
print(f"Model has {len(model.state_dict())} parameter keys")

# Try loading
model_state = model.state_dict()
loaded_count = 0
skipped_count = 0

for k, v in ckpt_state.items():
    if k in model_state:
        if hasattr(v, "shape") and v.shape == model_state[k].shape:
            loaded_count += 1
        else:
            skipped_count += 1
            print(f"  Shape mismatch: {k}")
            print(f"    Checkpoint: {v.shape}, Model: {model_state[k].shape}")
    else:
        skipped_count += 1

print(f"\n✅ Loaded {loaded_count} weights")
print(f"❌ Skipped {skipped_count} weights")

# Actually load the weights
model.load_state_dict(ckpt_state, strict=False)
model.eval()

print("\nModel after loading:")
print(f"  audio_encoder.proj.weight: min={model.audio_encoder.proj.weight.min():.6f}, max={model.audio_encoder.proj.weight.max():.6f}")
print(f"  text_encoder.proj.weight: min={model.text_encoder.proj.weight.min():.6f}, max={model.text_encoder.proj.weight.max():.6f}")

# Check what the classifier is named
for name, module in model.named_modules():
    if "class" in name.lower() or "pred" in name.lower() or "out" in name.lower():
        print(f"  Found: {name}")
        if hasattr(module, 'weight'):
            print(f"    Weight: min={module.weight.min():.6f}, max={module.weight.max():.6f}")

print("\n" + "=" * 70)
print("STEP 3: Check Forward Pass")
print("=" * 70)

# Test forward pass with real data
sample = dataset[0]
batch_audio = sample['audio'].unsqueeze(0)  # (1, max_samples)
batch_text = torch.randn(1, sample['text_input_ids'].shape[0], 768)  # (1, T, 768)
batch_vision = sample['vision'].unsqueeze(0)  # (1, 3, 224, 224)
batch_speaker_ids = torch.tensor([sample.get('speaker_id', 0)])

print(f"Input shapes:")
print(f"  Audio: {batch_audio.shape}")
print(f"  Text: {batch_text.shape}")
print(f"  Vision: {batch_vision.shape}")
print(f"  Speaker IDs: {batch_speaker_ids.shape}")

# Ensure audio is [B, T, 1]
if batch_audio.dim() == 2:
    batch_audio = batch_audio.unsqueeze(-1)
    print(f"  Audio reshaped to: {batch_audio.shape}")

with torch.no_grad():
    try:
        logits, *_ = model(batch_audio, batch_text, batch_vision, batch_speaker_ids, update=False, use_memories=False)
        print(f"\n✅ Forward pass successful")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Logits (avg over time): {logits.mean(dim=1)}")
        pred = logits.mean(dim=1).argmax(dim=-1)
        print(f"  Predicted class: {pred.item()}")
        print(f"  True label: {sample['label']}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("STEP 4: Check Label Distribution in Dataset")
print("=" * 70)

from collections import Counter
labels = []
for i in range(min(100, len(dataset))):
    labels.append(dataset[i]['label'])

label_counts = Counter(labels)
print(f"Label distribution (first 100 samples):")
for label in sorted(label_counts.keys()):
    print(f"  Class {label}: {label_counts[label]} samples")
