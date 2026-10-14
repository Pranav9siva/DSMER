"""
Check actual label distribution and data structure.
"""
import torch
import sys
import os
from collections import Counter

DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

from dsmr.data import MELDConversationDataset

dataset = MELDConversationDataset(
    dataset_name="meld",
    split="dev",
    tokenizer_name="bert-base-uncased",
)

print(f"Total samples: {len(dataset)}")

# Collect ALL labels
all_labels = []
for i in range(len(dataset)):
    label = dataset[i]['label']
    all_labels.append(label)

label_counts = Counter(all_labels)
print(f"\nLabel distribution (all {len(all_labels)} samples):")
for label in sorted(label_counts.keys()):
    count = label_counts[label]
    pct = 100.0 * count / len(all_labels)
    print(f"  Class {label}: {count:4d} samples ({pct:5.1f}%)")

total = sum(label_counts.values())
print(f"  Total: {total}")

# Check specific sample details
print(f"\nFirst 10 samples:")
for i in range(10):
    sample = dataset[i]
    print(f"  {i}: label={sample['label']}, speaker={sample.get('speaker_id', '?')}, text_len={sample['text_input_ids'].shape[0]}, audio_len={sample['audio'].shape[0]}")
