"""
Simple label check.
"""
import sys, os
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

# Count labels
label_counts = {}
for i in range(len(dataset)):
    label = int(dataset[i]['label'])
    label_counts[label] = label_counts.get(label, 0) + 1

print(f"\nLabel distribution:")
for label in sorted(label_counts.keys()):
    count = label_counts[label]
    pct = 100.0 * count / len(dataset)
    print(f"  Class {label}: {count:4d} ({pct:5.1f}%)")

print(f"\nEmotions in MELD (7 classes):")
print("  0=Neutral, 1=Surprise, 2=Fear, 3=Sadness, 4=Joy, 5=Disgust, 6=Anger")
