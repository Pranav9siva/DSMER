"""
Check all MELD splits for class distribution.
"""
import sys, os
DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

from dsmr.data import MELDConversationDataset

for split in ['train', 'dev', 'test']:
    print(f"\n{'='*60}")
    print(f"SPLIT: {split.upper()}")
    print('='*60)
    
    try:
        dataset = MELDConversationDataset(
            dataset_name="meld",
            split=split,
            tokenizer_name="bert-base-uncased",
        )
        
        print(f"Total samples: {len(dataset)}")
        
        # Count labels
        label_counts = {}
        for i in range(len(dataset)):
            label = int(dataset[i]['label'])
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nClass distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = 100.0 * count / len(dataset)
            print(f"  Class {label}: {count:5d} ({pct:5.1f}%)")
        
        all_classes = set(label_counts.keys())
        print(f"\nUnique classes in {split}: {sorted(all_classes)}")
        
    except Exception as e:
        print(f"Error loading {split}: {e}")
