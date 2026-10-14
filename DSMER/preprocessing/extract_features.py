from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import sys
import os

# Ensure DSMER package dir is on sys.path so local packages import correctly when
# running the script directly. `dsmr` lives under the DSMER directory.
DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

from dsmr.data.meld_dataset import MELDConversationDataset


def extract_processed(dataset: str = "meld", split: str = "train", tokenizer_name: str = "bert-base-uncased", out_dir: Optional[str] = None, max_examples: Optional[int] = None) -> Path:
    ds = MELDConversationDataset(dataset_name=dataset, split=split, tokenizer_name=tokenizer_name)
    out = Path(out_dir) if out_dir else Path.cwd() / "dataset" / "processed" / dataset
    out.mkdir(parents=True, exist_ok=True)

    total = len(ds)
    if max_examples is not None:
        total = min(total, int(max_examples))

    for i in range(total):
        item = ds[i]
        input_ids = item["text_input_ids"].cpu().numpy()
        attention_mask = item["text_attention_mask"].cpu().numpy()
        audio = item["audio"].cpu().numpy()
        vision = item["vision"].cpu().numpy()
        label = int(item["label"].cpu().item())
        np.savez_compressed(out / f"{split}_{i}.npz", input_ids=input_ids, attention_mask=attention_mask, audio=audio, vision=vision, label=label)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="meld")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()
    max_examples = args.max if args.max and args.max > 0 else None
    out = extract_processed(dataset=args.dataset, split=args.split, tokenizer_name=args.tokenizer, out_dir=args.out, max_examples=max_examples)
    print("Wrote processed features to", out)


if __name__ == "__main__":
    main()
