"""Small utility to extract and save processed features for the MELD dataset.

Saves per-sample compressed .npz files under `dataset/processed/<dataset_name>/` with
filenames `{split}_{index}.npz` containing arrays: input_ids, attention_mask, audio, vision, label.

Usage:
    .venv\\Scripts\\python DSMER\\tools\\extract_features.py --dataset meld --split train
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ensure DSMER is on sys.path so `import dsmr` works when run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsmr.data.meld_dataset import MELDConversationDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="meld")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=None)
    p.add_argument("--media-root", default=None)
    p.add_argument("--audio-root", default=None)
    p.add_argument("--video-root", default=None)
    args = p.parse_args()

    ds = MELDConversationDataset(
        dataset_name=args.dataset,
        split=args.split,
        media_root=args.media_root,
        audio_root=args.audio_root,
        video_root=args.video_root,
    )
    out_dir = Path(args.out) if args.out else Path.cwd() / "dataset" / "processed" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(ds)), desc="extract"):
        item = ds[i]
        # convert tensors to numpy
        input_ids = item["text_input_ids"].cpu().numpy()
        attention_mask = item["text_attention_mask"].cpu().numpy()
        audio = item["audio"].cpu().numpy()
        vision = item["vision"].cpu().numpy()
        label = int(item["label"].cpu().item())
        speaker_id = int(item.get("speaker_id", torch.tensor(0)).cpu().item())
        audio_available = bool(item.get("audio_available", torch.tensor(False)).item())
        video_available = bool(item.get("video_available", torch.tensor(False)).item())
        np.savez_compressed(
            out_dir / f"{args.split}_{i}.npz",
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio=audio,
            vision=vision,
            label=label,
            speaker_id=speaker_id,
            audio_available=audio_available,
            video_available=video_available,
        )

    print("Done")


if __name__ == "__main__":
    main()
