"""Validate local MELD media coverage against a MELD annotation CSV.

Usage:
    python DSMER/tools/validate_meld_media.py --media-root dataset/MELD/raw --split train
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


MEDIA_EXTENSIONS = {".wav", ".flac", ".mp3", ".mp4", ".avi", ".mkv", ".webm"}


def media_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            index.setdefault(path.stem.lower(), []).append(path)
    return index


def validate(csv_path: Path, media_root: Path) -> dict[str, int]:
    index = media_index(media_root)
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    missing = 0
    duplicate_keys = sum(len(paths) - 1 for paths in index.values() if len(paths) > 1)
    speakers = Counter(row.get("Speaker", "") for row in rows)
    emotions = Counter(row.get("Emotion", "") for row in rows)
    annotation_keys = set()
    for row in rows:
        key = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}".lower()
        annotation_keys.add(key)
        if key not in index:
            missing += 1
    extra = len(set(index) - annotation_keys)
    result = {
        "annotations": len(rows),
        "media_files": sum(len(paths) for paths in index.values()),
        "missing_annotations": missing,
        "extra_media_keys": extra,
        "duplicate_media_keys": duplicate_keys,
        "unique_speakers": len(speakers),
        "unique_emotions": len(emotions),
    }
    for name, value in result.items():
        print(f"{name}: {value}")
    print("emotion_counts:", dict(sorted(emotions.items())))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    csv_path = args.csv or Path("dataset/MELD") / f"{args.split}_sent_emo.csv"
    validate(csv_path, args.media_root)


if __name__ == "__main__":
    main()