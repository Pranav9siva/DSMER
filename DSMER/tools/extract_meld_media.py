"""Extract MELD split videos from MELD.Raw.tar.gz.

Usage:
    python DSMER/tools/extract_meld_media.py
"""
from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path


def extract_split(source: tarfile.TarFile, split: str, output_root: Path) -> int:
    member = source.getmember(f"MELD.Raw/{split}.tar.gz")
    nested_stream = source.extractfile(member)
    if nested_stream is None:
        raise RuntimeError(f"Cannot read nested archive for {split}")

    destination = output_root / split
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with gzip.GzipFile(fileobj=nested_stream) as gz:
        with tarfile.open(fileobj=gz, mode="r:") as nested:
            for item in nested:
                if not item.isfile() or "/._" in item.name or item.name.startswith("._"):
                    continue
                if not item.name.lower().endswith(".mp4"):
                    continue
                filename = Path(item.name).name
                if not filename.lower().startswith("dia"):
                    continue
                target = destination / filename
                if target.exists() and target.stat().st_size == item.size:
                    extracted += 1
                    continue
                stream = nested.extractfile(item)
                if stream is None:
                    continue
                with target.open("wb") as output:
                    while chunk := stream.read(1024 * 1024):
                        output.write(chunk)
                extracted += 1
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("dataset/MELD/raw/MELD.Raw.tar.gz"))
    parser.add_argument("--out", type=Path, default=Path("dataset/MELD/media"))
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="all")
    args = parser.parse_args()

    splits = ("train", "dev", "test") if args.split == "all" else (args.split,)
    with tarfile.open(args.archive, "r:gz") as source:
        for split in splits:
            count = extract_split(source, split, args.out)
            print(f"{split}: extracted {count} mp4 files to {args.out / split}")


if __name__ == "__main__":
    main()