"""Download MELD CSVs from the official GitHub repository and save to dataset/MELD.

Usage:
    python tools/download_meld.py
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import requests

FILES = [
    "train_sent_emo.csv",
    "dev_sent_emo.csv",
    "test_sent_emo.csv",
]
RAW_URL = "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
# try several possible raw repo paths
BASE_CANDIDATES = [
    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/",
    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/",
    "https://github.com/declare-lab/MELD/raw/master/data/MELD/",
    "https://github.com/declare-lab/MELD/raw/master/data/MELD/",
]


def download_raw_archive(out_dir: Path) -> Path:
    """Download the original MELD multimedia archive without touching CSVs."""
    import requests

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "MELD.Raw.tar.gz"
    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    print(f"Downloading {RAW_URL} -> {destination} (resume={existing_size} bytes)")
    with requests.get(RAW_URL, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        # A server may ignore Range; restart safely rather than corrupting the archive.
        append = existing_size > 0 and response.status_code == 206
        with destination.open("ab" if append else "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"Downloaded {destination} ({destination.stat().st_size} bytes)")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="also download MELD.Raw.tar.gz multimedia archive")
    args = parser.parse_args()
    out_dir = Path.cwd() / "dataset" / "MELD"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in FILES:
        dest = out_dir / fn
        success = False
        for base in BASE_CANDIDATES:
            url = base + fn
            print(f"Trying {url}")
            r = requests.get(url)
            if r.status_code == 200:
                print(f"Downloading {url} -> {dest}")
                dest.write_bytes(r.content)
                success = True
                break
        if not success:
            raise RuntimeError(f"Failed to download {fn} from known locations")
    if args.raw:
        download_raw_archive(out_dir)
    print("Done")


if __name__ == '__main__':
    main()
