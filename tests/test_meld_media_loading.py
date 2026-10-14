from __future__ import annotations

from pathlib import Path

from dsmr.data.meld_dataset import MELDConversationDataset


def test_real_meld_mp4_loads_audio_and_video():
    media_root = Path("dataset/MELD/media/train")
    if not media_root.exists() or not any(media_root.glob("*.mp4")):
        return
    dataset = MELDConversationDataset(
        dataset_name="meld",
        split="train",
        media_root=str(media_root),
        processed_dir="dataset/processed/test_media_fixture",
    )
    sample = dataset[0]
    assert sample["audio"].shape == (80000,)
    assert sample["vision"].shape == (3, 224, 224)
    assert bool(sample["audio_available"])
    assert bool(sample["video_available"])


def test_missing_media_is_explicit():
    dataset = MELDConversationDataset(
        dataset_name="meld",
        split="train",
        media_root="dataset/MELD/media/does-not-exist",
        processed_dir="dataset/processed/test_missing_media_fixture",
    )
    sample = dataset[0]
    assert not bool(sample["audio_available"])
    assert not bool(sample["video_available"])