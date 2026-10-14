from __future__ import annotations

import numpy as np

from dsmr.data.meld_dataset import MELDConversationDataset


def test_meld_dataset_loads_processed(tmp_path, synthetic_meld_dataset):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.savez(
        processed_dir / "train_0.npz",
        input_ids=np.zeros(128, dtype=np.int64),
        attention_mask=np.ones(128, dtype=np.int64),
        audio=np.zeros(80000, dtype=np.float32),
        vision=np.zeros((3, 224, 224), dtype=np.float32),
        label=np.array(4),
        speaker_id=np.array(0),
    )
    ds = MELDConversationDataset(processed_dir=str(processed_dir))
    assert len(ds) > 0
    item = ds[0]
    for k in ("text_input_ids", "text_attention_mask", "audio", "vision", "label"):
        assert k in item
    assert hasattr(item["text_input_ids"], "size")
