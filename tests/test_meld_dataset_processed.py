from __future__ import annotations

import os
from dsmr.data.meld_dataset import MELDConversationDataset


def test_meld_dataset_loads_processed():
    # Use the processed files created earlier
    processed_dir = os.path.join(os.getcwd(), "dataset", "processed", "meld")
    ds = MELDConversationDataset(processed_dir=processed_dir)
    assert len(ds) > 0
    item = ds[0]
    # expected keys
    for k in ("text_input_ids", "text_attention_mask", "audio", "vision", "label"):
        assert k in item
    # types
    assert hasattr(item["text_input_ids"], "size")
