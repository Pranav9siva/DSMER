from __future__ import annotations

import torch
import pytest

from dsmr.data import meld_dataset as meld_dataset_module


class _SyntheticSplit:
    column_names = ["Utterance", "Emotion", "Speaker", "Dialogue_ID", "Utterance_ID"]
    features = {name: "string" for name in column_names}

    def __init__(self):
        self.rows = [
            {
                "Utterance": "A synthetic MELD utterance.",
                "Emotion": "neutral",
                "Speaker": "speaker_1",
                "Dialogue_ID": 0,
                "Utterance_ID": 0,
            }
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _SyntheticTokenizer:
    def __call__(self, text, truncation, padding, max_length, return_tensors):
        return {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


@pytest.fixture
def synthetic_meld_dataset(monkeypatch):
    split = _SyntheticSplit()
    monkeypatch.setattr(
        meld_dataset_module.MELDConversationDataset,
        "_has_local_split",
        lambda self, split_name: True,
    )
    monkeypatch.setattr(
        meld_dataset_module.MELDConversationDataset,
        "_load_local_split",
        lambda self, split_name, raise_on_missing=False: split,
    )
    monkeypatch.setattr(
        meld_dataset_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _SyntheticTokenizer(),
    )
    return split
