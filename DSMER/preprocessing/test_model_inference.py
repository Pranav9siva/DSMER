from __future__ import annotations

import sys
import os

DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

import torch
import torch.nn.functional as F
from dsmr.data.meld_dataset import MELDConversationDataset
from dsmr.models.hmem_net import HMEMNet
from transformers import AutoTokenizer
import torch.nn as nn


def main(processed_dir: str | None = None):
    ds = MELDConversationDataset(processed_dir=processed_dir)
    item = ds[0]
    # prepare text embeddings
    tokenizer = AutoTokenizer.from_pretrained(ds.tokenizer_name)
    embed_dim = 64
    embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)

    input_ids = item["text_input_ids"].unsqueeze(0)
    text_feats = embedding(input_ids)

    # prepare audio: pool to text length
    audio = item["audio"].unsqueeze(0).float()
    B = audio.size(0)
    T = text_feats.size(1)
    audio_pooled = F.adaptive_avg_pool1d(audio.unsqueeze(1), T).transpose(1, 2)

    vision = item["vision"].unsqueeze(0).float()
    speaker = item.get("speaker_id")
    if speaker is None:
        speaker = torch.zeros(B, dtype=torch.long)
    else:
        speaker = speaker.unsqueeze(0)

    model = HMEMNet(audio_dim=1, text_dim=embed_dim, vision_dim=64, hidden_dim=64, num_speakers=16)
    model.eval()
    with torch.no_grad():
        logits, memory_weights, speech_attn, memories, temporal = model(audio_pooled, text_feats, vision, speaker, update=False, use_memories=False)

    probs = F.softmax(logits[:, 0, :], dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    conf = probs[0, pred].item()
    print("Predicted emotion:", pred)
    print("Confidence:", conf)
    print("Memory weights shape:", memory_weights.shape)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default=None)
    args = p.parse_args()
    main(processed_dir=args.processed_dir)
