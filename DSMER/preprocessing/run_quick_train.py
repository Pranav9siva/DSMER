from __future__ import annotations

import sys
import os
from pathlib import Path

DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dsmr.data.meld_dataset import MELDConversationDataset
from dsmr.models.hmem_net import HMEMNet
from transformers import AutoTokenizer


def collate_fn(batch):
    # default collate should work; keep placeholder
    return {k: torch.stack([b[k] for b in batch]) if isinstance(batch[0][k], torch.Tensor) else [b[k] for b in batch] for k in batch[0]}


def main(processed_dir: str | None = None, bs: int = 2, epochs: int = 1):
    device = torch.device("cpu")
    ds = MELDConversationDataset(processed_dir=processed_dir)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    # tokenizer to know vocab size for embedding
    tokenizer = AutoTokenizer.from_pretrained(ds.tokenizer_name)
    embed_dim = 64
    embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)

    model = HMEMNet(audio_dim=1, text_dim=embed_dim, vision_dim=64, hidden_dim=64, num_speakers=16)
    model.to(device)

    opt = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            # prepare inputs
            input_ids = batch["text_input_ids"].to(device)
            text_feats = embedding(input_ids)
            # audio: [B, N] -> [B, T, 1]
            audio_wave = batch["audio"].to(device).float()
            B = audio_wave.size(0)
            T = text_feats.size(1)
            audio_wave = audio_wave.view(B, -1)
            audio_pooled = F.adaptive_avg_pool1d(audio_wave.unsqueeze(1), T).transpose(1, 2)

            vision = batch["vision"].to(device).float()
            speaker = batch.get("speaker_id")
            if speaker is None:
                speaker = torch.zeros(B, dtype=torch.long)
            else:
                speaker = speaker.to(device)

            labels = batch["label"].to(device).long()

            opt.zero_grad()
            logits, mem_w, speech_attn, memories, temporal = model(audio_pooled, text_feats, vision, speaker)
            # use first timestep logits for per-utterance label
            loss = criterion(logits[:, 0, :], labels)
            loss.backward()
            opt.step()

            print(f"Epoch {epoch} Batch {i} Loss {loss.item():.4f}")
            if i >= 4:
                break


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default=None)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    main(processed_dir=args.processed_dir, bs=args.bs, epochs=args.epochs)
