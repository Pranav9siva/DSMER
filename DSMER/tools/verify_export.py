from __future__ import annotations

import sys
import os
from pathlib import Path

DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

import torch
from dsmr.models.hmem_net import HMEMNet


def load_model_from_checkpoint(checkpoint: Path):
    device = torch.device("cpu")
    ckpt_state = {}
    audio_dim = 1
    text_dim = 768
    vision_dim = 64
    hidden_dim = 256
    num_speakers = 100

    if checkpoint.exists():
        data = torch.load(str(checkpoint), map_location=device)
        if isinstance(data, dict):
            ckpt_state = data.get("model_state", data)
            for key, value in ckpt_state.items():
                if not hasattr(value, "shape"):
                    continue
                if key == "audio_encoder.proj.weight":
                    audio_dim = int(value.shape[1])
                elif key == "text_encoder.proj.weight":
                    text_dim = int(value.shape[1])
                elif key == "vision_encoder.proj.weight":
                    vision_dim = int(value.shape[1])
                elif key == "memory_manager.person_memory.memory_bank":
                    hidden_dim = int(value.shape[-1])
                    num_speakers = int(value.shape[0])

    class ExportableHMEMNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = HMEMNet(audio_dim=audio_dim, text_dim=text_dim, vision_dim=vision_dim, hidden_dim=hidden_dim, num_speakers=num_speakers)

        def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor, vision_features: torch.Tensor, speaker_ids: torch.Tensor):
            logits, *_ = self.model(audio_features, text_features, vision_features, speaker_ids, update=False, use_memories=False)
            return logits

    model = ExportableHMEMNet()
    if ckpt_state:
        model_state = model.state_dict()
        filtered = {k: v for k, v in ckpt_state.items() if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape}
        if filtered:
            model.load_state_dict(filtered, strict=False)
    return model


def main():
    outdir = Path("DSMER/exports")
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = Path("outputs/checkpoint_epoch_2.pth")
    model = load_model_from_checkpoint(ckpt)
    model.eval()

    B, T = 1, 8
    audio = torch.randn(B, T, 1)
    text = torch.randn(B, T, 768)
    vision = torch.randn(B, 3, 224, 224)
    speaker = torch.tensor([0])

    ts_path = outdir / "hmemnet_ts.pt"
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(ts_path))
        print("TorchScript saved to", ts_path)
    except Exception as e:
        print("TorchScript scripting/saving failed:", e)
        return

    try:
        loaded = torch.jit.load(str(ts_path))
        out = loaded(audio, text, vision, speaker)
        print("Loaded TorchScript OK. Output shape:", tuple(out.shape))
    except Exception as e:
        print("Loading TorchScript failed:", e)


if __name__ == "__main__":
    main()
