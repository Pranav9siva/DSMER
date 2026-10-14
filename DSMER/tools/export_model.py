"""Export a trained HMEMNet model to TorchScript and ONNX for deployment.

Usage:
    python tools/export_model.py --checkpoint path/to/ckpt.pth --outdir exports/
"""
from pathlib import Path
import argparse
import torch
import sys
import os

# Ensure DSMER package dir is on sys.path when running this script directly
DSMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DSMER_DIR not in sys.path:
    sys.path.insert(0, DSMER_DIR)

from dsmr.models import HMEMNet


def export_model(checkpoint: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ckpt_state = {}
    audio_dim = 1
    text_dim = 768
    vision_dim = 64
    hidden_dim = 256
    num_speakers = 100

    if checkpoint.exists():
        data = torch.load(str(checkpoint), map_location=device)
        if isinstance(data, dict) and "model_state" in data:
            ckpt_state = data["model_state"]
        elif isinstance(data, dict):
            ckpt_state = data

        if ckpt_state:
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

    B, T = 1, 8
    vision_c, vision_h, vision_w = 3, 224, 224

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
        filtered = {}
        skipped = []
        for k, v in ckpt_state.items():
            if k not in model_state and f"model.{k}" in model_state:
                k = f"model.{k}"
            if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if filtered:
            model.load_state_dict(filtered, strict=False)
            if skipped:
                print("Warning: skipped incompatible or missing keys:", skipped[:10])
        else:
            print("Warning: no compatible state keys found in checkpoint; skipping load")

    model.eval()

    audio = torch.randn(B, T, audio_dim, device=device)
    text = torch.randn(B, T, text_dim, device=device)
    vision = torch.randn(B, vision_c, vision_h, vision_w, device=device)
    speaker_ids = torch.tensor([0], device=device)

    try:
        scripted = torch.jit.script(model)
        scripted.save(str(outdir / "hmemnet_ts.pt"))
        print("Saved TorchScript at", outdir / "hmemnet_ts.pt")
    except Exception as exc:
        print("TorchScript export failed:", exc)

    try:
        torch.onnx.export(model, (audio, text, vision, speaker_ids), str(outdir / "hmemnet.onnx"), opset_version=18, input_names=["audio", "text", "vision", "speaker_ids"], output_names=["logits"])
        print("Saved ONNX at", outdir / "hmemnet.onnx")
    except Exception as exc:
        print("ONNX export failed:", exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="exports")
    args = parser.parse_args()
    export_model(Path(args.checkpoint), Path(args.outdir))


if __name__ == "__main__":
    main()
