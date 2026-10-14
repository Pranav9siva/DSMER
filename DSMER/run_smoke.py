"""Run a single-batch forward/backward pass and checkpoint save/load."""

from pathlib import Path
import torch
from dsmr.models.hmem_net import HMEMNet
from dsmr.trainers.trainer import HMEMTrainer

root = Path(__file__).resolve().parent


def run():
    B, T = 2, 8
    audio_dim = 1024
    text_dim = 768
    vision_input_channels = 3
    vision_h, vision_w = 224, 224
    hidden = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio = torch.randn(B, T, audio_dim, device=device)
    text = torch.randn(B, T, text_dim, device=device)
    # use realistic image tensor [B, C, H, W]
    vision = torch.randn(B, vision_input_channels, vision_h, vision_w, device=device)
    speaker_ids = torch.tensor([0, 1], device=device)
    labels = torch.randint(0, 7, (B, T), device=device)

    model = HMEMNet(audio_dim=audio_dim, text_dim=text_dim, vision_dim=2048, hidden_dim=hidden, num_speakers=4)
    trainer = HMEMTrainer(model=model, learning_rate=1e-4, device=device)

    batch = {
        "audio_features": audio,
        "text_features": text,
        "vision_features": vision,
        "speaker_ids": speaker_ids,
        "labels": labels,
    }

    loss = trainer.train_step(batch)
    print("Loss:", loss.item())

    ckpt = root / "smoke_ckpt.pth"
    trainer.save_checkpoint(str(ckpt), epoch=0)
    data = trainer.load_checkpoint(str(ckpt), map_location="cpu")
    print("Checkpoint saved and loaded; epoch:", data.get("epoch"))


if __name__ == '__main__':
    run()
