from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch
from torch.utils.data import DataLoader

from dsmr.data import MELDConversationDataset
from dsmr.evaluation import classification_report
from dsmr.models import HMEMNet
from dsmr.utils.config import ExperimentConfig
from dsmr.utils.seed import set_seed


def evaluate(config_path: str | Path, checkpoint_path: str | None = None) -> None:
    project_root = Path(__file__).resolve().parent
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    dataset = MELDConversationDataset(csv_path=data_dir / "MELD" / "test_sent_emo.csv")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    model = HMEMNet(
        audio_dim=config.audio_dim,
        text_dim=config.text_dim,
        vision_dim=config.vision_dim,
        hidden_dim=config.hidden_dim,
        num_speakers=config.num_speakers,
    )
    model.to(device)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if ckpt_path.exists():
            data = torch.load(str(ckpt_path), map_location=device)
            # support both wrapped checkpoint and raw state dict
            if isinstance(data, dict) and "model_state" in data:
                model.load_state_dict(data["model_state"])
                loaded_epoch = data.get("epoch")
                print(f"Loaded checkpoint {ckpt_path} (epoch={loaded_epoch})")
            elif isinstance(data, dict):
                try:
                    model.load_state_dict(data)
                    print(f"Loaded raw model state from {ckpt_path}")
                except Exception as exc:
                    print(f"Failed to load model state: {exc}")
            else:
                print(f"Unrecognized checkpoint format: {ckpt_path}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            audio_features = batch["audio_features"]
            text_features = batch["text_features"]
            vision_features = batch["vision_features"]
            speaker_ids = batch["speaker_ids"]
            labels = batch["labels"]

            logits, _, _, _, _ = model(audio_features, text_features, vision_features, speaker_ids)
            predictions.append(logits.cpu())
            targets.append(labels.cpu())

    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    report = classification_report(predictions_tensor, targets_tensor)

    print("Evaluation results:")
    for metric_name, metric_value in report.items():
        print(f"  {metric_name}: {metric_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to experiment yaml", default=str(Path(__file__).resolve().parent / "dsmr" / "configs" / "experiment.yaml"))
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (optional)", default=None)
    args = parser.parse_args()
    evaluate(args.config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
