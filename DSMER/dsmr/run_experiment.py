from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsmr.data import MELDConversationDataset
from dsmr.models import HMEMNet
from dsmr.trainers import HMEMTrainer
from dsmr.utils.config import ExperimentConfig
from dsmr.utils.seed import set_seed
import time


LOG_PATH = PROJECT_ROOT / "dsmr" / "run_experiment.log"


def _log(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
    except Exception:
        pass


def main() -> None:
    config_path = PROJECT_ROOT / "dsmr" / "configs" / "experiment.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    _log(f"Loaded config: {config_path}")
    set_seed(config.seed)
    _log(f"Seed set to {config.seed}")

    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    dataset = MELDConversationDataset(csv_path=data_dir / "MELD" / "train_sent_emo.csv")
    _log(f"Loaded dataset with {len(dataset)} items")
    # optional small subset for fast debug/fine-tune runs
    subset_size = getattr(config, "debug_subset_size", 0) or 0
    if subset_size and subset_size > 0:
        from torch.utils.data import Subset

        subset_size = min(len(dataset), int(subset_size))
        dataset = Subset(dataset, list(range(subset_size)))

    loader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    _log(f"Created DataLoader with batch_size={config.batch_size}")

    log_dir = Path(config.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    model = HMEMNet(
        audio_dim=config.audio_dim,
        text_dim=config.text_dim,
        vision_dim=config.vision_dim,
        hidden_dim=config.hidden_dim,
        num_speakers=config.num_speakers,
        pretrained_audio=config.pretrained_audio,
        pretrained_text=config.pretrained_text,
        pretrained_vision=config.pretrained_vision,
        fine_tune=config.fine_tune,
    )
    _log("HMEMNet instantiated")
    trainer = HMEMTrainer(model=model, learning_rate=config.learning_rate, device=config.device, log_dir=str(log_dir), use_wandb=getattr(config, "use_wandb", False))
    _log("Trainer created")
    history = []
    for epoch in range(config.epochs):
        _log(f"Starting epoch {epoch+1}/{config.epochs}")
        try:
            trainer.fit(loader, epochs=1)
        except Exception as exc:
            _log(f"Trainer.fit failed: {exc}")
            raise

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_results.json"
    trainer.save_training_summary(summary_path)
    print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()
