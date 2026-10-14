import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from dsmr.models.hmem_net import HMEMNet
from dsmr.models.losses.emotion_loss import EmotionClassificationLoss
from dsmr.trainers.trainer import HMEMTrainer
from train import _sampling_weights


def test_trainer_writes_json_training_summary(tmp_path: Path):
    model = HMEMNet(audio_dim=8, text_dim=8, vision_dim=8, hidden_dim=8, num_speakers=2)
    trainer = HMEMTrainer(model=model, learning_rate=1e-3, device="cpu")

    trainer.training_history = [
        {"epoch": 1, "loss": 1.25, "dev_macro_f1": 0.42},
        {"epoch": 2, "loss": 1.10, "dev_macro_f1": 0.39},
    ]

    summary_path = trainer.save_training_summary(tmp_path / "training_results.json")

    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_epochs"] == 2
    assert payload["epochs"][0]["loss"] == 1.25
    assert payload["best_loss"] == 1.10
    assert payload["best_dev_epoch"] == 1
    assert payload["best_dev_macro_f1"] == 0.42
    assert not any(path.suffix == ".pth" for path in tmp_path.iterdir())


def test_emotion_loss_applies_class_weights():
    logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0]], [[4.0, 0.0], [4.0, 0.0]]])
    labels = torch.tensor([0, 1])

    unweighted = EmotionClassificationLoss()(logits, labels)
    weighted = EmotionClassificationLoss(class_weights=torch.tensor([1.0, 5.0]))(logits, labels)

    assert weighted > unweighted


def test_sampling_weights_temper_imbalance_and_respect_subsets():
    class TinyRows:
        column_names = ["Emotion"]
        emotions = ["neutral", "fear", "fear", "fear", "joy"]

        def __getitem__(self, key):
            if key == "Emotion":
                return self.emotions
            raise KeyError(key)

    class TinyMeldDataset:
        dataset = TinyRows()

        def __len__(self):
            return len(self.dataset["Emotion"])

    weights, counts = _sampling_weights(Subset(TinyMeldDataset(), [0, 1, 2, 3]))

    assert counts["neutral"] == 1
    assert counts["fear"] == 3
    assert weights[0] > weights[1]
    assert torch.isclose(weights[0] / weights[1], torch.tensor(3.0**0.5, dtype=torch.double))
