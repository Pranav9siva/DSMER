from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dsmr.data.meld_dataset import MELDConversationDataset
from dsmr.evaluation.ablation import AblationConfig, AblationStudy
from dsmr.models import HMEMNet
from dsmr.trainers.trainer import HMEMTrainer
from dsmr.utils.config import ExperimentConfig


class NoopMemoryController(nn.Module):
    def forward(self, context, person_memory, dialogue_memory, conversation_memory):
        return torch.ones(context.size(0), 3, device=context.device, dtype=context.dtype) / 3.0


class NoopMemoryFusion(nn.Module):
    def forward(self, context, person_memory, dialogue_memory, conversation_memory, memory_weights=None):
        return context


class NoopMemoryUpdate(nn.Module):
    def forward(self, current_context, person_memory, dialogue_memory, conversation_memory):
        return person_memory, dialogue_memory, conversation_memory


class AblationRunner:
    def __init__(self, config_path: str | Path, device: str | None = None) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            self.config_path = Path(__file__).resolve().parents[2] / "dsmr" / "configs" / "experiment.yaml"
        self.config = ExperimentConfig.from_yaml(self.config_path)
        self.device = device or self.config.device
        self.results: List[Dict[str, Any]] = []

    def _resolve_dataset_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(self.config.data_dir)
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
        return data_dir / "MELD" / "train_sent_emo.csv"

    def _build_dataloader(self) -> DataLoader:
        csv_path = self._resolve_dataset_path()
        dataset = MELDConversationDataset(csv_path=csv_path, max_len=32)
        subset_size = self.config.debug_subset_size or 64
        if subset_size > 0 and subset_size < len(dataset):
            dataset = Subset(dataset, list(range(subset_size)))
        return DataLoader(dataset, batch_size=min(self.config.batch_size, 4), shuffle=True)

    def _evaluate(self, model: HMEMNet, dataloader: DataLoader) -> Dict[str, float]:
        model.eval()
        preds: List[int] = []
        targets: List[int] = []
        with torch.no_grad():
            for batch in dataloader:
                audio_features = batch["audio_features"].to(self.device)
                text_features = batch["text_features"].to(self.device)
                vision_features = batch["vision_features"].to(self.device)
                speaker_ids = batch["speaker_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits, _, _, _, _ = model(audio_features, text_features, vision_features, speaker_ids)
                logits = logits.mean(dim=1)
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
                if len(preds) >= 32:
                    break
        acc = sum(int(p == t) for p, t in zip(preds, targets)) / max(len(preds), 1)
        return {"accuracy": acc}

    def _run_one(self, ablation: AblationConfig) -> Dict[str, Any]:
        torch.manual_seed(42)
        model = HMEMNet(
            audio_dim=self.config.audio_dim,
            text_dim=self.config.text_dim,
            vision_dim=self.config.vision_dim,
            hidden_dim=self.config.hidden_dim,
            num_speakers=self.config.num_speakers,
            pretrained_audio=self.config.pretrained_audio,
            pretrained_text=self.config.pretrained_text,
            pretrained_vision=self.config.pretrained_vision,
            fine_tune=self.config.fine_tune,
        )
        model.memory_manager.person_memory.update_rate = 0.1 if ablation.use_person_memory else 0.0
        model.memory_manager.dialogue_memory.max_turns = 8
        model.memory_manager.conversation_memory.max_steps = 8

        if not ablation.use_person_memory:
            model.memory_manager.person_memory.reset_memory()
        if not ablation.use_dialogue_memory:
            model.memory_manager.dialogue_memory.buffer.zero_()
        if not ablation.use_conversation_memory:
            model.memory_manager.conversation_memory.reset_memory()

        if not ablation.use_memory_controller:
            model.memory_controller = NoopMemoryController()
        if not ablation.use_memory_fusion:
            model.memory_fusion = NoopMemoryFusion()
        if not ablation.use_memory_update:
            model.memory_update = NoopMemoryUpdate()

        trainer = HMEMTrainer(model=model, learning_rate=self.config.learning_rate, device=self.device, log_dir=None)
        dataloader = self._build_dataloader()
        start = time.time()
        trainer.fit(dataloader, epochs=1)
        elapsed = time.time() - start
        metrics = self._evaluate(model, dataloader)
        return {"name": ablation.name, "elapsed_s": elapsed, **metrics}

    def run(self) -> List[Dict[str, Any]]:
        study = AblationStudy()
        configs = study.build_default_configs()
        for config in configs:
            result = self._run_one(config)
            self.results.append(result)
        return self.results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMEM-Net ablation study")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "dsmr" / "configs" / "experiment.yaml"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "ablation_results.json"))
    args = parser.parse_args()

    runner = AblationRunner(args.config)
    results = runner.run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
