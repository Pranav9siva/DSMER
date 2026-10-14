from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    """Configuration object for structured HMEM-Net experiments."""

    name: str = "hmem_net_baseline"
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    log_every_n_steps: int = 10
    hidden_dim: int = 256
    num_speakers: int = 100
    audio_dim: int = 1024
    text_dim: int = 768
    vision_dim: int = 2048
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 20
    weight_decay: float = 1e-4
    debug_subset_size: int = 0
    fine_tune: bool = False
    pretrained_audio: str | None = None
    pretrained_text: str | None = None
    pretrained_vision: str | None = None
    use_wandb: bool = False
    use_person_memory: bool = True
    use_dialogue_memory: bool = True
    use_conversation_memory: bool = True
    use_memory_controller: bool = True
    use_memory_fusion: bool = True
    use_memory_update: bool = True
    data_dir: str = "dataset"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        exp_cfg = payload.get("experiment", {})
        model_cfg = payload.get("model", {})
        training_cfg = payload.get("training", {})
        ablation_cfg = payload.get("ablation", {})
        paths_cfg = payload.get("paths", {})
        return cls(
            name=exp_cfg.get("name", cls.name),
            seed=exp_cfg.get("seed", cls.seed),
            device=exp_cfg.get("device", cls.device),
            mixed_precision=exp_cfg.get("mixed_precision", cls.mixed_precision),
            gradient_accumulation_steps=exp_cfg.get("gradient_accumulation_steps", cls.gradient_accumulation_steps),
            early_stopping_patience=exp_cfg.get("early_stopping_patience", cls.early_stopping_patience),
            log_every_n_steps=exp_cfg.get("log_every_n_steps", cls.log_every_n_steps),
            hidden_dim=model_cfg.get("hidden_dim", cls.hidden_dim),
            num_speakers=model_cfg.get("num_speakers", cls.num_speakers),
            audio_dim=model_cfg.get("audio_dim", cls.audio_dim),
            text_dim=model_cfg.get("text_dim", cls.text_dim),
            vision_dim=model_cfg.get("vision_dim", cls.vision_dim),
            batch_size=training_cfg.get("batch_size", cls.batch_size),
            learning_rate=training_cfg.get("learning_rate", cls.learning_rate),
            epochs=training_cfg.get("epochs", cls.epochs),
            weight_decay=training_cfg.get("weight_decay", cls.weight_decay),
            debug_subset_size=training_cfg.get("debug_subset_size", cls.debug_subset_size),
            fine_tune=payload.get("fine_tune", cls.fine_tune),
            pretrained_audio=payload.get("pretrained_audio", cls.pretrained_audio),
            pretrained_text=payload.get("pretrained_text", cls.pretrained_text),
            pretrained_vision=payload.get("pretrained_vision", cls.pretrained_vision),
            use_wandb=payload.get("use_wandb", cls.use_wandb),
            use_person_memory=ablation_cfg.get("use_person_memory", cls.use_person_memory),
            use_dialogue_memory=ablation_cfg.get("use_dialogue_memory", cls.use_dialogue_memory),
            use_conversation_memory=ablation_cfg.get("use_conversation_memory", cls.use_conversation_memory),
            use_memory_controller=ablation_cfg.get("use_memory_controller", cls.use_memory_controller),
            use_memory_fusion=ablation_cfg.get("use_memory_fusion", cls.use_memory_fusion),
            use_memory_update=ablation_cfg.get("use_memory_update", cls.use_memory_update),
            data_dir=paths_cfg.get("data_dir", cls.data_dir),
            output_dir=paths_cfg.get("output_dir", cls.output_dir),
            log_dir=paths_cfg.get("log_dir", cls.log_dir),
        )
