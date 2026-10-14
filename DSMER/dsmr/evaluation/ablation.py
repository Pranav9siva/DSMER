from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AblationConfig:
    """Configuration for ablation studies on HMEM-Net."""

    name: str
    use_person_memory: bool = True
    use_dialogue_memory: bool = True
    use_conversation_memory: bool = True
    use_memory_controller: bool = True
    use_memory_fusion: bool = True
    use_memory_update: bool = True


class AblationStudy:
    """Generate a list of ablation settings for research experiments."""

    def __init__(self) -> None:
        self.configs: List[AblationConfig] = []

    def build_default_configs(self) -> List[AblationConfig]:
        base = AblationConfig(name="full_model")
        self.configs = [
            base,
            AblationConfig(name="no_person_memory", use_person_memory=False),
            AblationConfig(name="no_dialogue_memory", use_dialogue_memory=False),
            AblationConfig(name="no_conversation_memory", use_conversation_memory=False),
            AblationConfig(name="no_memory_controller", use_memory_controller=False),
            AblationConfig(name="no_memory_fusion", use_memory_fusion=False),
            AblationConfig(name="no_memory_update", use_memory_update=False),
        ]
        return self.configs
