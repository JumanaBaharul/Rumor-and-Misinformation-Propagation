"""Configuration dataclasses for rumor detection experiments."""

from __future__ import annotations

"""Configuration dataclasses used across the rumor detection package."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

import torch


@dataclass
class DataConfig:
    """Configuration for dataset loading and data loader construction."""

    dataset_path: str = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    dataset_name: str = "twitter15"
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    max_samples: Optional[int] = None
    seed: int = 42
    num_workers: int = 0
    normalise_features: bool = True
    use_balanced_sampler: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration shared by the enhanced model architectures."""

    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    attention_heads: int = 4
    pooling: str = "mean_max_std"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for optimisation and training loops."""

    epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 5e-3
    patience: int = 12
    grad_clip: float = 1.0
    device: str = "auto"
    log_every: int = 10
    label_smoothing: float = 0.05
    edge_dropout: float = 0.15
    feature_dropout: float = 0.1
    use_swa: bool = True
    swa_start: int = 30

    def resolve_device(self) -> torch.device:
        """Return the torch device specified by the configuration."""

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Bundle all configuration sections together."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
]
