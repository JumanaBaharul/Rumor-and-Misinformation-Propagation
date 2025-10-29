"""High-level imports for the rumor detection package."""

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from .data import (
    TwitterDatasetPreprocessor,
    TwitterRumorDataset,
    compute_class_weights,
    create_data_loaders,
    dataset_summary,
    load_twitter_dataset,
    normalise_node_features,
)
from .evaluation import build_detailed_report, compute_classification_metrics, evaluate_model
from .models import MODEL_REGISTRY, AdvancedRvNN, EnhancedTGNN, ImprovedTransformerGNN, build_model
from .training import EpochResult, ModelTrainer
from .utils import ensure_dir, save_json, set_global_seed

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "TwitterDatasetPreprocessor",
    "TwitterRumorDataset",
    "compute_class_weights",
    "create_data_loaders",
    "dataset_summary",
    "load_twitter_dataset",
    "normalise_node_features",
    "build_detailed_report",
    "compute_classification_metrics",
    "evaluate_model",
    "MODEL_REGISTRY",
    "AdvancedRvNN",
    "EnhancedTGNN",
    "ImprovedTransformerGNN",
    "build_model",
    "EpochResult",
    "ModelTrainer",
    "ensure_dir",
    "save_json",
    "set_global_seed",
]
