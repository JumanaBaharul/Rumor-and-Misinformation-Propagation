#!/usr/bin/env python3
"""Train the enhanced rumor detection models using the refactored package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from rumor_detection import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    build_model,
    compute_class_weights,
    create_data_loaders,
    dataset_summary,
    evaluate_model,
    load_twitter_dataset,
    normalise_node_features,
    save_json,
    set_global_seed,
)
from rumor_detection.training import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhanced rumor detection models")
    parser.add_argument("--dataset-path", default="data/Rumor Detection Dataset (Twitter15 and Twitter16)")
    parser.add_argument("--dataset-name", default="twitter15")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--no-feature-normalisation",
        dest="normalise_features",
        action="store_false",
        help="Disable z-score normalisation applied to node features.",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        dest="use_balanced_sampler",
        action="store_false",
        help="Disable the class-balanced sampler for the training loader.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["enhanced_tgnn", "improved_transformer_gnn", "advanced_rvnn"],
        help="List of model identifiers to train",
    )
    parser.set_defaults(normalise_features=True, use_balanced_sampler=True)
    return parser.parse_args()


def serialise_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    serialised: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serialised[key] = value.tolist()
        elif torch.is_tensor(value):
            serialised[key] = value.cpu().tolist()
        else:
            serialised[key] = value
    return serialised


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    data_config = DataConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
        normalise_features=args.normalise_features,
        use_balanced_sampler=args.use_balanced_sampler,
    )
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_heads=args.heads,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        device=args.device,
        log_every=max(1, args.log_every),
    )

    print("🚀 Loading dataset...")
    dataset = load_twitter_dataset(data_config)
    summary = dataset_summary(dataset)
    print(f"Loaded {summary['num_samples']} samples with {summary['num_features']} features")
    print(f"Label distribution: {summary['label_distribution']}")

    feature_stats_payload = None
    if data_config.normalise_features:
        print("🧮 Applying z-score normalisation to node features")
        raw_stats = normalise_node_features(dataset)
        mean_abs = raw_stats["mean"].abs().mean().item()
        std_mean = raw_stats["std"].mean().item()
        post_stack = torch.cat([data.x for data in dataset], dim=0)
        post_mean_abs = post_stack.abs().mean().item()
        post_std_mean = post_stack.std(dim=0).mean().item()
        print(
            "   → Pre-normalisation |mean|: "
            f"{mean_abs:.4e}, avg std: {std_mean:.4e}; "
            "post-normalisation |mean|: "
            f"{post_mean_abs:.4f}, avg std: {post_std_mean:.4f}"
        )
        feature_stats_payload = {
            "mean": raw_stats["mean"],
            "std": raw_stats["std"],
            "pre_mean_abs": mean_abs,
            "pre_std_mean": std_mean,
            "post_mean_abs": post_mean_abs,
            "post_std_mean": post_std_mean,
        }

    print("📦 Creating data loaders...")
    train_loader, val_loader, test_loader, splits = create_data_loaders(dataset, data_config)
    input_size = dataset[0].x.size(-1)
    num_classes = len(dataset.label_mapping)
    train_labels = [dataset.labels[int(idx)] for idx in splits["train"]]
    class_weights = compute_class_weights(train_labels, num_classes=num_classes)
    device = training_config.resolve_device()
    print(f"Using device: {device}")

    results: Dict[str, Dict[str, object]] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}
    summary_rows: Dict[str, Dict[str, float]] = {}
    best_model_name = None
    best_f1 = -float("inf")

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    result_dir = output_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n{'=' * 70}\nTraining {model_name}\n{'=' * 70}")
        model = build_model(model_name, input_size, num_classes, model_config)
        trainer = ModelTrainer(
            model,
            training_config=training_config,
            class_weights=class_weights,
        )

        history = trainer.fit(train_loader, val_loader)
        histories[model_name] = history

        metrics = evaluate_model(trainer.model, test_loader, device)
        results[model_name] = serialise_metrics(metrics)
        summary_rows[model_name] = {
            key: float(metrics[key]) if key in metrics else None
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            )
        }

        torch.save(trainer.model.state_dict(), model_dir / f"{model_name}.pt")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = model_name

        print(
            f"Results for {model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )
        print("Confusion matrix:")
        print(np.asarray(metrics["confusion_matrix"]))

    experiment_record = {
        "dataset": summary,
        "data_config": data_config.to_dict(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "results": results,
        "histories": histories,
        "best_model": {
            "name": best_model_name,
            "f1": best_f1,
        },
    }

    if feature_stats_payload is not None:
        experiment_record["feature_normalisation"] = {
            key: value.tolist() if torch.is_tensor(value) else value
            for key, value in feature_stats_payload.items()
        }

    save_json(experiment_record, result_dir / "enhanced_models_results.json")

    if summary_rows:
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        header = (
            f"{'Model':<28}"
            f"{'Acc.':>8}"
            f"{'Prec.':>8}"
            f"{'Recall':>8}"
            f"{'F1':>8}"
            f"{'mPrec.':>8}"
            f"{'mRecall':>9}"
            f"{'mF1':>8}"
        )
        print(header)
        print("-" * len(header))
        
        def format_value(value: float | None, width: int) -> str:
            return f"{value:>{width}.4f}" if value is not None else f"{'n/a':>{width}}"

        for model_name, metrics in summary_rows.items():
            print(
                f"{model_name:<28}"
                f"{format_value(metrics['accuracy'], 8)}"
                f"{format_value(metrics['precision'], 8)}"
                f"{format_value(metrics['recall'], 8)}"
                f"{format_value(metrics['f1'], 8)}"
                f"{format_value(metrics['macro_precision'], 8)}"
                f"{format_value(metrics['macro_recall'], 9)}"
                f"{format_value(metrics['macro_f1'], 8)}"
            )

    if best_model_name is not None:
        print(f"\n🏆 Best model: {best_model_name} (F1={best_f1:.4f})")
    print("✅ Training completed!")
    print(f"📁 Results saved to {result_dir / 'enhanced_models_results.json'}")


if __name__ == "__main__":
    main()
