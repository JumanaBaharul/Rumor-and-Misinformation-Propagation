"""Utility functions for evaluating rumor detection models."""

from __future__ import annotations

from typing import Dict, List, Sequence
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def build_detailed_report(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, object]:
    cm = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    metrics = compute_classification_metrics(labels, predictions)
    metrics.update({
        "confusion_matrix": cm,
        "classification_report": report,
    })
    return metrics


def evaluate_model(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())

    predictions = torch.cat(all_logits, dim=0).argmax(dim=1)
    labels = torch.cat(all_labels, dim=0)
    metrics = build_detailed_report(labels.tolist(), predictions.tolist())
    metrics["predictions"] = predictions.tolist()
    metrics["labels"] = labels.tolist()
    return metrics


__all__ = [
    "compute_classification_metrics",
    "build_detailed_report",
    "evaluate_model",
]
