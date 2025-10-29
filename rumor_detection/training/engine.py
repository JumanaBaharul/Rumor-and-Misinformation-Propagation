"""Training utilities for rumor detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..config import TrainingConfig
from ..evaluation.metrics import compute_classification_metrics


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]


class ModelTrainer:
    """Utility class encapsulating the training loop for a single model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        training_config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.training_config = training_config
        self.device = training_config.resolve_device()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=max(2, training_config.patience // 3),
            factor=0.5,
        )

    def fit(self, train_loader, val_loader) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "val_macro_f1": [],
        }

        best_macro_f1 = -float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.training_config.epochs + 1):
            train_result = self._run_epoch(train_loader, train=True)
            val_result = self._run_epoch(val_loader, train=False)

            history["train_loss"].append(train_result.loss)
            history["val_loss"].append(val_result.loss)
            history["train_f1"].append(train_result.metrics["f1"])
            history["val_f1"].append(val_result.metrics["f1"])
            history["val_macro_f1"].append(val_result.metrics["macro_f1"])

            if (epoch == 1) or (epoch % self.training_config.log_every == 0):
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss: {train_result.loss:.4f} | "
                    f"val_loss: {val_result.loss:.4f} | "
                    f"val_f1: {val_result.metrics['f1']:.4f} | "
                    f"val_macro_f1: {val_result.metrics['macro_f1']:.4f}"
                )

            self.scheduler.step(val_result.metrics["macro_f1"])

            if val_result.metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_result.metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.training_config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def _run_epoch(self, loader, *, train: bool) -> EpochResult:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_graphs = 0
        all_predictions: List[int] = []
        all_labels: List[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)
                self.optimizer.step()

            batch_size = batch.y.size(0)
            total_loss += loss.item() * batch_size
            total_graphs += batch_size

            preds = logits.argmax(dim=1)
            all_predictions.extend(preds.detach().cpu().tolist())
            all_labels.extend(batch.y.detach().cpu().tolist())

        avg_loss = total_loss / max(1, total_graphs)
        metrics = compute_classification_metrics(all_labels, all_predictions)
        return EpochResult(loss=avg_loss, metrics=metrics)

    def evaluate(self, loader) -> Dict[str, float]:
        result = self._run_epoch(loader, train=False)
        return {"loss": result.loss, **result.metrics}


__all__ = ["ModelTrainer", "EpochResult"]
