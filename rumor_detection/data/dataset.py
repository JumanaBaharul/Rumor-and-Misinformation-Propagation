"""Utilities for loading and preparing the Twitter rumor detection dataset."""

from __future__ import annotations

import os
import random
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from ..config import DataConfig


class TwitterDatasetPreprocessor:
    """Parse the raw Twitter15/16 files and generate propagation graphs."""

    def __init__(self, dataset_path: str, dataset_name: str) -> None:
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.labels: Dict[str, str] = {}
        self.source_tweets: Dict[str, str] = {}
        self._load_raw_files()

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_raw_files(self) -> None:
        dataset_dir = os.path.join(self.dataset_path, self.dataset_name)

        label_file = os.path.join(dataset_dir, "label.txt")
        with open(label_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                label, tweet_id = line.split(":")
                self.labels[tweet_id] = label

        source_file = os.path.join(dataset_dir, "source_tweets.txt")
        with open(source_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                tweet_id = parts[0]
                tweet_text = "\t".join(parts[1:])
                self.source_tweets[tweet_id] = tweet_text

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_text(text: str) -> str:
        text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
        text = re.sub(r"@\w+", "USER", text)
        text = re.sub(r"#\w+", "HASHTAG", text)
        return text

    def extract_features(self, tweet_text: str) -> Dict[str, float]:
        cleaned = self._normalise_text(tweet_text)

        blob = TextBlob(cleaned)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        word_count = len(cleaned.split())
        char_count = len(cleaned)
        hashtag_count = len(re.findall(r"#\w+", tweet_text))
        mention_count = len(re.findall(r"@\w+", tweet_text))
        url_count = len(re.findall(r"http\S+|www\S+|https\S+", tweet_text))
        exclamation_count = tweet_text.count("!")
        question_count = tweet_text.count("?")
        emoji_count = len(re.findall(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", tweet_text))
        avg_word_length = float(np.mean([len(word) for word in cleaned.split()])) if cleaned.split() else 0.0
        capital_ratio = float(sum(1 for char in cleaned if char.isupper()) / len(cleaned)) if cleaned else 0.0

        return {
            "sentiment": sentiment,
            "subjectivity": subjectivity,
            "word_count": float(word_count),
            "char_count": float(char_count),
            "hashtag_count": float(hashtag_count),
            "mention_count": float(mention_count),
            "url_count": float(url_count),
            "exclamation_count": float(exclamation_count),
            "question_count": float(question_count),
            "emoji_count": float(emoji_count),
            "avg_word_length": avg_word_length,
            "capital_ratio": capital_ratio,
        }

    @staticmethod
    def create_temporal_features(tweet_id: str) -> Dict[str, float]:
        numeric_id = int(tweet_id) if tweet_id.isdigit() else hash(tweet_id) % 1_000_000
        hour_of_day = numeric_id % 24
        day_of_week = numeric_id % 7

        return {
            "hour_of_day": float(hour_of_day),
            "day_of_week": float(day_of_week),
            "month": float((numeric_id % 12) + 1),
            "time_bin": float(numeric_id % 6),
            "is_weekend": float(1 if day_of_week >= 5 else 0),
            "is_business_hours": float(1 if 9 <= hour_of_day <= 17 else 0),
            "timestamp": float(numeric_id),
        }

    def build_graph(self, tweet_id: str, base_features: Dict[str, float], temporal_features: Dict[str, float]) -> nx.DiGraph:
        graph = nx.DiGraph()
        node_features = {**base_features, **temporal_features, "type": "source", "level": 0}
        graph.add_node(tweet_id, **node_features)

        rng = np.random.default_rng(abs(hash(tweet_id)) % 2**32)
        num_replies = int(rng.integers(1, 8))

        for idx in range(num_replies):
            reply_id = f"{tweet_id}_reply_{idx}"
            reply_features = base_features.copy()
            reply_features["sentiment"] = float(rng.uniform(-1, 1))
            reply_features["word_count"] = float(rng.integers(5, 40))

            reply_temporal = temporal_features.copy()
            reply_temporal["timestamp"] = temporal_features["timestamp"] + idx + 1
            reply_temporal["hour_of_day"] = float((temporal_features["hour_of_day"] + idx) % 24)

            node_attributes = {**reply_features, **reply_temporal, "type": "reply", "level": 1}
            graph.add_node(reply_id, **node_attributes)
            graph.add_edge(tweet_id, reply_id, type="reply", timestamp=idx + 1)

        return graph

    # ------------------------------------------------------------------
    # Dataset level helpers
    # ------------------------------------------------------------------
    def get_label_mapping(self) -> Dict[str, int]:
        unique_labels = sorted(set(self.labels.values()))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def iter_processed_graphs(self, max_samples: Optional[int] = None, seed: int = 42) -> Iterable[Data]:
        label_mapping = self.get_label_mapping()
        items = sorted(self.labels.items())
        if max_samples is not None:
            rng = random.Random(seed)
            items = rng.sample(items, k=min(max_samples, len(items)))

        for tweet_id, label_name in items:
            if tweet_id not in self.source_tweets:
                continue

            base_features = self.extract_features(self.source_tweets[tweet_id])
            temporal_features = self.create_temporal_features(tweet_id)
            graph = self.build_graph(tweet_id, base_features, temporal_features)
            label = label_mapping[label_name]
            yield self._graph_to_data(graph, label)

    @staticmethod
    def _graph_to_data(graph: nx.DiGraph, label: int) -> Data:
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        node_features: List[List[float]] = []

        for node in graph.nodes():
            node_data = graph.nodes[node]
            features: List[float] = [
                float(node_data.get("level", 0)),
                float(node_data.get("timestamp", 0.0)),
                float(node_data.get("sentiment", 0.0)),
                float(node_data.get("subjectivity", 0.0)),
                float(node_data.get("word_count", 0.0)),
                float(node_data.get("char_count", 0.0)),
                float(node_data.get("hashtag_count", 0.0)),
                float(node_data.get("mention_count", 0.0)),
                float(node_data.get("url_count", 0.0)),
                float(node_data.get("exclamation_count", 0.0)),
                float(node_data.get("question_count", 0.0)),
                float(node_data.get("emoji_count", 0.0)),
                float(node_data.get("avg_word_length", 0.0)),
                float(node_data.get("capital_ratio", 0.0)),
                float(node_data.get("hour_of_day", 0.0)),
                float(node_data.get("day_of_week", 0.0)),
                float(node_data.get("month", 0.0)),
                float(node_data.get("time_bin", 0.0)),
                float(node_data.get("is_weekend", 0.0)),
                float(node_data.get("is_business_hours", 0.0)),
            ]

            if node_data.get("type") == "source":
                features.extend([1.0, 0.0])
            else:
                features.extend([0.0, 1.0])

            node_features.append(features)

        edge_pairs: List[List[int]] = []
        for src, dst in graph.edges():
            edge_pairs.append([node_mapping[src], node_mapping[dst]])

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


class TwitterRumorDataset:
    """Light-weight in-memory dataset wrapper around processed graphs."""

    def __init__(self, dataset_path: str, dataset_name: str, *, max_samples: Optional[int] = None, seed: int = 42) -> None:
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.seed = seed
        self._data_list: List[Data] = []
        self._labels: List[int] = []
        self._label_mapping: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        preprocessor = TwitterDatasetPreprocessor(self.dataset_path, self.dataset_name)
        self._label_mapping = preprocessor.get_label_mapping()
        for data in preprocessor.iter_processed_graphs(max_samples=self.max_samples, seed=self.seed):
            self._data_list.append(data)
            self._labels.append(int(data.y.item()))

    @property
    def labels(self) -> Sequence[int]:
        return self._labels

    @property
    def label_mapping(self) -> Dict[str, int]:
        return self._label_mapping

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]

    def __iter__(self) -> Iterable[Data]:
        return iter(self._data_list)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_twitter_dataset(config: DataConfig) -> TwitterRumorDataset:
    """Load the dataset according to the supplied configuration."""

    return TwitterRumorDataset(
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
        max_samples=config.max_samples,
        seed=config.seed,
    )


def _stratified_split(indices: np.ndarray, labels: np.ndarray, *, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("Train and validation ratios must sum to less than 1.0")

    label_counts = np.bincount(labels)
    use_stratify = np.all(label_counts >= 2)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=test_ratio + val_ratio,
        random_state=seed,
        stratify=labels if use_stratify else None,
    )

    val_relative = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_relative,
        random_state=seed,
        stratify=labels[temp_idx] if use_stratify else None,
    )

    return train_idx, val_idx, test_idx


def create_data_loaders(
    dataset: TwitterRumorDataset, config: DataConfig
) -> Tuple[GeometricDataLoader, GeometricDataLoader, GeometricDataLoader, Dict[str, np.ndarray]]:
    indices = np.arange(len(dataset))
    labels = np.asarray(dataset.labels)

    train_idx, val_idx, test_idx = _stratified_split(
        indices,
        labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_subset = [dataset[int(idx)] for idx in train_idx]
    val_subset = [dataset[int(idx)] for idx in val_idx]
    test_subset = [dataset[int(idx)] for idx in test_idx]

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = GeometricDataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = GeometricDataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = GeometricDataLoader(test_subset, shuffle=False, **loader_kwargs)
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    return train_loader, val_loader, test_loader, splits


def compute_class_weights(labels: Sequence[int], num_classes: Optional[int] = None) -> torch.Tensor:
    label_array = np.asarray(labels)
    inferred_classes = (label_array.max() + 1) if label_array.size > 0 else 0
    if num_classes is None:
        num_classes = inferred_classes
    counts = np.bincount(label_array, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(label_array) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def dataset_summary(dataset: TwitterRumorDataset) -> Dict[str, object]:
    inverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
    distribution = Counter(dataset.labels)
    label_distribution = {inverse_mapping[label]: int(count) for label, count in distribution.items()}
    num_features = dataset[0].x.size(-1) if len(dataset) > 0 else 0
    return {
        "num_samples": len(dataset),
        "num_features": int(num_features),
        "label_distribution": label_distribution,
    }


__all__ = [
    "TwitterDatasetPreprocessor",
    "TwitterRumorDataset",
    "load_twitter_dataset",
    "create_data_loaders",
    "compute_class_weights",
    "dataset_summary",
]
