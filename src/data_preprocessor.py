import os
import re
import json
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from textblob import TextBlob
from torch_geometric.data import Data, Dataset


class TwitterDatasetPreprocessor:
    """Preprocessor that reconstructs real rumor cascades for Twitter15/16."""

    TWITTER_DATETIME_FORMATS = (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    )

    def __init__(self, dataset_path: str, dataset_name: str = "twitter15"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_path, dataset_name)

        self.labels: Dict[str, str] = {}
        self.source_text_fallback: Dict[str, str] = {}
        self.structure_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.tweet_cache: Dict[str, Dict[str, Any]] = {}

        # Node and edge feature definitions ensure consistent tensor ordering
        self.node_feature_keys: List[str] = [
            "depth",
            "temporal_order",
            "time_since_root_seconds",
            "time_since_root_hours",
            "time_since_parent_seconds",
            "time_since_parent_hours",
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "has_timestamp",
            "sentiment",
            "subjectivity",
            "word_count",
            "char_count",
            "hashtag_count",
            "mention_count",
            "url_count",
            "exclamation_count",
            "question_count",
            "emoji_count",
            "avg_word_length",
            "capital_ratio",
            "user_followers",
            "user_friends",
            "user_statuses",
            "user_verified",
            "in_degree",
            "out_degree",
            "cascade_size",
        ]
        self.edge_feature_keys: List[str] = [
            "type_reply",
            "type_retweet",
            "type_quote",
            "time_delay_hours",
        ]

        self._load_dataset()

    # ------------------------------------------------------------------
    # Dataset loading helpers
    # ------------------------------------------------------------------
    def _load_dataset(self) -> None:
        """Load labels and fallback source tweet text from the dataset."""
        label_path = os.path.join(self.dataset_dir, "label.txt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Could not locate label file at {label_path}. Ensure the Twitter15/16 dataset is available."
            )

        with open(label_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if ":" in line:
                    label, tweet_id = line.split(":", 1)
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    label, tweet_id = parts[0], parts[1]
                self.labels[tweet_id] = label

        # Optional text fallback file
        fallback_path = os.path.join(self.dataset_dir, "source_tweets.txt")
        if os.path.exists(fallback_path):
            with open(fallback_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        tweet_id, text = line.split("\t", 1)
                    elif ":" in line:
                        tweet_id, text = line.split(":", 1)
                    else:
                        parts = line.split(maxsplit=1)
                        if len(parts) != 2:
                            continue
                        tweet_id, text = parts
                    self.source_text_fallback[tweet_id] = text.strip()

    # ------------------------------------------------------------------
    # Tweet and structure loading
    # ------------------------------------------------------------------
    def _read_json(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        for fmt in self.TWITTER_DATETIME_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _normalise_tweet_record(
        self,
        tweet_id: str,
        payload: Optional[Dict[str, Any]],
        fallback_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if payload is None and fallback_text is None:
            return None

        text = ""
        created_at = None
        user = {}
        tweet_type = None

        if payload:
            text = payload.get("text") or payload.get("content") or ""
            created_at = payload.get("created_at") or payload.get("timestamp")
            user = payload.get("user") or {}
            tweet_type = payload.get("type") or payload.get("tweet_type")

        if not text and fallback_text:
            text = fallback_text

        text = text or ""
        created_at_dt = self._parse_datetime(created_at)

        return {
            "id": tweet_id,
            "text": text,
            "created_at": created_at,
            "created_at_dt": created_at_dt,
            "user": user if isinstance(user, dict) else {},
            "tweet_type": tweet_type,
        }

    def _get_tweet_record(self, tweet_id: str, is_source: bool = False) -> Optional[Dict[str, Any]]:
        if tweet_id in self.tweet_cache:
            return self.tweet_cache[tweet_id]

        directories = []
        if is_source:
            directories.extend(
                [
                    os.path.join(self.dataset_dir, "source_tweets"),
                    os.path.join(self.dataset_dir, "source-tweets"),
                ]
            )
        directories.extend(
            [
                os.path.join(self.dataset_dir, "tweets"),
                os.path.join(self.dataset_dir, "reactions"),
            ]
        )

        payload = None
        for directory in directories:
            if not os.path.isdir(directory):
                continue
            for extension in (".json", ".JSON"):
                candidate = os.path.join(directory, f"{tweet_id}{extension}")
                payload = self._read_json(candidate)
                if payload is not None:
                    break
            if payload is not None:
                break

        fallback_text = self.source_text_fallback.get(tweet_id) if is_source else None
        record = self._normalise_tweet_record(tweet_id, payload, fallback_text)
        if record is None:
            return None

        self.tweet_cache[tweet_id] = record
        return record

    def _parse_structure_lines(self, lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            parent_id: Optional[str] = None
            child_id: Optional[str] = None
            edge_type = "reply"

            # JSON-style payload
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    parent_id = str(payload.get("parent") or payload.get("source") or "").strip()
                    child_id = str(payload.get("child") or payload.get("target") or "").strip()
                    edge_type = str(payload.get("type") or "reply").lower()
                except json.JSONDecodeError:
                    pass
            elif line.startswith("[") and line.endswith("]"):
                try:
                    payload = json.loads(line)
                    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                        parent_id = str(payload[0]).strip()
                        child_id = str(payload[1]).strip()
                        if len(payload) >= 3 and isinstance(payload[2], str):
                            edge_type = payload[2].lower()
                except json.JSONDecodeError:
                    pass

            if parent_id is None or child_id is None:
                tokens = re.findall(r"\d+", line)
                if len(tokens) >= 2:
                    parent_id, child_id = tokens[0], tokens[1]
                else:
                    continue

            if not parent_id or not child_id:
                continue

            if "retweet" in line.lower():
                edge_type = "retweet"
            elif "quote" in line.lower():
                edge_type = "quote"

            adjacency[parent_id].append({"child": child_id, "type": edge_type})

        return adjacency

    def _load_structure(self, tweet_id: str) -> Dict[str, List[Dict[str, Any]]]:
        if tweet_id in self.structure_cache:
            return self.structure_cache[tweet_id]

        adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        structure_dir = os.path.join(self.dataset_dir, "structure")

        candidates: List[str] = []
        if os.path.isdir(structure_dir):
            for extension in (".json", ".txt", ".tree"):
                path = os.path.join(structure_dir, f"{tweet_id}{extension}")
                if os.path.exists(path):
                    candidates.append(path)

        global_structure_file = os.path.join(self.dataset_dir, "structure.txt")
        if not candidates and os.path.exists(global_structure_file):
            candidates.append(global_structure_file)

        for path in candidates:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()

            parsed = self._parse_structure_lines(lines)
            for parent, edges in parsed.items():
                for edge in edges:
                    adjacency[parent].append(edge)

        self.structure_cache[tweet_id] = adjacency
        return adjacency

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def extract_text_features(self, text: str) -> Dict[str, float]:
        cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "URL", text or "")
        cleaned_text = re.sub(r"@\w+", "USER", cleaned_text)
        cleaned_text = re.sub(r"#\w+", "HASHTAG", cleaned_text)

        blob = TextBlob(cleaned_text)
        sentiment = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)

        word_count = float(len(cleaned_text.split()))
        char_count = float(len(cleaned_text))
        hashtag_count = float(len(re.findall(r"#\w+", text or "")))
        mention_count = float(len(re.findall(r"@\w+", text or "")))
        url_count = float(len(re.findall(r"http\S+|www\S+|https\S+", text or "")))
        exclamation_count = float(text.count("!"))
        question_count = float(text.count("?"))
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        emoji_count = float(len(emoji_pattern.findall(text or "")))
        avg_word_length = float(np.mean([len(word) for word in cleaned_text.split()]) if cleaned_text.split() else 0.0)
        capital_ratio = float(sum(1 for c in cleaned_text if c.isupper()) / len(cleaned_text)) if cleaned_text else 0.0

        return {
            "sentiment": sentiment,
            "subjectivity": subjectivity,
            "word_count": word_count,
            "char_count": char_count,
            "hashtag_count": hashtag_count,
            "mention_count": mention_count,
            "url_count": url_count,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "emoji_count": emoji_count,
            "avg_word_length": avg_word_length,
            "capital_ratio": capital_ratio,
        }

    def extract_user_features(self, user: Dict[str, Any]) -> Dict[str, float]:
        followers = float(user.get("followers_count", 0) or 0)
        friends = float(user.get("friends_count", 0) or 0)
        statuses = float(user.get("statuses_count", 0) or 0)
        verified = 1.0 if user.get("verified", False) else 0.0
        return {
            "user_followers": followers,
            "user_friends": friends,
            "user_statuses": statuses,
            "user_verified": verified,
        }

    def _time_delta_seconds(
        self, current_time: Optional[datetime], reference_time: Optional[datetime]
    ) -> Optional[float]:
        if current_time is None or reference_time is None:
            return None
        delta = (current_time - reference_time).total_seconds()
        return float(delta)

    def create_temporal_features(
        self,
        current_time: Optional[datetime],
        root_time: Optional[datetime],
        parent_time: Optional[datetime],
    ) -> Dict[str, float]:
        time_since_root_seconds = self._time_delta_seconds(current_time, root_time) or 0.0
        time_since_parent_seconds = self._time_delta_seconds(current_time, parent_time) or 0.0

        if current_time is not None:
            hour_of_day = float(current_time.hour)
            day_of_week = float(current_time.weekday())
            day_of_month = float(current_time.day)
            has_timestamp = 1.0
        else:
            hour_of_day = 0.0
            day_of_week = 0.0
            day_of_month = 0.0
            has_timestamp = 0.0

        return {
            "time_since_root_seconds": time_since_root_seconds,
            "time_since_root_hours": time_since_root_seconds / 3600.0,
            "time_since_parent_seconds": time_since_parent_seconds,
            "time_since_parent_hours": time_since_parent_seconds / 3600.0,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "has_timestamp": has_timestamp,
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_propagation_graph(self, tweet_id: str) -> Optional[nx.DiGraph]:
        source_record = self._get_tweet_record(tweet_id, is_source=True)
        if source_record is None:
            return None

        adjacency = self._load_structure(tweet_id)
        graph = nx.DiGraph()
        root_time = source_record.get("created_at_dt")

        root_features = {}
        root_features.update(self.extract_text_features(source_record.get("text", "")))
        root_features.update(self.extract_user_features(source_record.get("user", {})))
        root_features.update(
            self.create_temporal_features(root_time, root_time, root_time)
        )
        root_features.update(
            {
                "depth": 0.0,
                "temporal_order": 0.0,
                "cascade_size": 1.0,
                "in_degree": 0.0,
                "out_degree": 0.0,
                "type": "source",
                "text": source_record.get("text", ""),
            }
        )

        graph.add_node(tweet_id, **root_features)

        queue: deque[Tuple[str, Optional[datetime], int]] = deque()
        queue.append((tweet_id, root_time, 0))
        visited = {tweet_id}

        chronological: List[Tuple[str, float]] = [(tweet_id, 0.0)]

        while queue:
            parent_id, parent_time, depth = queue.popleft()
            children = adjacency.get(parent_id, [])
            for child_info in children:
                child_id = child_info.get("child")
                if not child_id or child_id in visited:
                    continue

                child_record = self._get_tweet_record(child_id)
                if child_record is None:
                    continue

                child_time = child_record.get("created_at_dt")
                text_features = self.extract_text_features(child_record.get("text", ""))
                user_features = self.extract_user_features(child_record.get("user", {}))
                temporal_features = self.create_temporal_features(
                    child_time, root_time, parent_time
                )

                node_attributes = {}
                node_attributes.update(text_features)
                node_attributes.update(user_features)
                node_attributes.update(temporal_features)
                node_attributes.update(
                    {
                        "depth": float(depth + 1),
                        "type": child_info.get("type", "reply"),
                        "text": child_record.get("text", ""),
                    }
                )

                graph.add_node(child_id, **node_attributes)

                time_delay_seconds = self._time_delta_seconds(child_time, parent_time) or 0.0
                edge_type = child_info.get("type", "reply").lower()
                graph.add_edge(
                    parent_id,
                    child_id,
                    type=edge_type,
                    time_delay_seconds=time_delay_seconds,
                    time_delay_hours=time_delay_seconds / 3600.0,
                )

                visited.add(child_id)
                queue.append((child_id, child_time, depth + 1))
                chronological.append(
                    (
                        child_id,
                        self._time_delta_seconds(child_time, root_time) or len(chronological) + 1,
                    )
                )

        # Update cascade-level attributes now that full graph is built
        cascade_size = float(graph.number_of_nodes())
        # Temporal ordering
        chronological.sort(key=lambda item: item[1])
        for order, (node_id, _) in enumerate(chronological):
            if node_id in graph.nodes:
                graph.nodes[node_id]["temporal_order"] = float(order)
                graph.nodes[node_id]["cascade_size"] = cascade_size

        for node_id in graph.nodes:
            graph.nodes[node_id]["in_degree"] = float(graph.in_degree(node_id))
            graph.nodes[node_id]["out_degree"] = float(graph.out_degree(node_id))

        return graph

    # ------------------------------------------------------------------
    # Conversion to PyTorch Geometric
    # ------------------------------------------------------------------
    def graph_to_pytorch_geometric(self, graph: nx.DiGraph, label: int) -> Data:
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        node_features: List[List[float]] = []

        for node in graph.nodes:
            attributes = graph.nodes[node]
            feature_vector: List[float] = []
            for key in self.node_feature_keys:
                feature_vector.append(float(attributes.get(key, 0.0)))

            node_features.append(feature_vector)

        edge_index: List[List[int]] = []
        edge_attr: List[List[float]] = []

        for source, target in graph.edges:
            edge_index.append([node_mapping[source], node_mapping[target]])
            edge_data = graph.edges[(source, target)]
            edge_type = edge_data.get("type", "reply").lower()
            edge_features = [
                1.0 if edge_type == "reply" else 0.0,
                1.0 if edge_type == "retweet" else 0.0,
                1.0 if edge_type == "quote" else 0.0,
                float(edge_data.get("time_delay_hours", 0.0)),
            ]
            edge_attr.append(edge_features)

        x = torch.tensor(node_features, dtype=torch.float)
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, len(self.edge_feature_keys)), dtype=torch.float)

        y = torch.tensor([label], dtype=torch.long)

        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_label_mapping(self) -> Dict[str, int]:
        unique_labels = sorted(set(self.labels.values()))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def process_dataset(self) -> List[Data]:
        processed: List[Data] = []
        label_mapping = self.get_label_mapping()

        for tweet_id, label_str in self.labels.items():
            graph = self.build_propagation_graph(tweet_id)
            if graph is None or graph.number_of_nodes() == 0:
                continue
            numeric_label = label_mapping[label_str]
            processed.append(self.graph_to_pytorch_geometric(graph, numeric_label))

        return processed

    def get_dataset_stats(self) -> Dict[str, Any]:
        label_counts = Counter(self.labels.values())
        sentiments: List[float] = []
        word_counts: List[float] = []
        hashtag_counts: List[float] = []

        for tweet_id in self.labels:
            record = self._get_tweet_record(tweet_id, is_source=True)
            if record is None:
                continue
            features = self.extract_text_features(record.get("text", ""))
            sentiments.append(features["sentiment"])
            word_counts.append(features["word_count"])
            hashtag_counts.append(features["hashtag_count"])

        def _aggregate(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "total_samples": len(self.labels),
            "label_distribution": dict(label_counts),
            "dataset_name": self.dataset_name,
            "feature_stats": {
                "sentiment": _aggregate(sentiments),
                "word_count": _aggregate(word_counts),
                "hashtag_count": _aggregate(hashtag_counts),
            },
        }


class TwitterDataset(Dataset):
    """PyTorch Geometric dataset for rumor propagation cascades."""

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "twitter15",
        transform=None,
        pre_transform=None,
    ):
        super().__init__(transform, pre_transform)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_list: List[Data] = []
        self.preprocessor = TwitterDatasetPreprocessor(dataset_path, dataset_name)
        self.data_list = self.preprocessor.process_dataset()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def load_twitter_dataset(dataset_path: str, dataset_name: str = "twitter15") -> TwitterDataset:
    return TwitterDataset(dataset_path, dataset_name)


if __name__ == "__main__":
    dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    dataset_name = "twitter15"

    print("Loading Twitter dataset...")
    dataset = load_twitter_dataset(dataset_path, dataset_name)

    print(f"Loaded {len(dataset)} cascades")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample node feature shape: {sample.x.shape}")
        print(f"Sample edge index shape: {sample.edge_index.shape}")
        print(f"Sample edge attr shape: {sample.edge_attr.shape}")
        print(f"Label: {sample.y.item()}")

    stats = dataset.preprocessor.get_dataset_stats()
    print("Dataset statistics:")
    print(stats)
