import os
import re
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import torch
from torch_geometric.data import Data, Dataset
import nltk
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TwitterDatasetPreprocessor:
    """Preprocessor for Twitter15/16 rumor detection datasets"""
    
    def __init__(self, dataset_path: str, dataset_name: str = "twitter15"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.labels = {}
        self.source_tweets = {}
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load labels and source tweets from dataset files"""
        dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
        
        # Load labels
        label_file = os.path.join(dataset_dir, "label.txt")
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label, tweet_id = line.split(':')
                    self.labels[tweet_id] = label
        
        # Load source tweets
        source_file = os.path.join(dataset_dir, "source_tweets.txt")
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        tweet_id = parts[0]
                        tweet_text = parts[1]
                        self.source_tweets[tweet_id] = tweet_text
        
        print(f"Loaded {len(self.labels)} labels and {len(self.source_tweets)} source tweets from {self.dataset_name}")
    
    def extract_features(self, tweet_text: str) -> Dict[str, Any]:
        """Extract comprehensive features from tweet text"""
        # Basic text preprocessing
        text = re.sub(r'http\S+|www\S+|https\S+', 'URL', tweet_text)
        text = re.sub(r'@\w+', 'USER', text)
        text = re.sub(r'#\w+', 'HASHTAG', text)
        
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Text length features
        word_count = len(text.split())
        char_count = len(text)
        
        # Social media features
        hashtag_count = len(re.findall(r'#\w+', tweet_text))
        mention_count = len(re.findall(r'@\w+', tweet_text))
        url_count = len(re.findall(r'http\S+|www\S+|https\S+', tweet_text))
        
        # Linguistic features
        exclamation_count = tweet_text.count('!')
        question_count = tweet_text.count('?')
        
        # Emoji detection
        emoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
        emoji_count = len(emoji_pattern.findall(tweet_text))
        
        # Text complexity features
        avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return {
            'text': text,
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'word_count': word_count,
            'char_count': char_count,
            'hashtag_count': hashtag_count,
            'mention_count': mention_count,
            'url_count': url_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'emoji_count': emoji_count,
            'avg_word_length': avg_word_length,
            'capital_ratio': capital_ratio
        }
    
    def create_temporal_features(self, tweet_id: str) -> Dict[str, Any]:
        """Create temporal features for the tweet"""
        # Simulate temporal features (in real scenario, you'd have actual timestamps)
        numeric_id = int(tweet_id) if tweet_id.isdigit() else hash(tweet_id) % 1000000
        
        temporal_features = {
            'hour_of_day': (numeric_id % 24),
            'day_of_week': (numeric_id % 7),
            'month': (numeric_id % 12) + 1,
            'timestamp': numeric_id,
            'time_bin': (numeric_id % 6),
            'is_weekend': 1 if (numeric_id % 7) >= 5 else 0,
            'is_business_hours': 1 if 9 <= (numeric_id % 24) <= 17 else 0
        }
        
        return temporal_features
    
    def build_propagation_graph(self, tweet_id: str, features: Dict, 
                               temporal_features: Dict) -> nx.DiGraph:
        """Build a propagation graph with temporal information"""
        G = nx.DiGraph()
        
        # Add source tweet as root
        # Merge features and temporal_features, with temporal_features taking precedence
        node_attributes = {**features, **temporal_features}
        node_attributes.update({
            'type': 'source',
            'level': 0
        })
        
        G.add_node(tweet_id, **node_attributes)
        
        # Simulate reply structure (in real scenario, you'd have actual reply data)
        num_replies = np.random.randint(1, 8)
        
        for i in range(num_replies):
            reply_id = f"{tweet_id}_reply_{i}"
            
            # Simulate reply features
            reply_features = features.copy()
            reply_features['sentiment'] = np.random.uniform(-1, 1)
            reply_features['word_count'] = np.random.randint(5, 40)
            
            # Simulate temporal features for replies
            reply_temporal = temporal_features.copy()
            reply_temporal['timestamp'] = temporal_features['timestamp'] + i + 1
            reply_temporal['hour_of_day'] = (temporal_features['hour_of_day'] + i) % 24
            
            # Add reply node
            # Merge features and temporal features, with temporal features taking precedence
            reply_node_attributes = {**reply_features, **reply_temporal}
            reply_node_attributes.update({
                'type': 'reply',
                'level': 1
            })
            
            G.add_node(reply_id, **reply_node_attributes)
            
            # Add edge from source to reply
            G.add_edge(tweet_id, reply_id, type='reply', timestamp=i+1)
        
        return G
    
    def graph_to_pytorch_geometric(self, graph: nx.DiGraph, 
                                  label: int) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        # Node features
        node_features = []
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        
        for node in graph.nodes():
            features = []
            node_data = graph.nodes[node]
            
            # Basic features
            features.extend([
                node_data.get('level', 0),
                node_data.get('timestamp', 0),
                node_data.get('sentiment', 0.0),
                node_data.get('subjectivity', 0.0),
                node_data.get('word_count', 0),
                node_data.get('char_count', 0),
                node_data.get('hashtag_count', 0),
                node_data.get('mention_count', 0),
                node_data.get('url_count', 0),
                node_data.get('exclamation_count', 0),
                node_data.get('question_count', 0),
                node_data.get('emoji_count', 0),
                node_data.get('avg_word_length', 0.0),
                node_data.get('capital_ratio', 0.0),
                node_data.get('hour_of_day', 0),
                node_data.get('day_of_week', 0),
                node_data.get('month', 1),
                node_data.get('time_bin', 0),
                node_data.get('is_weekend', 0),
                node_data.get('is_business_hours', 0)
            ])
            
            # Type encoding (one-hot)
            if node_data.get('type') == 'source':
                features.extend([1, 0])
            else:
                features.extend([0, 1])
            
            node_features.append(features)
        
        # Edge indices
        edge_index = []
        edge_attr = []
        
        for edge in graph.edges():
            source, target = edge
            edge_index.append([node_mapping[source], node_mapping[target]])
            
            # Edge type encoding
            edge_type = graph.edges[edge].get('type', 'reply')
            if edge_type == 'reply':
                edge_attr.append([1, 0])
            else:
                edge_attr.append([0, 1])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def process_dataset(self) -> List[Data]:
        """Process the entire dataset"""
        processed_data = []
        
        # Get label mapping
        label_mapping = self.get_label_mapping()
        
        print(f"Processing {len(self.labels)} tweets...")
        
        for i, (tweet_id, label_str) in enumerate(self.labels.items()):
            if tweet_id in self.source_tweets:
                # Get numerical label
                label = label_mapping[label_str]
                
                # Extract features
                tweet_text = self.source_tweets[tweet_id]
                features = self.extract_features(tweet_text)
                
                # Create temporal features
                temporal_features = self.create_temporal_features(tweet_id)
                
                # Build propagation graph
                graph = self.build_propagation_graph(tweet_id, features, temporal_features)
                
                # Convert to PyTorch Geometric format
                data = self.graph_to_pytorch_geometric(graph, label)
                processed_data.append(data)
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(self.labels)} tweets")
        
        print(f"Successfully processed {len(processed_data)} samples")
        return processed_data
    
    def get_label_mapping(self) -> Dict[str, int]:
        """Convert string labels to numerical indices"""
        unique_labels = sorted(set(self.labels.values()))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics"""
        label_counts = Counter(self.labels.values())
        
        # Calculate feature statistics
        all_sentiments = []
        all_word_counts = []
        all_hashtag_counts = []
        
        for tweet_id, text in self.source_tweets.items():
            features = self.extract_features(text)
            all_sentiments.append(features['sentiment'])
            all_word_counts.append(features['word_count'])
            all_hashtag_counts.append(features['hashtag_count'])
        
        return {
            'total_samples': len(self.labels),
            'label_distribution': dict(label_counts),
            'dataset_name': self.dataset_name,
            'feature_stats': {
                'sentiment': {
                    'mean': np.mean(all_sentiments),
                    'std': np.std(all_sentiments),
                    'min': np.min(all_sentiments),
                    'max': np.max(all_sentiments)
                },
                'word_count': {
                    'mean': np.mean(all_word_counts),
                    'std': np.std(all_word_counts),
                    'min': np.min(all_word_counts),
                    'max': np.max(all_word_counts)
                },
                'hashtag_count': {
                    'mean': np.mean(all_hashtag_counts),
                    'std': np.std(all_hashtag_counts),
                    'min': np.min(all_hashtag_counts),
                    'max': np.max(all_hashtag_counts)
                }
            }
        }

class TwitterDataset(Dataset):
    """PyTorch Geometric Dataset for Twitter rumor detection"""
    
    def __init__(self, dataset_path: str, dataset_name: str = "twitter15", 
                 transform=None, pre_transform=None):
        super(TwitterDataset, self).__init__(transform, pre_transform)
        
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_list = []
        
        # Load and process data
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Load and process the dataset"""
        preprocessor = TwitterDatasetPreprocessor(self.dataset_path, self.dataset_name)
        self.data_list = preprocessor.process_dataset()
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

def load_twitter_dataset(dataset_path: str, dataset_name: str = "twitter15"):
    """Convenience function to load Twitter dataset"""
    return TwitterDataset(dataset_path, dataset_name)

if __name__ == "__main__":
    # Example usage
    dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    
    # Load dataset
    print("Loading Twitter dataset...")
    dataset = load_twitter_dataset(dataset_path, "twitter15")
    
    # Get dataset statistics
    preprocessor = TwitterDatasetPreprocessor(dataset_path, "twitter15")
    stats = preprocessor.get_dataset_stats()
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Label distribution: {stats['label_distribution']}")
    
    print(f"\nFeature Statistics:")
    for feature, values in stats['feature_stats'].items():
        print(f"{feature}: {values}")
    
    # Example of processing a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample features shape: {sample.x.shape}")
        print(f"Sample edge index shape: {sample.edge_index.shape}")
        print(f"Sample label: {sample.y.item()}")
        print(f"Number of features per node: {sample.x.shape[1]}")
# import os
# import re
# import json
# import pickle
# from typing import Dict, List, Tuple, Optional, Any
# import pandas as pd
# import numpy as np
# import networkx as nx
# from collections import defaultdict, Counter
# import torch
# from torch_geometric.data import Data, Dataset
# import nltk
# from textblob import TextBlob
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# class TwitterDatasetPreprocessor:
#     """Preprocessor for Twitter15/16 rumor detection datasets"""
    
#     def __init__(self, dataset_path: str, dataset_name: str = "twitter15"):
#         self.dataset_path = dataset_path
#         self.dataset_name = dataset_name
#         self.labels = {}
#         self.source_tweets = {}
#         self.tree_dir = os.path.join(dataset_path, dataset_name, 'tree')
        
#         self._load_dataset()
        
#     def _load_dataset(self):
#         """Load labels and source tweets from dataset files"""
#         dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
        
#         # Load labels
#         label_file = os.path.join(dataset_dir, "label.txt")
#         with open(label_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     label, tweet_id = line.split(':')
#                     self.labels[tweet_id] = label
        
#         # Load source tweets
#         source_file = os.path.join(dataset_dir, "source_tweets.txt")
#         with open(source_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     parts = line.split('\t')
#                     if len(parts) >= 2:
#                         tweet_id = parts[0]
#                         tweet_text = '\t'.join(parts[1:])  # Handle tabs in text
#                         self.source_tweets[tweet_id] = tweet_text
        
#         print(f"Loaded {len(self.labels)} labels and {len(self.source_tweets)} source tweets from {self.dataset_name}")
    
#     def extract_features(self, tweet_text: str) -> Dict[str, Any]:
#         """Extract comprehensive features from tweet text"""
#         # Basic text preprocessing
#         text = re.sub(r'http\S+|www\S+|https\S+', 'URL', tweet_text)
#         text = re.sub(r'@\w+', 'USER', text)
#         text = re.sub(r'#\w+', 'HASHTAG', text)
        
#         # Sentiment analysis
#         blob = TextBlob(text)
#         sentiment = blob.sentiment.polarity
#         subjectivity = blob.sentiment.subjectivity
        
#         # Text length features
#         word_count = len(text.split())
#         char_count = len(text)
        
#         # Social media features
#         hashtag_count = len(re.findall(r'#\w+', tweet_text))
#         mention_count = len(re.findall(r'@\w+', tweet_text))
#         url_count = len(re.findall(r'http\S+|www\S+|https\S+', tweet_text))
        
#         # Linguistic features
#         exclamation_count = tweet_text.count('!')
#         question_count = tweet_text.count('?')
#         punctuation_ratio = sum(1 for c in text if c in '.,!?') / len(text) if len(text) > 0 else 0
        
#         # Emoji detection
#         emoji_pattern = re.compile("["
#                                   u"\U0001F600-\U0001F64F"  # emoticons
#                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                                   u"\U00002702-\U000027B0"
#                                   u"\U000024C2-\U0001F251"
#                                   "]+", flags=re.UNICODE)
#         emoji_count = len(emoji_pattern.findall(tweet_text))
        
#         # Text complexity features
#         avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
#         capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
#         return {
#             'text': text,
#             'sentiment': sentiment,
#             'subjectivity': subjectivity,
#             'word_count': word_count,
#             'char_count': char_count,
#             'hashtag_count': hashtag_count,
#             'mention_count': mention_count,
#             'url_count': url_count,
#             'exclamation_count': exclamation_count,
#             'question_count': question_count,
#             'punctuation_ratio': punctuation_ratio,
#             'emoji_count': emoji_count,
#             'avg_word_length': avg_word_length,
#             'capital_ratio': capital_ratio
#         }
    
#     def parse_timestamp(self, ts_str: str) -> float:
#         """Parse timestamp string to Unix time"""
#         try:
#             # Assume format like 'Wed Jul 23 12:34:56 +0000 2014' or Unix
#             if ts_str.isdigit():
#                 return float(ts_str)
#             return datetime.strptime(ts_str, '%a %b %d %H:%M:%S %z %Y').timestamp()
#         except:
#             return 0.0  # Fallback
    
#     def build_propagation_graph(self, tweet_id: str) -> nx.DiGraph:
#         """Build propagation graph from tree file"""
#         G = nx.DiGraph()
#         tree_file = os.path.join(self.tree_dir, f"{tweet_id}.txt")
        
#         if not os.path.exists(tree_file):
#             print(f"Warning: Tree file not found for {tweet_id}, using source only")
#             source_text = self.source_tweets.get(tweet_id, "")
#             features = self.extract_features(source_text)
#             ts = 0.0  # Default
#             G.add_node(tweet_id, **features, timestamp=ts, type='source', level=0)
#             return G
        
#         with open(tree_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
        
#         node_times = {}  # For normalization
#         for line in lines:
#             parts = line.strip().split('\t')
#             if len(parts) != 6:
#                 continue  # Skip malformed
#             parent_id, child_id, ts_parent_str, ts_child_str, text_parent, text_child = parts
            
#             ts_parent = self.parse_timestamp(ts_parent_str)
#             ts_child = self.parse_timestamp(ts_child_str)
            
#             # Add parent if not exists
#             if not G.has_node(parent_id):
#                 features = self.extract_features(text_parent)
#                 G.add_node(parent_id, **features, timestamp=ts_parent, type='reply' if parent_id != tweet_id else 'source', level=0 if parent_id == tweet_id else -1)
            
#             # Add child
#             features = self.extract_features(text_child)
#             G.add_node(child_id, **features, timestamp=ts_child, type='reply', level=-1)
            
#             # Add edge
#             G.add_edge(parent_id, child_id, timestamp_diff=ts_child - ts_parent)
            
#             node_times[parent_id] = ts_parent
#             node_times[child_id] = ts_child
        
#         # Set levels (BFS from source)
#         if G.has_node(tweet_id):
#             levels = nx.single_source_shortest_path_length(G, tweet_id)
#             for node, level in levels.items():
#                 G.nodes[node]['level'] = level
        
#         # Normalize timestamps relative to min (source)
#         min_ts = min(node_times.values()) if node_times else 0
#         max_ts = max(node_times.values()) if node_times else 1
#         for node in G.nodes():
#             ts = G.nodes[node].get('timestamp', min_ts)
#             norm_ts = (ts - min_ts) / (max_ts - min_ts + 1e-6)  # Avoid div0
#             G.nodes[node]['norm_timestamp'] = norm_ts
#             G.nodes[node]['hour_of_day'] = datetime.fromtimestamp(ts).hour if ts > 0 else 0
#             G.nodes[node]['day_of_week'] = datetime.fromtimestamp(ts).weekday() if ts > 0 else 0
#             G.nodes[node]['is_weekend'] = 1 if G.nodes[node]['day_of_week'] >= 5 else 0
        
#         return G
    
#     def graph_to_pytorch_geometric(self, graph: nx.DiGraph, label: int) -> Data:
#         """Convert NetworkX graph to PyTorch Geometric Data object"""
#         # Node features
#         node_features = []
#         node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        
#         for node in graph.nodes():
#             features = []
#             node_data = graph.nodes[node]
            
#             # Basic features (expanded)
#             features.extend([
#                 node_data.get('level', 0),
#                 node_data.get('norm_timestamp', 0.0),
#                 node_data.get('sentiment', 0.0),
#                 node_data.get('subjectivity', 0.0),
#                 node_data.get('word_count', 0),
#                 node_data.get('char_count', 0),
#                 node_data.get('hashtag_count', 0),
#                 node_data.get('mention_count', 0),
#                 node_data.get('url_count', 0),
#                 node_data.get('exclamation_count', 0),
#                 node_data.get('question_count', 0),
#                 node_data.get('punctuation_ratio', 0.0),
#                 node_data.get('emoji_count', 0),
#                 node_data.get('avg_word_length', 0.0),
#                 node_data.get('capital_ratio', 0.0),
#                 node_data.get('hour_of_day', 0),
#                 node_data.get('day_of_week', 0),
#                 node_data.get('is_weekend', 0)
#             ])
            
#             # Type encoding (one-hot: source, reply)
#             if node_data.get('type') == 'source':
#                 features.extend([1, 0])
#             else:
#                 features.extend([0, 1])
            
#             node_features.append(features)
        
#         # Edge indices and attr
#         edge_index = []
#         edge_attr = []
        
#         for source, target in graph.edges():
#             edge_index.append([node_mapping[source], node_mapping[target]])
#             ts_diff = graph.edges[(source, target)].get('timestamp_diff', 0.0)
#             edge_attr.append([ts_diff])  # Temporal diff as attr
        
#         # Convert to tensors
#         x = torch.tensor(node_features, dtype=torch.float)
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
#         edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)
#         y = torch.tensor([label], dtype=torch.long)
        
#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
#     def process_dataset(self) -> List[Data]:
#         """Process the entire dataset"""
#         processed_data = []
        
#         # Get label mapping
#         label_mapping = self.get_label_mapping()
        
#         print(f"Processing {len(self.labels)} tweets...")
        
#         for i, (tweet_id, label_str) in enumerate(self.labels.items()):
#             # Get numerical label
#             label = label_mapping[label_str]
            
#             # Build propagation graph from tree file
#             graph = self.build_propagation_graph(tweet_id)
            
#             # Convert to PyTorch Geometric format
#             data = self.graph_to_pytorch_geometric(graph, label)
#             processed_data.append(data)
            
#             # Progress update
#             if (i + 1) % 100 == 0:
#                 print(f"Processed {i + 1}/{len(self.labels)} tweets")
        
#         print(f"Successfully processed {len(processed_data)} samples")
#         return processed_data
    
#     def get_label_mapping(self) -> Dict[str, int]:
#         """Convert string labels to numerical indices"""
#         unique_labels = sorted(set(self.labels.values()))
#         return {label: idx for idx, label in enumerate(unique_labels)}
    
#     def get_dataset_stats(self) -> Dict:
#         """Get comprehensive dataset statistics"""
#         label_counts = Counter(self.labels.values())
        
#         # Calculate feature statistics
#         all_sentiments = []
#         all_word_counts = []
#         all_hashtag_counts = []
        
#         for tweet_id, text in self.source_tweets.items():
#             features = self.extract_features(text)
#             all_sentiments.append(features['sentiment'])
#             all_word_counts.append(features['word_count'])
#             all_hashtag_counts.append(features['hashtag_count'])
        
#         return {
#             'total_samples': len(self.labels),
#             'label_distribution': dict(label_counts),
#             'dataset_name': self.dataset_name,
#             'feature_stats': {
#                 'sentiment': {
#                     'mean': np.mean(all_sentiments),
#                     'std': np.std(all_sentiments),
#                     'min': np.min(all_sentiments),
#                     'max': np.max(all_sentiments)
#                 },
#                 'word_count': {
#                     'mean': np.mean(all_word_counts),
#                     'std': np.std(all_word_counts),
#                     'min': np.min(all_word_counts),
#                     'max': np.max(all_word_counts)
#                 },
#                 'hashtag_count': {
#                     'mean': np.mean(all_hashtag_counts),
#                     'std': np.std(all_hashtag_counts),
#                     'min': np.min(all_hashtag_counts),
#                     'max': np.max(all_hashtag_counts)
#                 }
#             }
#         }

# class TwitterDataset(Dataset):
#     """PyTorch Geometric Dataset for Twitter rumor detection"""
    
#     def __init__(self, dataset_path: str, dataset_name: str = "twitter15", 
#                  transform=None, pre_transform=None):
#         super(TwitterDataset, self).__init__(transform, pre_transform)
        
#         self.dataset_path = dataset_path
#         self.dataset_name = dataset_name
#         self.data_list = []
        
#         # Load and process data
#         self._load_and_process_data()
    
#     def _load_and_process_data(self):
#         """Load and process the dataset"""
#         preprocessor = TwitterDatasetPreprocessor(self.dataset_path, self.dataset_name)
#         self.data_list = preprocessor.process_dataset()
    
#     def len(self):
#         return len(self.data_list)
    
#     def get(self, idx):
#         return self.data_list[idx]

# def load_twitter_dataset(dataset_path: str, dataset_name: str = "twitter15"):
#     """Convenience function to load Twitter dataset"""
#     return TwitterDataset(dataset_path, dataset_name)

# if __name__ == "__main__":
#     # Example usage
#     dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    
#     # Load dataset
#     print("Loading Twitter dataset...")
#     dataset = load_twitter_dataset(dataset_path, "twitter15")
    
#     # Get dataset statistics
#     preprocessor = TwitterDatasetPreprocessor(dataset_path, "twitter15")
#     stats = preprocessor.get_dataset_stats()
    
#     print(f"\nDataset Statistics:")
#     print(f"Total samples: {stats['total_samples']}")
#     print(f"Label distribution: {stats['label_distribution']}")
    
#     print(f"\nFeature Statistics:")
#     for feature, values in stats['feature_stats'].items():
#         print(f"{feature}: {values}")
    
#     # Example of processing a sample
#     if len(dataset) > 0:
#         sample = dataset[0]
#         print(f"\nSample features shape: {sample.x.shape}")
#         print(f"Sample edge index shape: {sample.edge_index.shape}")
#         print(f"Sample label: {sample.y.item()}")
#         print(f"Number of features per node: {sample.x.shape[1]}")