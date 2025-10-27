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
import warnings
warnings.filterwarnings('ignore')

class TemporalGraphBuilder:
    """Advanced temporal graph builder for rumor propagation networks"""
    
    def __init__(self, dataset_path: str, dataset_name: str = "twitter15"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.temporal_graphs = {}
        self.temporal_snapshots = {}
        
    def build_temporal_propagation_graph(self, tweet_id: str, features: Dict, 
                                        temporal_features: Dict, 
                                        propagation_depth: int = 3) -> nx.DiGraph:
        """Build a comprehensive temporal propagation graph"""
        G = nx.DiGraph()
        
        # Add source tweet as root
        # Merge features and temporal_features, with temporal_features taking precedence
        node_attributes = {**features, **temporal_features}
        node_attributes.update({
            'type': 'source',
            'level': 0
        })
        
        G.add_node(tweet_id, **node_attributes)
        
        # Build propagation tree with temporal constraints
        self._build_propagation_tree(G, tweet_id, features, temporal_features, 
                                    propagation_depth, current_level=0)
        
        return G
    
    def _build_propagation_tree(self, G: nx.DiGraph, parent_id: str, 
                               parent_features: Dict, parent_temporal: Dict,
                               max_depth: int, current_level: int):
        """Recursively build propagation tree with temporal constraints"""
        if current_level >= max_depth:
            return
        
        # Simulate reply structure with temporal progression
        num_replies = np.random.randint(1, 6)  # Fewer replies at deeper levels
        
        for i in range(num_replies):
            reply_id = f"{parent_id}_reply_{i}_level_{current_level}"
            
            # Create temporal progression
            time_offset = (current_level + 1) * np.random.randint(1, 4)
            reply_temporal = self._create_temporal_progression(parent_temporal, time_offset)
            
            # Create reply features with propagation effects
            reply_features = self._create_propagated_features(parent_features, current_level)
            
            # Add reply node
            # Merge features and temporal features, with temporal features taking precedence
            reply_node_attributes = {**reply_features, **reply_temporal}
            reply_node_attributes.update({
                'type': 'reply',
                'level': current_level + 1
            })
            
            G.add_node(reply_id, **reply_node_attributes)
            
            # Add edge with temporal attributes
            G.add_edge(parent_id, reply_id, 
                      type='reply', 
                      timestamp=reply_temporal['timestamp'],
                      time_delay=time_offset,
                      propagation_level=current_level)
            
            # Recursively build deeper levels
            self._build_propagation_tree(G, reply_id, reply_features, reply_temporal,
                                       max_depth, current_level + 1)
    
    def _create_temporal_progression(self, parent_temporal: Dict, time_offset: int) -> Dict:
        """Create temporal features for propagated content"""
        new_temporal = parent_temporal.copy()
        
        # Progressive time advancement
        new_temporal['timestamp'] = parent_temporal['timestamp'] + time_offset
        new_temporal['hour_of_day'] = (parent_temporal['hour_of_day'] + time_offset) % 24
        new_temporal['day_of_week'] = (parent_temporal['day_of_week'] + (time_offset // 24)) % 7
        
        # Time bin progression
        new_temporal['time_bin'] = (parent_temporal['time_bin'] + 1) % 6
        
        # Update temporal indicators
        new_temporal['is_weekend'] = 1 if new_temporal['day_of_week'] >= 5 else 0
        new_temporal['is_business_hours'] = 1 if 9 <= new_temporal['hour_of_day'] <= 17 else 0
        
        return new_temporal
    
    def _create_propagated_features(self, parent_features: Dict, level: int) -> Dict:
        """Create features for propagated content with level-based modifications"""
        propagated_features = parent_features.copy()
        
        # Sentiment evolution (can become more extreme or moderate)
        sentiment_change = np.random.normal(0, 0.2)
        propagated_features['sentiment'] = np.clip(
            parent_features['sentiment'] + sentiment_change, -1, 1
        )
        
        # Content length variation
        length_factor = 0.8 + (level * 0.1)  # Slightly shorter at deeper levels
        propagated_features['word_count'] = max(3, int(parent_features['word_count'] * length_factor))
        propagated_features['char_count'] = max(10, int(parent_features['char_count'] * length_factor))
        
        # Engagement features (typically decrease with depth)
        engagement_decay = 0.9 ** level
        propagated_features['hashtag_count'] = max(0, int(parent_features['hashtag_count'] * engagement_decay))
        propagated_features['mention_count'] = max(0, int(parent_features['mention_count'] * engagement_decay))
        
        # Linguistic features
        propagated_features['exclamation_count'] = max(0, int(parent_features['exclamation_count'] * engagement_decay))
        propagated_features['question_count'] = max(0, int(parent_features['question_count'] * engagement_decay))
        
        return propagated_features
    
    def create_temporal_snapshots(self, graph: nx.DiGraph, 
                                 time_window: int = 2, 
                                 overlap: float = 0.5) -> List[nx.DiGraph]:
        """Create overlapping temporal snapshots of the propagation graph"""
        snapshots = []
        
        # Get all timestamps
        timestamps = sorted(set(nx.get_node_attributes(graph, 'timestamp').values()))
        
        if len(timestamps) <= time_window:
            # If graph is small, return the whole graph
            return [graph]
        
        # Create overlapping windows
        step_size = max(1, int(time_window * (1 - overlap)))
        
        for i in range(0, len(timestamps) - time_window + 1, step_size):
            window_timestamps = timestamps[i:i+time_window]
            
            # Create snapshot for this time window
            snapshot = nx.DiGraph()
            
            # Add nodes within the window
            for node in graph.nodes():
                node_timestamp = graph.nodes[node]['timestamp']
                if node_timestamp in window_timestamps:
                    # Copy node attributes
                    snapshot.add_node(node, **graph.nodes[node])
            
            # Add edges within the window
            for edge in graph.edges():
                source, target = edge
                if (source in snapshot.nodes and target in snapshot.nodes):
                    # Copy edge attributes
                    snapshot.add_edge(source, target, **graph.edges[edge])
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def extract_temporal_patterns(self, graph: nx.DiGraph) -> Dict:
        """Extract temporal patterns from the propagation graph"""
        patterns = {}
        
        # Get all timestamps
        timestamps = list(nx.get_node_attributes(graph, 'timestamp').values())
        
        if not timestamps:
            return patterns
        
        # Temporal distribution analysis
        patterns['temporal_span'] = max(timestamps) - min(timestamps)
        patterns['temporal_density'] = len(timestamps) / patterns['temporal_span'] if patterns['temporal_span'] > 0 else 0
        
        # Propagation speed analysis
        levels = nx.get_node_attributes(graph, 'level')
        level_timestamps = defaultdict(list)
        
        for node, level in levels.items():
            timestamp = graph.nodes[node]['timestamp']
            level_timestamps[level].append(timestamp)
        
        # Calculate propagation speed between levels
        propagation_speeds = []
        for level in sorted(level_timestamps.keys())[1:]:
            prev_level = level - 1
            if prev_level in level_timestamps:
                prev_timestamps = level_timestamps[prev_level]
                curr_timestamps = level_timestamps[level]
                
                if prev_timestamps and curr_timestamps:
                    avg_prev_time = np.mean(prev_timestamps)
                    avg_curr_time = np.mean(curr_timestamps)
                    speed = avg_curr_time - avg_prev_time
                    propagation_speeds.append(speed)
        
        patterns['avg_propagation_speed'] = np.mean(propagation_speeds) if propagation_speeds else 0
        patterns['propagation_speed_std'] = np.std(propagation_speeds) if propagation_speeds else 0
        
        # Temporal clustering analysis
        hour_distribution = Counter([t % 24 for t in timestamps])
        patterns['peak_hour'] = max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else 0
        patterns['hour_entropy'] = self._calculate_entropy(list(hour_distribution.values()))
        
        return patterns
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution"""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy
    
    def create_temporal_features_tensor(self, graph: nx.DiGraph) -> torch.Tensor:
        """Create temporal features tensor for the graph"""
        num_nodes = len(graph.nodes())
        temporal_features = torch.zeros(num_nodes, 8)  # 8 temporal features
        
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        
        for node, idx in node_mapping.items():
            node_data = graph.nodes[node]
            
            # Extract temporal features
            temporal_features[idx, 0] = node_data.get('timestamp', 0)
            temporal_features[idx, 1] = node_data.get('hour_of_day', 0)
            temporal_features[idx, 2] = node_data.get('day_of_week', 0)
            temporal_features[idx, 3] = node_data.get('month', 1)
            temporal_features[idx, 4] = node_data.get('time_bin', 0)
            temporal_features[idx, 5] = node_data.get('is_weekend', 0)
            temporal_features[idx, 6] = node_data.get('is_business_hours', 0)
            temporal_features[idx, 7] = node_data.get('level', 0)
        
        return temporal_features
    
    def build_comprehensive_temporal_graph(self, tweet_id: str, features: Dict, 
                                         temporal_features: Dict) -> Tuple[nx.DiGraph, Dict]:
        """Build comprehensive temporal graph with analysis"""
        # Build the main propagation graph
        graph = self.build_temporal_propagation_graph(tweet_id, features, temporal_features)
        
        # Create temporal snapshots
        snapshots = self.create_temporal_snapshots(graph)
        
        # Extract temporal patterns
        patterns = self.extract_temporal_patterns(graph)
        
        # Create temporal features tensor
        temporal_tensor = self.create_temporal_features_tensor(graph)
        
        return graph, {
            'snapshots': snapshots,
            'patterns': patterns,
            'temporal_tensor': temporal_tensor,
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'max_depth': max(nx.get_node_attributes(graph, 'level').values()) if graph.nodes() else 0
        }

def main():
    """Example usage of temporal graph builder"""
    dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    
    # Create temporal graph builder
    builder = TemporalGraphBuilder(dataset_path, "twitter15")
    
    # Example features and temporal data
    example_features = {
        'sentiment': 0.2,
        'word_count': 15,
        'hashtag_count': 2,
        'mention_count': 1
    }
    
    example_temporal = {
        'timestamp': 1000,
        'hour_of_day': 14,
        'day_of_week': 2,
        'month': 6
    }
    
    # Build comprehensive temporal graph
    graph, analysis = builder.build_comprehensive_temporal_graph(
        "example_tweet_123", example_features, example_temporal
    )
    
    print(f"Built temporal graph with {analysis['num_nodes']} nodes and {analysis['num_edges']} edges")
    print(f"Maximum propagation depth: {analysis['max_depth']}")
    print(f"Temporal span: {analysis['patterns']['temporal_span']}")
    print(f"Number of snapshots: {len(analysis['snapshots'])}")

if __name__ == "__main__":
    main()
