"""
Enhanced Graph-Based Rumor Detection System

This package contains all the components for:
- Data preprocessing and temporal graph construction
- Dataset visualization and analysis
- Baseline RvNN model (base paper replication)
- Novel TGNN and Transformer-GNN models
- Comprehensive training and evaluation framework

Author: AI Assistant
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Import main components
from .data_preprocessor import TwitterDatasetPreprocessor, TwitterDataset, load_twitter_dataset
from .data_visualizer import TwitterDataVisualizer
from .temporal_graph_builder import TemporalGraphBuilder
from .baseline_rvnn import BaselineRvNNModel, BaselineRvNNWithAttention, create_baseline_model
from .novel_tgnn import TemporalGraphNeuralNetwork, TemporalGNNWithMemory, create_tgnn_model
from .novel_transformer_gnn import TransformerGNNModel, HierarchicalTransformerGNN, create_transformer_gnn_model
from .model_trainer import RumorDetectionTrainer

__all__ = [
    # Data preprocessing
    'TwitterDatasetPreprocessor',
    'TwitterDataset',
    'load_twitter_dataset',
    
    # Visualization
    'TwitterDataVisualizer',
    
    # Temporal graph building
    'TemporalGraphBuilder',
    
    # Baseline models
    'BaselineRvNNModel',
    'BaselineRvNNWithAttention',
    'create_baseline_model',
    
    # Novel models
    'TemporalGraphNeuralNetwork',
    'TemporalGNNWithMemory',
    'create_tgnn_model',
    'TransformerGNNModel',
    'HierarchicalTransformerGNN',
    'create_transformer_gnn_model',
    
    # Training and evaluation
    'RumorDetectionTrainer'
]
