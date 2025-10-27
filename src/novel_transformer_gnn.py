import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class TransformerGNNLayer(nn.Module):
    """Transformer-GNN layer combining graph convolution and self-attention"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(TransformerGNNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        # Graph convolution component
        self.gnn_conv = GATConv(hidden_size, hidden_size // num_heads, heads=num_heads)
        
        # Self-attention component
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through Transformer-GNN layer"""
        # Graph convolution
        x_gnn = self.gnn_conv(x, edge_index)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.dropout(x_gnn)
        
        # First residual connection and normalization
        x = self.norm1(x + x_gnn)
        
        # Self-attention
        x_reshaped = x.unsqueeze(0)  # [1, num_nodes, hidden_size]
        x_attended, _ = self.self_attention(x_reshaped, x_reshaped, x_reshaped)
        x_attended = x_attended.squeeze(0)  # [num_nodes, hidden_size]
        x_attended = self.dropout(x_attended)
        
        # Second residual connection and normalization
        x = self.norm2(x + x_attended)
        
        # Feed-forward network
        x_ff = self.feed_forward(x)
        x_ff = self.dropout(x_ff)
        
        # Third residual connection and normalization
        x = self.norm3(x + x_ff)
        
        return x

class TransformerGNNModel(nn.Module):
    """Novel Transformer-GNN model combining transformer and graph neural networks"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.5):
        super(TransformerGNNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # Transformer-GNN layers
        self.transformer_gnn_layers = nn.ModuleList([
            TransformerGNNLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Global attention mechanism
        self.global_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Temporal modeling with transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Graph-level attention
        self.graph_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Output projection for temporal importance
        self.temporal_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer-GNN
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            output: Classification logits
            temporal_importance: Temporal importance scores
        """
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_projection(x)
        
        # Process through Transformer-GNN layers
        for layer in self.transformer_gnn_layers:
            x = layer(x, edge_index)
        
        # Global attention for graph-level representation
        x_reshaped = x.unsqueeze(0)  # [1, num_nodes, hidden_size]
        x_global, _ = self.global_attention(x_reshaped, x_reshaped, x_reshaped)
        x_global = x_global.squeeze(0)  # [num_nodes, hidden_size]
        
        # Global pooling
        if hasattr(data, 'batch') and data.batch is not None:
            # Batch processing
            x_pooled = global_mean_pool(x_global, data.batch)
        else:
            # Single graph
            x_pooled = torch.mean(x_global, dim=0, keepdim=True)
        
        # Temporal modeling with transformer
        x_temporal = x_pooled.unsqueeze(1)  # [batch_size, 1, hidden_size]
        x_temporal = self.temporal_transformer(x_temporal)
        x_temporal = x_temporal.squeeze(1)  # [batch_size, hidden_size]
        
        # Graph-level attention for node importance
        node_weights = self.graph_attention(x_global)  # [num_nodes, 1]
        
        # Weighted pooling
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
            x_weighted = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
            for i in range(batch_size):
                mask = (data.batch == i)
                if mask.sum() > 0:
                    x_weighted[i] = torch.sum(x_global[mask] * node_weights[mask], dim=0) / node_weights[mask].sum()
        else:
            x_weighted = torch.sum(x_global * node_weights, dim=0, keepdim=True) / node_weights.sum()
        
        # Combine temporal and weighted representations
        x_combined = torch.cat([x_temporal, x_weighted], dim=-1)
        
        # Classification
        output = self.classifier(x_combined)
        
        # Temporal importance scoring
        temporal_importance = self.temporal_projection(x_temporal)
        
        return output, temporal_importance

class HierarchicalTransformerGNN(nn.Module):
    """Hierarchical Transformer-GNN with multi-scale processing"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 3, num_heads: int = 8, dropout: float = 0.5):
        super(HierarchicalTransformerGNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # Hierarchical layers
        self.hierarchical_layers = nn.ModuleList()
        current_size = hidden_size
        
        for i in range(num_layers):
            # Graph convolution
            gnn_layer = GATConv(current_size, current_size // num_heads, heads=num_heads)
            
            # Transformer layer
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=current_size,
                nhead=num_heads,
                dim_feedforward=current_size * 2,
                dropout=dropout,
                batch_first=True
            )
            
            # Pooling layer for hierarchy
            if i < num_layers - 1:
                pool_layer = nn.Sequential(
                    nn.Linear(current_size, current_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                current_size = current_size // 2
            else:
                pool_layer = nn.Identity()
            
            self.hierarchical_layers.append(nn.ModuleDict({
                'gnn': gnn_layer,
                'transformer': transformer_layer,
                'pool': pool_layer
            }))
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(current_size, current_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(current_size // 2, num_classes)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through hierarchical model"""
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_projection(x)
        
        # Process through hierarchical layers
        for i, layer in enumerate(self.hierarchical_layers):
            # Graph convolution
            x = layer['gnn'](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            
            # Transformer processing
            x_reshaped = x.unsqueeze(0)  # [1, num_nodes, hidden_size]
            x = layer['transformer'](x_reshaped)
            x = x.squeeze(0)  # [num_nodes, hidden_size]
            
            # Hierarchical pooling (except last layer)
            if i < len(self.hierarchical_layers) - 1:
                x = layer['pool'](x)
                
                # Update edge_index for next layer (simplified)
                # In practice, you might want to implement proper graph pooling
                if hasattr(data, 'batch') and data.batch is not None:
                    # Simple pooling: take mean of connected components
                    batch_size = data.batch.max().item() + 1
                    x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
                    
                    for j in range(batch_size):
                        mask = (data.batch == j)
                        if mask.sum() > 0:
                            x_pooled[j] = x[mask].mean(dim=0)
                    
                    x = x_pooled
                    # Note: edge_index would need to be updated accordingly
                    # This is a simplified version
        
        # Global pooling
        if hasattr(data, 'batch') and data.batch is not None:
            x = global_mean_pool(x, data.batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classification
        output = self.classifier(x)
        
        return output

def create_transformer_gnn_model(input_size: int, hidden_size: int, num_classes: int, 
                                model_type: str = "transformer_gnn", **kwargs) -> nn.Module:
    """Factory function to create Transformer-GNN models"""
    
    if model_type == "transformer_gnn":
        return TransformerGNNModel(input_size, hidden_size, num_classes, **kwargs)
    elif model_type == "hierarchical_transformer_gnn":
        return HierarchicalTransformerGNN(input_size, hidden_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown Transformer-GNN model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    input_size = 22  # Based on our feature extraction
    hidden_size = 128
    num_classes = 4  # false, true, unverified, non-rumor
    
    print("Testing Novel Transformer-GNN Models...")
    
    # Test Transformer-GNN
    print("\n--- Transformer-GNN Model ---")
    transformer_gnn = create_transformer_gnn_model(input_size, hidden_size, num_classes, 
                                                model_type="transformer_gnn")
    
    total_params = sum(p.numel() for p in transformer_gnn.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with dummy data
    num_nodes = 5
    x = torch.randn(num_nodes, input_size)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    y = torch.randint(0, num_classes, (1,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    with torch.no_grad():
        output, temporal_importance = transformer_gnn(data)
        print(f"Output shape: {output.shape}")
        print(f"Temporal importance shape: {temporal_importance.shape}")
    
    # Test Hierarchical Transformer-GNN
    print("\n--- Hierarchical Transformer-GNN Model ---")
    hierarchical_tgnn = create_transformer_gnn_model(input_size, hidden_size, num_classes, 
                                                   model_type="hierarchical_transformer_gnn")
    
    total_params = sum(p.numel() for p in hierarchical_tgnn.parameters())
    print(f"Total parameters: {total_params:,}")
    
    with torch.no_grad():
        output = hierarchical_tgnn(data)
        print(f"Output shape: {output.shape}")
    
    print("\n✅ All Transformer-GNN models created successfully!")
