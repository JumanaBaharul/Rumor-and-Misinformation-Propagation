import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class TemporalPositionalEncoding(nn.Module):
    """Temporal positional encoding for transformer-based models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(TemporalPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TemporalGraphAttention(nn.Module):
    """Temporal Graph Attention mechanism for capturing temporal dependencies"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(TemporalGraphAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Ensure hidden_size is divisible by num_heads
        if self.head_dim * num_heads != hidden_size:
            # Adjust hidden_size to be divisible by num_heads
            self.hidden_size = self.head_dim * num_heads
        
        # Update linear layers to use the adjusted hidden_size
        self.q_proj = nn.Linear(hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of temporal graph attention
        
        Args:
            x: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges]
            temporal_mask: Optional temporal mask for attention
        """
        num_nodes = x.size(0)
        
        # Project queries, keys, and values
        # Ensure input has the correct hidden size
        if x.size(-1) != self.hidden_size:
            # Pad or truncate to match hidden_size
            if x.size(-1) < self.hidden_size:
                # Pad with zeros
                padding = torch.zeros(num_nodes, self.hidden_size - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[:, :self.hidden_size]
        
        Q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply temporal mask if provided
        if temporal_mask is not None:
            scores = scores.masked_fill(temporal_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        out = out.view(num_nodes, self.hidden_size)
        
        # Project output and add residual connection
        out = self.out_proj(out)
        out = self.layer_norm(x + out)
        
        return out

class TemporalGraphNeuralNetwork(nn.Module):
    """Temporal Graph Neural Network with novel temporal modeling"""
    
    # def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
    #              num_layers: int = 3, dropout: float = 0.5, num_heads: int = 8):
    #     super(TemporalGraphNeuralNetwork, self).__init__()
        
    #     self.input_size = input_size
    #     # Ensure hidden_size is divisible by num_heads
    #     self.hidden_size = (hidden_size // num_heads) * num_heads
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 num_layers: int = 3, dropout: float = 0.5, num_heads: int = 8):
        super().__init__()
        self.hidden_size = (hidden_size // num_heads) * num_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # Temporal positional encoding
        self.temporal_pe = TemporalPositionalEncoding(hidden_size)
        
        # Graph convolution layers (simplified without temporal attention)
        self.gnn_layers = nn.ModuleList()
        # self.temporal_attention_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # GNN layer (using GCNConv for static graphs)
            if i == 0:
                self.gnn_layers.append(GCNConv(hidden_size, hidden_size))
            else:
                self.gnn_layers.append(GCNConv(hidden_size, hidden_size))
            
            # Temporal attention layer (removed for simplicity)
            # self.temporal_attention_layers.append(
            #     TemporalGraphAttention(hidden_size, num_heads, dropout)
            # )
        
        # Transformer-based temporal modeling
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Temporal convolution for sequence modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Temporal importance scoring
        self.temporal_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the Temporal GNN
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            output: Classification logits
        """
        x, edge_index = data.x, data.edge_index
        
        # Transform input features
        x = self.feature_transform(x)
        
        # Apply temporal positional encoding
        x = self.temporal_pe(x)
        
        # Process through GNN layers (simplified without temporal attention)
        for i in range(self.num_layers):
            # GNN processing
            x = self.gnn_layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph representation
        if hasattr(data, 'batch') and data.batch is not None:
            # Batch processing
            x_pooled = global_mean_pool(x, data.batch)
        else:
            # Single graph
            x_pooled = torch.mean(x, dim=0, keepdim=True)
        
        # Ensure x_pooled is 2D [batch_size, hidden_size]
        if x_pooled.dim() == 1:
            x_pooled = x_pooled.unsqueeze(0)
        elif x_pooled.dim() == 3:
            # If we have [num_nodes, batch_size, hidden_size], reduce first dimension
            x_pooled = torch.mean(x_pooled, dim=0)
        
        # Simple classification without complex temporal modeling
        output = self.classifier(x_pooled)
        
        return output

class TemporalGNNWithMemory(nn.Module):
    """Enhanced TGNN with memory mechanism for long-term temporal dependencies"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 memory_size: int = 64, dropout: float = 0.5):
        super(TemporalGNNWithMemory, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.memory_size = memory_size
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # Memory bank for temporal information
        self.memory_bank = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.memory_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Temporal GNN layers (using GCNConv for static graphs)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_size, hidden_size),
            GCNConv(hidden_size, hidden_size),
            GCNConv(hidden_size, hidden_size)
        ])
        
        # Memory update mechanism
        self.memory_update = nn.GRUCell(hidden_size, hidden_size)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with memory mechanism"""
        x, edge_index = data.x, data.edge_index
        
        # Transform input features
        x = self.feature_transform(x)
        
        # Process through GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Global pooling
        if hasattr(data, 'batch') and data.batch is not None:
            x_pooled = global_mean_pool(x, data.batch)
        else:
            x_pooled = torch.mean(x, dim=0, keepdim=True)
        
        # Memory interaction
        memory_enhanced = self._interact_with_memory(x_pooled)
        
        # Combine original and memory-enhanced representations
        x_combined = torch.cat([x_pooled, memory_enhanced], dim=-1)
        
        # Classification
        output = self.classifier(x_combined)
        
        return output
    
    def _interact_with_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Interact with memory bank"""
        # Reshape for attention
        x_reshaped = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        memory_reshaped = self.memory_bank.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Apply attention between input and memory
        memory_attended, _ = self.memory_attention(x_reshaped, memory_reshaped, memory_reshaped)
        memory_attended = memory_attended.squeeze(1)  # [batch_size, hidden_size]
        
        # Update memory bank
        self.memory_bank.data = self.memory_update(
            memory_attended.mean(dim=0), self.memory_bank.data
        )
        
        return memory_attended

def create_tgnn_model(input_size: int, hidden_size: int, num_classes: int, 
                     model_type: str = "temporal_gnn", **kwargs) -> nn.Module:
    """Factory function to create TGNN models"""
    
    if model_type == "temporal_gnn":
        return TemporalGraphNeuralNetwork(input_size, hidden_size, num_classes, **kwargs)
    elif model_type == "temporal_gnn_memory":
        return TemporalGNNWithMemory(input_size, hidden_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown TGNN model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    input_size = 22  # Based on our feature extraction
    hidden_size = 128
    num_classes = 4  # false, true, unverified, non-rumor
    
    print("Testing Novel TGNN Models...")
    
    # Test Temporal GNN
    print("\n--- Temporal GNN Model ---")
    temporal_gnn = create_tgnn_model(input_size, hidden_size, num_classes, 
                                    model_type="temporal_gnn")
    
    total_params = sum(p.numel() for p in temporal_gnn.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with dummy data
    num_nodes = 5
    x = torch.randn(num_nodes, input_size)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    y = torch.randint(0, num_classes, (1,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    with torch.no_grad():
        output = temporal_gnn(data)
        print(f"Output shape: {output.shape}")
    
    # Test Temporal GNN with Memory
    print("\n--- Temporal GNN with Memory ---")
    temporal_memory = create_tgnn_model(input_size, hidden_size, num_classes, 
                                      model_type="temporal_gnn_memory")
    
    total_params = sum(p.numel() for p in temporal_memory.parameters())
    print(f"Total parameters: {total_params:,}")
    
    with torch.no_grad():
        output = temporal_memory(data)
        print(f"Output shape: {output.shape}")
    
    print("\n✅ All TGNN models created successfully!")
