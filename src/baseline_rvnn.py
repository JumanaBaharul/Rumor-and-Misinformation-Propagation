import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class TreeLSTMCell(nn.Module):
    """Tree LSTM cell for processing tree-structured data (base paper replication)"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates for Tree LSTM (following base paper architecture)
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size)
        
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, h_children: List[torch.Tensor], 
                c_children: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Tree LSTM cell"""
        batch_size = x.size(0)
        
        if len(h_children) == 0:
            # Leaf node - initialize with zeros
            h_children = [torch.zeros(batch_size, self.hidden_size, device=x.device)]
            c_children = [torch.zeros(batch_size, self.hidden_size, device=x.device)]
        
        # Concatenate children states
        h_children_concat = torch.cat(h_children, dim=0)
        c_children_concat = torch.cat(c_children, dim=0)
        
        # Compute gates (input, output, update)
        h_sum = torch.sum(h_children_concat, dim=0)
        iou = self.W_iou(x) + self.U_iou(h_sum)
        
        # Ensure iou is 2D for splitting
        if iou.dim() == 1:
            iou = iou.unsqueeze(0)
        
        i, o, u = torch.split(iou, self.hidden_size, dim=1)
        
        # Compute forget gates for each child
        f = []
        for h_child in h_children:
            f.append(torch.sigmoid(self.W_f(x) + self.U_f(h_child)))
        
        # Compute cell state - simplified approach
        if f:
            f_stack = torch.stack(f)
            # Ensure dimensions match for multiplication
            if f_stack.dim() == 2 and c_children_concat.dim() == 2:
                c = torch.sigmoid(i) * torch.tanh(u) + torch.sum(f_stack * c_children_concat, dim=0)
            else:
                c = torch.sigmoid(i) * torch.tanh(u)
        else:
            c = torch.sigmoid(i) * torch.tanh(u)
        
        # Compute hidden state
        h = torch.sigmoid(o) * torch.tanh(c)
        
        return h, c

class BaselineRvNNModel(nn.Module):
    """Baseline RvNN model replicating the base research paper"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 dropout: float = 0.5, use_tree_lstm: bool = True):
        super(BaselineRvNNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_tree_lstm = use_tree_lstm
        
        # Feature transformation layers (base paper approach)
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        if use_tree_lstm:
            # Tree LSTM cell (core of base paper)
            self.tree_lstm = TreeLSTMCell(hidden_size, hidden_size)
        else:
            # Simple RNN fallback
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Classification layers (base paper architecture)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the baseline RvNN model"""
        x, edge_index = data.x, data.edge_index
        
        # Transform input features
        x = self.feature_transform(x)
        
        if self.use_tree_lstm:
            # Process with Tree LSTM (base paper approach)
            h = self._process_tree_lstm(x, edge_index)
        else:
            # Process with simple RNN
            h = self._process_rnn(x, edge_index)
        
        # Global pooling to get graph representation
        if hasattr(data, 'batch') and data.batch is not None:
            # Batch processing
            h = global_mean_pool(h, data.batch)
        else:
            # Single graph
            h = torch.mean(h, dim=0, keepdim=True)
        
        # Classification
        output = self.classifier(h)
        
        return output
    
    def _process_tree_lstm(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process graph with Tree LSTM (base paper implementation)"""
        num_nodes = x.size(0)
        device = x.device
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        
        # Handle edge_index tensor issues
        if edge_index.numel() == 0:
            # If no edges, treat as isolated nodes
            return self._process_rnn(x, edge_index)
        
        # Ensure edge_index has correct shape [2, num_edges]
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            print(f"Warning: edge_index has unexpected shape {edge_index.shape}, using RNN fallback")
            return self._process_rnn(x, edge_index)
        
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            adj_list[source].append(target)
        
        # Initialize hidden and cell states
        h = torch.zeros(num_nodes, self.hidden_size, device=device)
        c = torch.zeros(num_nodes, self.hidden_size, device=device)
        
        # Process nodes in topological order (BFS)
        processed = set()
        queue = []
        
        # Find root nodes (nodes with no incoming edges)
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index.size(1)):
            target = edge_index[1, i].item()
            in_degree[target] += 1
        
        for i in range(num_nodes):
            if in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            node = queue.pop(0)
            if node in processed:
                continue
            
            # Get children
            children = adj_list[node]
            h_children = [h[child] for child in children if child in processed]
            c_children = [c[child] for child in children if child in processed]
            
            # Process with Tree LSTM
            h[node], c[node] = self.tree_lstm(x[node], h_children, c_children)
            processed.add(node)
            
            # Add children to queue if all their parents are processed
            for child in children:
                if child not in processed:
                    all_parents_processed = True
                    for parent in range(num_nodes):
                        if parent in adj_list[child] and parent not in processed:
                            all_parents_processed = False
                            break
                    if all_parents_processed:
                        queue.append(child)
        
        return h
    
    def _process_rnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process graph with simple RNN (fallback method)"""
        num_nodes = x.size(0)
        
        # Sort nodes by timestamp (assuming x[:, 1] contains timestamp)
        if x.size(1) > 1:
            timestamps = x[:, 1]
            sorted_indices = torch.argsort(timestamps)
            x_sorted = x[sorted_indices]
        else:
            x_sorted = x
            sorted_indices = torch.arange(num_nodes)
        
        # Process with LSTM
        x_reshaped = x_sorted.unsqueeze(0)  # [1, num_nodes, hidden_size]
        lstm_out, _ = self.rnn(x_reshaped)
        
        # Restore original order
        h = lstm_out.squeeze(0)  # [num_nodes, hidden_size]
        h_restored = torch.zeros_like(h)
        h_restored[sorted_indices] = h
        
        return h_restored

class BaselineRvNNWithAttention(nn.Module):
    """Enhanced baseline RvNN with attention mechanism (base paper + attention)"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                 dropout: float = 0.5, num_heads: int = 4):
        super(BaselineRvNNWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Tree LSTM core
        self.tree_lstm = TreeLSTMCell(hidden_size, hidden_size)
        
        # Multi-head attention for node importance
        self.node_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Graph-level attention
        self.graph_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with attention mechanism"""
        x, edge_index = data.x, data.edge_index
        
        # Transform input features
        x = self.feature_transform(x)
        
        # Process with Tree LSTM
        h = self._process_tree_lstm(x, edge_index)
        
        # Apply node attention
        h_reshaped = h.unsqueeze(0)  # [1, num_nodes, hidden_size]
        h_attended, _ = self.node_attention(h_reshaped, h_reshaped, h_reshaped)
        h_attended = h_attended.squeeze(0)  # [num_nodes, hidden_size]
        
        # Graph-level attention for node importance
        node_weights = self.graph_attention(h_attended)  # [num_nodes, 1]
        
        # Weighted pooling
        if hasattr(data, 'batch') and data.batch is not None:
            # Batch processing with weighted pooling
            batch_size = data.batch.max().item() + 1
            h_pooled = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
            for i in range(batch_size):
                mask = (data.batch == i)
                if mask.sum() > 0:
                    h_pooled[i] = torch.sum(h_attended[mask] * node_weights[mask], dim=0) / node_weights[mask].sum()
        else:
            # Single graph
            h_pooled = torch.sum(h_attended * node_weights, dim=0, keepdim=True) / node_weights.sum()
        
        # Classification
        output = self.classifier(h_pooled)
        
        return output
    
    def _process_tree_lstm(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process graph with Tree LSTM (same as baseline)"""
        num_nodes = x.size(0)
        device = x.device
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        
        # Handle empty edge_index case
        if edge_index.size(0) == 0 or edge_index.size(1) == 0:
            # If no edges, treat as isolated nodes
            return self._process_rnn(x, edge_index)
        
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            adj_list[source].append(target)
        
        # Initialize hidden and cell states
        h = torch.zeros(num_nodes, self.hidden_size, device=device)
        c = torch.zeros(num_nodes, self.hidden_size, device=device)
        
        # Process nodes in topological order
        processed = set()
        queue = []
        
        # Find root nodes
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index.size(1)):
            target = edge_index[1, i].item()
            in_degree[target] += 1
        
        for i in range(num_nodes):
            if in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            node = queue.pop(0)
            if node in processed:
                continue
            
            # Get children
            children = adj_list[node]
            h_children = [h[child] for child in children if child in processed]
            c_children = [c[child] for child in children if child in processed]
            
            # Process with Tree LSTM
            h[node], c[node] = self.tree_lstm(x[node], h_children, c_children)
            processed.add(node)
            
            # Add children to queue
            for child in children:
                if child not in processed:
                    all_parents_processed = True
                    for parent in range(num_nodes):
                        if parent in adj_list[child] and parent not in processed:
                            all_parents_processed = False
                            break
                    if all_parents_processed:
                        queue.append(child)
        
        return h

def create_baseline_model(input_size: int, hidden_size: int, num_classes: int, 
                         model_type: str = "baseline", **kwargs) -> nn.Module:
    """Factory function to create baseline models"""
    
    if model_type == "baseline":
        return BaselineRvNNModel(input_size, hidden_size, num_classes, **kwargs)
    elif model_type == "baseline_attention":
        return BaselineRvNNWithAttention(input_size, hidden_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    input_size = 22  # Based on our feature extraction
    hidden_size = 128
    num_classes = 4  # false, true, unverified, non-rumor
    
    print("Testing Baseline RvNN Models...")
    
    # Test Baseline RvNN
    print("\n--- Baseline RvNN Model ---")
    baseline_rvnn = create_baseline_model(input_size, hidden_size, num_classes, 
                                        model_type="baseline")
    
    total_params = sum(p.numel() for p in baseline_rvnn.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with dummy data
    num_nodes = 5
    x = torch.randn(num_nodes, input_size)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    y = torch.randint(0, num_classes, (1,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    with torch.no_grad():
        output = baseline_rvnn(data)
        print(f"Output shape: {output.shape}")
    
    # Test Baseline RvNN with Attention
    print("\n--- Baseline RvNN with Attention ---")
    baseline_attention = create_baseline_model(input_size, hidden_size, num_classes, 
                                            model_type="baseline_attention")
    
    total_params = sum(p.numel() for p in baseline_attention.parameters())
    print(f"Total parameters: {total_params:,}")
    
    with torch.no_grad():
        output = baseline_attention(data)
        print(f"Output shape: {output.shape}")
    
    print("\n✅ All baseline models created successfully!")
