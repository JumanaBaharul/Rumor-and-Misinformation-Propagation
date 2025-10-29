"""Enhanced graph-based models for rumor detection."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GraphNorm,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

from ..config import ModelConfig

try:  # pragma: no cover - optional dependency availability
    from torch_geometric.nn import GATv2Conv
except ImportError:  # pragma: no cover
    GATv2Conv = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _get_attention_conv(in_channels: int, heads: int, dropout: float) -> nn.Module:
    conv_cls = GATv2Conv if GATv2Conv is not None else GATConv
    return conv_cls(
        in_channels,
        in_channels,
        heads=heads,
        concat=False,
        dropout=dropout,
    )


class ResidualAttentionBlock(nn.Module):
    """Graph attention block with residual connection and normalisation."""

    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.conv = _get_attention_conv(channels, heads=heads, dropout=dropout)
        self.norm = GraphNorm(channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.elu(out)
        if out.shape == residual.shape:
            out = out + residual
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class TransformerBlock(nn.Module):
    """Transformer-style graph convolution block with residual connection."""

    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.conv = TransformerConv(
            channels,
            channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            beta=True,
        )
        self.norm = GraphNorm(channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.gelu(out)
        if out.shape == residual.shape:
            out = out + residual
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class GraphReadout(nn.Module):
    """Combine multiple global pooling strategies into one representation."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean = global_mean_pool(x, batch)
        max_ = global_max_pool(x, batch)
        add = global_add_pool(x, batch)

        mean_centered = x - mean[batch]
        std = torch.sqrt(global_mean_pool(mean_centered.pow(2), batch) + 1e-6)

        return torch.cat([mean, max_, add, std], dim=-1)

    @property
    def output_dim(self) -> int:
        return self.input_dim * 4


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class EnhancedTGNN(nn.Module):
    """Enhanced TGNN with deeper attention stacks and robust pooling."""

    def __init__(self, input_size: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(config.hidden_size, config.attention_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        self.readout = GraphReadout(config.hidden_size)
        readout_dim = self.readout.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(config.hidden_size // 2, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)
        graph_repr = self.readout(x, batch)
        return self.classifier(graph_repr)


class ImprovedTransformerGNN(nn.Module):
    """Hybrid transformer and GNN architecture for rumor detection."""

    def __init__(self, input_size: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.hidden_size, config.attention_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_size * 2,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.readout = GraphReadout(config.hidden_size)
        readout_dim = self.readout.output_dim + config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(config.hidden_size // 2, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)

        dense_x, mask = to_dense_batch(x, batch)
        transformer_out = self.temporal_encoder(dense_x, src_key_padding_mask=~mask)
        transformer_out = transformer_out.masked_fill(~mask.unsqueeze(-1), 0.0)
        sequence_mean = transformer_out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        graph_repr = self.readout(x, batch)
        combined = torch.cat([graph_repr, sequence_mean], dim=-1)
        return self.classifier(combined)


class AdvancedRvNN(nn.Module):
    """Enhanced recursive neural network style architecture."""

    def __init__(self, input_size: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.attention_layers = nn.ModuleList(
            [
                ResidualAttentionBlock(config.hidden_size, config.attention_heads, config.dropout)
                for _ in range(max(1, config.num_layers - 1))
            ]
        )

        self.root_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.readout = GraphReadout(config.hidden_size)
        readout_dim = self.readout.output_dim + config.hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_classes),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.feature_proj(x)
        for layer in self.attention_layers:
            x = layer(x, edge_index)

        dense_x, mask = to_dense_batch(x, batch)
        packed_outputs, _ = self.root_gru(dense_x)
        packed_outputs = packed_outputs.masked_fill(~mask.unsqueeze(-1), 0.0)
        rnn_summary = packed_outputs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        graph_repr = self.readout(x, batch)
        combined = torch.cat([graph_repr, rnn_summary], dim=-1)
        return self.classifier(combined)


MODEL_REGISTRY: Dict[str, nn.Module] = {
    "enhanced_tgnn": EnhancedTGNN,
    "improved_transformer_gnn": ImprovedTransformerGNN,
    "advanced_rvnn": AdvancedRvNN,
}


def build_model(model_name: str, input_size: int, num_classes: int, config: ModelConfig) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(input_size=input_size, num_classes=num_classes, config=config)


__all__ = [
    "EnhancedTGNN",
    "ImprovedTransformerGNN",
    "AdvancedRvNN",
    "MODEL_REGISTRY",
    "build_model",
]
