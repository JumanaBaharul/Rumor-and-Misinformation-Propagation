# Model Improvements for Better Accuracy

## Overview

The models were experiencing low accuracy (around 25-30%) due to several issues. This document explains the improvements made and how to use the enhanced models.

## Problems Identified

1. **novel_tgnn vs novel_transformer_gnn**: These are DIFFERENT models:
   - **novel_tgnn**: Uses GCN layers with temporal positional encoding
   - **novel_transformer_gnn**: Uses Transformer-GNN hybrid architecture with GAT layers
   - They have different architectures and perform differently

2. **Low Accuracy Causes**:
   - Models were too simple (limited layers)
   - Missing residual connections
   - Inadequate regularization (dropout, batch normalization)
   - Suboptimal pooling strategies
   - Limited attention mechanisms
   - No gradient clipping

## Improvements Made

### 1. Enhanced Architectures (`src/enhanced_models.py`)

Three new enhanced models were created:

#### a) **EnhancedTGNN** (`enhanced_tgnn`)
- **Improvements**:
  - Graph Attention Networks (GAT) instead of GCN for better feature learning
  - Residual connections in GNN layers
  - Batch normalization for stable training
  - Multi-head attention-based pooling
  - Three pooling strategies (mean, max, sum) combined
  - Deeper classifier with better regularization
  - GELU activation instead of ReLU

#### b) **ImprovedTransformerGNN** (`improved_transformer_gnn`)
- **Improvements**:
  - Enhanced input projection with batch normalization
  - Graph convolution layers with skip connections
  - Transformer encoder for better temporal modeling
  - Graph-level attention weighting
  - Four representation strategies (mean, max, transformer, weighted)
  - Deeper classifier architecture
  - Gradient clipping for stability

#### c) **AdvancedRvNN** (`advanced_rvnn`)
- **Improvements**:
  - Graph attention layers (GAT) with multiple heads
  - Self-attention mechanism
  - Attention-weighted pooling
  - Enhanced feature transformation with batch norm
  - Multiple pooling strategies combined
  - Better regularization

### 2. Training Improvements (`src/model_trainer.py`)

- Better learning rates (0.0005 for enhanced models)
- Optimized weight decay (0.005 for enhanced models)
- Learning rate scheduler with factor=0.7 (more gradual decay)
- Improved patience settings
- Gradient clipping (max_norm=1.0)

### 3. Better Activations

- **GELU** instead of ReLU: Better for deep networks
- **Tanh** for attention scores: Better gradient flow

### 4. Better Regularization

- Batch normalization: Stabilizes training
- Layer normalization: For transformer components
- Strategic dropout placement
- Weight decay: Prevents overfitting

## How to Use the Enhanced Models

### Option 1: Train All Enhanced Models

```bash
python train_enhanced_models.py
```

This will:
1. Train only the 3 enhanced models
2. Use optimized hyperparameters
3. Save best models
4. Generate comprehensive results

### Option 2: Train All Models (Original + Enhanced)

```bash
python main.py
```

This trains all 6 models:
- Original models: `baseline_rvnn`, `novel_tgnn`, `novel_transformer_gnn`
- Enhanced models: `enhanced_tgnn`, `improved_transformer_gnn`, `advanced_rvnn`

### Option 3: Train Single Model

Edit `train_enhanced_models.py` and modify the `model_configs` dictionary to train only specific models.

## Expected Improvements

With these enhancements, you should see:

1. **Better Training Stability**: Models converge more smoothly
2. **Higher Accuracy**: Expected 35-50% on Twitter15/16 datasets
3. **Better Generalization**: Improved validation performance
4. **More Consistent Results**: Less variance across runs

## Architecture Comparison

| Model | GNN Type | Attention | Pooling | F1-Score (Expected) |
|-------|----------|-----------|---------|-------------------|
| baseline_rvnn | Tree LSTM | No | Mean | 0.09-0.12 |
| novel_tgnn | GCN | Temporal PE | Mean | 0.09-0.15 |
| novel_transformer_gnn | GAT | Multi-head | Multiple | 0.10-0.18 |
|简单enhanced_tgnn | GAT | Multi-head | Multiple | 0.18-0.30 |
| improved_transformer_gnn | GCN + Transformer | Graph + Self | Multiple | 0.20-0.35 |
| advanced_rvnn | GAT + Self-attn | Attention-weighted | Multiple | 0.18-0.32 |

## Key Tips for Better Results

1. **Increase Training Epochs**: Try 50-100 epochs instead of 30
2. **Data Augmentation**: Add more training examples
3. **Feature Engineering**: Add more useful features from the data
4. **Ensemble Methods**: Combine predictions from multiple models
5. **Hyperparameter Tuning**: Adjust learning rates, hidden sizes, etc.
6. **Class Balancing**: If dataset is imbalanced, use weighted loss

## Next Steps

1. Run `python train_enhanced_models.py` to train enhanced models
2. Check results in `outputs/results/enhanced_models_results.json`
3. Compare with original models
4. Fine-tune hyperparameters based on results

## Additional Improvements to Try

1. **Focal Loss**: For handling class imbalance
2. **Label Smoothing**: For better generalization
3. **Mixup Augmentation**: For data augmentation
4. **Different Optimizers**: Try Adam, SGD with momentum, or RMSprop
5. **Learning Rate Finder**: Use LR range test to find optimal LR
6. **Weight Ensembling**: Average weights from multiple training runs

