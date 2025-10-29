# Further Improvements to Boost Accuracy (Current: 48%)

## Current Status ✅

Great progress! You've achieved **48% accuracy** (vs 25% before). Here's how to push toward 60-70%:

## Current Results
- **Best Model**: improved_transformer_gnn
- **Accuracy**: 47.77%
- **F1-Score**: 0.4226

---

## 🚀 Quick Wins (Easy Improvements)

### 1. **Train Longer** (5 minutes to implement)
- Current: 50 epochs
- **Change to**: 100-200 epochs
- **Expected gain**: +3-5% accuracy

**How to do it**:
```python
# In train_enhanced_models.py, line 220
num_epochs = 100  # Change from 50 to 100
```

### 2. **Use Class Weights** (2 minutes to implement)
- Dataset is imbalanced (4 classes)
- **Use weighted loss** instead of regular CrossEntropyLoss

**How to do it** (add this to `train_enhanced_models.py`):
```python
# Calculate class weights
from collections import Counter
all_labels = [data.y.item() for data in dataset]
class_counts = Counter(all_labels)
total = len(all_labels)
class_weights = torch.tensor([
    total / (4 * class_counts[i]) for i in range(4)
], dtype=torch.float32)

# Use in training
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### 3. **Data Augmentation** (15 minutes to implement)
- Add noise to features
- Random edge dropout

**Implementation** (create new file `src/data_augmentation.py`):
```python
import torch
import torch.nn.functional as F

def augment_graph_features(data, noise_std=0.01):
    """Add small noise to features"""
    noisy_x = data.x + torch.randn_like(data.x) * noise_std
    return data.__class__(x=noisy_x, edge_index=data.edge_index, 
                         y=data.y, batch=data.batch)

def random_edge_dropout(data, p=0.1):
    """Randomly drop edges"""
    if data.edge_index.size(1) == 0:
        return data
    
    num_edges = data.edge_index.size(1)
    edge_mask = torch.rand(num_edges) > p
    return data.__class__(x=data.x, edge_index=data.edge_index[:, edge_mask],
                         y=data.y, batch=data.batch)
```

### 4. **Ensemble Models** (10 minutes to implement)
- Combine predictions from all 3 enhanced models
- **Expected gain**: +5-7% accuracy

**Implementation**:
```python
def ensemble_predict(models, test_loader, device):
    """Combine predictions from multiple models"""
    all_probs = []
    
    for model in models:
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                probs.append(torch.softmax(output, dim=1))
        all_probs.append(torch.cat(probs))
    
    # Average probabilities
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    predictions = torch.argmax(avg_probs, dim=1)
    
    return predictions
```

---

## 🔧 Architecture Improvements

### 5. **Add More GNN Layers** (5 minutes)
Current models have 3-4 layers. Add more for deeper feature learning:

```python
# In EnhancedTGNN, change num_layers
model = create_enhanced_model(
    input_size, hidden_size, num_classes,
    model_type="enhanced_tgnn",
    num_layers=5  # Increase from 4 to 5
)
```

### 6. **Use Larger Hidden Size** (2 minutes)
- Current: hidden_size=128
- **Change to**: hidden_size=256

**Warning**: Will take longer to train and use more memory.

### 7. **Add Skip Connections** ✅ Already done!
- Your enhanced models already have residual connections

### 8. **Better Pooling Strategies**
Current models use mean + max pooling. Add:
- Attention pooling
- Sum pooling  
- Learnable weighted pooling

---

## 🎯 Advanced Techniques

### 9. **Focal Loss** (for imbalanced data)
Instead of CrossEntropyLoss, use focal loss:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 10. **Learning Rate Scheduler**
Use Cosine Annealing with Warm Restarts:

```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

### 11. **Label Smoothing**
Add label smoothing during training en en

```python
# In train_enhanced_model function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 12. **Mixup Augmentation**
Mix features from different samples:

```python
def mixup(data1, data2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * data1.x + (1 - lam) * data2.x
    # Mix labels similarly
    return mixed_x, data1.y, data2.y, lam
```

---

## 📊 Hyperparameter Tuning

### Most Effective Changes:

1. **Learning Rate**: Try 0.0001, 0.0005, 0.001
2. **Weight Decay**: Try 0.001, 0.005, 0.01
3. **Dropout**: Try 0.2, 0.3, 0.4
4. **Batch Size**: Try 16, 32, 64

### Automatic Hyperparameter Tuning:

Use Optuna for automated tuning:

```bash
pip install optuna
```

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 3, 6)
    
    # Train model with these hyperparameters
    # Return validation accuracy
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 🔍 Feature Engineering

### Better Features = Better Accuracy

Current: 22 features. Consider adding:

1. **Temporal features**:
   - Time between tweet and replies
   - Propagation velocity
   - Peak activity time

2. **User features**:
   - User verification status
   - User activity level
   - User influence score

3. **Content features**:
   - URL presence
   - Hashtag count
   - Mention count
   - Question marks / exclamation marks

---

## 📈 Quick Implementation Priority

### Priority 1 (Do First - Biggest Impact):
1. ✅ Train longer (200 epochs)
2. ✅ Use class weights
3. ✅ Ensemble all 3 models

**Expected result: 55-60% accuracy**

### Priority 2 (Good ROI):
4. Add more layers (5-6 layers)
5. Use focal loss
6. Add data augmentation

**Expected result: 60-65% accuracy**

### Priority 3 (Advanced):
7. Hyperparameter tuning with Optuna
8. Larger hidden size (256)
9. Better feature engineering

**Expected result: 65-70% accuracy**

---

## 🎯 Recommended Quick Start

Here's what I recommend you do RIGHT NOW:

```bash
# 1. Modify train_enhanced_models.py line 220
num_epochs = 100  # Change from 50

# 2. Add class weights (around line 55)
# Calculate class weights
all_labels = [data.y.item() for data in dataset]
class_counts = Counter(all_labels)
total = len(all_labels)
class_weights = torch.tensor([
    total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)
], dtype=torch.float32, device=device)

# 3. Update criterion
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 4. Re-run
python train_enhanced_models.py
```

**This single change should get you to 55-60% accuracy!**

---

## 📊 Expected Final Results

| Method | Accuracy | F1-Score | Time |
|--------|----------|----------|------|
| Current | 48% | 0.42 | - |
| + Longer training | 52-55% | 0.45-0.50 | +30 min |
| + Class weights | 55-58% | 0.48-0.52 | +0 min |
| + Ensemble | 58-62% | 0.52-0.55 | +5 min |
| + All improvements | 65-70% | 0.58-0.65 | +2-3 hrs |

---

## 🏆 Summary

You've already achieved **2x improvement** (25% → 48%)!

To go from **48% → 60-70%**:
1. Train longer (100-200 epochs)
2. Use class weights
3. Ensemble models
4. Add more layers

**Estimated effort**: 1-2 hours of changes
**Expected gain**: +12-22% accuracy

The code is working great - now it's mostly hyperparameter tuning and ensembling! 🚀

