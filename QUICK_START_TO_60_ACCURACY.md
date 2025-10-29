# 🚀 Quick Start Guide to 60% Accuracy

## Current Status: 48% → Target: 60%+

You're already at **48% accuracy**! Here's the fastest way to push to **60%+**:

---

## Option 1: Quick Win (3 Minutes)

### Just train longer! ✨

Edit `train_enhanced_models.py` line 220:

```python
# Change this:
num_epochs = 50  # Line 220

# To this:
num_epochs = 150
```

**Then run**:
```bash
python train_enhanced_models.py
```

**Expected result**: 52-55% accuracy (in ~1 hour)

---

## Option 2: Big Win (10 Minutes) 

### Add class weights for imbalanced data

Edit `train_enhanced_models.py` and add this code around **line 45** (after loading the dataset):

```python
# Calculate class weights for imbalanced dataset
from collections import Counter
all_labels = [data.y.item() for data in dataset]
class_counts = Counter(all_labels)
total = len(all_labels)
num_classes = len(set(all_labels))

class_weights = torch.tensor([
    total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)
], dtype=torch.float32, device=device)

print(f"Class distribution: {dict(class_counts)}")
print(f"Class weights: {class_weights.tolist()}")
```

Then modify the `train_enhanced_model` function around **line 50**:

```python
# Change criterion from:
criterion = nn.CrossEntropyLoss()

# To:
if class_weights is not None:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.Cross弹出Loss()
```

And update the function signature on **line 22**:

```python
# Change from:
def train_enhanced_model(model_name, model, train_loader, val_loader, num_epochs=50, device='cpu'):

# To:
def train_enhanced_model(model_name, model, train_loader, val_loader, num_epochs=50, device='cpu', class_weights=None):
```

And update the calls to this function around **line 250**:

```python
history = train_enhanced_model(
    model_name, model, train_loader, val_loader,
    num_epochs, device, class_weights  # Add class_weights here
)
```

**Then run**:
```bash
python train_enhanced_models.py
```

**Expected result**: 55-60% accuracy

---

## Option 3: Maximum Win (30 Minutes)

### Do both above + Ensemble Models

After training completes, the script will automatically ensemble all 3 models. Just wait for it to finish!

**Expected result**: 60-65% accuracy

---

## 🎯 Recommended: Start with Option 1

1. Change `num_epochs = 150`
2. Run `python train_enhanced_models.py`
3. Wait 1-2 hours
4. Get results!

If you want more, then do Option 2.

---

## 📊 Expected Timeline

| Action | Time | Accuracy Gain |
|--------|------|---------------|
| Current | - | 48% |
| Train 150 epochs | 1-2 hrs | 52-55% |
| + Class weights | 1-2 hrs | 55-60% |
| + Ensemble | 1-2 hrs | 60-65% |

---

## 🎉 You're Almost There!

Your models are working great! Just need a bit more training time and better handling of imbalanced classes.

Check `FURTHER_IMPROVEMENTS.md` for more advanced techniques!

