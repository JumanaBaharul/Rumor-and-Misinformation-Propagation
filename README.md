# Rumor and Misinformation Propagation Detection

This repository contains an end-to-end PyTorch Geometric pipeline for training advanced rumor detection models on the Twitter15/16 datasets. The code base has been refactored into a reusable Python package (`rumor_detection`) that exposes data loaders, model factories, and training utilities.

## Highlights
- **Modular package design** – reusable components for data preprocessing, model construction, training, and evaluation.
- **Enhanced architectures** – attention-based temporal GNN, transformer-GNN hybrid, and an improved recursive neural network baseline.
- **Robust training loop** – stratified splits, class-balanced loss + sampler, feature normalisation, scheduled learning rates, gradient clipping, stochastic weight averaging, and early stopping by macro F1 score.
- **Regularised optimisation** – label smoothing, feature/edge dropout augmentation, and automatic batch-norm updates for the averaged model to squeeze out extra accuracy.
- **Balanced inputs** – automatic z-score normalisation of node features and an optional class-balanced sampler for every batch.
- **Structured experiment outputs** – automatic checkpointing and JSON summaries containing metrics and training histories.

## Project layout
```
Rumor-and-Misinformation-Propagation/
├── data/                              # Twitter15/16 dataset assets
├── rumor_detection/                   # Core Python package
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   └── dataset.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   └── enhanced.py
│   ├── training/
│   │   └── engine.py
│   └── utils.py
├── train_enhanced_models.py           # CLI entrypoint for training experiments
├── requirements.txt
└── README.md
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
Run the enhanced training pipeline (defaults: Twitter15, AdamW, 120 epochs):
```bash
python train_enhanced_models.py
```

Useful flags:
```
--dataset-name twitter16     # choose dataset split
--epochs 60                  # override training duration
--max-samples 200            # limit samples for quick smoke tests
--models enhanced_tgnn improved_transformer_gnn
--output-dir custom_outputs  # change output location
--no-feature-normalisation   # disable z-score normalisation if needed
--no-balanced-sampler        # fall back to uniform random sampling
--label-smoothing 0.02       # tune label smoothing for cross-entropy
--edge-dropout 0.1           # drop edges during training for regularisation
--feature-dropout 0.15       # drop node features during training
--swa-start 50               # delay stochastic weight averaging start epoch
--no-swa                     # disable SWA entirely
```

The script writes model checkpoints to `outputs/models/` and a JSON experiment report to `outputs/results/enhanced_models_results.json`.

## Configuration overview
- `DataConfig`: dataset path, batching, train/validation ratios, and reproducibility settings.
- `ModelConfig`: hidden size, depth, dropout, and attention head count shared across enhanced models.
- `TrainingConfig`: optimisation hyper-parameters, device selection, gradient clipping, logging cadence, label smoothing, feature/edge dropout, and stochastic weight averaging controls.

All configurations are created internally from CLI flags and serialised in the experiment report.

## Outputs
Each run produces a JSON document with:
- Dataset summary (label distribution, feature count).
- Resolved data/model/training configuration.
- Per-model metrics (accuracy, precision, recall, weighted & macro F1, confusion matrix, classification report).
- Training histories (loss/F1 curves per epoch).
- Best-performing model and its F1 score.

Model weights are stored alongside the report for easy reuse.

## Reproducibility
Use `--seed` to fix randomness across dataset sampling, splitting, and training. The package seeds Python, NumPy, PyTorch CPU, and CUDA RNGs.

## License
MIT
