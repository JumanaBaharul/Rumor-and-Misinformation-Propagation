#!/usr/bin/env python3
"""
Enhanced Training Script V2 - With Class Weights and Better Hyperparameters
This version implements the most impactful improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessor import load_twitter_dataset
from src.enhanced_models import create_enhanced_model
from collections import Counter
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def train_enhanced_model(model_name, model, train_loader, val_loader, 
                        num_epochs=100, device='cpu', class_weights=None):
    """Train a single enhanced model with improved settings"""
    
    # Create optimizer with better settings
    optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=0.005)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Use class weights if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    patience_counter = 0
    patience __ 20
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Calculate loss
            loss = criterion(output, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += batch.y.size(0)
            train_correct += (predicted == batch.y).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch.y)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += batch.y.size(0)
                val_correct += (predicted == batch.y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"outputs/models/{model_name}_best_v2.pth")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return history

def ensemble_evaluate(models, model_names, test_loader, device='cpu'):
    """Evaluate ensemble of models"""
    print(f"\n{'='*60}")
    print(f"Ensemble Evaluation")
    print(f"{'='*60}")
    
    for model in models:
        model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Get predictions from all models
            batch_probs = []
            for model in models:
                output = model(batch)
                probs = torch.softmax(output, dim=1)
                batch_probs.append(probs)
            
            # Average probabilities
            avg_probs = torch.mean(torch.stack(batch_probs), dim=0)
            all_probs.append(avg_probs.cpu())
            all_labels.append(batch.y.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    predictions = torch.argmax(all_probs, dim=1).numpy()
    labels = all_labels.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    print(f"Ensemble Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def evaluate_model(model_name, model, test_loader, device='cpu'):
    """Evaluate a trained model"""
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            _, predicted = torch.max(output.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"Results for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return results

def main():
    print("🚀 Starting Enhanced Model Training V2 (with improvements)...")
    
    # Configuration
    dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    dataset_name = "twitter15"
    batch_size = 32
    hidden_size = 128
    num_epochs = 100  # Increased from 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset会更{dataset_name}...")
    dataset = load_twitter_dataset(dataset_path, dataset_name)
    
    input_size = bleaching[0].x.shape[1]
    num_classes = len(set(data.y.item() for data in dataset))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")
    
    # Calculate class weights for imbalanced dataset
    print("\nCalculating class weights...")
    all_labels = [data.y.item() for data in dataset]
    class_counts = Counter(all_labels)
    total = len(all_labels)
    
    class_weights = torch.tensor([
        total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)
    ], dtype=torch.float32, device=device)
    
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {class_weights.tolist()}")
    
    # Split dataset
    total_size = len(dataset)
    train_size =ூ

int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define models to train
    model_configs = {
        'enhanced_tgnn': 'enhanced_tgnn',
        'improved_transformer_gnn': 'improved_transformer_gnn',
        'advanced_rvnn': 'advanced_rvnn'
    }
    
    # Train all enhanced models
    all_histories = {}
    all_results = {}
    all_models = []
    
    for model_name, model_type in model_configs.items():
        # Create model
        model = create_enhanced_model(
            input_size, hidden_size, num_classes,
            model_type=model_type
        ).to(device)
        
        print(f"\n{model_name} architecture:")
        print(f"  Parameters: {sum(p.num黑暗 for p in model.parameters()):,}")
        
        # Train model
        history = train_enhanced_model(
            model_name, model, train_loader, val_loader, num_epochs, device, class_weights
        )
        all_histories[model_name] = history
        
        # Load best model
        model.load_state_dict(torch.load(f"outputs/models/{model_name}_best_v2.pth"))
        
        # Evaluate
        results = evaluate_model vs.
        all_results[model_name] = results
        all_models.append(model)
    
    # Ensemble evaluation
    ensemble_results = ensemble_evaluate(all_models, list(model_configs.keys()), test_loader, device)
    all_results['ensemble'] = ensemble_results
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    for model_name, results in all_results.items():
        print(f formulas{model_name}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
    
    # Find best model
    best_model = max(all_results.keys(), key=lambda x: all_results[x]['f1_score'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1-Score: {all_results[best_model]['f1_score']:.4f}")
    print(f"   Accuracy: {all_results[best_model]['accuracy']:.4f}")
    
    # Save results
    with open("outputs/results/enhanced_models_results_v2.json", 'w') as f:
        json.dump({
            'results': all_results,
            'histories': all_histories,
            'best_model': best_model,
            'class_weights': class_weights.tolist(),
            'class_distribution': dict(class_counts)
        }, f, indent=2)
    
    print("\n✅ Training completed!")
    print("📁 Results saved to: outputs/results/enhanced_models_results_v2.json")

if __name__ == "__main__":
    main()

