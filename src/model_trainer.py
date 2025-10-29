import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import os
import json
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .data_preprocessor import load_twitter_dataset
from .baseline_rvnn import create_baseline_model
from .novel_tgnn import create_tgnn_model
from .novel_transformer_gnn import create_transformer_gnn_model
from .enhanced_models import create_enhanced_model

class RumorDetectionTrainer:
    """Comprehensive trainer for rumor detection models with comparison"""
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.training_history = {}
        
    def prepare_data(self, dataset_path: str, dataset_name: str = "twitter15",
                    batch_size: int = 32, train_split: float = 0.7, 
                    val_split: float = 0.15):
        """Prepare data loaders for training"""
        print(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_twitter_dataset(dataset_path, dataset_name)
        
        # Get dataset statistics
        input_size = dataset[0].x.shape[1] if len(dataset) > 0 else 22
        num_classes = len(set(data.y.item() for data in dataset))
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Input size: {input_size}")
        print(f"Number of classes: {num_classes}")
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, input_size, num_classes
    
    def create_models(self, input_size: int, hidden_size: int, num_classes: int):
        """Create both baseline and novel models"""
        print("Creating models...")
        
        # Baseline RvNN model (base paper replication)
        self.models['baseline_rvnn'] = create_baseline_model(
            input_size, hidden_size, num_classes, model_type="baseline"
        ).to(self.device)
        
        # Novel TGNN model
        self.models['novel_tgnn'] = create_tgnn_model(
            input_size, hidden_size, num_classes, model_type="temporal_gnn"
        ).to(self.device)
        
        # Novel Transformer-GNN model
        self.models['novel_transformer_gnn'] = create_transformer_gnn_model(
            input_size, hidden_size, num_classes, model_type="transformer_gnn"
        ).to(self.device)
        
        # Enhanced models with better architectures
        self.models['enhanced_tgnn'] = create_enhanced_model(
            input_size, hidden_size, num_classes, model_type="enhanced_tgnn"
        ).to(self.device)
        
        self.models['improved_transformer_gnn'] = create_enhanced_model(
            input_size, hidden_size, num_classes, model_type="improved_transformer_gnn"
        ).to(self.device)
        
        self.models['advanced_rvnn'] = create_enhanced_model(
            input_size, hidden_size, num_classes, model_type="advanced_rvnn"
        ).to(self.device)
        
        # Create optimizers and schedulers for each model
        for name, model in self.models.items():
            # Use better learning rates for enhanced models
            if 'enhanced' in name or 'improved' in name or 'advanced' in name:
                lr = 0.0005  # Slightly lower learning rate for stability
                weight_decay = 0.005  # Moderate regularization
            else:
                lr = 0.001
                weight_decay = 0.01
            
            self.optimizers[name] = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[name], mode='min', factor=0.7, patience=5
            )
            self.training_history[name] = {
                'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []
            }
        
        # Print model information
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name}: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    def train_model(self, model_name: str, train_loader: GeometricDataLoader, 
                   val_loader: GeometricDataLoader, num_epochs: int = 50,
                   early_stopping_patience: int = 10):
        """Train a specific model"""
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        scheduler = self.schedulers[model_name]
        
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                if model_name == 'novel_transformer_gnn':
                    output, _ = model(batch)
                else:
                    output = model(batch)
                
                # Calculate loss
                loss = criterion(output, batch.y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += batch.y.size(0)
                train_correct += (predicted == batch.y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    if model_name == 'novel_transformer_gnn':
                        output, _ = model(batch)
                    else:
                        output = model(batch)
                    
                    # Calculate loss
                    loss = criterion(output, batch.y)
                    
                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += batch.y.size(0)
                    val_correct += (predicted == batch.y).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Store history
            self.training_history[model_name]['train_loss'].append(avg_train_loss)
            self.training_history[model_name]['val_loss'].append(avg_val_loss)
            self.training_history[model_name]['train_acc'].append(train_acc)
            self.training_history[model_name]['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"outputs/models/{model_name}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print(f"Training completed for {model_name}")
        return self.training_history[model_name]
    
    def evaluate_model(self, model_name: str, test_loader: GeometricDataLoader) -> Dict:
        """Evaluate a trained model"""
        print(f"\nEvaluating {model_name}...")
        
        model = self.models[model_name]
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                if model_name == 'novel_transformer_gnn':
                    output, _ = model(batch)
                else:
                    output = model(batch)
                
                # Get predictions
                probabilities = torch.softmax(output, dim=1)
                _, predictions = torch.max(output, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        print(f"Results for {model_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def train_and_evaluate_all_models(self, dataset_path: str, dataset_name: str = "twitter15",
                                    hidden_size: int = 128, num_epochs: int = 50):
        """Train and evaluate all models"""
        print("🚀 Starting comprehensive training and evaluation...")
        
        # Create output directories
        os.makedirs("outputs/models", exist_ok=True)
        os.makedirs("outputs/results", exist_ok=True)
        os.makedirs("outputs/visualizations", exist_ok=True)
        
        # Prepare data
        train_loader, val_loader, test_loader, input_size, num_classes = self.prepare_data(
            dataset_path, dataset_name
        )
        
        # Create models
        self.create_models(input_size, hidden_size, num_classes)
        
        # Train all models
        training_results = {}
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            training_results[model_name] = self.train_model(
                model_name, train_loader, val_loader, num_epochs
            )
        
        # Evaluate all models
        evaluation_results = {}
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}")
            print(f"{'='*50}")
            
            evaluation_results[model_name] = self.evaluate_model(model_name, test_loader)
        
        # Create comprehensive comparison
        self.create_model_comparison(evaluation_results, training_results)
        
        # Save results
        self.save_results(evaluation_results, training_results, dataset_name)
        
        return evaluation_results, training_results
    
    def create_model_comparison(self, evaluation_results: Dict, training_results: Dict):
        """Create comprehensive model comparison"""
        print("\n📊 Creating model comparison...")
        
        # Plot training curves
        self.plot_training_curves(training_results)
        
        # Plot confusion matrices
        self.plot_confusion_matrices(evaluation_results)
        
        # Create performance comparison
        self.plot_performance_comparison(evaluation_results)
        
        # Create comprehensive report
        self.create_comprehensive_report(evaluation_results, training_results)
    
    def plot_training_curves(self, training_results: Dict):
        """Plot training and validation curves for all models"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        model_names = list(training_results.keys())
        
        for i, (model_name, results) in enumerate(training_results.items()):
            epochs = range(1, len(results['train_loss']) + 1)
            color = colors[i % len(colors)]
            
            # Loss curves
            ax1.plot(epochs, results['train_loss'], color=color, label=f'{model_name} (Train)', alpha=0.8)
            ax1.plot(epochs, results['val_loss'], color=color, label=f'{model_name} (Val)', linestyle='--', alpha=0.8)
            
            # Accuracy curves
            ax2.plot(epochs, results['train_acc'], color=color, label=f'{model_name} (Train)', alpha=0.8)
            ax2.plot(epochs, results['val_acc'], color=color, label=f'{model_name} (Val)', linestyle='--', alpha=0.8)
        
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Training and Validation Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final metrics comparison
        final_train_acc = [results['train_acc'][-1] for results in training_results.values()]
        final_val_acc = [results['val_acc'][-1] for results in training_results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, final_train_acc, width, label='Final Train Acc', alpha=0.8)
        ax3.bar(x + width/2, final_val_acc, width, label='Final Val Acc', alpha=0.8)
        ax3.set_title('Final Training and Validation Accuracy', fontweight='bold')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Training time comparison (simulated)
        training_times = [len(results['train_loss']) * 2 for results in training_results.values()]  # Simulated
        ax4.bar(model_names, training_times, color=colors[:len(model_names)], alpha=0.8)
        ax4.set_title('Training Time Comparison (Simulated)', fontweight='bold')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Time (minutes)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("outputs/visualizations/training_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, evaluation_results: Dict):
        """Plot confusion matrices for all models"""
        num_models = len(evaluation_results)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 4))
        
        if num_models == 1:
            axes = [axes]
        
        class_names = ['False', 'True', 'Unverified', 'Non-rumor']
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[i])
            axes[i].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig("outputs/visualizations/confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, evaluation_results: Dict):
        """Plot performance metrics comparison"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values, color=colors[:len(model_names)], alpha=0.8)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("outputs/visualizations/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, evaluation_results: Dict, training_results: Dict):
        """Create comprehensive evaluation report"""
        print("\n📋 Creating comprehensive report...")
        
        report = {
            'summary': {},
            'detailed_results': {},
            'recommendations': []
        }
        
        # Summary statistics
        for model_name, results in evaluation_results.items():
            report['summary'][model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
        
        # Find best model for each metric
        best_models = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            best_model = max(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x][metric])
            best_models[metric] = best_model
        
        # Detailed results
        for model_name, results in evaluation_results.items():
            report['detailed_results'][model_name] = {
                'metrics': {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                },
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'training_history': training_results[model_name]
            }
        
        # Recommendations
        overall_best = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['f1_score'])
        
        report['recommendations'] = [
            f"Best overall model: {overall_best} (F1-Score: {evaluation_results[overall_best]['f1_score']:.4f})",
            f"Best accuracy: {best_models['accuracy']} ({evaluation_results[best_models['accuracy']]['accuracy']:.4f})",
            f"Best precision: {best_models['precision']} ({evaluation_results[best_models['precision']]['precision']:.4f})",
            f"Best recall: {best_models['recall']} ({evaluation_results[best_models['recall']]['recall']:.4f})"
        ]
        
        # Print report
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        for model_name, metrics in report['summary'].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\n🏆 BEST MODELS BY METRIC:")
        for metric, model in best_models.items():
            value = evaluation_results[model][metric]
            print(f"  {metric.replace('_', ' ').title()}: {model} ({value:.4f})")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        return report
    
    def save_results(self, evaluation_results: Dict, training_results: Dict, dataset_name: str):
        """Save all results to files"""
        print(f"\n💾 Saving results for {dataset_name}...")
        
        # Save evaluation results
        results_file = f"outputs/results/evaluation_results_{dataset_name}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, results in evaluation_results.items():
                json_results[model_name] = {
                    'metrics': {
                        'accuracy': float(results['accuracy']),
                        'precision': float(results['precision']),
                        'recall': float(results['recall']),
                        'f1_score': float(results['f1_score'])
                    },
                    'confusion_matrix': results['confusion_matrix'].tolist()
                }
            json.dump(json_results, f, indent=2)
        
        # Save training history
        history_file = f"outputs/results/training_history_{dataset_name}.json"
        with open(history_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        print(f"Training history saved to {history_file}")

def main():
    """Main function to run the trainer"""
    # Initialize trainer
    trainer = RumorDetectionTrainer()
    
    # Dataset path
    dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    
    # Train and evaluate on Twitter15
    print("🚀 Training and evaluating on Twitter15 dataset...")
    results_15, history_15 = trainer.train_and_evaluate_all_models(
        dataset_path, "twitter15", hidden_size=128, num_epochs=30
    )
    
    # Train and evaluate on Twitter16
    print("\n🚀 Training and evaluating on Twitter16 dataset...")
    results_16, history_16 = trainer.train_and_evaluate_all_models(
        dataset_path, "twitter16", hidden_size=128, num_epochs=30
    )
    
    print("\n✅ All training and evaluation completed!")
    print("Check the outputs/ directory for results and visualizations.")

if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader as GeometricDataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
# from sklearn.model_selection import StratifiedKFold
# import os
# import json
# from typing import Dict, List, Tuple, Optional
# import time
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# from .data_preprocessor import load_twitter_dataset
# from .baseline_rvnn import create_baseline_model
# from .novel_tgnn import create_tgnn_model
# from .novel_transformer_gnn import create_transformer_gnn_model

# class RumorDetectionTrainer:
#     """Comprehensive trainer for rumor detection models with comparison"""
    
#     def __init__(self, device: str = None):
#         self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")
        
#         self.models = {}
#         self.optimizers = {}
#         self.schedulers = {}
#         self.training_history = {}
        
#     def prepare_data(self, dataset_path: str, dataset_name: str = "twitter15",
#                     batch_size: int = 32, train_split: float = 0.8, 
#                     val_split: float = 0.1):
#         """Prepare data loaders for training with increased train split for small datasets"""
#         print(f"Loading dataset: {dataset_name}")
        
#         # Load dataset
#         dataset = load_twitter_dataset(dataset_path, dataset_name)
        
#         # Get dataset statistics
#         input_size = dataset[0].x.shape[1] if len(dataset) > 0 else 22
#         num_classes = len(set(data.y.item() for data in dataset))
        
#         print(f"Dataset size: {len(dataset)}")
#         print(f"Input size: {input_size}")
#         print(f"Number of classes: {num_classes}")
        
#         # Split dataset
#         total_size = len(dataset)
#         train_size = int(total_size * train_split)
#         val_size = int(total_size * val_split)
#         test_size = total_size - train_size - val_size
        
#         train_dataset, val_dataset, test_dataset = random_split(
#             dataset, [train_size, val_size, test_size]
#         )
        
#         # Create data loaders
#         train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#         test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
#         return train_loader, val_loader, test_loader, input_size, num_classes
    
#     def create_models(self, input_size: int, hidden_size: int, num_classes: int):
#         """Create both baseline and novel models with lower learning rate"""
#         print("Creating models...")
        
#         # Baseline RvNN model (base paper replication)
#         self.models['baseline_rvnn'] = create_baseline_model(
#             input_size, hidden_size, num_classes, model_type="baseline"
#         ).to(self.device)
        
#         # Novel TGNN model
#         self.models['novel_tgnn'] = create_tgnn_model(
#             input_size, hidden_size, num_classes, model_type="temporal_gnn"
#         ).to(self.device)
        
#         # Novel Transformer-GNN model
#         self.models['novel_transformer_gnn'] = create_transformer_gnn_model(
#             input_size, hidden_size, num_classes, model_type="transformer_gnn"
#         ).to(self.device)
        
#         # Create optimizers and schedulers for each model with lower learning rate
#         for name, model in self.models.items():
#             self.optimizers[name] = optim.AdamW(
#                 model.parameters(), lr=1e-4, weight_decay=0.01  # Lower learning rate
#             )
#             self.schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
#                 self.optimizers[name], mode='min', factor=0.5, patience=5
#             )
#             self.training_history[name] = {
#                 'train_loss': [], 'val_loss': [], 
#                 'train_acc': [], 'val_acc': [],
#                 'val_f1': []  # Added for macro F1 tracking
#             }
        
#         # Print model information
#         for name, model in self.models.items():
#             total_params = sum(p.numel() for p in model.parameters())
#             trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#             print(f"{name}: {total_params:,} total parameters, {trainable_params:,} trainable")
    
#     def train_model(self, model_name: str, train_loader: GeometricDataLoader, 
#                    val_loader: GeometricDataLoader, num_epochs: int = 100,
#                    early_stopping_patience: int = 15):
#         """Train a specific model with class weights and macro F1 tracking"""
#         print(f"\nTraining {model_name}...")
        
#         model = self.models[model_name]
#         optimizer = self.optimizers[model_name]
#         scheduler = self.schedulers[model_name]
        
#         # Compute class weights for imbalance
#         all_labels = torch.cat([batch.y for batch in train_loader])
#         class_weights = torch.tensor(
#             [len(all_labels) / (len(torch.unique(all_labels)) * count) 
#              for count in torch.bincount(all_labels)], 
#             dtype=torch.float
#         ).to(self.device)
#         criterion = nn.CrossEntropyLoss(weight=class_weights)
        
#         best_val_f1 = 0.0
#         best_val_loss = float('inf')
#         patience_counter = 0
        
#         for epoch in range(num_epochs):
#             # Training phase
#             model.train()
#             train_loss = 0.0
#             train_correct = 0
#             train_total = 0
            
#             for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
#                 batch = batch.to(self.device)
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 if model_name == 'novel_transformer_gnn':
#                     output, _ = model(batch)
#                 else:
#                     output = model(batch)
                
#                 # Calculate loss
#                 loss = criterion(output, batch.y)
                
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
                
#                 # Statistics
#                 train_loss += loss.item()
#                 _, predicted = torch.max(output.data, 1)
#                 train_total += batch.y.size(0)
#                 train_correct += (predicted == batch.y).sum().item()
            
#             # Validation phase
#             model.eval()
#             val_loss = 0.0
#             val_correct = 0
#             val_total = 0
#             all_val_preds = []
#             all_val_labels = []
            
#             with torch.no_grad():
#                 for batch in val_loader:
#                     batch = batch.to(self.device)
                    
#                     # Forward pass
#                     if model_name == 'novel_transformer_gnn':
#                         output, _ = model(batch)
#                     else:
#                         output = model(batch)
                    
#                     # Calculate loss
#                     loss = criterion(output, batch.y)
                    
#                     # Statistics
#                     val_loss += loss.item()
#                     _, predicted = torch.max(output.data, 1)
#                     val_total += batch.y.size(0)
#                     val_correct += (predicted == batch.y).sum().item()
                    
#                     # Store predictions and labels for F1 calculation
#                     all_val_preds.extend(predicted.cpu().numpy())
#                     all_val_labels.extend(batch.y.cpu().numpy())
            
#             # Calculate metrics
#             avg_train_loss = train_loss / len(train_loader)
#             avg_val_loss = val_loss / len(val_loader)
#             train_acc = 100 * train_correct / train_total
#             val_acc = 100 * val_correct / val_total
            
#             # Calculate macro F1 score
#             val_f1 = precision_recall_fscore_support(
#                 all_val_labels, all_val_preds, average='macro'
#             )[2]
            
#             # Update learning rate based on validation loss
#             scheduler.step(avg_val_loss)
            
#             # Store history
#             self.training_history[model_name]['train_loss'].append(avg_train_loss)
#             self.training_history[model_name]['val_loss'].append(avg_val_loss)
#             self.training_history[model_name]['train_acc'].append(train_acc)
#             self.training_history[model_name]['val_acc'].append(val_acc)
#             self.training_history[model_name]['val_f1'].append(val_f1)
            
#             # Print progress
#             print(f"Epoch {epoch+1}/{num_epochs}:")
#             print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#             print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            
#             # Early stopping based on validation F1
#             if val_f1 > best_val_f1:
#                 best_val_f1 = val_f1
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#                 # Save best model
#                 torch.save(model.state_dict(), f"outputs/models/{model_name}_best.pth")
#             else:
#                 patience_counter += 1
#                 if patience_counter >= early_stopping_patience:
#                     print(f"Early stopping triggered after {epoch+1} epochs")
#                     break
        
#         print(f"Training completed for {model_name}")
#         return self.training_history[model_name]
    
#     def evaluate_model(self, model_name: str, test_loader: GeometricDataLoader) -> Dict:
#         """Evaluate a trained model using macro metrics"""
#         print(f"\nEvaluating {model_name}...")
        
#         model = self.models[model_name]
#         model.eval()
        
#         all_predictions = []
#         all_labels = []
#         all_probabilities = []
        
#         with torch.no_grad():
#             for batch in test_loader:
#                 batch = batch.to(self.device)
                
#                 # Forward pass
#                 if model_name == 'novel_transformer_gnn':
#                     output, _ = model(batch)
#                 else:
#                     output = model(batch)
                
#                 # Get predictions
#                 probabilities = torch.softmax(output, dim=1)
#                 _, predictions = torch.max(output, 1)
                
#                 all_predictions.extend(predictions.cpu().numpy())
#                 all_labels.extend(batch.y.cpu().numpy())
#                 all_probabilities.extend(probabilities.cpu().numpy())
        
#         # Calculate metrics using macro averaging
#         accuracy = accuracy_score(all_labels, all_predictions)
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             all_labels, all_predictions, average='macro'  # Macro for multi-class
#         )
        
#         # Confusion matrix
#         cm = confusion_matrix(all_labels, all_predictions)
        
#         # Classification report
#         class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
#         results = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'confusion_matrix': cm,
#             'classification_report': class_report,
#             'predictions': all_predictions,
#             'labels': all_labels,
#             'probabilities': all_probabilities
#         }
        
#         print(f"Results for {model_name}:")
#         print(f"  Accuracy: {accuracy:.4f}")
#         print(f"  Precision: {precision:.4f}")
#         print(f"  Recall: {recall:.4f}")
#         print(f"  F1-Score: {f1:.4f}")
        
#         return results
    
#     def train_and_evaluate_all_models(self, dataset_path: str, dataset_name: str = "twitter15",
#                                     hidden_size: int = 128, num_epochs: int = 100):
#         """Train and evaluate all models with increased epochs"""
#         print("🚀 Starting comprehensive training and evaluation...")
        
#         # Create output directories
#         os.makedirs("outputs/models", exist_ok=True)
#         os.makedirs("outputs/results", exist_ok=True)
#         os.makedirs("outputs/visualizations", exist_ok=True)
        
#         # Prepare data
#         train_loader, val_loader, test_loader, input_size, num_classes = self.prepare_data(
#             dataset_path, dataset_name
#         )
        
#         # Create models
#         self.create_models(input_size, hidden_size, num_classes)
        
#         # Train all models
#         training_results = {}
#         for model_name in self.models.keys():
#             print(f"\n{'='*50}")
#             print(f"Training {model_name}")
#             print(f"{'='*50}")
            
#             training_results[model_name] = self.train_model(
#                 model_name, train_loader, val_loader, num_epochs
#             )
        
#         # Evaluate all models
#         evaluation_results = {}
#         for model_name in self.models.keys():
#             print(f"\n{'='*50}")
#             print(f"Evaluating {model_name}")
#             print(f"{'='*50}")
            
#             evaluation_results[model_name] = self.evaluate_model(model_name, test_loader)
        
#         # Create comprehensive comparison
#         self.create_model_comparison(evaluation_results, training_results)
        
#         # Save results
#         self.save_results(evaluation_results, training_results, dataset_name)
        
#         return evaluation_results, training_results
    
#     # ... (keep all the remaining methods unchanged: create_model_comparison, plot_training_curves, 
#     # plot_confusion_matrices, plot_performance_comparison, create_comprehensive_report, save_results)

# def main():
#     """Main function to run the trainer"""
#     # Initialize trainer
#     trainer = RumorDetectionTrainer()
    
#     # Dataset path
#     dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    
#     # Train and evaluate on Twitter15
#     print("🚀 Training and evaluating on Twitter15 dataset...")
#     results_15, history_15 = trainer.train_and_evaluate_all_models(
#         dataset_path, "twitter15", hidden_size=128, num_epochs=100  # Increased epochs
#     )
    
#     # Train and evaluate on Twitter16
#     print("\n🚀 Training and evaluating on Twitter16 dataset...")
#     results_16, history_16 = trainer.train_and_evaluate_all_models(
#         dataset_path, "twitter16", hidden_size=128, num_epochs=100  # Increased epochs
#     )
    
#     print("\n✅ All training and evaluation completed!")
#     print("Check the outputs/ directory for results and visualizations.")

# if __name__ == "__main__":
#     main()