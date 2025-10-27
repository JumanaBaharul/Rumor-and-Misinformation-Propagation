#!/usr/bin/env python3
"""
Main execution script for the Enhanced Graph-Based Rumor Detection System

This script orchestrates the complete pipeline:
1. Data preprocessing and temporal graph construction
2. Dataset visualization and analysis
3. Model training and evaluation (Baseline RvNN + Novel models)
4. Comprehensive comparison and reporting

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function"""
    print("🚀 Enhanced Graph-Based Rumor Detection System")
    print("=" * 60)
    print("Based on research paper: https://arxiv.org/pdf/2508.07201")
    print("=" * 60)
    
    # Check if data directory exists
    data_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure the Twitter15/16 datasets are in the correct location.")
        print("Expected structure:")
        print("  data/")
        print("  └── Rumor Detection Dataset (Twitter15 and Twitter16)/")
        print("      ├── twitter15/")
        print("      │   ├── label.txt")
        print("      │   └── source_tweets.txt")
        print("      └── twitter16/")
        print("          ├── label.txt")
        print("          └── source_tweets.txt")
        return
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)
    
    try:
        # Step 1: Data Preprocessing and Temporal Graph Construction
        print("\n📊 Step 1: Data Preprocessing and Temporal Graph Construction")
        print("-" * 60)
        
        from src.data_preprocessor import TwitterDatasetPreprocessor
        from src.temporal_graph_builder import TemporalGraphBuilder
        
        # Initialize preprocessor and temporal graph builder
        preprocessor = TwitterDatasetPreprocessor(data_path, "twitter15")
        temporal_builder = TemporalGraphBuilder(data_path, "twitter15")
        
        # Get dataset statistics
        stats = preprocessor.get_dataset_stats()
        print(f"Dataset loaded: {stats['total_samples']} samples")
        print(f"Label distribution: {stats['label_distribution']}")
        
        # Example temporal graph construction
        if len(preprocessor.labels) > 0:
            tweet_id = list(preprocessor.labels.keys())[0]
            tweet_text = preprocessor.source_tweets[tweet_id]
            features = preprocessor.extract_features(tweet_text)
            temporal_features = preprocessor.create_temporal_features(tweet_id)
            
            # Build temporal propagation graph
            graph, analysis = temporal_builder.build_comprehensive_temporal_graph(
                tweet_id, features, temporal_features
            )
            print(f"Example temporal graph built: {analysis['num_nodes']} nodes, {analysis['num_edges']} edges")
        
        # Step 2: Dataset Visualization
        print("\n📈 Step 2: Dataset Visualization and Analysis")
        print("-" * 60)
        
        from src.data_visualizer import TwitterDataVisualizer
        
        # Create visualizer for Twitter15
        visualizer_15 = TwitterDataVisualizer(data_path, "twitter15")
        print("Creating visualizations for Twitter15...")
        
        # Generate comprehensive visualization report
        stats_15 = visualizer_15.create_comprehensive_report("outputs/visualizations_twitter15")
        
        # Create visualizer for Twitter16
        visualizer_16 = TwitterDataVisualizer(data_path, "twitter16")
        print("Creating visualizations for Twitter16...")
        
        # Generate comprehensive visualization report
        stats_16 = visualizer_16.create_comprehensive_report("outputs/visualizations_twitter16")
        
        # Step 3: Model Training and Evaluation
        print("\n🤖 Step 3: Model Training and Evaluation")
        print("-" * 60)
        
        from src.model_trainer import RumorDetectionTrainer
        
        # Initialize trainer
        trainer = RumorDetectionTrainer()
        
        # Train and evaluate on Twitter15
        print("🚀 Training and evaluating on Twitter15 dataset...")
        results_15, history_15 = trainer.train_and_evaluate_all_models(
            data_path, "twitter15", hidden_size=128, num_epochs=30
        )
        
        # Train and evaluate on Twitter16
        print("\n🚀 Training and evaluating on Twitter16 dataset...")
        results_16, history_16 = trainer.train_and_evaluate_all_models(
            data_path, "twitter16", hidden_size=128, num_epochs=30
        )
        
        # Step 4: Final Summary and Comparison
        print("\n📋 Step 4: Final Summary and Model Comparison")
        print("-" * 60)
        
        print("\n🏆 FINAL RESULTS SUMMARY")
        print("=" * 40)
        
        # Twitter15 Results
        print(f"\n📊 Twitter15 Dataset Results:")
        for model_name, results in results_15.items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall: {results['recall']:.4f}")
            print(f"    F1-Score: {results['f1_score']:.4f}")
        
        # Twitter16 Results
        print(f"\n📊 Twitter16 Dataset Results:")
        for model_name, results in results_16.items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall: {results['recall']:.4f}")
            print(f"    F1-Score: {results['f1_score']:.4f}")
        
        # Find best overall model
        all_results = {**results_15, **results_16}
        best_model = max(all_results.keys(), 
                        key=lambda x: all_results[x]['f1_score'])
        best_f1 = all_results[best_model]['f1_score']
        
        print(f"\n🏅 BEST OVERALL MODEL: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")
        
        print(f"\n✅ All steps completed successfully!")
        print(f"📁 Check the outputs/ directory for detailed results and visualizations.")
        print(f"🤖 Best model saved as: outputs/models/{best_model}_best.pth")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_quick_demo():
    """Run a quick demo for testing"""
    print("🚀 Running Quick Demo Mode")
    print("=" * 40)
    
    # Check if data exists
    data_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    if not os.path.exists(data_path):
        print("❌ Data directory not found. Please ensure datasets are available.")
        return False
    
    try:
        # Quick preprocessing test
        print("📊 Testing data preprocessing...")
        from src.data_preprocessor import TwitterDatasetPreprocessor
        
        preprocessor = TwitterDatasetPreprocessor(data_path, "twitter15")
        stats = preprocessor.get_dataset_stats()
        print(f"✅ Dataset loaded: {stats['total_samples']} samples")
        
        # Quick model creation test
        print("🤖 Testing model creation...")
        from src.baseline_rvnn import create_baseline_model
        from src.novel_tgnn import create_tgnn_model
        from src.novel_transformer_gnn import create_transformer_gnn_model
        
        # Create models
        baseline = create_baseline_model(22, 64, 4, model_type="baseline")
        tgnn = create_tgnn_model(22, 64, 4, model_type="temporal_gnn")
        transformer_gnn = create_transformer_gnn_model(22, 64, 4, model_type="transformer_gnn")
        
        print("✅ All models created successfully!")
        
        # Quick visualization test
        print("📈 Testing visualization...")
        from src.data_visualizer import TwitterDataVisualizer
        
        visualizer = TwitterDataVisualizer(data_path, "twitter15")
        print("✅ Visualizer created successfully!")
        
        print("\n🎉 Quick demo completed successfully!")
        print("All components are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_help():
    """Show help information"""
    print("Enhanced Graph-Based Rumor Detection System")
    print("=" * 50)
    print("Usage:")
    print("  python main.py              # Run full pipeline")
    print("  python main.py --demo       # Run quick demo")
    print("  python main.py --help       # Show this help")
    print("\nOptions:")
    print("  --demo    Run quick demo mode for testing")
    print("  --help    Show this help message")
    print("\nDescription:")
    print("  This system implements:")
    print("  1. Data preprocessing for Twitter15/16 datasets")
    print("  2. Temporal propagation graph construction")
    print("  3. Baseline RvNN model (base paper replication)")
    print("  4. Novel TGNN model with temporal modeling")
    print("  5. Novel Transformer-GNN model")
    print("  6. Comprehensive model comparison and evaluation")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            success = run_quick_demo()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help":
            show_help()
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            show_help()
            sys.exit(1)
    
    # Run main pipeline
    success = main()
    sys.exit(0 if success else 1)
