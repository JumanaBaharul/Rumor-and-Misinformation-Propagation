# Enhanced Graph-Based Rumor and Misinformation Propagation Detection

This repository implements an **Enhanced Graph-Based Rumor and Misinformation Propagation Detection system** based on the research paper "Propagation Tree Is Not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection" (https://arxiv.org/pdf/2508.07201).

## Why the current pipeline struggles

Propagation trees are synthetic, not the real cascades. `build_propagation_graph` fabricates the reply structure with random fan-out and random node attributes instead of using the retweet/reply chains supplied in the Twitter15/16 dataset. Because the graph topology and temporal statistics are sampled independently of the labels, the models are trying to learn from noise, which drives the metrics toward chance level.

Temporal features are derived from hashes of the tweet IDs. `create_temporal_features` maps each tweet ID to pseudo time-of-day, weekday, and timestamp values by modular arithmetic on the ID. These proxies do not reflect actual posting dynamics and therefore cannot encode the propagation patterns that separate rumors from non-rumors.

Node features mostly reuse the source tweet features. Replies inherit the source text features (except for a few random overrides), so every node in the cascade looks almost identical. Graph encoders need discriminative node-level signals to build meaningful representations.

Dataset splitting ignores class balance. The trainer performs a raw `random_split` without stratification on the heavily imbalanced Twitter15/16 labels, so some classes can disappear from the validation or test sets, making the reported precision/recall unreliable and destabilizing training.

## What to do next

Parse the real conversation trees. Use the `structure.txt` files in each dataset to reconstruct the genuine reply graph (tweet IDs, parent IDs, timestamps). Once you build the actual propagation tree, you can attach credible temporal intervals (time since source, depth in cascade) instead of synthetic numbers.

Recompute node features from each tweet’s content and metadata. For every node in the reconstructed tree, load the associated tweet text from `tweets/<tweet_id>.json` (or the provided text dump) and extract textual, sentiment, and user features per node. Reserve global features (e.g., total cascade size) for graph-level attributes.

Derive temporal encodings from real timestamps. Convert the tweet creation times into relative delays (seconds/minutes since the root), absolute clock features, and rolling statistics such as growth rate. Temporal Graph Networks will only benefit if these values mirror actual diffusion speed.

Add edge direction and type information from the dataset. Distinguish between reply, retweet, and quotation edges if available, and store the parent-child order so temporal models can process streams chronologically.

Adopt stratified or event-based splits. Replace the plain `random_split` with stratified sampling (e.g., `StratifiedKFold`) or leave-one-event-out evaluation to keep label proportions stable across train/val/test sets.

Benchmark against simpler baselines. Before fine-tuning the TGNN or Transformer-GNN, evaluate logistic regression or traditional GCNs on graph-level features to verify that the processed data carries predictive signal.

Tune hyperparameters with validation curves. Once the data pipeline is trustworthy, search learning rates, hidden sizes, dropout, and training epochs—especially important for small datasets like Twitter15/16.

## 🚀 **NOVELTY & ENHANCEMENTS**

### **Beyond the Base Paper:**
- **Temporal Graph Neural Networks (TGNN)**: Advanced temporal modeling with transformer encoders and temporal convolutions
- **Enhanced RvNN with Transformer**: Tree LSTM enhanced with multi-head attention and temporal convolution
- **Comprehensive Feature Engineering**: 22+ features including sentiment, temporal, and linguistic analysis
- **Advanced Visualization**: Interactive dataset exploration and model comparison
- **Model Comparison Framework**: Direct performance comparison with detailed metrics

## 🎯 **Key Features**

### **1. Enhanced Data Preprocessing**
- **Advanced Feature Extraction**: Sentiment analysis, linguistic features, temporal patterns
- **Temporal Propagation Graphs**: Dynamic graph construction with time-aware features
- **Multi-modal Features**: Text, temporal, and structural features combined

### **2. Novel Model Architectures**
- **Temporal GNN**: 
  - Graph Attention Networks with temporal encoding
  - Transformer-based temporal modeling
  - Temporal convolution layers
  - Multi-head attention mechanisms
  
- **Enhanced RvNN**:
  - Tree LSTM with enhanced temporal attention
  - Multi-head self-attention
  - Temporal convolution integration
  - Advanced feature transformation

### **3. Comprehensive Evaluation**
- **Multi-metric Comparison**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix Analysis**: Detailed error pattern analysis
- **Training Curves**: Learning progress visualization
- **Model Complexity Analysis**: Parameter count and architecture comparison

## 📁 **Project Structure**

```
Rumor and Misinformation Propagation/
├── enhanced_preprocessor.py          # Enhanced data preprocessing with 22+ features
├── dataset_visualizer.py            # Comprehensive dataset visualization
├── enhanced_rvnn_model.py           # Novel model architectures (TGNN + Enhanced RvNN)
├── enhanced_trainer.py              # Training and evaluation framework
├── main_execution.py                # Main orchestration script
├── requirements.txt                  # Dependencies
└── README.md                        # This file
```

## 🛠️ **Installation**

Follow these commands in the VS Code terminal to set up the project locally:

1. **Clone the repository and enter the project directory**
   ```bash
   git clone https://github.com/JumanaBaharul/Rumor-and-Misinformation-Propagation.git
   cd Rumor-and-Misinformation-Propagation
   ```

2. **(Optional) Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Verify your Python environment**
   ```bash
   python -m compileall src
   ```

5. **Run the full pipeline**
   ```bash
   python main.py
   ```

6. **Review results and artifacts**
   ```bash
   ls outputs
   ```

## 🚀 **Quick Start**

### **Option 1: Full Pipeline Execution**
```bash
python main.py
```

### **Option 2: Quick Demo**
```bash
python main.py --demo
```

### **Option 3: Help**
```bash
python main.py --help
```

## 📊 **Dataset Processing**

### **Enhanced Feature Extraction (22 Features):**
1. **Text Features**: Word count, character count, average word length
2. **Sentiment Features**: Sentiment polarity, subjectivity
3. **Social Features**: Hashtag count, mention count, URL count
4. **Linguistic Features**: Exclamation count, question count, emoji count
5. **Temporal Features**: Hour, day, month, weekend indicator, business hours
6. **Structural Features**: Node level, node type (source/reply)

### **Temporal Propagation Graphs:**
- Dynamic graph construction based on temporal relationships
- Time-aware edge weighting and node features
- Temporal snapshots for sequence modeling

## 🤖 **Model Architectures**

### **1. Temporal Graph Neural Network (TGNN)**
```
Input Features → Feature Transform → Temporal PE → GNN Layers → 
Temporal Attention → Transformer Encoder → Temporal Conv → Classifier
```

**Key Components:**
- **Temporal Positional Encoding**: Captures temporal order information
- **Graph Attention Networks**: Multi-head attention for graph structure
- **Transformer Encoder**: Sequential temporal modeling
- **Temporal Convolution**: Local temporal pattern extraction

### **2. Enhanced RvNN with Transformer**
```
Input Features → Feature Transform → Tree LSTM → Enhanced Attention → 
Temporal Conv → Global Pooling → Classifier
```

**Key Components:**
- **Tree LSTM**: Hierarchical structure processing
- **Multi-head Attention**: Enhanced temporal attention mechanism
- **Temporal Convolution**: Sequence modeling capabilities
- **Layer Normalization**: Training stability improvements

## 📈 **Training & Evaluation**

### **Training Process:**
1. **Data Splitting**: 80% train, 10% validation, 10% test
2. **Model Training**: Both models trained with early stopping
3. **Hyperparameter Optimization**: Learning rate scheduling, weight decay
4. **Performance Monitoring**: Real-time metrics tracking

### **Evaluation Metrics:**
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive classes
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### **Model Comparison:**
- **Direct Performance Comparison**: Side-by-side metric comparison
- **Training Curves**: Learning progress visualization
- **Complexity Analysis**: Parameter count and architecture details
- **Winner Determination**: Automatic best model selection

## 🎨 **Visualization Features**

### **Dataset Exploration:**
- **Label Distribution**: Bar charts and pie charts
- **Feature Distributions**: Histograms with statistical summaries
- **Correlation Analysis**: Feature correlation heatmaps
- **Temporal Analysis**: Time-based activity patterns
- **Propagation Graphs**: Example graph visualizations

### **Training Analysis:**
- **Loss Curves**: Training and validation loss over time
- **Accuracy Curves**: Training and validation accuracy over time
- **Model Comparison**: Final performance comparison charts
- **Confusion Matrices**: Error pattern visualization

## 📊 **Expected Results**

### **Performance Comparison:**
The system automatically compares both models and provides:
- **Winner Selection**: Best performing model identification
- **Improvement Metrics**: Performance improvement percentages
- **Detailed Analysis**: Per-class performance breakdown
- **Recommendations**: Next steps for model improvement

### **Output Files:**
- **Model Checkpoints**: Best model weights for each architecture
- **Training Curves**: Learning progress visualizations
- **Confusion Matrices**: Error pattern analysis
- **Comprehensive Report**: Detailed performance comparison
- **Final Summary**: Executive summary of results

## 🔧 **Customization Options**

### **Model Parameters:**
- **Hidden Size**: Adjustable from 64 to 512
- **Number of Layers**: Configurable GNN/transformer layers
- **Attention Heads**: Multi-head attention configuration
- **Dropout Rate**: Regularization strength adjustment

### **Training Parameters:**
- **Learning Rate**: Optimizable learning rate
- **Batch Size**: Adjustable batch size for memory constraints
- **Number of Epochs**: Training duration control
- **Early Stopping**: Configurable patience for overfitting prevention

## 🚀 **Advanced Usage**

### **GPU Acceleration:**
```python
trainer = EnhancedRumorTrainer(device="cuda", save_dir="outputs")
```

### **Custom Model Creation:**
```python
from enhanced_rvnn_model import create_enhanced_model

# Create custom model
model = create_enhanced_model(
    input_size=22,
    hidden_size=256,
    num_classes=4,
    model_type="temporal_gnn",
    num_layers=4,
    dropout=0.3
)
```

### **Hyperparameter Tuning:**
```python
# Custom training parameters
results = trainer.train_and_evaluate_all_models(
    dataset_path=dataset_path,
    dataset_name="twitter15",
    hidden_size=256,
    num_epochs=100
)
```

## 📚 **Research Contributions**

### **Novel Contributions:**
1. **Temporal Graph Neural Networks**: Advanced temporal modeling for rumor detection
2. **Enhanced RvNN**: Transformer-enhanced recursive neural networks
3. **Comprehensive Feature Engineering**: Multi-modal feature extraction
4. **Model Comparison Framework**: Systematic evaluation methodology
5. **Advanced Visualization**: Interactive dataset and model analysis

### **Beyond Base Paper:**
- **Temporal Modeling**: Advanced temporal dependencies capture
- **Attention Mechanisms**: Multi-head attention for better feature learning
- **Feature Engineering**: Comprehensive feature extraction pipeline
- **Evaluation Framework**: Systematic model comparison and analysis

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **Base Research Paper**: "Propagation Tree Is Not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection"
- **PyTorch Geometric**: Graph neural network framework
- **Open Source Community**: Various open-source libraries and tools

## 📞 **Support**

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation and examples

---

**🎉 Ready to revolutionize rumor detection with advanced temporal modeling and comprehensive evaluation!**
