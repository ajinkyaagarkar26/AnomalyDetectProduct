# Anomaly Detection Model Explanation

## Overview

This document provides a comprehensive explanation of the log anomaly detection system based on DeepLog architecture. The system uses an LSTM-based neural network to learn normal log patterns and detect anomalies through next log event prediction.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Training Process](#training-process)
3. [Loss Calculation](#loss-calculation)
4. [Weight Updates](#weight-updates)
5. [Prediction Methods](#prediction-methods)
6. [TopK Configuration](#topk-configuration)
7. [Data Pipeline](#data-pipeline)
8. [Configuration Parameters](#configuration-parameters)

## Model Architecture

### DeepLog Model Structure

The main model is implemented in `/logdeep/models/lstm.py` as the `Deeplog` class:

```python
class Deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(Deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer to convert log event IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Output layer for next log event prediction
        self.fc0 = nn.Linear(hidden_size, vocab_size)
```

### Model Components

| Component | Purpose | Parameters |
|-----------|---------|------------|
| **Embedding Layer** | Converts log event IDs to dense vectors | vocab_size × embedding_dim |
| **LSTM Layer** | Models sequential dependencies in log events | Input-to-hidden, hidden-to-hidden weights + biases |
| **Linear Layer** | Predicts probability distribution over next events | hidden_size × vocab_size + bias |

### Default Architecture Parameters

- **Hidden Size**: 64
- **Number of Layers**: 2
- **Embedding Dimension**: 50
- **Vocabulary Size**: 24 (configurable based on data)

## Training Process

### Problem Formulation

The system treats anomaly detection as a **next log event prediction** problem:
- **Input**: Sliding window of log events (e.g., [E1, E2, E3, ..., E9, E10])
- **Target**: The next log event in the sequence (e.g., E11)
- **Training Objective**: Learn to predict the next log event given a sequence

### Training Loop Structure

**Location**: `/logdeep/tools/train.py`

```python
def start_train(self):
    for epoch in range(self.start_epoch, self.max_epoch):
        if self.early_stopping:
            break
        self.train(epoch)    # Training phase
        self.valid(epoch)    # Validation phase
        self.save_log()      # Log metrics
```

### Data Preparation

1. **Log Parsing**: Raw logs → Structured events using Drain parser
2. **Event Mapping**: Log events → Integer IDs using vocabulary
3. **Sliding Window**: Create sequences of fixed length (default: 10 events)
4. **Dataset Creation**: Generate (input_sequence, next_event) pairs

## Loss Calculation

### Loss Function: Cross-Entropy Loss

**Location**: `/logdeep/tools/train.py`, lines 227-240

```python
# Forward pass through model
output = self.model(features=features, device=self.device)
output = output.squeeze()  # Shape: [batch_size, vocab_size]
label = label.view(-1).to(self.device)  # Shape: [batch_size] - ground truth

# Cross-entropy loss calculation
loss = self.criterion(output, label)  # nn.CrossEntropyLoss(ignore_index=0)
```

### Mathematical Formula

For each sample in the batch:
```
Loss = -log(P(true_next_event))
```

Where `P(true_next_event)` is the predicted probability of the correct next log event.

### Loss Function Configuration

```python
self.criterion = nn.CrossEntropyLoss(ignore_index=0)
```

- **ignore_index=0**: Ignores padding tokens in loss calculation
- **Reduction**: Default 'mean' across batch

## Weight Updates

### Primary Update Location

**File**: `/logdeep/tools/train.py`, lines 232-240

```python
loss = self.criterion(output, label)
total_losses += float(loss)
loss /= self.accumulation_step  # Scale for gradient accumulation
loss.backward()                 # Compute gradients

# Update weights after accumulating gradients
if (i + 1) % self.accumulation_step == 0:
    self.optimizer.step()      # ← WEIGHTS & BIASES UPDATED HERE
    self.optimizer.zero_grad() # Reset gradients
```

### Optimizer Configuration

**Location**: `/logdeep/tools/train.py`, lines 137-148

```python
if options['optimizer'] == 'adam':
    self.optimizer = torch.optim.Adam(
        self.model.parameters(),  # All trainable parameters
        lr=options['lr'],         # Learning rate: 0.001
        betas=(0.9, 0.999),      # Adam momentum parameters
    )
```

### Updated Parameters

During `optimizer.step()`, the following parameters are updated:

1. **Embedding Weights**: `self.embedding.weight` (vocab_size × embedding_dim)
2. **LSTM Parameters**:
   - Input-to-hidden weights and biases
   - Hidden-to-hidden weights and biases
   - Gate parameters (forget, input, output, cell)
3. **Linear Layer**: `self.fc0.weight` and `self.fc0.bias`

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Learning Rate** | 0.001 | Base learning rate |
| **Batch Size** | 32 | Number of samples per batch |
| **Max Epochs** | 200 | Maximum training epochs |
| **Early Stopping** | 10 epochs | Stop if no improvement |
| **Accumulation Steps** | 1 | Gradient accumulation (disabled by default) |

## Prediction Methods

### 1. Unsupervised Prediction (`predict_unsupervised`)

**Method**: Next log event prediction with top-K candidate checking

```python
def detect_logkey_anomaly(self, output, label):
    for i in range(len(label)):
        # Get top-K most likely next events
        predicted = torch.argsort(output[i])[-self.num_candidates:].clone().detach().cpu()
        # Check if actual event is in top-K predictions
        if label[i] not in predicted:
            num_anomaly += 1  # Flag as anomaly
```

**Usage**: 
- Loads test_normal and test_abnormal data
- Computes anomaly detection metrics
- Finds optimal threshold for classification

### 2. Unsupervised Prediction 2 (`predict_unsupervised_2`)

**Method**: Real-time prediction on new data

```python
# Loads event sequences from CSV
pred_loader, _ = generate_2('./predOutput/', 'event_sequence.csv')
# Returns list of anomalous sequence indices
pred_results, pred_errors = self.unsupervised_helper_2(model, pred_loader, vocab, ...)
```

**Usage**: 
- Processes new log data for anomaly detection
- Returns indices of anomalous sequences
- Used for production/real-time detection

### 3. Supervised Prediction (`predict_supervised`)

**Method**: Binary classification with labeled data

```python
# Apply sigmoid to get probabilities
output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
# Use fixed threshold for classification
predicted = (output < 0.2).astype(int)
```

**Usage**:
- Requires labeled test data with ground truth
- Uses binary classification approach
- Outputs precision, recall, F1-score metrics

## TopK Configuration

### Configuration Location

**File**: `/deeplog.py`, line 76
```python
options['num_candidates'] = 9  # TopK = 9 candidates
```

### How TopK Works

1. **Model Output**: Probability distribution over all log event types
2. **TopK Selection**: Extract top 9 most probable next events
3. **Anomaly Detection**: If actual event ∉ top-9 predictions → Anomaly

```python
# Get indices sorted by probability (ascending)
predicted = torch.argsort(output[i])[-self.num_candidates:]  # Last 9 elements
```

### TopK Impact

| K Value | Detection Behavior | Precision | Recall |
|---------|-------------------|-----------|--------|
| Low (3-5) | Strict anomaly detection | Higher | Lower |
| Medium (9) | Balanced approach | Medium | Medium |
| High (15-20) | Lenient detection | Lower | Higher |

**Current Setting**: K=9 provides balanced anomaly detection sensitivity.

## Data Pipeline

### 1. Raw Log Processing

```
Raw Logs → Drain Parser → Structured Events → Event Mapping → Integer Sequences
```

### 2. Training Data Generation

```
Event Sequences → Sliding Window → (Input, Target) Pairs → DataLoader → Model
```

### 3. Prediction Data Flow

```
New Logs → Parser → Event Mapping → Sliding Window → Model → Anomaly Detection
```

### File Structure

```
output/deeplog/
├── train                    # Training sequences
├── test_normal             # Normal test sequences  
├── test_abnormal           # Abnormal test sequences
├── vocab.pkl               # Event vocabulary
├── deeplog_log_templates.json  # Event mappings
└── deeplog/
    ├── bestloss.pth        # Trained model checkpoint
    └── scale.pkl           # Feature scaling (if used)
```

## Configuration Parameters

### Model Parameters

```python
# Model architecture
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['embedding_dim'] = 50
options['vocab_size'] = 24  # Determined from data
options['num_classes'] = options['vocab_size']
```

### Training Parameters

```python
# Training configuration
options['batch_size'] = 32
options['accumulation_step'] = 1
options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 200
options['n_epochs_stop'] = 10  # Early stopping patience
```

### Data Parameters

```python
# Data processing
options['window_size'] = 10      # Sliding window size
options['min_len'] = 5          # Minimum sequence length
options['train_ratio'] = 1      # Use all training data
options['valid_ratio'] = 0.1    # 10% for validation
options['test_ratio'] = 0.1     # 10% for testing
```

### Prediction Parameters

```python
# Anomaly detection
options['num_candidates'] = 9   # TopK for anomaly detection
options['model_path'] = options['save_dir'] + 'bestloss.pth'
options['threshold'] = None     # Dynamic threshold search
```

## Performance Metrics

### Unsupervised Metrics

- **Best Threshold**: Optimal anomaly detection threshold
- **Confusion Matrix**: TP, TN, FP, FN counts
- **Performance**: Precision, Recall, F1-measure
- **Execution Time**: Processing time for evaluation

### Supervised Metrics

- **Binary Classification**: Direct 0/1 predictions
- **Fixed Threshold**: Uses threshold = 0.2
- **Standard Metrics**: Precision, Recall, F1-score

## Model Variants

The codebase includes multiple model implementations:

1. **Deeplog**: Main LSTM-based model with embedding
2. **deeplog**: Simple LSTM without embedding  
3. **robustlog**: Basic LSTM variant
4. **loganomaly**: Multi-LSTM with concatenation

**Current Default**: `Deeplog` with embedding layer for better representation learning.

## Usage Examples

### Training a Model

```python
from deeplog import train
train()  # Trains model and saves to output/deeplog/deeplog/bestloss.pth
```

### Running Prediction

```python
from deeplog import predict
result = predict()  # Runs unsupervised anomaly detection
```

### Custom Prediction

```python
from logdeep.tools.predict import Predicter
from deeplog import Model, options

predicter = Predicter(Model, options)
results = predicter.predict_unsupervised_2()  # For new data
```

## API Integration

The system is integrated with a FastAPI web service:

```python
@app.post("/deeplog-predict/")
async def run_deeplog_predict(request: Request):
    result = ui_deeplog_predict()
    return {"status": "success", "result": result}
```

This enables web-based anomaly detection for uploaded log files.

---

**Note**: This system is designed for log anomaly detection in distributed systems, particularly effective for detecting unusual patterns in system logs, application logs, and security logs.
