# Design Document - Food Classification Training System

## Overview

The Food Classification Training System is a production-ready PyTorch-based machine learning pipeline designed to train deep learning models for food image recognition. The system implements transfer learning using pre-trained convolutional neural networks (MobileNetV2 or EfficientNet-B0), integrates MLOps practices through MLflow for experiment tracking and model versioning, and provides comprehensive evaluation metrics for model performance analysis.

The training pipeline is designed to be:
- **Reproducible**: All hyperparameters and random seeds are tracked
- **Configurable**: Command-line arguments allow easy experimentation
- **Production-ready**: Includes proper error handling, logging, and artifact management
- **Resource-efficient**: Optimized for both GPU and CPU execution
- **Interview-ready**: Demonstrates end-to-end MLOps capabilities

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Script (train_model.py)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Dataset    │───▶│    Model     │───▶│   Training   │    │
│  │   Manager    │    │   Builder    │    │    Engine    │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ Data Loaders │    │ Frozen Layers│    │  Optimizer   │    │
│  │ Augmentation │    │  Classifier  │    │  Scheduler   │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              MLflow Tracking                         │     │
│  │  • Log hyperparameters                               │     │
│  │  • Log metrics per epoch                             │     │
│  │  • Save model artifacts                              │     │
│  │  • Track confusion matrix                            │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              Model Artifacts                         │     │
│  │  • model_v{version}.pth                              │     │
│  │  • class_to_idx.json                                 │     │
│  │  • model_config.json                                 │     │
│  │  • evaluation_results.json                           │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Initialization Phase**:
   - Parse command-line arguments
   - Initialize MLflow experiment
   - Set random seeds for reproducibility
   - Detect and configure device (GPU/CPU)

2. **Data Preparation Phase**:
   - Load Food-101 dataset
   - Create train/validation split
   - Apply transformations and augmentation
   - Create data loaders with batching

3. **Model Setup Phase**:
   - Load pre-trained model (MobileNetV2/EfficientNet-B0)
   - Freeze early layers
   - Replace classifier head
   - Initialize optimizer and scheduler

4. **Training Phase**:
   - Iterate through epochs
   - Forward pass, loss calculation, backward pass
   - Update weights, log metrics
   - Validate after each epoch
   - Save checkpoints for best model

5. **Evaluation Phase**:
   - Calculate comprehensive metrics
   - Generate confusion matrix
   - Identify worst-performing classes
   - Save evaluation results

6. **Artifact Saving Phase**:
   - Save model state dict
   - Save class mappings and config
   - Log artifacts to MLflow
   - Create timestamped directory

## Components and Interfaces

### 1. DatasetManager

**Responsibility**: Handle dataset loading, splitting, and transformation

```python
class DatasetManager:
    """Manages Food-101 dataset loading and preprocessing."""
    
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store/load dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for parallel loading
        """
        
    def get_transforms(self, is_training: bool) -> transforms.Compose:
        """
        Get image transformations for training or validation.
        
        Args:
            is_training: Whether to apply training augmentation
            
        Returns:
            Composed transformations
        """
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
        """
        Prepare train and validation data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, idx_to_class mapping)
        """
```

**Key Methods**:
- `get_transforms()`: Returns appropriate transforms for train/val
- `prepare_data()`: Creates and returns data loaders
- `get_class_names()`: Returns mapping of indices to class names

### 2. ModelBuilder

**Responsibility**: Create and configure the neural network model

```python
class ModelBuilder:
    """Builds and configures transfer learning models."""
    
    def __init__(self, model_name: str, num_classes: int, freeze_layers: int):
        """
        Initialize model builder.
        
        Args:
            model_name: Name of pre-trained model ('mobilenet_v2' or 'efficientnet_b0')
            num_classes: Number of output classes
            freeze_layers: Number of early layers to freeze
        """
        
    def build_model(self) -> nn.Module:
        """
        Build and configure the model.
        
        Returns:
            Configured PyTorch model
        """
        
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for saving.
        
        Returns:
            Dictionary with model metadata
        """
```

**Key Methods**:
- `build_model()`: Creates model with frozen layers and new classifier
- `freeze_backbone()`: Freezes specified number of layers
- `replace_classifier()`: Replaces final layer for target classes
- `get_model_config()`: Returns model configuration dict

### 3. TrainingEngine

**Responsibility**: Execute training loop with optimization and validation

```python
class TrainingEngine:
    """Handles model training and validation loops."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        num_epochs: int,
        early_stopping_patience: int
    ):
        """Initialize training engine with all components."""
        
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        
    def train(self) -> Dict[str, List[float]]:
        """
        Execute full training loop.
        
        Returns:
            Dictionary with training history
        """
```

**Key Methods**:
- `train_epoch()`: Executes one training epoch
- `validate()`: Runs validation and returns metrics
- `train()`: Main training loop with early stopping
- `save_checkpoint()`: Saves model checkpoint

### 4. MLflowTracker

**Responsibility**: Track experiments and log artifacts to MLflow

```python
class MLflowTracker:
    """Manages MLflow experiment tracking."""
    
    def __init__(self, experiment_name: str, run_name: str):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
            run_name: Name for this specific run
        """
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for a specific step/epoch."""
        
    def log_model(self, model_path: str, artifact_path: str) -> None:
        """Log model artifact."""
        
    def log_figure(self, figure: plt.Figure, artifact_file: str) -> None:
        """Log matplotlib figure."""
```

**Key Methods**:
- `log_params()`: Logs hyperparameters at start
- `log_metrics()`: Logs metrics after each epoch
- `log_model()`: Saves model as MLflow artifact
- `log_figure()`: Saves plots (confusion matrix, etc.)

### 5. ModelEvaluator

**Responsibility**: Compute comprehensive evaluation metrics

```python
class ModelEvaluator:
    """Evaluates trained model and generates metrics."""
    
    def __init__(self, model: nn.Module, data_loader: DataLoader, device: torch.device):
        """Initialize evaluator."""
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute and return confusion matrix."""
        
    def get_per_class_metrics(self) -> pd.DataFrame:
        """Get precision, recall, F1 for each class."""
        
    def get_top_k_accuracy(self, k: int = 5) -> float:
        """Compute top-k accuracy."""
```

**Key Methods**:
- `evaluate()`: Runs full evaluation suite
- `compute_confusion_matrix()`: Generates confusion matrix
- `get_per_class_metrics()`: Calculates per-class metrics
- `get_top_k_accuracy()`: Computes top-k accuracy

### 6. InferenceHelper

**Responsibility**: Provide inference capabilities for single images

```python
class InferenceHelper:
    """Helper for model inference on single images."""
    
    def __init__(self, model_path: str, config_path: str, class_mapping_path: str):
        """
        Initialize inference helper.
        
        Args:
            model_path: Path to saved model
            config_path: Path to model config JSON
            class_mapping_path: Path to class mapping JSON
        """
        
    def load_model(self) -> None:
        """Load model and configuration."""
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor
        """
        
    def predict(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict top-k classes for an image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
```

**Key Methods**:
- `load_model()`: Loads saved model and config
- `preprocess_image()`: Applies same transforms as training
- `predict()`: Returns top-k predictions with probabilities

## Data Models

### Configuration Model

```python
@dataclass
class TrainingConfig:
    """Configuration for training run."""
    
    # Data parameters
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    
    # Model parameters
    model_name: str = "mobilenet_v2"  # or "efficientnet_b0"
    num_classes: int = 101
    freeze_layers: int = 10
    dropout_rate: float = 0.2
    
    # Training parameters
    num_epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    
    # Scheduler parameters
    scheduler_patience: int = 3
    scheduler_factor: float = 0.1
    
    # Checkpoint parameters
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    
    # MLflow parameters
    experiment_name: str = "food_classification"
    run_name: Optional[str] = None
    
    # Device parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    random_seed: int = 42
```

### Model Artifact Structure

```python
# model_config.json
{
    "model_name": "mobilenet_v2",
    "num_classes": 101,
    "input_size": [224, 224],
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "version": "v1",
    "timestamp": "2024-01-15T10:30:00",
    "training_config": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 20
    }
}

# class_to_idx.json
{
    "apple_pie": 0,
    "baby_back_ribs": 1,
    "baklava": 2,
    ...
    "waffles": 100
}

# evaluation_results.json
{
    "accuracy": 0.8542,
    "top_3_accuracy": 0.9621,
    "top_5_accuracy": 0.9812,
    "per_class_metrics": {
        "apple_pie": {
            "precision": 0.87,
            "recall": 0.85,
            "f1_score": 0.86,
            "support": 250
        },
        ...
    },
    "worst_classes": [
        {"class": "pork_chop", "f1_score": 0.62},
        ...
    ],
    "confusion_matrix_path": "confusion_matrix.png"
}
```

### Checkpoint Structure

```python
# checkpoint.pth
{
    "epoch": 15,
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "best_val_loss": 0.4523,
    "best_val_acc": 0.8542,
    "training_history": {
        "train_loss": [1.2, 0.8, 0.6, ...],
        "train_acc": [0.65, 0.75, 0.82, ...],
        "val_loss": [1.1, 0.7, 0.5, ...],
        "val_acc": [0.68, 0.78, 0.85, ...]
    },
    "config": {...}
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, I identified the following testable properties and performed redundancy elimination:

**Redundant Properties Identified:**
- Properties about data loader configuration and model configuration can be combined into single comprehensive properties
- Multiple properties about file saving can be consolidated into artifact completeness properties
- Properties about metric logging can be unified into comprehensive logging properties

**Consolidated Properties:**
The following properties provide unique validation value without redundancy:

### Property 1: Train/Validation Split Ratio Consistency

*For any* dataset size and class distribution, when the Training System creates a train/validation split, the ratio SHALL be 80/20 (±1% tolerance) and each class SHALL be proportionally represented in both splits with stratified sampling.

**Validates: Requirements 1.2**

### Property 2: Frozen Layers Immutability

*For any* model architecture (MobileNetV2 or EfficientNet-B0), after the Training System configures the model for transfer learning, the first 10 feature layers SHALL have `requires_grad=False` and SHALL NOT update during training.

**Validates: Requirements 2.2**

### Property 3: Classifier Output Dimension Correctness

*For any* number of food classes N, when the Training System replaces the classifier head, the final Linear layer output dimension SHALL equal N and the classifier SHALL contain both Dropout and Linear layers.

**Validates: Requirements 2.3**

### Property 4: Optimizer Learning Rate Configuration

*For any* specified learning rate value, the Training System SHALL initialize the Adam optimizer with exactly that learning rate value.

**Validates: Requirements 3.2**

### Property 5: Early Stopping Trigger Condition

*For any* training run, if validation loss does not improve for 5 consecutive epochs, the Training System SHALL terminate training before reaching the maximum epoch count.

**Validates: Requirements 3.4**

### Property 6: Epoch Count Execution

*For any* specified number of epochs N (when early stopping does not trigger), the Training System SHALL execute exactly N training epochs.

**Validates: Requirements 3.5**

### Property 7: Per-Epoch Metric Logging Completeness

*For any* completed training epoch, the Training System SHALL log exactly four metrics to MLflow: train_loss, train_accuracy, val_loss, and val_accuracy.

**Validates: Requirements 4.2**

### Property 8: Class Mapping Serialization Round-Trip

*For any* set of class names and their index mappings, when the Training System saves the class_to_idx dictionary as JSON and then loads it, the loaded dictionary SHALL be equivalent to the original dictionary.

**Validates: Requirements 5.2**

### Property 9: Checkpoint Completeness

*For any* saved checkpoint, the checkpoint file SHALL contain all required keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_val_loss, and best_val_acc.

**Validates: Requirements 5.4**

### Property 10: Evaluation Metric Completeness

*For any* model evaluation on a validation set with N classes, the Training System SHALL calculate and return accuracy, precision, recall, and F1-score for all N classes.

**Validates: Requirements 6.1**

### Property 11: Confusion Matrix Dimensionality

*For any* model with N classes, the generated confusion matrix SHALL be a square matrix of dimensions N×N, and when normalized, each row SHALL sum to approximately 1.0 (±0.01 tolerance).

**Validates: Requirements 6.2**

### Property 12: Image Preprocessing Output Shape

*For any* valid input image, when the Training System preprocesses it for inference, the output tensor SHALL have shape [3, 224, 224] and values SHALL be normalized according to ImageNet statistics.

**Validates: Requirements 7.2**

### Property 13: Top-K Prediction Probability Sum

*For any* input image, when the Training System makes a prediction and returns top-3 classes, the sum of the three probability scores SHALL be less than or equal to 1.0 and each individual probability SHALL be between 0.0 and 1.0.

**Validates: Requirements 7.3**

## Error Handling

### Dataset Loading Errors

**Error Type**: `DatasetNotFoundError`
- **Trigger**: Food-101 dataset not found in specified directory
- **Handling**: Log error with download instructions, raise exception with helpful message
- **Recovery**: Provide automatic download option if network available

**Error Type**: `InvalidSplitRatioError`
- **Trigger**: Train/val split results in empty validation set
- **Handling**: Log warning, adjust split ratio automatically
- **Recovery**: Use minimum 10% validation split

### Model Configuration Errors

**Error Type**: `UnsupportedModelError`
- **Trigger**: Invalid model name provided
- **Handling**: Log error with list of supported models, raise exception
- **Recovery**: None - user must provide valid model name

**Error Type**: `ModelLoadError`
- **Trigger**: Pre-trained model download fails
- **Handling**: Log error with network diagnostics, raise exception
- **Recovery**: Retry download with exponential backoff

### Training Errors

**Error Type**: `OutOfMemoryError`
- **Trigger**: GPU memory exhausted during training
- **Handling**: Log error with memory usage stats, clear cache, reduce batch size
- **Recovery**: Automatically reduce batch size by 50% and retry

**Error Type**: `NaNLossError`
- **Trigger**: Loss becomes NaN during training
- **Handling**: Log error with last valid loss, save emergency checkpoint
- **Recovery**: Reduce learning rate by 10x and resume from last checkpoint

**Error Type**: `CheckpointLoadError`
- **Trigger**: Corrupted or incompatible checkpoint file
- **Handling**: Log error with checkpoint details, raise exception
- **Recovery**: None - start training from scratch

### MLflow Errors

**Error Type**: `MLflowConnectionError`
- **Trigger**: Cannot connect to MLflow tracking server
- **Handling**: Log warning, continue training without MLflow tracking
- **Recovery**: Save metrics locally, allow manual upload later

**Error Type**: `ArtifactSaveError`
- **Trigger**: Failed to save artifact to MLflow
- **Handling**: Log error, save artifact locally as backup
- **Recovery**: Retry artifact upload at end of training

### Inference Errors

**Error Type**: `InvalidImageError`
- **Trigger**: Image file corrupted or unsupported format
- **Handling**: Log error with file details, raise descriptive exception
- **Recovery**: None - user must provide valid image

**Error Type**: `ModelNotFoundError`
- **Trigger**: Saved model file not found at specified path
- **Handling**: Log error with expected path, raise exception
- **Recovery**: None - user must train model first

### General Error Handling Strategy

1. **Validation First**: Validate all inputs before processing
2. **Fail Fast**: Detect errors early and provide clear messages
3. **Graceful Degradation**: Continue with reduced functionality when possible
4. **Comprehensive Logging**: Log all errors with context and stack traces
5. **User-Friendly Messages**: Provide actionable error messages with solutions
6. **State Preservation**: Save checkpoints before risky operations
7. **Resource Cleanup**: Always release GPU memory and close file handles

## Testing Strategy

### Unit Testing

The training system will use **pytest** as the testing framework with the following unit test coverage:

**DatasetManager Tests**:
- Test transform composition for training and validation
- Test data loader creation with various batch sizes
- Test class mapping generation
- Test edge cases: empty dataset, single class, unbalanced classes

**ModelBuilder Tests**:
- Test model creation for both MobileNetV2 and EfficientNet-B0
- Test layer freezing with different freeze counts
- Test classifier replacement with various class counts
- Test model configuration serialization

**TrainingEngine Tests**:
- Test single epoch training with mock data
- Test validation loop with mock data
- Test early stopping logic with simulated loss patterns
- Test checkpoint saving and loading

**MLflowTracker Tests**:
- Test parameter logging with mock MLflow
- Test metric logging with various metric types
- Test artifact saving with mock files
- Test error handling when MLflow unavailable

**ModelEvaluator Tests**:
- Test metric calculation with known predictions
- Test confusion matrix generation
- Test top-k accuracy calculation
- Test per-class metric computation

**InferenceHelper Tests**:
- Test model loading from saved artifacts
- Test image preprocessing with various image sizes
- Test prediction output format
- Test error handling for invalid inputs

### Property-Based Testing

The training system will use **Hypothesis** for property-based testing with a minimum of 100 iterations per property:

**Property Test Configuration**:
```python
from hypothesis import given, settings, strategies as st

@settings(max_examples=100, deadline=None)
```

**Property Test Implementation**:
- Each property-based test MUST be tagged with a comment referencing the design document property
- Tag format: `# Feature: food-classification-training, Property {number}: {property_text}`
- Each correctness property MUST be implemented by a SINGLE property-based test

**Test Generators**:
- `dataset_sizes`: Generate random dataset sizes (100-10000)
- `class_counts`: Generate random number of classes (2-200)
- `learning_rates`: Generate valid learning rates (1e-5 to 1e-1)
- `batch_sizes`: Generate valid batch sizes (powers of 2, 8-128)
- `image_tensors`: Generate random image tensors with valid shapes
- `probability_distributions`: Generate valid probability distributions

**Property Tests to Implement**:
1. Train/validation split ratio property (Property 1)
2. Frozen layers immutability property (Property 2)
3. Classifier output dimension property (Property 3)
4. Optimizer learning rate property (Property 4)
5. Early stopping trigger property (Property 5)
6. Epoch count execution property (Property 6)
7. Per-epoch metric logging property (Property 7)
8. Class mapping serialization property (Property 8)
9. Checkpoint completeness property (Property 9)
10. Evaluation metric completeness property (Property 10)
11. Confusion matrix dimensionality property (Property 11)
12. Image preprocessing output property (Property 12)
13. Top-K prediction probability property (Property 13)

### Integration Testing

**End-to-End Training Test**:
- Test complete training pipeline with small dataset (10 classes, 100 images)
- Verify all artifacts are created correctly
- Verify MLflow logging works end-to-end
- Verify model can be loaded and used for inference

**Checkpoint Resume Test**:
- Train for 5 epochs, save checkpoint
- Resume training from checkpoint
- Verify training continues from correct epoch
- Verify optimizer and scheduler states are restored

**Multi-GPU Test** (if available):
- Test training with DataParallel
- Verify batch size scaling
- Verify gradient synchronization

### Performance Testing

**Training Speed Benchmark**:
- Measure time per epoch with different batch sizes
- Measure GPU utilization during training
- Identify bottlenecks in data loading

**Memory Usage Test**:
- Monitor GPU memory usage during training
- Test maximum batch size before OOM
- Verify memory is released after training

**Inference Speed Test**:
- Measure inference time for single image
- Measure batch inference throughput
- Compare CPU vs GPU inference speed

### Test Execution

**Running Tests**:
```bash
# Run all unit tests
pytest tests/ -v

# Run property-based tests
pytest tests/test_properties.py -v

# Run with coverage
pytest tests/ --cov=train_model --cov-report=html

# Run specific test file
pytest tests/test_model_builder.py -v
```

**Continuous Integration**:
- Run unit tests on every commit
- Run property tests on pull requests
- Generate coverage reports
- Fail build if coverage < 80%

### Test Data

**Mock Datasets**:
- Create small synthetic datasets for fast testing
- Use deterministic random seeds for reproducibility
- Include edge cases: single class, unbalanced, empty

**Fixtures**:
- Provide pre-trained model fixtures for testing
- Provide sample images in various formats
- Provide mock MLflow tracking server

## Implementation Notes

### Performance Optimizations

1. **Data Loading**:
   - Use multiple workers for parallel data loading
   - Pin memory for faster GPU transfer
   - Prefetch batches to overlap computation and I/O

2. **Model Training**:
   - Use mixed precision training (torch.cuda.amp) for faster training
   - Accumulate gradients for effective larger batch sizes
   - Use torch.compile() for PyTorch 2.0+ speedup

3. **Memory Management**:
   - Clear GPU cache between epochs
   - Use gradient checkpointing for large models
   - Delete intermediate tensors explicitly

### Reproducibility

1. **Random Seeds**:
   ```python
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

2. **Version Tracking**:
   - Log PyTorch version, CUDA version, Python version
   - Log all package versions from requirements.txt
   - Save git commit hash with model artifacts

### Google Colab Compatibility

1. **GPU Detection**:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   if device.type == "cuda":
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

2. **Drive Mounting**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   data_dir = '/content/drive/MyDrive/food-101'
   ```

3. **Progress Display**:
   - Use tqdm with notebook=True for Colab
   - Display sample predictions with matplotlib
   - Show training curves in real-time

### MLflow Best Practices

1. **Experiment Organization**:
   - Use descriptive experiment names
   - Tag runs with model architecture and dataset version
   - Use nested runs for hyperparameter tuning

2. **Artifact Management**:
   - Save model in both PyTorch and ONNX formats
   - Include sample predictions with artifacts
   - Save training curves as images

3. **Metric Tracking**:
   - Log metrics at consistent intervals
   - Include both training and validation metrics
   - Track system metrics (GPU usage, memory)

### Deployment Considerations

1. **Model Export**:
   - Export to ONNX for cross-platform compatibility
   - Quantize model for mobile deployment
   - Create TorchScript version for production

2. **API Integration**:
   - Provide inference wrapper compatible with FastAPI
   - Include preprocessing in model artifact
   - Support batch inference for efficiency

3. **Monitoring**:
   - Log prediction confidence distributions
   - Track inference latency
   - Monitor for data drift

## Future Enhancements

1. **Advanced Augmentation**:
   - Implement MixUp and CutMix augmentation
   - Add AutoAugment policies
   - Use test-time augmentation for inference

2. **Model Improvements**:
   - Experiment with Vision Transformers (ViT)
   - Implement ensemble methods
   - Add attention mechanisms

3. **Training Enhancements**:
   - Implement learning rate warmup
   - Add cosine annealing scheduler
   - Support distributed training

4. **Dataset Expansion**:
   - Add custom Indian food categories
   - Implement active learning for data collection
   - Support multi-label classification

5. **MLOps Features**:
   - Implement A/B testing framework
   - Add model performance monitoring dashboard
   - Automate model retraining pipeline
