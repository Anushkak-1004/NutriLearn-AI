# Design Document - Model Training Pipeline

## Overview

The Model Training Pipeline is a production-ready PyTorch-based system for training food classification models using transfer learning. The pipeline integrates with MLflow for experiment tracking, supports multiple backbone architectures, and is optimized for both local training and Google Colab execution with GPU acceleration.

The system follows MLOps best practices by tracking all experiments, logging comprehensive metrics, and saving versioned model artifacts that can be seamlessly integrated into the NutriLearn AI backend for inference.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Dataset    │───▶│   Training   │───▶│  Evaluation  │ │
│  │   Loader     │    │    Loop      │    │   & Metrics  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │     Data     │    │    MLflow    │    │    Model     │ │
│  │ Augmentation │    │   Tracking   │    │   Artifacts  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  MLflow Server   │
                    │  (Experiments,   │
                    │   Runs, Models)  │
                    └──────────────────┘
```

### Component Interaction Flow

1. **Configuration Phase**: Parse command-line arguments and set up training configuration
2. **Data Loading Phase**: Download/load Food-101 dataset, apply transformations, create data loaders
3. **Model Setup Phase**: Load pre-trained backbone, modify classifier, move to device (CPU/GPU)
4. **Training Phase**: Execute training loop with forward/backward passes, log to MLflow
5. **Evaluation Phase**: Calculate metrics, generate confusion matrix, identify best model
6. **Saving Phase**: Save model weights, class mappings, and configuration files
7. **Inference Testing Phase**: Load saved model and test on sample images

## Components and Interfaces

### 1. Configuration Manager

**Purpose**: Handle command-line arguments and training configuration

**Interface**:
```python
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.
    
    Returns:
        Namespace containing all training parameters
    """
    pass

def get_device() -> torch.device:
    """
    Detect and return the appropriate device (CUDA/CPU).
    
    Returns:
        torch.device object
    """
    pass
```

**Configuration Parameters**:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--model-name`: Backbone architecture (choices: mobilenet_v2, efficientnet_b0, resnet50)
- `--resume`: Path to checkpoint for resuming training
- `--data-dir`: Directory for dataset storage
- `--output-dir`: Directory for saving models and artifacts

### 2. Dataset Manager

**Purpose**: Handle dataset loading, preprocessing, and augmentation

**Interface**:
```python
def get_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train and validation data loaders with augmentation.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, class_to_idx)
    """
    pass
```

**Data Augmentation Pipeline**:
- **Training Transforms**:
  - RandomResizedCrop(224)
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  - ToTensor()
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

- **Validation Transforms**:
  - Resize(256)
  - CenterCrop(224)
  - ToTensor()
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 3. Model Builder

**Purpose**: Create and configure neural network models

**Interface**:
```python
def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True
) -> nn.Module:
    """
    Build a model with transfer learning.
    
    Args:
        model_name: Name of the backbone architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Configured PyTorch model
    """
    pass

def freeze_layers(model: nn.Module, freeze_until: int) -> None:
    """
    Freeze early layers of the model.
    
    Args:
        model: PyTorch model
        freeze_until: Layer index to freeze until
    """
    pass
```

**Supported Architectures**:
- **MobileNetV2**: Lightweight, optimized for mobile/edge deployment
  - Input: 224x224x3
  - Feature extractor: 1280 features
  - Classifier: Dropout(0.2) + Linear(1280, num_classes)
  
- **EfficientNet-B0**: Balanced accuracy and efficiency
  - Input: 224x224x3
  - Feature extractor: 1280 features
  - Classifier: Dropout(0.2) + Linear(1280, num_classes)
  
- **ResNet50**: Higher accuracy, more parameters
  - Input: 224x224x3
  - Feature extractor: 2048 features
  - Classifier: Linear(2048, num_classes)

### 4. Training Engine

**Purpose**: Execute the training loop with optimization and monitoring

**Interface**:
```python
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    pass

def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    pass
```

**Training Configuration**:
- Loss Function: CrossEntropyLoss()
- Optimizer: Adam with configurable learning rate
- Learning Rate Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=3)
- Early Stopping: Patience of 5 epochs on validation loss
- Gradient Clipping: Max norm of 1.0 to prevent exploding gradients

### 5. MLflow Integration

**Purpose**: Track experiments, log metrics, and save model artifacts

**Interface**:
```python
def setup_mlflow(experiment_name: str) -> str:
    """
    Initialize MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
    """
    pass

def log_training_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    learning_rate: float
) -> None:
    """
    Log metrics for the current epoch.
    
    Args:
        epoch: Current epoch number
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        learning_rate: Current learning rate
    """
    pass

def log_model_artifact(
    model_path: str,
    model_name: str,
    version: str
) -> None:
    """
    Log trained model as MLflow artifact.
    
    Args:
        model_path: Path to saved model
        model_name: Name of the model
        version: Model version
    """
    pass
```

**Logged Information**:
- **Parameters**: model_name, learning_rate, batch_size, epochs, optimizer, scheduler
- **Metrics**: train_loss, train_acc, val_loss, val_acc, learning_rate (per epoch)
- **Artifacts**: model weights (.pth), class mappings (.json), config (.json), confusion matrix (.png)
- **Tags**: version, dataset, backbone, training_time

### 6. Evaluation Module

**Purpose**: Calculate comprehensive performance metrics

**Interface**:
```python
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to evaluate on
        class_names: List of class names
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    pass

def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str
) -> None:
    """
    Generate and save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    pass
```

**Metrics Calculated**:
- Top-1 Accuracy: Percentage of correct predictions
- Top-5 Accuracy: Percentage where correct class is in top 5 predictions
- Per-Class Metrics: Precision, Recall, F1-Score for each food category
- Confusion Matrix: Visual representation of prediction patterns
- Worst Performing Classes: Classes with lowest F1 scores

### 7. Model Persistence

**Purpose**: Save and load model artifacts

**Interface**:
```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    save_path: str
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_acc: Best validation accuracy
        save_path: Path to save checkpoint
    """
    pass

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Tuple[int, float]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (start_epoch, best_accuracy)
    """
    pass

def save_model_artifacts(
    model: nn.Module,
    class_to_idx: Dict[str, int],
    config: Dict[str, Any],
    version: str,
    output_dir: str
) -> None:
    """
    Save all model artifacts for deployment.
    
    Args:
        model: Trained model
        class_to_idx: Class to index mapping
        config: Model configuration
        version: Model version
        output_dir: Output directory
    """
    pass
```

**Saved Artifacts**:
1. **Model Weights** (`food_model_v{version}.pth`):
   - Complete model state_dict
   - Can be loaded for inference
   
2. **Class Mappings** (`class_to_idx.json`):
   ```json
   {
     "apple_pie": 0,
     "biryani": 1,
     "pizza": 2,
     ...
   }
   ```

3. **Model Configuration** (`model_config.json`):
   ```json
   {
     "model_name": "mobilenet_v2",
     "num_classes": 101,
     "input_size": [224, 224],
     "mean": [0.485, 0.456, 0.406],
     "std": [0.229, 0.224, 0.225],
     "version": "v1",
     "training_date": "2024-01-01",
     "best_accuracy": 0.85
   }
   ```

### 8. Inference Module

**Purpose**: Test trained models on individual images

**Interface**:
```python
def load_model_for_inference(
    model_path: str,
    config_path: str,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load trained model for inference.
    
    Args:
        model_path: Path to model weights
        config_path: Path to model config
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    pass

def predict_image(
    model: nn.Module,
    image_path: str,
    config: Dict[str, Any],
    device: torch.device,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Predict food class for an image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        config: Model configuration
        device: Device to run inference on
        top_k: Number of top predictions to return
        
    Returns:
        List of (class_name, probability) tuples
    """
    pass
```

## Data Models

### Training Configuration

```python
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model parameters
    model_name: str = "mobilenet_v2"
    num_classes: int = 101
    pretrained: bool = True
    freeze_layers: int = 10
    
    # Training parameters
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimization parameters
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 5
    
    # Data parameters
    data_dir: str = "./data"
    train_split: float = 0.8
    num_workers: int = 4
    
    # Output parameters
    output_dir: str = "./ml-models"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    
    # MLflow parameters
    experiment_name: str = "nutrilearn-food-training"
    tracking_uri: str = "file:./mlruns"
    
    # Device parameters
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = False
```

### Training Metrics

```python
@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float
```

### Evaluation Results

```python
@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""
    
    top1_accuracy: float
    top5_accuracy: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    worst_classes: List[Tuple[str, float]]
    best_classes: List[Tuple[str, float]]
    total_samples: int
    inference_time_ms: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Model architecture consistency

*For any* backbone model and number of classes, the final classifier layer output dimension should equal the number of classes.

**Validates: Requirements 1.2**

### Property 2: Data augmentation preservation

*For any* training image, applying data augmentation should preserve the image dimensions at 224x224 and maintain valid pixel value ranges [0, 1] after normalization.

**Validates: Requirements 1.4**

### Property 3: MLflow logging completeness

*For any* training epoch, all required metrics (train_loss, train_accuracy, val_loss, val_accuracy) should be logged to MLflow.

**Validates: Requirements 2.2**

### Property 4: Checkpoint resumption consistency

*For any* saved checkpoint, loading and resuming training should restore the exact model state, optimizer state, and epoch number.

**Validates: Requirements 3.2**

### Property 5: Model artifact completeness

*For any* trained model, saving artifacts should create all three required files: model weights (.pth), class mappings (.json), and configuration (.json).

**Validates: Requirements 5.1, 5.2, 5.3**

### Property 6: Inference preprocessing consistency

*For any* input image, the inference preprocessing pipeline should apply the same transformations as validation preprocessing (resize, center crop, normalize).

**Validates: Requirements 6.2**

### Property 7: Top-K predictions validity

*For any* inference prediction with top_k=3, the returned predictions should contain exactly 3 items, probabilities should sum to approximately 1.0, and be sorted in descending order.

**Validates: Requirements 6.3**

### Property 8: Device compatibility

*For any* available device (CPU or CUDA), the training script should successfully move all tensors and models to that device without errors.

**Validates: Requirements 7.1, 7.4**

### Property 9: Learning rate scheduling

*For any* training run with ReduceLROnPlateau, when validation loss doesn't improve for 3 consecutive epochs, the learning rate should be reduced by the specified factor.

**Validates: Requirements 8.3**

### Property 10: Early stopping trigger

*For any* training run with early stopping enabled, when validation loss doesn't improve for 5 consecutive epochs, training should terminate and save the best model.

**Validates: Requirements 8.4**

### Property 11: Class mapping bijection

*For any* saved class_to_idx mapping, each class name should map to a unique index, and the number of mappings should equal the number of classes.

**Validates: Requirements 5.2**

### Property 12: Confusion matrix dimensions

*For any* generated confusion matrix with N classes, the matrix should have dimensions NxN and row sums should equal the number of samples per class.

**Validates: Requirements 4.2**

## Error Handling

### Dataset Errors

- **Missing Dataset**: Download Food-101 automatically or provide clear instructions
- **Corrupted Images**: Skip corrupted images and log warnings
- **Insufficient Disk Space**: Check available space before downloading
- **Invalid Data Directory**: Create directory if it doesn't exist

### Training Errors

- **Out of Memory**: Suggest reducing batch size or using gradient accumulation
- **NaN Loss**: Implement gradient clipping and learning rate adjustment
- **Model Divergence**: Early detection and checkpoint rollback
- **Device Errors**: Graceful fallback from CUDA to CPU

### MLflow Errors

- **Connection Failures**: Continue training without MLflow, log locally
- **Logging Errors**: Catch exceptions and continue training
- **Artifact Upload Failures**: Retry with exponential backoff

### File System Errors

- **Permission Denied**: Clear error message with suggested fix
- **Disk Full**: Check space before saving, clean up old checkpoints
- **Path Not Found**: Create necessary directories automatically

## Testing Strategy

### Unit Tests

1. **Model Building Tests**:
   - Test each backbone architecture loads correctly
   - Verify classifier head has correct output dimensions
   - Test layer freezing functionality

2. **Data Loading Tests**:
   - Test data augmentation pipeline
   - Verify batch dimensions
   - Test train/val split ratios

3. **Checkpoint Tests**:
   - Test save and load functionality
   - Verify state restoration
   - Test checkpoint versioning

4. **Inference Tests**:
   - Test image preprocessing
   - Verify prediction format
   - Test top-k predictions

### Property-Based Tests

The model will use pytest with hypothesis for property-based testing. Each property-based test will run a minimum of 100 iterations.

1. **Property Test 1: Model Architecture Consistency**
   - **Feature: model-training-pipeline, Property 1: Model architecture consistency**
   - Generate random num_classes values (10-200)
   - Build model and verify output dimension matches num_classes

2. **Property Test 2: Data Augmentation Preservation**
   - **Feature: model-training-pipeline, Property 2: Data augmentation preservation**
   - Generate random images of various sizes
   - Apply augmentation and verify output is 224x224

3. **Property Test 3: MLflow Logging Completeness**
   - **Feature: model-training-pipeline, Property 3: MLflow logging completeness**
   - Generate random metric values
   - Log to MLflow and verify all metrics are present

4. **Property Test 4: Checkpoint Resumption Consistency**
   - **Feature: model-training-pipeline, Property 4: Checkpoint resumption consistency**
   - Create random model states
   - Save, load, and verify exact restoration

5. **Property Test 5: Model Artifact Completeness**
   - **Feature: model-training-pipeline, Property 5: Model artifact completeness**
   - Save model artifacts
   - Verify all three files exist and are valid

6. **Property Test 6: Inference Preprocessing Consistency**
   - **Feature: model-training-pipeline, Property 6: Inference preprocessing consistency**
   - Generate random images
   - Verify inference and validation preprocessing produce same output

7. **Property Test 7: Top-K Predictions Validity**
   - **Feature: model-training-pipeline, Property 7: Top-K predictions validity**
   - Generate random predictions
   - Verify count, sum, and ordering

8. **Property Test 8: Device Compatibility**
   - **Feature: model-training-pipeline, Property 8: Device compatibility**
   - Test on available devices
   - Verify no device-related errors

9. **Property Test 9: Learning Rate Scheduling**
   - **Feature: model-training-pipeline, Property 9: Learning rate scheduling**
   - Simulate plateau scenarios
   - Verify LR reduction occurs correctly

10. **Property Test 10: Early Stopping Trigger**
    - **Feature: model-training-pipeline, Property 10: Early stopping trigger**
    - Simulate non-improving validation loss
    - Verify training stops at correct epoch

11. **Property Test 11: Class Mapping Bijection**
    - **Feature: model-training-pipeline, Property 11: Class mapping bijection**
    - Generate random class mappings
    - Verify uniqueness and completeness

12. **Property Test 12: Confusion Matrix Dimensions**
    - **Feature: model-training-pipeline, Property 12: Confusion matrix dimensions**
    - Generate random predictions
    - Verify confusion matrix shape and row sums

### Integration Tests

1. **End-to-End Training Test**:
   - Run training for 2 epochs on small subset
   - Verify all artifacts are created
   - Check MLflow logging

2. **Resume Training Test**:
   - Train for 2 epochs, save checkpoint
   - Resume and train 2 more epochs
   - Verify continuity

3. **Inference Pipeline Test**:
   - Train small model
   - Save and load for inference
   - Test predictions on sample images

### Manual Testing Checklist

- [ ] Train on Google Colab with GPU
- [ ] Train locally on CPU
- [ ] Resume from checkpoint
- [ ] Test all three backbone models
- [ ] Verify MLflow UI shows experiments
- [ ] Test inference on new images
- [ ] Verify all artifacts are created
- [ ] Check error handling for edge cases

## Performance Considerations

### Training Optimization

- **Mixed Precision Training**: Use torch.cuda.amp for faster training on compatible GPUs
- **Gradient Accumulation**: Simulate larger batch sizes on limited memory
- **DataLoader Workers**: Use multiple workers for faster data loading
- **Pin Memory**: Enable for faster CPU-to-GPU transfer

### Memory Management

- **Batch Size Tuning**: Automatically reduce if OOM errors occur
- **Gradient Checkpointing**: Trade compute for memory on large models
- **Model Pruning**: Optional post-training compression

### Inference Optimization

- **TorchScript**: Convert model for faster inference
- **ONNX Export**: Enable deployment on various platforms
- **Quantization**: Reduce model size for edge deployment

## Deployment Integration

### Backend Integration

The trained model integrates with the existing NutriLearn AI backend:

1. **Replace Mock Predictor**: Update `backend/app/ml/predictor.py` to load trained model
2. **Load Artifacts**: Read model weights, class mappings, and config on startup
3. **Preprocessing**: Apply same transformations as training
4. **Prediction**: Return top-3 predictions with confidence scores
5. **MLflow Logging**: Continue logging predictions for monitoring

### Model Versioning

- Models are versioned as v1, v2, v3, etc.
- Each version is tracked in MLflow
- Backend can be configured to use specific version
- A/B testing supported through version switching

## Future Enhancements

1. **Distributed Training**: Multi-GPU training with DataParallel or DistributedDataParallel
2. **AutoML**: Hyperparameter tuning with Optuna or Ray Tune
3. **Active Learning**: Identify uncertain predictions for labeling
4. **Model Ensemble**: Combine multiple models for better accuracy
5. **Continual Learning**: Update model with new data without forgetting
6. **Explainability**: Add Grad-CAM for visual explanations
7. **Custom Indian Foods**: Fine-tune on regional cuisine dataset
8. **Real-time Training**: Stream training data from production

---

**Design Status**: Complete and ready for implementation
**Next Step**: Create implementation task list
