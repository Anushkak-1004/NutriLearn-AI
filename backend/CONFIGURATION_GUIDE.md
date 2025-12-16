# Configuration Management Guide

## Overview

The training pipeline uses a comprehensive configuration management system built around the `TrainingConfig` dataclass. This provides:

- **Type-safe configuration** with validation
- **Automatic device detection** (CUDA/CPU)
- **Command-line argument parsing**
- **Configuration persistence** (save/load from JSON)
- **Sensible defaults** for all parameters

## TrainingConfig Dataclass

### Key Features

1. **Automatic Validation**: All parameters are validated on initialization
2. **Device Resolution**: Automatically detects CUDA availability and falls back to CPU
3. **Directory Creation**: Creates output and checkpoint directories automatically
4. **Serialization**: Can be saved to and loaded from JSON files

### Configuration Parameters

#### Model Parameters
- `model_name`: Architecture to use (`mobilenet_v2`, `efficientnet_b0`, `resnet50`)
- `num_classes`: Number of output classes (default: 101)
- `pretrained`: Use pretrained weights (default: True)
- `freeze_layers`: Number of early layers to freeze (default: 10)

#### Training Hyperparameters
- `epochs`: Number of training epochs (default: 20)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Initial learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 1e-4)

#### Optimization Parameters
- `optimizer`: Optimizer type (default: "adam")
- `scheduler`: LR scheduler type (default: "reduce_on_plateau")
- `scheduler_patience`: Epochs to wait before reducing LR (default: 3)
- `scheduler_factor`: Factor to reduce LR by (default: 0.5)
- `early_stopping_patience`: Epochs to wait before stopping (default: 5)
- `gradient_clip_max_norm`: Max gradient norm for clipping (default: 1.0)

#### Data Parameters
- `data_dir`: Directory containing dataset (default: "./data/food-101")
- `train_split`: Fraction of data for training (default: 0.8)
- `num_workers`: Number of data loading workers (default: 4)

#### Output Parameters
- `output_dir`: Directory for saving models (default: "./ml-models")
- `checkpoint_dir`: Directory for checkpoints (default: "./checkpoints")
- `log_interval`: Logging frequency (default: 10)

#### MLflow Parameters
- `experiment_name`: MLflow experiment name (default: "nutrilearn-food-training")
- `tracking_uri`: MLflow tracking URI (default: "file:./mlruns")

#### Device Parameters
- `device`: Device to use (`auto`, `cuda`, `cpu`) (default: "auto")
- `mixed_precision`: Use mixed precision training (default: False)

#### Resume Training
- `resume_from`: Path to checkpoint to resume from (default: None)

## Usage Examples

### 1. Using Default Configuration

```python
from train_model import TrainingConfig, FoodClassificationTrainer

# Create config with defaults
config = TrainingConfig()

# Create trainer
trainer = FoodClassificationTrainer(config)
trainer.train()
```

### 2. Custom Configuration

```python
from train_model import TrainingConfig, FoodClassificationTrainer

# Create custom config
config = TrainingConfig(
    model_name="efficientnet_b0",
    epochs=30,
    batch_size=64,
    learning_rate=0.0001,
    device="cuda"
)

# Create trainer
trainer = FoodClassificationTrainer(config)
trainer.train()
```

### 3. Command-Line Usage

```bash
# Basic training with defaults
python train_model.py

# Custom parameters
python train_model.py --model_name efficientnet_b0 --epochs 30 --batch_size 64

# Resume training
python train_model.py --resume ./checkpoints/checkpoint_latest.pth

# Train on CPU
python train_model.py --device cpu

# Full example with all parameters
python train_model.py \
  --model_name resnet50 \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --scheduler_patience 5 \
  --early_stopping_patience 10 \
  --data_dir ./data/food-101 \
  --output_dir ./ml-models \
  --device auto
```

### 4. Save and Load Configuration

```python
from train_model import TrainingConfig

# Create and save config
config = TrainingConfig(
    model_name="mobilenet_v2",
    epochs=20,
    batch_size=32
)
config.save("my_config.json")

# Load config later
loaded_config = TrainingConfig.from_file("my_config.json")
```

### 5. Convert to Dictionary

```python
from train_model import TrainingConfig

config = TrainingConfig()
config_dict = config.to_dict()

# Use for logging or serialization
print(config_dict)
```

## Input Validation

The configuration system validates all inputs:

### Model Name Validation
```python
# Valid
config = TrainingConfig(model_name="mobilenet_v2")

# Invalid - raises ValueError
config = TrainingConfig(model_name="invalid_model")
```

### Numeric Range Validation
```python
# Valid
config = TrainingConfig(epochs=20, batch_size=32, learning_rate=0.001)

# Invalid - raises ValueError
config = TrainingConfig(epochs=-5)  # Negative epochs
config = TrainingConfig(batch_size=0)  # Zero batch size
config = TrainingConfig(learning_rate=-0.001)  # Negative learning rate
```

### Train Split Validation
```python
# Valid
config = TrainingConfig(train_split=0.8)

# Invalid - raises ValueError
config = TrainingConfig(train_split=1.5)  # Must be between 0 and 1
```

### Device Validation
```python
# Valid
config = TrainingConfig(device="auto")  # Auto-detect
config = TrainingConfig(device="cuda")  # Use CUDA (falls back to CPU if unavailable)
config = TrainingConfig(device="cpu")   # Force CPU

# Invalid - raises ValueError
config = TrainingConfig(device="gpu")  # Invalid device name
```

## Device Detection

The configuration system automatically handles device detection:

```python
config = TrainingConfig(device="auto")

# If CUDA is available:
# - config.device will be "cuda"
# - Logs GPU name and memory

# If CUDA is not available:
# - config.device will be "cpu"
# - Logs fallback message

# Get torch.device object
device = config.get_device()  # Returns torch.device("cuda") or torch.device("cpu")
```

## Error Handling

The configuration system provides clear error messages:

```python
try:
    config = TrainingConfig(
        model_name="invalid_model",
        epochs=-5,
        batch_size=0
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Output: Configuration error: Invalid model_name 'invalid_model'. Must be one of ['mobilenet_v2', 'efficientnet_b0', 'resnet50']
```

## Integration with Trainer

The `FoodClassificationTrainer` class accepts a `TrainingConfig` instance:

```python
from train_model import TrainingConfig, FoodClassificationTrainer

# Create configuration
config = TrainingConfig(
    model_name="mobilenet_v2",
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)

# Create trainer with config
trainer = FoodClassificationTrainer(config)

# All config parameters are accessible
print(f"Training for {trainer.epochs} epochs")
print(f"Using device: {trainer.device}")
print(f"Batch size: {trainer.batch_size}")

# Start training
trainer.train()
```

## Best Practices

1. **Always validate configuration**: Let the `TrainingConfig` class handle validation
2. **Use auto device detection**: Set `device="auto"` for automatic CUDA/CPU selection
3. **Save configurations**: Save successful configurations for reproducibility
4. **Start with defaults**: Use default values and adjust only what's needed
5. **Check warnings**: Pay attention to validation warnings (e.g., very large batch sizes)

## Troubleshooting

### Issue: "CUDA requested but not available"
**Solution**: The system automatically falls back to CPU. Check CUDA installation if GPU training is required.

### Issue: "Resume checkpoint not found"
**Solution**: Verify the checkpoint path exists. Use absolute paths if relative paths fail.

### Issue: "Invalid model_name"
**Solution**: Use one of the supported models: `mobilenet_v2`, `efficientnet_b0`, `resnet50`

### Issue: "Out of memory errors"
**Solution**: Reduce `batch_size` or use `gradient_accumulation` (future feature)

## Command-Line Help

Get full list of available arguments:

```bash
python train_model.py --help
```

This displays all available parameters with descriptions and default values.
