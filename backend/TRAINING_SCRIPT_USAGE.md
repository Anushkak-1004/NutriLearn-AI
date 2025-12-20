# Food Classification Training Script - Usage Guide

## Overview
The `train_model.py` script provides a complete end-to-end training pipeline for food classification models using transfer learning with PyTorch.

## Quick Start

### Basic Training
```bash
python train_model.py
```

This will train a MobileNetV2 model with default settings:
- 20 epochs
- Batch size: 32
- Learning rate: 0.001
- Data directory: ./data
- Model: MobileNetV2

### Custom Configuration
```bash
python train_model.py \
  --model-name efficientnet_b0 \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --data-dir /path/to/food101
```

## Command-Line Arguments

### Data Parameters
- `--data-dir`: Directory to store/load Food-101 dataset (default: ./data)
- `--batch-size`: Batch size for training and validation (default: 32)
- `--num-workers`: Number of worker processes for data loading (default: 4)

### Model Parameters
- `--model-name`: Pre-trained model architecture (choices: mobilenet_v2, efficientnet_b0, default: mobilenet_v2)
- `--num-classes`: Number of food classes to classify (default: 101)
- `--freeze-layers`: Number of early feature layers to freeze (default: 10)
- `--dropout-rate`: Dropout rate in classifier head (default: 0.2)

### Training Parameters
- `--epochs`: Maximum number of training epochs (default: 20)
- `--learning-rate`: Initial learning rate for Adam optimizer (default: 0.001)
- `--weight-decay`: L2 regularization weight decay (default: 1e-4)
- `--early-stopping-patience`: Epochs without improvement before stopping (default: 5)

### Learning Rate Scheduler Parameters
- `--scheduler-patience`: Epochs without improvement before reducing LR (default: 3)
- `--scheduler-factor`: Factor to reduce learning rate by (default: 0.1)

### Checkpoint Parameters
- `--checkpoint-path`: Path to checkpoint file to resume training from (optional)
- `--checkpoint-dir`: Directory to save model checkpoints (default: ./checkpoints)
- `--save-best-only`: Save only the best model (default: True)

### MLflow Parameters
- `--experiment-name`: MLflow experiment name (default: food_classification)
- `--run-name`: MLflow run name (auto-generated if not provided)
- `--mlflow-tracking-uri`: MLflow tracking server URI (optional)

### Miscellaneous Parameters
- `--device`: Device to train on (choices: cuda, cpu, auto-detected if not specified)
- `--random-seed`: Random seed for reproducibility (default: 42)

## Usage Examples

### Example 1: Quick Training with Small Batch Size
```bash
python train_model.py --epochs 10 --batch-size 16
```

### Example 2: Training with EfficientNet-B0
```bash
python train_model.py \
  --model-name efficientnet_b0 \
  --epochs 25 \
  --learning-rate 0.0005
```

### Example 3: Training with Custom Data Directory
```bash
python train_model.py \
  --data-dir /mnt/datasets/food101 \
  --batch-size 64 \
  --num-workers 8
```

### Example 4: Fine-tuning with Fewer Frozen Layers
```bash
python train_model.py \
  --freeze-layers 5 \
  --learning-rate 0.0001 \
  --epochs 30
```

### Example 5: Resume Training from Checkpoint
```bash
python train_model.py \
  --checkpoint-path ./checkpoints/checkpoint_mobilenet_v2_20231215.pth \
  --epochs 10
```

### Example 6: Training with Custom MLflow Experiment
```bash
python train_model.py \
  --experiment-name my_food_experiment \
  --run-name experiment_v1 \
  --epochs 20
```

## Output

### Console Output
The script provides detailed logging including:
- Configuration summary
- Dataset loading progress
- Model architecture details
- Training progress with tqdm progress bars
- Epoch summaries (train/val loss and accuracy)
- Learning rate updates
- Early stopping notifications
- Evaluation metrics
- Final summary with best metrics

### Saved Artifacts
All artifacts are saved in timestamped directories under `ml-models/`:

```
ml-models/
└── training_20231215_143022_v1/
    ├── food_model_v1.pth          # Model weights
    ├── class_to_idx.json          # Class mapping
    ├── model_config.json          # Model configuration
    └── evaluation_results.json    # Evaluation metrics
```

### Checkpoints
Checkpoints are saved in the checkpoint directory:

```
checkpoints/
└── checkpoint_mobilenet_v2_20231215_143022.pth
```

Each checkpoint includes:
- Epoch number
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Best validation loss and accuracy
- Training history
- Configuration

### MLflow Tracking
All experiments are tracked in MLflow:
- Hyperparameters logged at start
- Metrics logged after each epoch
- Artifacts logged at end
- Confusion matrix saved as image
- Training time tracked

## Training Process

### 1. Initialization
- Parse arguments and create configuration
- Set random seeds for reproducibility
- Initialize MLflow experiment

### 2. Data Preparation
- Download Food-101 dataset (if not present)
- Create 80/20 train/validation split
- Apply data augmentation to training set
- Create data loaders with parallel loading

### 3. Model Setup
- Load pre-trained model (MobileNetV2 or EfficientNet-B0)
- Freeze early layers
- Replace classifier head
- Move model to device (GPU/CPU)

### 4. Training
- Initialize optimizer (Adam) and scheduler (ReduceLROnPlateau)
- Train for specified epochs
- Validate after each epoch
- Update learning rate based on validation loss
- Apply early stopping if no improvement

### 5. Evaluation
- Calculate accuracy, precision, recall, F1-score per class
- Generate confusion matrix
- Compute top-1, top-3, top-5 accuracy
- Identify worst-performing classes

### 6. Artifact Saving
- Save model weights
- Save class mappings
- Save model configuration
- Save evaluation results
- Save checkpoint with full training state
- Log all artifacts to MLflow

## Performance Tips

### For Faster Training
- Increase batch size (if GPU memory allows)
- Increase number of workers for data loading
- Use GPU if available
- Reduce number of epochs for quick experiments

### For Better Accuracy
- Train for more epochs
- Use lower learning rate
- Freeze fewer layers for fine-tuning
- Increase model capacity (use EfficientNet-B0)

### For Memory Constraints
- Reduce batch size
- Reduce number of workers
- Use MobileNetV2 (smaller model)
- Train on CPU if GPU memory is insufficient

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train_model.py --batch-size 16

# Or use CPU
python train_model.py --device cpu
```

### Dataset Download Issues
```bash
# Specify custom data directory
python train_model.py --data-dir /path/with/more/space
```

### MLflow Connection Issues
The script will continue training even if MLflow is unavailable, saving metrics locally.

### Training Too Slow
```bash
# Increase number of workers
python train_model.py --num-workers 8

# Or reduce batch size for faster iterations
python train_model.py --batch-size 16
```

## Requirements

### Python Packages
- torch >= 2.2.0
- torchvision >= 0.17.0
- mlflow >= 2.9.2
- tqdm >= 4.66.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- numpy
- Pillow

### Hardware
- Minimum: CPU with 8GB RAM
- Recommended: NVIDIA GPU with 8GB+ VRAM
- Storage: 5GB+ for dataset and models

## Google Colab Usage

```python
# Install dependencies
!pip install torch torchvision mlflow tqdm matplotlib seaborn scikit-learn

# Clone repository
!git clone https://github.com/your-repo/nutrilearn-ai.git
%cd nutrilearn-ai/backend

# Run training
!python train_model.py --epochs 10 --batch-size 32
```

## Next Steps

After training completes:
1. Review evaluation metrics in the console output
2. Check MLflow UI for experiment tracking
3. Examine confusion matrix for problem classes
4. Use the trained model for inference with `InferenceHelper`
5. Deploy the model to the FastAPI backend

## Support

For issues or questions:
- Check the logs in `training.log`
- Review the MLflow experiment tracking
- Examine the saved artifacts
- Consult the design document in `.kiro/specs/food-classification-training/design.md`
