"""
Food Classification Model Training Script

This script trains a PyTorch-based deep learning model for food image classification
using transfer learning with pre-trained models (MobileNetV2 or EfficientNet-B0).
Includes MLOps integration with MLflow for experiment tracking and model versioning.

Usage:
    python train_model.py --epochs 20 --batch-size 32 --learning-rate 0.001
    python train_model.py --model-name efficientnet_b0 --checkpoint-path ./checkpoints/model.pth

Requirements:
    - PyTorch >= 2.2.0
    - torchvision >= 0.17.0
    - mlflow >= 2.9.2
    - hypothesis >= 6.92.0 (for property-based testing)
    - tqdm >= 4.66.0
    - matplotlib >= 3.7.0
    - seaborn >= 0.12.0
    - scikit-learn >= 1.3.0

Google Colab Installation:
    !pip install torch torchvision mlflow tqdm matplotlib seaborn scikit-learn

Google Colab Usage:
    # Mount Google Drive for persistent storage
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Run training with Drive storage
    !python train_model.py --data-dir /content/drive/MyDrive/food-101 --epochs 20

Author: NutriLearn AI Team
Purpose: B.Tech Final Year AI/ML Project - Production-ready MLOps demonstration
"""

import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Any, Tuple, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def is_colab_environment() -> bool:
    """
    Detect if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def load_checkpoint_for_resume(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device
) -> Tuple[int, Dict[str, List[float]], float, float]:
    """
    Load checkpoint to resume training.
    
    Restores model weights, optimizer state, scheduler state, epoch number,
    and training history from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state
        scheduler: Scheduler to restore state
        device: Device to load checkpoint on
        
    Returns:
        Tuple of (start_epoch, training_history, best_val_loss, best_val_acc)
        
    Raises:
        CheckpointLoadError: If checkpoint is corrupted or incompatible
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate checkpoint has required keys
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 
                        'scheduler_state_dict', 'best_val_loss', 'best_val_acc']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            raise CheckpointLoadError(
                f"Checkpoint missing required keys: {missing_keys}\n"
                "This checkpoint may be corrupted or from an incompatible version."
            )
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model weights restored")
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state restored")
        
        # Restore scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state restored")
        
        # Get training state
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        })
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"Training history: {len(training_history['train_loss'])} epochs")
        
        return start_epoch, training_history, best_val_loss, best_val_acc
        
    except FileNotFoundError:
        raise CheckpointLoadError(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
        raise CheckpointLoadError(
            f"Failed to load checkpoint: {str(e)}\n"
            "The checkpoint may be corrupted or incompatible with current code version."
        ) from e


def setup_colab_environment():
    """
    Configure environment for Google Colab compatibility.
    
    - Detects GPU and prints GPU name
    - Configures tqdm for notebook display
    - Sets up GPU cache clearing
    - Provides Drive mounting instructions
    """
    if not is_colab_environment():
        return
    
    logger.info("=" * 80)
    logger.info("Google Colab Environment Detected")
    logger.info("=" * 80)
    
    # Detect and print GPU information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Available: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        logger.info("No GPU available - training will use CPU")
        logger.warning("Training on CPU will be significantly slower")
    
    # Check if Google Drive is mounted
    import os
    if os.path.exists('/content/drive'):
        logger.info("Google Drive is mounted at /content/drive")
    else:
        logger.info("Google Drive not mounted. To mount:")
        logger.info("  from google.colab import drive")
        logger.info("  drive.mount('/content/drive')")
    
    logger.info("=" * 80)


def parse_arguments():
    """
    Parse command-line arguments for training configuration.
    
    Supports all training parameters with sensible defaults and validation.
    
    Returns:
        Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train a food classification model using transfer learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to store/load Food-101 dataset'
    )
    data_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training and validation (must be >= 1)'
    )
    data_group.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of worker processes for data loading (must be >= 0)'
    )
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument(
        '--model-name',
        type=str,
        default='mobilenet_v2',
        choices=['mobilenet_v2', 'efficientnet_b0'],
        help='Pre-trained model architecture to use'
    )
    model_group.add_argument(
        '--num-classes',
        type=int,
        default=101,
        help='Number of food classes to classify (must be >= 2)'
    )
    model_group.add_argument(
        '--freeze-layers',
        type=int,
        default=10,
        help='Number of early feature layers to freeze (must be >= 0)'
    )
    model_group.add_argument(
        '--dropout-rate',
        type=float,
        default=0.2,
        help='Dropout rate in classifier head (must be in [0, 1))'
    )
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Maximum number of training epochs (must be >= 1)'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate for Adam optimizer (must be > 0)'
    )
    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='L2 regularization weight decay (must be >= 0)'
    )
    train_group.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Epochs without improvement before stopping (must be >= 1)'
    )
    
    # Scheduler parameters
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler Parameters')
    scheduler_group.add_argument(
        '--scheduler-patience',
        type=int,
        default=3,
        help='Epochs without improvement before reducing LR (must be >= 1)'
    )
    scheduler_group.add_argument(
        '--scheduler-factor',
        type=float,
        default=0.1,
        help='Factor to reduce learning rate by (must be in (0, 1))'
    )
    
    # Checkpoint parameters
    checkpoint_group = parser.add_argument_group('Checkpoint Parameters')
    checkpoint_group.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    checkpoint_group.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    checkpoint_group.add_argument(
        '--save-best-only',
        action='store_true',
        default=True,
        help='Save only the best model (based on validation loss)'
    )
    
    # MLflow parameters
    mlflow_group = parser.add_argument_group('MLflow Parameters')
    mlflow_group.add_argument(
        '--experiment-name',
        type=str,
        default='food_classification',
        help='MLflow experiment name'
    )
    mlflow_group.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name (auto-generated if not provided)'
    )
    mlflow_group.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking server URI (optional)'
    )
    
    # Device and reproducibility
    misc_group = parser.add_argument_group('Miscellaneous Parameters')
    misc_group.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to train on (auto-detected if not specified)'
    )
    misc_group.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    _validate_arguments(args)
    
    return args


def _validate_arguments(args):
    """
    Validate command-line arguments with range checks.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        ValueError: If any argument is invalid
    """
    # Validate batch_size
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    
    # Validate num_workers
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    
    # Validate num_classes
    if args.num_classes < 2:
        raise ValueError(f"--num-classes must be >= 2, got {args.num_classes}")
    
    # Validate freeze_layers
    if args.freeze_layers < 0:
        raise ValueError(f"--freeze-layers must be >= 0, got {args.freeze_layers}")
    
    # Validate dropout_rate
    if not 0 <= args.dropout_rate < 1:
        raise ValueError(f"--dropout-rate must be in [0, 1), got {args.dropout_rate}")
    
    # Validate epochs
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    
    # Validate learning_rate
    if args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be > 0, got {args.learning_rate}")
    
    # Validate weight_decay
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be >= 0, got {args.weight_decay}")
    
    # Validate early_stopping_patience
    if args.early_stopping_patience < 1:
        raise ValueError(f"--early-stopping-patience must be >= 1, got {args.early_stopping_patience}")
    
    # Validate scheduler_patience
    if args.scheduler_patience < 1:
        raise ValueError(f"--scheduler-patience must be >= 1, got {args.scheduler_patience}")
    
    # Validate scheduler_factor
    if not 0 < args.scheduler_factor < 1:
        raise ValueError(f"--scheduler-factor must be in (0, 1), got {args.scheduler_factor}")
    
    # Validate checkpoint_path exists if provided
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {args.checkpoint_path}")
    
    logger.info("All command-line arguments validated successfully")


def main():
    """
    Main entry point for the training script.
    
    Orchestrates the complete training pipeline:
    1. Parse command-line arguments and create configuration
    2. Set random seeds for reproducibility
    3. Initialize MLflow experiment and start run
    4. Initialize DatasetManager and prepare data loaders
    5. Initialize ModelBuilder and build model
    6. Initialize optimizer (Adam) and scheduler (ReduceLROnPlateau)
    7. Initialize TrainingEngine and execute training
    8. Run ModelEvaluator after training completes
    9. Save all artifacts and log to MLflow
    10. Print final summary with best metrics
    
    Requirements: 3.1, 3.2, 3.3, 9.4, 9.5
    """
    training_start_time = time.time()
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        logger.info("=" * 80)
        logger.info("Food Classification Training System")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info(f"Data Directory: {args.data_dir}")
        if args.checkpoint_path:
            logger.info(f"Resuming from: {args.checkpoint_path}")
        logger.info("=" * 80)
        
        # Create training configuration from arguments
        config = TrainingConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_name=args.model_name,
            num_classes=args.num_classes,
            freeze_layers=args.freeze_layers,
            dropout_rate=args.dropout_rate,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            checkpoint_dir=args.checkpoint_dir,
            save_best_only=args.save_best_only,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
            random_seed=args.random_seed
        )
        
        logger.info("Training configuration created successfully")
        logger.info(f"Device: {config.device}")
        logger.info(f"Random Seed: {config.random_seed}")
        
        # Set random seeds for reproducibility (torch, numpy, random)
        # This is already done in TrainingConfig.__post_init__, but we log it here
        logger.info("Random seeds set for reproducibility")
        
        # Setup Google Colab environment if detected
        setup_colab_environment()
        
        # Initialize MLflow experiment and start run
        logger.info("Initializing MLflow tracking...")
        mlflow_tracker = MLflowTracker(
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            tracking_uri=args.mlflow_tracking_uri if hasattr(args, 'mlflow_tracking_uri') else None
        )
        
        # Log hyperparameters to MLflow
        mlflow_tracker.log_params(config.to_dict())
        
        # Initialize DatasetManager and prepare data loaders
        logger.info("Preparing dataset...")
        dataset_manager = DatasetManager(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        train_loader, val_loader, idx_to_class = dataset_manager.prepare_data()
        
        # Convert idx_to_class to class_to_idx for saving
        class_to_idx = {class_name: idx for idx, class_name in idx_to_class.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        # Initialize ModelBuilder and build model
        logger.info("Building model...")
        model_builder = ModelBuilder(
            model_name=config.model_name,
            num_classes=config.num_classes,
            freeze_layers=config.freeze_layers,
            dropout_rate=config.dropout_rate
        )
        
        model = model_builder.build_model()
        device = torch.device(config.device)
        model = model.to(device)
        
        # Initialize optimizer (Adam) and scheduler (ReduceLROnPlateau)
        logger.info("Initializing optimizer and scheduler...")
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            verbose=True
        )
        
        logger.info(f"Optimizer: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})")
        logger.info(f"Scheduler: ReduceLROnPlateau (patience={config.scheduler_patience}, factor={config.scheduler_factor})")
        logger.info(f"Loss function: CrossEntropyLoss")
        
        # Initialize TrainingEngine and execute training
        logger.info("Initializing training engine...")
        training_engine = TrainingEngine(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config.num_epochs,
            early_stopping_patience=config.early_stopping_patience
        )
        
        # Load checkpoint if resuming training
        if args.checkpoint_path:
            logger.info("=" * 80)
            logger.info("RESUMING TRAINING FROM CHECKPOINT")
            logger.info("=" * 80)
            
            start_epoch, training_history, best_val_loss, best_val_acc = load_checkpoint_for_resume(
                checkpoint_path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )
            
            # Restore training engine state
            training_engine.training_history = training_history
            training_engine.best_val_loss = best_val_loss
            training_engine.best_val_acc = best_val_acc
            
            # Adjust number of epochs to train from checkpoint
            remaining_epochs = config.num_epochs - start_epoch + 1
            if remaining_epochs <= 0:
                logger.warning(f"Checkpoint is already at or past target epochs ({config.num_epochs})")
                logger.warning("No additional training needed. Proceeding to evaluation...")
                training_history = training_engine.training_history
            else:
                logger.info(f"Will train for {remaining_epochs} more epochs (from {start_epoch} to {config.num_epochs})")
                training_engine.num_epochs = config.num_epochs
                
                # Execute training from checkpoint
                logger.info("Resuming training...")
                training_history = training_engine.train()
        else:
            # Execute training from scratch
            logger.info("Starting training from scratch...")
            training_history = training_engine.train()
        
        # Log metrics to MLflow for each epoch
        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(
            zip(
                training_history['train_loss'],
                training_history['train_acc'],
                training_history['val_loss'],
                training_history['val_acc']
            ),
            start=1
        ):
            mlflow_tracker.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)
        
        # Calculate total training time
        training_time = time.time() - training_start_time
        mlflow_tracker.log_training_time(training_time)
        
        # Run ModelEvaluator after training completes
        logger.info("Running model evaluation...")
        evaluator = ModelEvaluator(
            model=model,
            data_loader=val_loader,
            device=device,
            class_names=class_names
        )
        
        evaluation_results = evaluator.evaluate()
        
        # Generate and save confusion matrix
        conf_matrix = evaluator.compute_confusion_matrix()
        conf_matrix_fig = evaluator.plot_confusion_matrix(
            conf_matrix,
            save_path="confusion_matrix.png"
        )
        
        # Log confusion matrix to MLflow
        mlflow_tracker.log_figure(conf_matrix_fig, "confusion_matrix.png")
        plt.close(conf_matrix_fig)
        
        # Plot and save training curves
        logger.info("Generating training curves...")
        training_curves_fig = plot_training_curves(
            training_history,
            save_path="training_curves.png"
        )
        mlflow_tracker.log_figure(training_curves_fig, "training_curves.png")
        plt.close(training_curves_fig)
        
        # Plot and save sample predictions
        logger.info("Generating sample predictions...")
        sample_predictions_fig = plot_sample_predictions(
            model=model,
            data_loader=val_loader,
            class_names=class_names,
            device=device,
            num_samples=16,
            save_path="sample_predictions.png"
        )
        mlflow_tracker.log_figure(sample_predictions_fig, "sample_predictions.png")
        plt.close(sample_predictions_fig)
        
        # Save all artifacts
        logger.info("Saving model artifacts...")
        artifact_manager = ModelArtifactManager(base_dir="ml-models")
        
        # Get model configuration
        model_config = model_builder.get_model_config()
        model_config['training_config'] = {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'best_val_loss': training_engine.best_val_loss,
            'best_val_acc': training_engine.best_val_acc
        }
        
        # Save all artifacts in timestamped directory
        artifact_paths = artifact_manager.save_all_artifacts(
            model=model,
            class_to_idx=class_to_idx,
            model_config=model_config,
            use_timestamped_dir=True
        )
        
        # Save evaluation results
        evaluator.save_evaluation_results(
            evaluation_results,
            filepath=str(Path(artifact_paths['model']).parent / "evaluation_results.json")
        )
        
        # Save checkpoint with full training state
        checkpoint = {
            'epoch': len(training_history['train_loss']),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': training_engine.best_val_loss,
            'best_val_acc': training_engine.best_val_acc,
            'training_history': training_history,
            'config': config.to_dict()
        }
        
        checkpoint_path = Path(config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path / f"checkpoint_{config.run_name}.pth"
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Log artifacts to MLflow
        logger.info("Logging artifacts to MLflow...")
        for artifact_name, artifact_path in artifact_paths.items():
            mlflow_tracker.log_artifact(artifact_path)
        
        mlflow_tracker.log_artifact(str(checkpoint_file), "checkpoints")
        
        # End MLflow run
        mlflow_tracker.end_run()
        
        # Print final summary with best metrics
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE - FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Total Training Time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
        logger.info(f"Epochs Completed: {len(training_history['train_loss'])}")
        logger.info("")
        logger.info("Best Validation Metrics:")
        logger.info(f"  Validation Loss: {training_engine.best_val_loss:.4f}")
        logger.info(f"  Validation Accuracy: {training_engine.best_val_acc:.2f}%")
        logger.info("")
        logger.info("Evaluation Metrics:")
        logger.info(f"  Overall Accuracy: {evaluation_results['accuracy']:.2f}%")
        logger.info(f"  Top-3 Accuracy: {evaluation_results['top_3_accuracy']:.2f}%")
        logger.info(f"  Top-5 Accuracy: {evaluation_results['top_5_accuracy']:.2f}%")
        logger.info("")
        logger.info("Artifacts Saved:")
        logger.info(f"  Model: {artifact_paths['model']}")
        logger.info(f"  Class Mapping: {artifact_paths['class_mapping']}")
        logger.info(f"  Config: {artifact_paths['config']}")
        logger.info(f"  Checkpoint: {checkpoint_file}")
        logger.info("")
        logger.info("Worst Performing Classes (Top 5):")
        for i, cls in enumerate(evaluation_results['worst_classes'][:5], 1):
            logger.info(f"  {i}. {cls['class']}: F1={cls['f1_score']:.3f}")
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
    except ValueError as e:
        logger.error(f"Invalid argument: {str(e)}")
        sys.exit(1)
    except DatasetNotFoundError as e:
        logger.error(f"Dataset error: {str(e)}")
        logger.error("Please ensure you have internet connection for automatic download,")
        logger.error("or manually download the dataset as instructed above.")
        sys.exit(2)
    except UnsupportedModelError as e:
        logger.error(f"Model error: {str(e)}")
        logger.error(f"Supported models: {', '.join(ModelBuilder.SUPPORTED_MODELS)}")
        sys.exit(3)
    except OutOfMemoryError as e:
        logger.error(f"Memory error: {str(e)}")
        logger.error("Training stopped due to insufficient GPU memory.")
        logger.error("Please restart with a smaller batch size.")
        sys.exit(4)
    except NaNLossError as e:
        logger.error(f"Training error: {str(e)}")
        logger.error("Training stopped due to numerical instability.")
        logger.error("Emergency checkpoint has been saved.")
        sys.exit(5)
    except CheckpointLoadError as e:
        logger.error(f"Checkpoint error: {str(e)}")
        logger.error("Cannot resume from corrupted checkpoint.")
        logger.error("Please start training from scratch or use a different checkpoint.")
        sys.exit(6)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        logger.error("An unexpected error occurred. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()


import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Dict
import numpy as np


class DatasetManager:
    """
    Manages Food-101 dataset loading and preprocessing.
    
    Handles dataset downloading, train/validation splitting with stratified sampling,
    data augmentation for training, and data loader creation with parallel loading.
    
    Attributes:
        data_dir: Directory to store/load the Food-101 dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for parallel data loading
        train_split: Ratio of training data (default 0.8 for 80/20 split)
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8
    ):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store/load dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for parallel loading
            train_split: Ratio of training data (0.0 to 1.0)
            
        Raises:
            ValueError: If train_split is not between 0 and 1
        """
        if not 0 < train_split < 1:
            raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        
        logger.info(f"DatasetManager initialized with batch_size={batch_size}, "
                   f"num_workers={num_workers}, train_split={train_split}")
    
    def get_transforms(self, is_training: bool) -> transforms.Compose:
        """
        Get image transformations for training or validation.
        
        Training transforms include data augmentation:
        - RandomResizedCrop(224): Random crop and resize for scale invariance
        - RandomHorizontalFlip: Horizontal flip for orientation invariance
        - ColorJitter: Random brightness, contrast, saturation changes
        - Normalize: ImageNet mean and std normalization
        
        Validation transforms are deterministic:
        - Resize(256): Resize shorter side to 256
        - CenterCrop(224): Center crop to 224x224
        - Normalize: ImageNet mean and std normalization
        
        Args:
            is_training: Whether to apply training augmentation
            
        Returns:
            Composed transformations
        """
        # ImageNet normalization statistics (pre-trained models expect these)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if is_training:
            # Training augmentation pipeline
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                normalize
            ])
            logger.debug("Created training transforms with augmentation")
        else:
            # Validation pipeline (deterministic)
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            logger.debug("Created validation transforms (no augmentation)")
        
        return transform
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
        """
        Prepare train and validation data loaders with stratified sampling.
        
        Downloads Food-101 dataset if not present, creates 80/20 train/val split
        with stratified sampling to maintain class distribution, and creates
        data loaders with parallel loading.
        
        Returns:
            Tuple of (train_loader, val_loader, idx_to_class mapping)
            
        Raises:
            RuntimeError: If dataset download or loading fails
        """
        logger.info("Preparing Food-101 dataset...")
        
        try:
            # Load full training dataset (Food-101 has predefined train/test split)
            # We'll use the training split and further divide it into train/val
            full_dataset = datasets.Food101(
                root=self.data_dir,
                split='train',
                download=True,
                transform=None  # We'll apply transforms later
            )
            
            logger.info(f"Loaded Food-101 dataset with {len(full_dataset)} training images")
            
            # Get class information
            num_classes = len(full_dataset.classes)
            idx_to_class = {i: class_name for i, class_name in enumerate(full_dataset.classes)}
            
            logger.info(f"Dataset contains {num_classes} food classes")
            
            # Create stratified train/val split
            train_indices, val_indices = self._stratified_split(full_dataset)
            
            logger.info(f"Split: {len(train_indices)} training, {len(val_indices)} validation")
            logger.info(f"Split ratio: {len(train_indices)/len(full_dataset):.2%} train, "
                       f"{len(val_indices)/len(full_dataset):.2%} val")
            
            # Create subsets with appropriate transforms
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            
            # Apply transforms by wrapping datasets
            train_dataset.dataset.transform = self.get_transforms(is_training=True)
            val_dataset.dataset.transform = self.get_transforms(is_training=False)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            logger.info(f"Created data loaders: {len(train_loader)} train batches, "
                       f"{len(val_loader)} val batches")
            
            return train_loader, val_loader, idx_to_class
            
        except FileNotFoundError as e:
            error_msg = (
                f"Food-101 dataset not found in {self.data_dir}.\n"
                "The dataset will be automatically downloaded on first run.\n"
                "If download fails, you can manually download from:\n"
                "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz\n"
                f"Extract to: {self.data_dir}"
            )
            logger.error(error_msg, exc_info=True)
            raise DatasetNotFoundError(error_msg) from e
        except ConnectionError as e:
            error_msg = (
                f"Failed to download Food-101 dataset due to network error.\n"
                "Please check your internet connection and try again.\n"
                "Alternatively, manually download from:\n"
                "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
            )
            logger.error(error_msg, exc_info=True)
            raise DatasetNotFoundError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {str(e)}", exc_info=True)
            raise RuntimeError(f"Dataset preparation failed: {str(e)}") from e
    
    def _stratified_split(self, dataset: datasets.Food101) -> Tuple[list, list]:
        """
        Create stratified train/validation split maintaining class distribution.
        
        Ensures each class is proportionally represented in both train and val sets.
        
        Args:
            dataset: Food-101 dataset to split
            
        Returns:
            Tuple of (train_indices, val_indices)
        """
        # Get all labels
        labels = np.array([label for _, label in dataset])
        
        # Get unique classes
        classes = np.unique(labels)
        
        train_indices = []
        val_indices = []
        
        # For each class, split proportionally
        for class_idx in classes:
            # Get indices for this class
            class_indices = np.where(labels == class_idx)[0]
            
            # Shuffle indices for this class
            np.random.shuffle(class_indices)
            
            # Split based on train_split ratio
            split_point = int(len(class_indices) * self.train_split)
            
            train_indices.extend(class_indices[:split_point])
            val_indices.extend(class_indices[split_point:])
        
        # Shuffle the final indices
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        logger.debug(f"Stratified split created: {len(train_indices)} train, "
                    f"{len(val_indices)} val samples")
        
        return train_indices, val_indices
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get mapping of class indices to class names.
        
        Returns:
            Dictionary mapping class index to class name
        """
        try:
            dataset = datasets.Food101(
                root=self.data_dir,
                split='train',
                download=False
            )
            return {i: class_name for i, class_name in enumerate(dataset.classes)}
        except Exception as e:
            logger.error(f"Failed to get class names: {str(e)}")
            raise


import torch.nn as nn
from torchvision import models
from typing import Any
from datetime import datetime


class ModelBuilder:
    """
    Builds and configures transfer learning models for food classification.
    
    Supports pre-trained MobileNetV2 and EfficientNet-B0 architectures with
    configurable layer freezing and custom classifier heads.
    
    Attributes:
        model_name: Name of pre-trained model ('mobilenet_v2' or 'efficientnet_b0')
        num_classes: Number of output classes for classification
        freeze_layers: Number of early feature layers to freeze
        dropout_rate: Dropout rate for classifier head
    """
    
    SUPPORTED_MODELS = ['mobilenet_v2', 'efficientnet_b0']
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        freeze_layers: int = 10,
        dropout_rate: float = 0.2
    ):
        """
        Initialize model builder.
        
        Args:
            model_name: Name of pre-trained model ('mobilenet_v2' or 'efficientnet_b0')
            num_classes: Number of output classes
            freeze_layers: Number of early layers to freeze
            dropout_rate: Dropout rate for regularization
            
        Raises:
            ValueError: If model_name is not supported or parameters are invalid
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )
        
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        
        if freeze_layers < 0:
            raise ValueError(f"freeze_layers must be >= 0, got {freeze_layers}")
        
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.dropout_rate = dropout_rate
        
        logger.info(f"ModelBuilder initialized: {model_name}, "
                   f"{num_classes} classes, freeze {freeze_layers} layers")
    
    def build_model(self) -> nn.Module:
        """
        Build and configure the model with frozen layers and custom classifier.
        
        Loads pre-trained model, freezes specified number of early layers,
        and replaces the classifier head with a custom one for the target
        number of classes.
        
        Returns:
            Configured PyTorch model ready for training
            
        Raises:
            RuntimeError: If model loading or configuration fails
        """
        try:
            logger.info(f"Building {self.model_name} model...")
            
            # Load pre-trained model
            if self.model_name == 'mobilenet_v2':
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                in_features = model.classifier[1].in_features
                logger.info(f"Loaded MobileNetV2 (in_features={in_features})")
            elif self.model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                in_features = model.classifier[1].in_features
                logger.info(f"Loaded EfficientNet-B0 (in_features={in_features})")
            
            # Freeze backbone layers
            self._freeze_backbone(model)
            
            # Replace classifier head
            self._replace_classifier(model, in_features)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"Model built successfully:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model building failed: {str(e)}") from e
    
    def _freeze_backbone(self, model: nn.Module) -> None:
        """
        Freeze the first N feature layers to preserve pre-trained weights.
        
        Args:
            model: PyTorch model to freeze layers in
        """
        if self.freeze_layers == 0:
            logger.info("No layers frozen (training all layers)")
            return
        
        # Get feature layers
        if self.model_name == 'mobilenet_v2':
            feature_layers = list(model.features.children())
        elif self.model_name == 'efficientnet_b0':
            feature_layers = list(model.features.children())
        
        # Freeze first N layers
        layers_to_freeze = min(self.freeze_layers, len(feature_layers))
        
        for i, layer in enumerate(feature_layers[:layers_to_freeze]):
            for param in layer.parameters():
                param.requires_grad = False
        
        logger.info(f"Froze first {layers_to_freeze} feature layers")
    
    def _replace_classifier(self, model: nn.Module, in_features: int) -> None:
        """
        Replace the classifier head with custom layers for target classes.
        
        Creates a new classifier with Dropout for regularization and Linear
        layer with output dimension matching num_classes.
        
        Args:
            model: PyTorch model to replace classifier in
            in_features: Number of input features to classifier
        """
        # Create new classifier head
        new_classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, self.num_classes)
        )
        
        # Replace classifier
        model.classifier = new_classifier
        
        logger.info(f"Replaced classifier: Dropout({self.dropout_rate}) -> "
                   f"Linear({in_features}, {self.num_classes})")
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for saving with artifacts.
        
        Returns:
            Dictionary with model metadata including architecture, input size,
            normalization parameters, and training configuration
        """
        config = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'freeze_layers': self.freeze_layers,
            'dropout_rate': self.dropout_rate,
            'input_size': [224, 224],
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'timestamp': datetime.now().isoformat(),
            'architecture_details': {
                'backbone': self.model_name,
                'pretrained': 'ImageNet-1K',
                'classifier': f'Dropout({self.dropout_rate}) + Linear({self.num_classes})'
            }
        }
        
        logger.debug(f"Generated model config: {config}")
        return config


class DatasetNotFoundError(Exception):
    """
    Raised when dataset cannot be found or downloaded.
    
    Provides instructions for manual dataset download.
    """
    pass


class UnsupportedModelError(Exception):
    """Raised when an unsupported model architecture is requested."""
    pass


class ModelLoadError(Exception):
    """Raised when pre-trained model loading fails."""
    pass


from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class TrainingConfig:
    """
    Configuration for model training run.
    
    Centralizes all hyperparameters and settings for reproducible training.
    Includes data parameters, model architecture settings, training hyperparameters,
    scheduler configuration, checkpoint management, and MLflow tracking settings.
    
    Attributes:
        data_dir: Directory to store/load Food-101 dataset
        batch_size: Batch size for training and validation
        num_workers: Number of worker processes for data loading
        
        model_name: Pre-trained model architecture ('mobilenet_v2' or 'efficientnet_b0')
        num_classes: Number of food classes to classify
        freeze_layers: Number of early feature layers to freeze
        dropout_rate: Dropout rate in classifier head
        
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate for Adam optimizer
        weight_decay: L2 regularization weight decay
        early_stopping_patience: Epochs without improvement before stopping
        
        scheduler_patience: Epochs without improvement before reducing LR
        scheduler_factor: Factor to reduce learning rate by
        
        checkpoint_dir: Directory to save model checkpoints
        save_best_only: Whether to save only the best model or all checkpoints
        
        experiment_name: MLflow experiment name
        run_name: MLflow run name (auto-generated if None)
        
        device: Device to train on ('cuda' or 'cpu', auto-detected)
        random_seed: Random seed for reproducibility
    """
    
    # Data parameters
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    
    # Model parameters
    model_name: str = "mobilenet_v2"
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
    
    # Device and reproducibility
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate model name
        if self.model_name not in ModelBuilder.SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model_name: {self.model_name}. "
                f"Must be one of {ModelBuilder.SUPPORTED_MODELS}"
            )
        
        # Validate numeric ranges
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        
        if self.freeze_layers < 0:
            raise ValueError(f"freeze_layers must be >= 0, got {self.freeze_layers}")
        
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        
        if self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}")
        
        if self.scheduler_patience < 1:
            raise ValueError(f"scheduler_patience must be >= 1, got {self.scheduler_patience}")
        
        if self.scheduler_factor <= 0 or self.scheduler_factor >= 1:
            raise ValueError(f"scheduler_factor must be in (0, 1), got {self.scheduler_factor}")
        
        # Auto-generate run name if not provided
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_name}_{timestamp}"
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        logger.info("TrainingConfig initialized and validated")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Random seed: {self.random_seed}")
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility across all libraries."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            # Make CUDA operations deterministic (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.debug(f"Random seeds set to {self.random_seed} for reproducibility")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for logging.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'data_dir': self.data_dir,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'freeze_layers': self.freeze_layers,
            'dropout_rate': self.dropout_rate,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor,
            'checkpoint_dir': self.checkpoint_dir,
            'save_best_only': self.save_best_only,
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'device': self.device,
            'random_seed': self.random_seed
        }


from tqdm.auto import tqdm  # Auto-detects notebook environment
import time
from typing import List


class TrainingEngine:
    """
    Handles model training and validation loops with early stopping.
    
    Manages the complete training process including forward/backward passes,
    optimization, validation, early stopping, and checkpoint saving.
    
    Attributes:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for weight updates
        scheduler: Learning rate scheduler
        device: Device to train on (CPU/GPU)
        num_epochs: Maximum number of training epochs
        early_stopping_patience: Epochs without improvement before stopping
    """
    
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
        early_stopping_patience: int = 5
    ):
        """
        Initialize training engine with all components.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (e.g., CrossEntropyLoss)
            optimizer: Optimizer (e.g., Adam)
            scheduler: Learning rate scheduler (e.g., ReduceLROnPlateau)
            device: Device to train on
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info("TrainingEngine initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Performs forward pass, loss calculation, backward pass, and weight updates
        for all batches in the training set. Displays progress bar with tqdm.
        
        Args:
            epoch: Current epoch number (for logging)
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs} [Train]",
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    error_msg = (
                        f"NaN or Inf loss detected at epoch {epoch}, batch {batch_idx}.\n"
                        f"Last valid loss: {running_loss / max(batch_idx, 1):.4f}\n"
                        "This usually indicates:\n"
                        "  - Learning rate too high (try reducing by 10x)\n"
                        "  - Gradient explosion (try gradient clipping)\n"
                        "  - Numerical instability in the model\n"
                        "Saving emergency checkpoint..."
                    )
                    logger.error(error_msg)
                    
                    # Save emergency checkpoint
                    emergency_checkpoint_path = f"emergency_checkpoint_epoch{epoch}_batch{batch_idx}.pth"
                    self.save_checkpoint(emergency_checkpoint_path, epoch, {'error': 'NaN loss'})
                    
                    raise NaNLossError(error_msg)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update running loss
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    error_msg = (
                        f"GPU out of memory at epoch {epoch}, batch {batch_idx}.\n"
                        f"Current batch size: {images.size(0)}\n"
                        "Suggestions:\n"
                        "  - Reduce batch size (try half of current: {images.size(0) // 2})\n"
                        "  - Reduce number of workers\n"
                        "  - Use gradient accumulation\n"
                        "  - Train on CPU (slower but uses system RAM)"
                    )
                    logger.error(error_msg, exc_info=True)
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    raise OutOfMemoryError(error_msg) from e
                else:
                    # Re-raise other runtime errors
                    raise
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model on validation set.
        
        Runs model in evaluation mode (no gradient computation) and calculates
        loss and accuracy on validation data.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            
            for images, labels in pbar:
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update running loss
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, List[float]]:
        """
        Execute full training loop with early stopping.
        
        Trains for specified number of epochs, validates after each epoch,
        updates learning rate scheduler, checks for early stopping, and
        tracks best model.
        
        Returns:
            Dictionary with training history (losses and accuracies)
        """
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            logger.info(f"Epoch {epoch}/{self.num_epochs} - {epoch_time:.2f}s")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                logger.info(f"  ✓ New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info("=" * 80)
                logger.info(f"Early stopping triggered after {epoch} epochs")
                logger.info(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.2f}%")
                logger.info("=" * 80)
                break
            
            # Clear GPU cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("-" * 80)
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best Val Acc: {self.best_val_acc:.2f}%")
        logger.info("=" * 80)
        
        return self.training_history
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        additional_info: Dict[str, Any] = None
    ) -> None:
        """
        Save model checkpoint with training state.
        
        Saves model weights, optimizer state, scheduler state, and training
        history for resuming training later.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            additional_info: Additional information to save (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
            logger.info(f"  Epoch: {epoch}")
            logger.info(f"  Val Loss: {self.best_val_loss:.4f}")
            logger.info(f"  Val Acc: {self.best_val_acc:.2f}%")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise


class NaNLossError(Exception):
    """Raised when loss becomes NaN during training."""
    pass


class OutOfMemoryError(Exception):
    """Raised when GPU memory is exhausted."""
    pass


import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from pathlib import Path


class MLflowTracker:
    """
    Manages MLflow experiment tracking for model training.
    
    Handles experiment creation, parameter logging, metric tracking,
    artifact saving, and graceful fallback when MLflow is unavailable.
    
    Attributes:
        experiment_name: Name of MLflow experiment
        run_name: Name for this specific run
        tracking_uri: MLflow tracking server URI
        mlflow_available: Whether MLflow is available for tracking
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: str = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
            run_name: Name for this specific run
            tracking_uri: MLflow tracking server URI (optional)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.mlflow_available = True
        self.local_metrics = []  # Fallback storage
        
        try:
            # Set tracking URI if provided
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            # Set or create experiment
            mlflow.set_experiment(experiment_name)
            
            # Start run
            mlflow.start_run(run_name=run_name)
            
            logger.info(f"MLflow tracking initialized")
            logger.info(f"  Experiment: {experiment_name}")
            logger.info(f"  Run: {run_name}")
            logger.info(f"  Tracking URI: {mlflow.get_tracking_uri()}")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {str(e)}")
            logger.warning("Continuing without MLflow tracking (metrics saved locally)")
            self.mlflow_available = False
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to MLflow.
        
        Args:
            params: Dictionary of hyperparameters to log
        """
        if not self.mlflow_available:
            logger.debug("MLflow unavailable, skipping parameter logging")
            return
        
        try:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters to MLflow")
            logger.debug(f"Parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics for a specific step/epoch.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step/epoch number
        """
        # Store locally as fallback
        self.local_metrics.append({'step': step, **metrics})
        
        if not self.mlflow_available:
            logger.debug(f"MLflow unavailable, metrics saved locally (step {step})")
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics for step {step}: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_model(self, model_path: str, artifact_path: str = "model") -> None:
        """
        Log model artifact to MLflow.
        
        Args:
            model_path: Path to saved model file
            artifact_path: Artifact path in MLflow (default: "model")
        """
        if not self.mlflow_available:
            logger.debug("MLflow unavailable, skipping model logging")
            return
        
        try:
            mlflow.log_artifact(model_path, artifact_path)
            logger.info(f"Logged model artifact: {model_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """
        Log any file as an artifact to MLflow.
        
        Args:
            local_path: Path to local file
            artifact_path: Artifact path in MLflow (optional)
        """
        if not self.mlflow_available:
            logger.debug("MLflow unavailable, skipping artifact logging")
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {str(e)}")
    
    def log_figure(
        self,
        figure: plt.Figure,
        artifact_file: str,
        artifact_path: str = "plots"
    ) -> None:
        """
        Log matplotlib figure as an artifact.
        
        Saves figure to temporary file and logs to MLflow.
        
        Args:
            figure: Matplotlib figure to log
            artifact_file: Filename for the artifact (e.g., "confusion_matrix.png")
            artifact_path: Artifact directory in MLflow (default: "plots")
        """
        if not self.mlflow_available:
            logger.debug("MLflow unavailable, skipping figure logging")
            # Still save locally
            try:
                figure.savefig(artifact_file, dpi=150, bbox_inches='tight')
                logger.info(f"Figure saved locally: {artifact_file}")
            except Exception as e:
                logger.error(f"Failed to save figure: {str(e)}")
            return
        
        try:
            # Save figure temporarily
            temp_path = Path(artifact_file)
            figure.savefig(temp_path, dpi=150, bbox_inches='tight')
            
            # Log to MLflow
            mlflow.log_artifact(str(temp_path), artifact_path)
            logger.info(f"Logged figure: {artifact_file}")
            
        except Exception as e:
            logger.error(f"Failed to log figure: {str(e)}")
    
    def log_training_time(self, training_time: float) -> None:
        """
        Log total training time as a metric.
        
        Args:
            training_time: Total training time in seconds
        """
        if not self.mlflow_available:
            logger.debug("MLflow unavailable, skipping training time logging")
            return
        
        try:
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("training_time_minutes", training_time / 60)
            logger.info(f"Logged training time: {training_time:.2f}s ({training_time/60:.2f}m)")
        except Exception as e:
            logger.error(f"Failed to log training time: {str(e)}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.mlflow_available:
            # Save local metrics to file
            self._save_local_metrics()
            return
        
        try:
            mlflow.end_run()
            logger.info("MLflow run ended successfully")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {str(e)}")
    
    def _save_local_metrics(self) -> None:
        """Save metrics locally when MLflow is unavailable."""
        try:
            import json
            metrics_file = f"metrics_{self.run_name}.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.local_metrics, f, indent=2)
            logger.info(f"Metrics saved locally: {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save local metrics: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


class MLflowConnectionError(Exception):
    """Raised when MLflow connection fails."""
    pass


class ArtifactSaveError(Exception):
    """Raised when artifact saving fails."""
    pass


import json
import os


class ModelArtifactManager:
    """
    Manages saving and loading of model artifacts.
    
    Handles saving model weights, class mappings, model configuration,
    and organizing artifacts in timestamped directories.
    
    Attributes:
        base_dir: Base directory for saving artifacts (default: ml-models)
        version: Model version identifier
    """
    
    def __init__(self, base_dir: str = "ml-models", version: str = "v1"):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for artifacts
            version: Model version identifier
        """
        self.base_dir = Path(base_dir)
        self.version = version
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelArtifactManager initialized")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Version: {version}")
    
    def save_model(
        self,
        model: nn.Module,
        filename: str = None
    ) -> str:
        """
        Save model state_dict to file.
        
        Args:
            model: PyTorch model to save
            filename: Custom filename (optional, auto-generated if None)
            
        Returns:
            Path to saved model file
        """
        if filename is None:
            filename = f"food_model_{self.version}.pth"
        
        model_path = self.base_dir / filename
        
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved: {model_path}")
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise
    
    def save_class_mapping(
        self,
        class_to_idx: Dict[str, int],
        filename: str = "class_to_idx.json"
    ) -> str:
        """
        Save class to index mapping as JSON.
        
        Args:
            class_to_idx: Dictionary mapping class names to indices
            filename: Filename for class mapping
            
        Returns:
            Path to saved mapping file
        """
        mapping_path = self.base_dir / filename
        
        try:
            with open(mapping_path, 'w') as f:
                json.dump(class_to_idx, f, indent=2)
            
            logger.info(f"Class mapping saved: {mapping_path}")
            logger.info(f"  Total classes: {len(class_to_idx)}")
            return str(mapping_path)
        except Exception as e:
            logger.error(f"Failed to save class mapping: {str(e)}", exc_info=True)
            raise
    
    def save_model_config(
        self,
        config: Dict[str, Any],
        filename: str = "model_config.json"
    ) -> str:
        """
        Save model configuration as JSON.
        
        Includes architecture details, input size, normalization parameters,
        and training configuration.
        
        Args:
            config: Model configuration dictionary
            filename: Filename for config
            
        Returns:
            Path to saved config file
        """
        config_path = self.base_dir / filename
        
        try:
            # Add version and timestamp if not present
            if 'version' not in config:
                config['version'] = self.version
            if 'timestamp' not in config:
                config['timestamp'] = datetime.now().isoformat()
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Model config saved: {config_path}")
            return str(config_path)
        except Exception as e:
            logger.error(f"Failed to save model config: {str(e)}", exc_info=True)
            raise
    
    def create_timestamped_directory(self) -> Path:
        """
        Create timestamped directory for organizing artifacts.
        
        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"training_{timestamp}_{self.version}"
        dir_path = self.base_dir / dir_name
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created timestamped directory: {dir_path}")
            return dir_path
        except Exception as e:
            logger.error(f"Failed to create directory: {str(e)}", exc_info=True)
            raise
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filename: str = "checkpoint.pth"
    ) -> str:
        """
        Save training checkpoint with all required keys.
        
        Checkpoint must include: epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict, best_val_loss, best_val_acc.
        
        Args:
            checkpoint: Checkpoint dictionary
            filename: Filename for checkpoint
            
        Returns:
            Path to saved checkpoint
            
        Raises:
            ValueError: If required keys are missing
        """
        required_keys = [
            'epoch',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'best_val_loss',
            'best_val_acc'
        ]
        
        # Validate checkpoint has all required keys
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
        
        checkpoint_path = self.base_dir / filename
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            logger.info(f"  Epoch: {checkpoint['epoch']}")
            logger.info(f"  Val Loss: {checkpoint['best_val_loss']:.4f}")
            logger.info(f"  Val Acc: {checkpoint['best_val_acc']:.2f}%")
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            CheckpointLoadError: If checkpoint is corrupted
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
            raise CheckpointLoadError(f"Corrupted checkpoint: {str(e)}") from e
    
    def save_all_artifacts(
        self,
        model: nn.Module,
        class_to_idx: Dict[str, int],
        model_config: Dict[str, Any],
        use_timestamped_dir: bool = True
    ) -> Dict[str, str]:
        """
        Save all model artifacts in one call.
        
        Args:
            model: PyTorch model
            class_to_idx: Class mapping dictionary
            model_config: Model configuration dictionary
            use_timestamped_dir: Whether to use timestamped directory
            
        Returns:
            Dictionary with paths to all saved artifacts
        """
        # Create timestamped directory if requested
        if use_timestamped_dir:
            artifact_dir = self.create_timestamped_directory()
            # Temporarily change base_dir
            original_base_dir = self.base_dir
            self.base_dir = artifact_dir
        
        try:
            paths = {}
            
            # Save model
            paths['model'] = self.save_model(model)
            
            # Save class mapping
            paths['class_mapping'] = self.save_class_mapping(class_to_idx)
            
            # Save model config
            paths['config'] = self.save_model_config(model_config)
            
            logger.info("All artifacts saved successfully")
            return paths
            
        finally:
            # Restore original base_dir if changed
            if use_timestamped_dir:
                self.base_dir = original_base_dir


class CheckpointLoadError(Exception):
    """Raised when checkpoint loading fails."""
    pass


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns


class ModelEvaluator:
    """
    Evaluates trained model and generates comprehensive metrics.
    
    Computes accuracy, precision, recall, F1-score per class, confusion matrix,
    top-k accuracy, and identifies worst-performing classes.
    
    Attributes:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: List of class names for reporting
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        class_names: List[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader for evaluation data
            device: Device to run on
            class_names: List of class names (optional)
        """
        self.model = model.to(device)
        self.device = device
        self.data_loader = data_loader
        self.class_names = class_names
        
        logger.info("ModelEvaluator initialized")
        logger.info(f"  Evaluation batches: {len(data_loader)}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.
        
        Computes all metrics including per-class metrics, confusion matrix,
        top-k accuracies, and worst-performing classes.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("Starting model evaluation...")
        logger.info("=" * 80)
        
        # Get predictions and labels
        all_preds, all_labels, all_probs = self._get_predictions()
        
        # Calculate overall accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        # Calculate per-class metrics
        per_class_metrics = self._get_per_class_metrics(all_labels, all_preds)
        
        # Generate confusion matrix
        conf_matrix = self.compute_confusion_matrix(all_labels, all_preds)
        
        # Calculate top-k accuracies
        top_1_acc = accuracy
        top_3_acc = self.get_top_k_accuracy(all_probs, all_labels, k=3)
        top_5_acc = self.get_top_k_accuracy(all_probs, all_labels, k=5)
        
        # Identify worst classes
        worst_classes = self._identify_worst_classes(per_class_metrics, n=10)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'top_3_accuracy': top_3_acc,
            'top_5_accuracy': top_5_acc,
            'per_class_metrics': per_class_metrics,
            'worst_classes': worst_classes,
            'confusion_matrix_shape': conf_matrix.shape,
            'num_samples': len(all_labels)
        }
        
        # Log summary
        logger.info("Evaluation Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.2f}%")
        logger.info(f"  Top-3 Accuracy: {top_3_acc:.2f}%")
        logger.info(f"  Top-5 Accuracy: {top_5_acc:.2f}%")
        logger.info(f"  Samples evaluated: {len(all_labels)}")
        logger.info("=" * 80)
        
        return results
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions for all data.
        
        Returns:
            Tuple of (predictions, labels, probabilities)
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.data_loader, desc="Evaluating")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs)
        )
    
    def compute_confusion_matrix(
        self,
        labels: np.ndarray = None,
        predictions: np.ndarray = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute and return confusion matrix.
        
        Args:
            labels: True labels (if None, will compute from data_loader)
            predictions: Predicted labels (if None, will compute from data_loader)
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Confusion matrix as numpy array
        """
        if labels is None or predictions is None:
            predictions, labels, _ = self._get_predictions()
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Normalize if requested
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        logger.info(f"Confusion matrix computed: shape {conf_matrix.shape}")
        
        # Verify normalization
        if normalize:
            row_sums = conf_matrix.sum(axis=1)
            logger.debug(f"Normalized confusion matrix row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}")
        
        return conf_matrix
    
    def _get_per_class_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Get precision, recall, F1 for each class.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            
        Returns:
            Dictionary mapping class names to metrics
        """
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0
        )
        
        # Organize by class
        per_class = {}
        num_classes = len(precision)
        
        for i in range(num_classes):
            class_name = self.class_names[i] if self.class_names else f"class_{i}"
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        logger.info(f"Per-class metrics computed for {num_classes} classes")
        return per_class
    
    def get_top_k_accuracy(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            probabilities: Predicted probabilities (N x num_classes)
            labels: True labels (N,)
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy as percentage
        """
        # Get top-k predictions
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_preds[i]:
                correct += 1
        
        accuracy = (correct / len(labels)) * 100
        logger.debug(f"Top-{k} accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def _identify_worst_classes(
        self,
        per_class_metrics: Dict[str, Dict[str, float]],
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify worst-performing classes by F1-score.
        
        Args:
            per_class_metrics: Per-class metrics dictionary
            n: Number of worst classes to return
            
        Returns:
            List of worst classes with their F1-scores
        """
        # Sort classes by F1-score
        sorted_classes = sorted(
            per_class_metrics.items(),
            key=lambda x: x[1]['f1_score']
        )
        
        # Get worst N classes
        worst = [
            {
                'class': class_name,
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'support': metrics['support']
            }
            for class_name, metrics in sorted_classes[:n]
        ]
        
        logger.info(f"Identified {len(worst)} worst-performing classes")
        for i, cls in enumerate(worst[:5], 1):
            logger.info(f"  {i}. {cls['class']}: F1={cls['f1_score']:.3f}")
        
        return worst
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        filepath: str = "ml-models/evaluation_results.json"
    ) -> None:
        """
        Save evaluation results as JSON.
        
        Args:
            results: Evaluation results dictionary
            filepath: Path to save results
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {str(e)}", exc_info=True)
            raise
    
    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        save_path: str = "confusion_matrix.png",
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            conf_matrix: Confusion matrix to plot
            save_path: Path to save figure
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            conf_matrix,
            annot=False,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix plot: {str(e)}")
        
        return fig


from PIL import Image


def plot_training_curves(
    training_history: Dict[str, List[float]],
    save_path: str = "training_curves.png"
) -> plt.Figure:
    """
    Plot training and validation loss and accuracy curves.
    
    Creates a 2x1 subplot showing:
    - Top: Training and validation loss over epochs
    - Bottom: Training and validation accuracy over epochs
    
    Args:
        training_history: Dictionary with train_loss, train_acc, val_loss, val_acc lists
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    try:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training curves saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save training curves: {str(e)}")
    
    return fig


def plot_sample_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    num_samples: int = 16,
    save_path: str = "sample_predictions.png"
) -> plt.Figure:
    """
    Display sample predictions with ground truth labels.
    
    Creates a grid of images showing model predictions vs actual labels.
    Correct predictions are shown in green, incorrect in red.
    
    Args:
        model: Trained model
        data_loader: DataLoader to sample from
        class_names: List of class names
        device: Device to run inference on
        num_samples: Number of samples to display
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Move to CPU for plotting
    images = images.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()
    
    # Create figure
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Denormalize images for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Denormalize image
        img = images[idx].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Get prediction info
        pred_class = class_names[predictions[idx]]
        true_class = class_names[labels[idx]]
        confidence = probabilities[idx][predictions[idx]] * 100
        
        # Set title color based on correctness
        is_correct = predictions[idx] == labels[idx]
        color = 'green' if is_correct else 'red'
        
        # Set title
        title = f"Pred: {pred_class}\n({confidence:.1f}%)\nTrue: {true_class}"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    try:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sample predictions saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save sample predictions: {str(e)}")
    
    return fig


class InferenceHelper:
    """
    Helper for model inference on single images.
    
    Loads trained model and provides prediction capabilities for single images
    with preprocessing and top-k prediction support.
    
    Attributes:
        model_path: Path to saved model file
        config_path: Path to model config JSON
        class_mapping_path: Path to class mapping JSON
        model: Loaded PyTorch model
        config: Model configuration dictionary
        idx_to_class: Index to class name mapping
        device: Device for inference
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        class_mapping_path: str,
        device: str = None
    ):
        """
        Initialize inference helper.
        
        Args:
            model_path: Path to saved model (.pth file)
            config_path: Path to model config JSON
            class_mapping_path: Path to class mapping JSON
            device: Device for inference (auto-detected if None)
            
        Raises:
            FileNotFoundError: If any required file is missing
            ModelNotFoundError: If model file doesn't exist
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.class_mapping_path = Path(class_mapping_path)
        
        # Validate files exist
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize (will be loaded)
        self.model = None
        self.config = None
        self.idx_to_class = None
        self.transform = None
        
        logger.info("InferenceHelper initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load model and configuration files.
        
        Loads model weights, configuration, and class mappings.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Load config
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info("Loaded model configuration")
            logger.debug(f"Config: {self.config}")
            
            # Load class mapping
            with open(self.class_mapping_path, 'r') as f:
                class_to_idx = json.load(f)
            
            # Create reverse mapping (idx to class)
            self.idx_to_class = {int(idx): class_name for class_name, idx in class_to_idx.items()}
            
            logger.info(f"Loaded class mapping: {len(self.idx_to_class)} classes")
            
            # Build model architecture
            model_builder = ModelBuilder(
                model_name=self.config['model_name'],
                num_classes=self.config['num_classes'],
                freeze_layers=self.config.get('freeze_layers', 10),
                dropout_rate=self.config.get('dropout_rate', 0.2)
            )
            
            self.model = model_builder.build_model()
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create preprocessing transform
            self.transform = self._create_transform()
            
            logger.info("Model loaded successfully and ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def _create_transform(self) -> transforms.Compose:
        """
        Create preprocessing transform from config.
        
        Returns:
            Composed transformations for inference
        """
        # Get normalization parameters from config
        norm_params = self.config.get('normalization', {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        })
        
        # Get input size from config
        input_size = self.config.get('input_size', [224, 224])[0]
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_params['mean'],
                std=norm_params['std']
            )
        ])
        
        logger.debug(f"Created inference transform: input_size={input_size}")
        return transform
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Loads image, resizes to 224x224, applies normalization, and converts to tensor.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor ready for model input
            
        Raises:
            InvalidImageError: If image cannot be loaded or is invalid
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            logger.debug(f"Preprocessed image: {image_path}, shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}", exc_info=True)
            raise InvalidImageError(f"Invalid image file: {image_path}") from e
    
    def predict(
        self,
        image_path: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k classes for an image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples, sorted by probability
            
        Raises:
            RuntimeError: If model not loaded
            InvalidImageError: If image is invalid
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            tensor = self.preprocess_image(image_path)
            tensor = tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            # Convert to list of tuples
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.idx_to_class[idx.item()]
                probability = prob.item()
                predictions.append((class_name, probability))
            
            logger.info(f"Prediction for {image_path}:")
            for i, (class_name, prob) in enumerate(predictions, 1):
                logger.info(f"  {i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
            
            return predictions
            
        except InvalidImageError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
    
    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict for multiple images.
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions per image
            
        Returns:
            List of predictions for each image
        """
        predictions = []
        
        for image_path in tqdm(image_paths, desc="Predicting"):
            try:
                preds = self.predict(image_path, top_k=top_k)
                predictions.append(preds)
            except Exception as e:
                logger.warning(f"Failed to predict {image_path}: {str(e)}")
                predictions.append([])
        
        return predictions


class InvalidImageError(Exception):
    """Raised when image file is corrupted or unsupported format."""
    pass


class ModelNotFoundError(Exception):
    """Raised when saved model file is not found."""
    pass
