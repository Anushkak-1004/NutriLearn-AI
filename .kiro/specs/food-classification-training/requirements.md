# Requirements Document

## Introduction

This document specifies the requirements for a PyTorch-based food classification model training system for NutriLearn AI. The system will train a deep learning model to recognize food items from images, with integrated MLOps practices for experiment tracking, model versioning, and performance monitoring. The training pipeline must be production-ready, reproducible, and suitable for demonstration in job interviews.

## Glossary

- **Training System**: The complete PyTorch-based pipeline that handles data loading, model training, evaluation, and artifact saving
- **Food-101 Dataset**: A publicly available dataset containing 101 food categories with 1000 images each
- **Transfer Learning**: The technique of using a pre-trained model and fine-tuning it for a specific task
- **MLflow**: An open-source platform for managing the ML lifecycle, including experiment tracking and model versioning
- **Model Artifact**: The saved model file along with associated metadata (class mappings, configuration, preprocessing parameters)
- **Checkpoint**: A saved state of the model during training that allows resuming training from that point
- **Confusion Matrix**: A table showing the performance of a classification model across all classes
- **Top-K Accuracy**: The percentage of predictions where the correct class appears in the top K predictions

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to load and preprocess the Food-101 dataset with appropriate augmentation, so that the model can learn robust features from diverse food images.

#### Acceptance Criteria

1. WHEN the Training System initializes the dataset, THEN the Training System SHALL load the Food-101 dataset using torchvision.datasets.Food101
2. WHEN the Training System splits the dataset, THEN the Training System SHALL create an 80/20 train/validation split with stratified sampling
3. WHEN the Training System applies data augmentation to training images, THEN the Training System SHALL apply RandomResizedCrop, RandomHorizontalFlip, ColorJitter, and Normalize transformations
4. WHEN the Training System processes validation images, THEN the Training System SHALL apply only Resize, CenterCrop, and Normalize transformations
5. WHEN the Training System creates data loaders, THEN the Training System SHALL use configurable batch size with num_workers for parallel loading

### Requirement 2

**User Story:** As a data scientist, I want to use transfer learning with a pre-trained model architecture, so that I can achieve high accuracy with limited training time and computational resources.

#### Acceptance Criteria

1. WHEN the Training System initializes the model, THEN the Training System SHALL load a pre-trained MobileNetV2 or EfficientNet-B0 model from torchvision
2. WHEN the Training System configures the model for transfer learning, THEN the Training System SHALL freeze the first 10 feature layers to preserve pre-trained weights
3. WHEN the Training System replaces the classifier head, THEN the Training System SHALL create a new classifier with Dropout and Linear layers matching the number of food classes
4. WHEN the Training System supports multiple architectures, THEN the Training System SHALL accept model_name as a parameter to switch between MobileNetV2 and EfficientNet-B0

### Requirement 3

**User Story:** As a data scientist, I want to train the model with appropriate loss functions and optimizers, so that the model converges efficiently to high accuracy.

#### Acceptance Criteria

1. WHEN the Training System configures the loss function, THEN the Training System SHALL use CrossEntropyLoss for multi-class classification
2. WHEN the Training System configures the optimizer, THEN the Training System SHALL use Adam optimizer with configurable learning rate
3. WHEN the Training System implements learning rate scheduling, THEN the Training System SHALL use ReduceLROnPlateau with patience of 3 epochs
4. WHEN the Training System implements early stopping, THEN the Training System SHALL stop training if validation loss does not improve for 5 consecutive epochs
5. WHEN the Training System runs training epochs, THEN the Training System SHALL train for a configurable number of epochs with default of 15-20

### Requirement 4

**User Story:** As an ML engineer, I want to track all experiments with MLflow, so that I can compare different training runs and reproduce successful experiments.

#### Acceptance Criteria

1. WHEN the Training System starts a training run, THEN the Training System SHALL log hyperparameters including learning_rate, batch_size, model_architecture, epochs, and optimizer_name to MLflow
2. WHEN the Training System completes each epoch, THEN the Training System SHALL log metrics including train_loss, train_accuracy, val_loss, and val_accuracy to MLflow
3. WHEN the Training System completes training, THEN the Training System SHALL save the best model as an MLflow artifact with version identifier
4. WHEN the Training System generates evaluation results, THEN the Training System SHALL log the confusion matrix as an image artifact to MLflow
5. WHEN the Training System finishes training, THEN the Training System SHALL log total training time as a metric to MLflow

### Requirement 5

**User Story:** As an ML engineer, I want to save trained models with all necessary artifacts, so that I can deploy the model for inference without missing dependencies.

#### Acceptance Criteria

1. WHEN the Training System saves the best model, THEN the Training System SHALL save the model state_dict to ml-models/food_model_v{version}.pth
2. WHEN the Training System saves class mappings, THEN the Training System SHALL save the class_to_idx dictionary as JSON to ml-models/class_to_idx.json
3. WHEN the Training System saves model configuration, THEN the Training System SHALL save model architecture, input size, and normalization parameters as JSON to ml-models/model_config.json
4. WHEN the Training System creates a model checkpoint, THEN the Training System SHALL include epoch number, optimizer state, and scheduler state for resuming training
5. WHEN the Training System organizes model artifacts, THEN the Training System SHALL create a timestamped directory containing all related files

### Requirement 6

**User Story:** As a data scientist, I want comprehensive evaluation metrics for the trained model, so that I can understand model performance across different food categories.

#### Acceptance Criteria

1. WHEN the Training System evaluates the model, THEN the Training System SHALL calculate accuracy, precision, recall, and F1-score for each class
2. WHEN the Training System generates a confusion matrix, THEN the Training System SHALL create a normalized confusion matrix showing prediction patterns across all classes
3. WHEN the Training System calculates top-K accuracy, THEN the Training System SHALL compute top-1, top-3, and top-5 accuracy metrics
4. WHEN the Training System identifies problematic classes, THEN the Training System SHALL report the 10 worst-performing classes by F1-score
5. WHEN the Training System saves evaluation results, THEN the Training System SHALL save all metrics as JSON to ml-models/evaluation_results.json

### Requirement 7

**User Story:** As a developer, I want to test model inference on single images, so that I can verify the model works correctly before integration into the API.

#### Acceptance Criteria

1. WHEN the Training System provides an inference function, THEN the Training System SHALL load the saved model and preprocessing configuration
2. WHEN the Training System preprocesses an input image, THEN the Training System SHALL resize to 224x224, apply normalization, and convert to tensor
3. WHEN the Training System makes a prediction, THEN the Training System SHALL return the top-3 predicted classes with their probability scores
4. WHEN the Training System handles inference errors, THEN the Training System SHALL raise descriptive exceptions for invalid images or missing model files

### Requirement 8

**User Story:** As a developer, I want configurable training parameters via command-line arguments, so that I can easily experiment with different hyperparameters without modifying code.

#### Acceptance Criteria

1. WHEN the Training System accepts command-line arguments, THEN the Training System SHALL support parameters for epochs, batch_size, learning_rate, and model_name
2. WHEN the Training System supports checkpoint resuming, THEN the Training System SHALL accept a checkpoint_path argument to resume training from a saved state
3. WHEN the Training System provides default values, THEN the Training System SHALL use sensible defaults for all parameters when not specified
4. WHEN the Training System validates arguments, THEN the Training System SHALL check parameter ranges and raise errors for invalid values

### Requirement 9

**User Story:** As a user, I want clear progress indication and logging during training, so that I can monitor training progress and diagnose issues.

#### Acceptance Criteria

1. WHEN the Training System trains the model, THEN the Training System SHALL display a progress bar using tqdm for each epoch showing batch progress
2. WHEN the Training System logs information, THEN the Training System SHALL use Python logging module with INFO level for important events
3. WHEN the Training System encounters errors, THEN the Training System SHALL log ERROR level messages with full stack traces
4. WHEN the Training System completes each epoch, THEN the Training System SHALL print a summary showing train/val loss and accuracy
5. WHEN the Training System saves checkpoints, THEN the Training System SHALL log the save location and model performance metrics

### Requirement 10

**User Story:** As a developer, I want the training script to be compatible with Google Colab, so that I can leverage free GPU resources for faster training.

#### Acceptance Criteria

1. WHEN the Training System detects available hardware, THEN the Training System SHALL automatically use CUDA GPU if available, otherwise CPU
2. WHEN the Training System runs in Google Colab, THEN the Training System SHALL support mounting Google Drive for dataset and model storage
3. WHEN the Training System handles dependencies, THEN the Training System SHALL include all required packages in requirements.txt with version specifications
4. WHEN the Training System manages memory, THEN the Training System SHALL clear GPU cache between epochs to prevent out-of-memory errors
