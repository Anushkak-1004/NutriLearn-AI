# Implementation Plan - Food Classification Training System

- [x] 1. Set up project structure and dependencies



  - Create backend/train_model.py as main training script
  - Add required packages to backend/requirements.txt: torch, torchvision, mlflow, hypothesis, pytest, tqdm, matplotlib, seaborn, scikit-learn
  - Create backend/tests/ directory for test files
  - Set up logging configuration with Python logging module
  - _Requirements: 9.2, 10.3_

- [x] 2. Implement DatasetManager component



  - Create DatasetManager class to handle Food-101 dataset loading
  - Implement get_transforms() method with training augmentation (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize)
  - Implement get_transforms() method with validation transforms (Resize, CenterCrop, Normalize)
  - Implement prepare_data() method to create 80/20 train/val split with stratified sampling
  - Create data loaders with configurable batch_size and num_workers
  - Return idx_to_class mapping for class names
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ]* 2.1 Write property test for train/validation split ratio
  - **Property 1: Train/Validation Split Ratio Consistency**
  - **Validates: Requirements 1.2**

- [x] 3. Implement ModelBuilder component



  - Create ModelBuilder class to build transfer learning models
  - Implement support for loading pre-trained MobileNetV2 from torchvision
  - Implement support for loading pre-trained EfficientNet-B0 from torchvision
  - Implement freeze_backbone() method to freeze first 10 feature layers
  - Implement replace_classifier() method with Dropout(0.2) and Linear layer for num_classes
  - Implement get_model_config() method to return model metadata dictionary
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 3.1 Write property test for frozen layers immutability
  - **Property 2: Frozen Layers Immutability**
  - **Validates: Requirements 2.2**

- [ ]* 3.2 Write property test for classifier output dimension
  - **Property 3: Classifier Output Dimension Correctness**
  - **Validates: Requirements 2.3**

- [x] 4. Implement TrainingConfig dataclass



  - Create TrainingConfig dataclass with all hyperparameters
  - Include data parameters: data_dir, batch_size, num_workers
  - Include model parameters: model_name, num_classes, freeze_layers, dropout_rate
  - Include training parameters: num_epochs, learning_rate, weight_decay, early_stopping_patience
  - Include scheduler parameters: scheduler_patience, scheduler_factor
  - Include checkpoint and MLflow parameters
  - Add device detection (CUDA/CPU) and random seed
  - _Requirements: 8.1, 8.3, 10.1_

- [x] 5. Implement TrainingEngine component



  - Create TrainingEngine class to handle training and validation loops
  - Implement train_epoch() method with forward pass, loss calculation, backward pass
  - Add tqdm progress bar for batch iteration within epochs
  - Implement validate() method to compute validation loss and accuracy
  - Implement train() method with main training loop
  - Add early stopping logic with patience of 5 epochs
  - Implement save_checkpoint() method with epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 9.1, 9.4_

- [ ]* 5.1 Write property test for optimizer learning rate configuration
  - **Property 4: Optimizer Learning Rate Configuration**
  - **Validates: Requirements 3.2**

- [ ]* 5.2 Write property test for early stopping trigger
  - **Property 5: Early Stopping Trigger Condition**
  - **Validates: Requirements 3.4**

- [ ]* 5.3 Write property test for epoch count execution
  - **Property 6: Epoch Count Execution**
  - **Validates: Requirements 3.5**

- [x] 6. Implement MLflowTracker component



  - Create MLflowTracker class for experiment tracking
  - Implement log_params() method to log hyperparameters at training start
  - Implement log_metrics() method to log train_loss, train_accuracy, val_loss, val_accuracy per epoch
  - Implement log_model() method to save best model as MLflow artifact
  - Implement log_figure() method to save confusion matrix and training curves
  - Add log_training_time() to track total training duration
  - Handle MLflowConnectionError gracefully with local fallback
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 6.1 Write property test for per-epoch metric logging completeness
  - **Property 7: Per-Epoch Metric Logging Completeness**
  - **Validates: Requirements 4.2**

- [x] 7. Implement model artifact saving functionality



  - Create function to save model state_dict to ml-models/food_model_v{version}.pth
  - Create function to save class_to_idx dictionary as JSON to ml-models/class_to_idx.json
  - Create function to save model_config as JSON to ml-models/model_config.json with architecture, input_size, normalization params
  - Implement timestamped directory creation for organizing artifacts
  - Add checkpoint saving with all required keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_val_loss, best_val_acc
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 7.1 Write property test for class mapping serialization round-trip
  - **Property 8: Class Mapping Serialization Round-Trip**
  - **Validates: Requirements 5.2**

- [ ]* 7.2 Write property test for checkpoint completeness
  - **Property 9: Checkpoint Completeness**
  - **Validates: Requirements 5.4**

- [x] 8. Implement ModelEvaluator component



  - Create ModelEvaluator class for comprehensive evaluation
  - Implement evaluate() method to run full evaluation suite
  - Implement compute_confusion_matrix() method to generate normalized confusion matrix
  - Implement get_per_class_metrics() method to calculate accuracy, precision, recall, F1-score for each class
  - Implement get_top_k_accuracy() method for top-1, top-3, and top-5 accuracy
  - Implement identify_worst_classes() method to find 10 worst-performing classes by F1-score
  - Save evaluation results as JSON to ml-models/evaluation_results.json
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 8.1 Write property test for evaluation metric completeness
  - **Property 10: Evaluation Metric Completeness**
  - **Validates: Requirements 6.1**

- [ ]* 8.2 Write property test for confusion matrix dimensionality
  - **Property 11: Confusion Matrix Dimensionality**
  - **Validates: Requirements 6.2**

- [x] 9. Implement InferenceHelper component





  - Create InferenceHelper class for single image inference
  - Implement load_model() method to load saved model, config, and class mappings
  - Implement preprocess_image() method to resize to 224x224, normalize, and convert to tensor
  - Implement predict() method to return top-3 predictions with probability scores
  - Add error handling for InvalidImageError and ModelNotFoundError
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ]* 9.1 Write property test for image preprocessing output shape
  - **Property 12: Image Preprocessing Output Shape**
  - **Validates: Requirements 7.2**

- [ ]* 9.2 Write property test for top-K prediction probability sum
  - **Property 13: Top-K Prediction Probability Sum**
  - **Validates: Requirements 7.3**

- [x] 10. Implement command-line argument parsing





  - Add argparse configuration for all training parameters
  - Support arguments: --epochs, --batch-size, --learning-rate, --model-name
  - Add --checkpoint-path argument for resuming training
  - Add --data-dir argument for dataset location
  - Implement argument validation with range checks
  - Set sensible defaults for all parameters
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 11. Implement main training script orchestration





  - Create main() function to orchestrate all components
  - Parse command-line arguments and create TrainingConfig
  - Set random seeds for reproducibility (torch, numpy, random)
  - Initialize MLflow experiment and start run
  - Initialize DatasetManager and prepare data loaders
  - Initialize ModelBuilder and build model
  - Initialize optimizer (Adam) and scheduler (ReduceLROnPlateau)
  - Initialize TrainingEngine and execute training
  - Run ModelEvaluator after training completes
  - Save all artifacts and log to MLflow
  - Print final summary with best metrics
  - _Requirements: 3.1, 3.2, 3.3, 9.4, 9.5_

- [x] 12. Add comprehensive error handling


  - Implement DatasetNotFoundError with download instructions
  - Implement UnsupportedModelError with list of valid models
  - Implement OutOfMemoryError with automatic batch size reduction
  - Implement NaNLossError with emergency checkpoint saving
  - Implement CheckpointLoadError for corrupted checkpoints
  - Add try-except blocks around all major operations
  - Log all errors with full stack traces using logging.error()
  - _Requirements: 9.3_

- [x] 13. Add Google Colab compatibility features


  - Add automatic device detection (CUDA/CPU) with GPU name printing
  - Add Google Drive mounting support for dataset and model storage
  - Configure tqdm with notebook=True for Colab progress bars
  - Add GPU cache clearing between epochs to prevent OOM
  - Include installation commands for dependencies in docstring
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 14. Implement checkpoint resume functionality


  - Add load_checkpoint() function to restore training state
  - Restore model_state_dict, optimizer_state_dict, scheduler_state_dict
  - Restore epoch number and training history
  - Validate checkpoint compatibility with current config
  - Continue training from restored epoch
  - _Requirements: 8.2_

- [x] 15. Add visualization and reporting


  - Create function to plot training curves (loss and accuracy over epochs)
  - Create function to visualize confusion matrix with seaborn heatmap
  - Create function to display sample predictions with ground truth
  - Save all plots as PNG files in artifacts directory
  - Log plots to MLflow as artifacts
  - _Requirements: 4.4, 6.2_

- [x] 16. Create comprehensive documentation

  - Add module-level docstring with overview, usage examples, and requirements
  - Add Google-style docstrings to all classes and functions
  - Include type hints for all function parameters and returns
  - Add inline comments for complex logic sections
  - Create example usage section showing training and inference
  - Document all command-line arguments with examples
  - _Requirements: 9.2_

- [ ]* 17. Write unit tests for all components
  - Write unit tests for DatasetManager (transforms, data loaders, class mapping)
  - Write unit tests for ModelBuilder (model creation, layer freezing, classifier replacement)
  - Write unit tests for TrainingEngine (train_epoch, validate, early stopping)
  - Write unit tests for MLflowTracker (parameter logging, metric logging, artifact saving)
  - Write unit tests for ModelEvaluator (metrics calculation, confusion matrix, top-k accuracy)
  - Write unit tests for InferenceHelper (model loading, preprocessing, prediction)
  - Use pytest fixtures for mock data and models
  - Aim for 80%+ code coverage

- [ ]* 18. Write integration tests
  - Write end-to-end test with small dataset (10 classes, 100 images)
  - Test checkpoint save and resume functionality
  - Test MLflow logging end-to-end
  - Test inference after training completes
  - Verify all artifacts are created correctly

- [x] 19. Final checkpoint - Ensure all tests pass



  - Run all unit tests with pytest
  - Run all property-based tests with Hypothesis
  - Run integration tests
  - Verify code coverage meets 80% threshold
  - Fix any failing tests
  - Ensure all tests pass, ask the user if questions arise.