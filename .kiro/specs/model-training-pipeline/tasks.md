# Implementation Plan - Model Training Pipeline

- [x] 1. Set up project structure and dependencies





  - Create `backend/train_model.py` as main training script
  - Update `backend/requirements.txt` with PyTorch, torchvision, MLflow, tqdm, matplotlib, seaborn, scikit-learn
  - Create `backend/MODEL_TRAINING_GUIDE.md` with comprehensive documentation
  - Create quick start scripts: `train_quick_start.sh` and `train_quick_start.bat`
  - _Requirements: 1.1, 3.1, 9.1_

- [x] 2. Implement configuration management









  - Create argument parser with all training parameters (epochs, batch_size, learning_rate, model_name, resume, data_dir, output_dir)
  - Implement device detection (CUDA/CPU) with automatic fallback
  - Create TrainingConfig dataclass for configuration management
  - Add input validation for all parameters
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.1_

- [x] 3. Implement dataset loading and preprocessing





  - Create function to download and load Food-101 dataset using torchvision
  - Implement train/validation split (80/20)
  - Create training data augmentation pipeline (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize)
  - Create validation preprocessing pipeline (Resize, CenterCrop, Normalize)
  - Create DataLoader instances with configurable batch size and workers
  - _Requirements: 1.3, 1.4_

- [x]* 3.1 Write property test for data augmentation
  - **Property 2: Data augmentation preservation**
  - **Validates: Requirements 1.4**

- [x] 4. Implement model building and architecture


  - Create function to load pre-trained backbone models (MobileNetV2, EfficientNet-B0, ResNet50)
  - Implement layer freezing for transfer learning (freeze first 10 layers)
  - Replace classifier head with custom layer matching number of classes
  - Add Dropout(0.2) before final linear layer
  - Move model to appropriate device (CPU/GPU)
  - _Requirements: 1.1, 1.2, 3.3_

- [ ]* 4.1 Write property test for model architecture
  - **Property 1: Model architecture consistency**
  - **Validates: Requirements 1.2**

- [ ]* 4.2 Write property test for device compatibility
  - **Property 8: Device compatibility**
  - **Validates: Requirements 7.1, 7.4**

- [x] 5. Implement training loop


  - Create train_epoch function with forward pass, loss calculation, backpropagation
  - Implement gradient clipping (max norm 1.0)
  - Add progress bar with tqdm showing loss and accuracy
  - Calculate and return epoch training loss and accuracy
  - _Requirements: 8.1, 8.5, 9.3_

- [x] 6. Implement validation loop


  - Create validate_epoch function with forward pass only (no gradients)
  - Calculate validation loss and accuracy
  - Return metrics for monitoring
  - _Requirements: 1.5, 4.1_

- [x] 7. Implement optimization and scheduling


  - Set up CrossEntropyLoss as loss function
  - Initialize Adam optimizer with configurable learning rate
  - Implement ReduceLROnPlateau scheduler (patience=3, factor=0.5)
  - Implement early stopping logic (patience=5 epochs)
  - Track best validation accuracy for model saving
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ]* 7.1 Write property test for learning rate scheduling
  - **Property 9: Learning rate scheduling**
  - **Validates: Requirements 8.3**

- [ ]* 7.2 Write property test for early stopping
  - **Property 10: Early stopping trigger**
  - **Validates: Requirements 8.4**

- [x] 8. Implement MLflow integration


  - Initialize MLflow experiment with name "nutrilearn-food-training"
  - Log all hyperparameters at training start (model_name, learning_rate, batch_size, epochs)
  - Log metrics after each epoch (train_loss, train_acc, val_loss, val_acc, learning_rate)
  - Log training time and final performance metrics
  - Handle MLflow errors gracefully (continue training if logging fails)
  - _Requirements: 2.1, 2.2, 2.5, 9.1_

- [ ]* 8.1 Write property test for MLflow logging
  - **Property 3: MLflow logging completeness**
  - **Validates: Requirements 2.2**

- [x] 9. Implement checkpoint management


  - Create save_checkpoint function to save model, optimizer, epoch, and best accuracy
  - Create load_checkpoint function to resume training from saved state
  - Save checkpoints periodically and when best model is found
  - Implement checkpoint versioning
  - _Requirements: 3.2, 7.5, 9.4_

- [ ]* 9.1 Write property test for checkpoint resumption
  - **Property 4: Checkpoint resumption consistency**
  - **Validates: Requirements 3.2**

- [x] 10. Implement model evaluation


  - Calculate per-class precision, recall, and F1-score
  - Compute top-1 and top-5 accuracy
  - Generate confusion matrix using sklearn
  - Identify worst and best performing classes
  - Save evaluation results to JSON file
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 10.1 Write property test for confusion matrix
  - **Property 12: Confusion Matrix dimensions**
  - **Validates: Requirements 4.2**

- [x] 11. Implement confusion matrix visualization


  - Create function to generate confusion matrix heatmap
  - Use matplotlib/seaborn for visualization
  - Save confusion matrix as PNG image
  - Log confusion matrix to MLflow as artifact
  - _Requirements: 2.4, 4.2_

- [x] 12. Implement model artifact saving


  - Save best model weights to `ml-models/food_model_v{version}.pth`
  - Save class-to-index mappings to `ml-models/class_to_idx.json`
  - Save model configuration (architecture, preprocessing params) to `ml-models/model_config.json`
  - Implement version numbering (auto-increment)
  - Log all artifacts to MLflow
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 2.3_

- [ ]* 12.1 Write property test for artifact completeness
  - **Property 5: Model artifact completeness**
  - **Validates: Requirements 5.1, 5.2, 5.3**

- [ ]* 12.2 Write property test for class mapping bijection
  - **Property 11: Class mapping bijection**
  - **Validates: Requirements 5.2**

- [x] 13. Implement inference testing module


  - Create load_model_for_inference function to load saved model and config
  - Implement predict_image function with preprocessing pipeline
  - Return top-3 predictions with confidence probabilities
  - Add example usage in main script
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ]* 13.1 Write property test for inference preprocessing
  - **Property 6: Inference preprocessing consistency**
  - **Validates: Requirements 6.2**

- [ ]* 13.2 Write property test for top-K predictions
  - **Property 7: Top-K predictions validity**
  - **Validates: Requirements 6.3**

- [x] 14. Implement comprehensive logging and error handling


  - Set up Python logging with INFO level
  - Log training configuration at start
  - Log epoch summaries with metrics
  - Implement try-except blocks for all major operations
  - Provide clear error messages with suggested fixes
  - Log checkpoint saves and model artifact locations
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 5.5_

- [x] 15. Create Google Colab notebook


  - Create `backend/NutriLearn_Training_Colab.ipynb`
  - Add cells for dependency installation
  - Add GPU detection and configuration
  - Include training script with example parameters
  - Add visualization cells for training curves
  - Include inference testing examples
  - Add instructions for downloading trained models
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 16. Create comprehensive documentation


  - Write MODEL_TRAINING_GUIDE.md with setup instructions
  - Document all command-line arguments
  - Provide training examples for different scenarios
  - Include troubleshooting section
  - Add MLflow UI usage instructions
  - Document model deployment integration
  - _Requirements: 9.1_

- [x] 17. Create quick start scripts


  - Create `train_quick_start.sh` for Linux/Mac
  - Create `train_quick_start.bat` for Windows
  - Include dependency installation
  - Add example training commands
  - Make scripts executable
  - _Requirements: 3.1_

- [x] 18. Implement optional custom Indian food support


  - Add command-line flag for custom dataset path
  - Create function to merge custom data with Food-101
  - Update class mappings to include custom categories
  - Validate custom data format
  - Handle missing custom data gracefully
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 19. Integrate trained model with backend predictor


  - Update `backend/app/ml/predictor.py` to load trained model
  - Replace mock predictions with real model inference
  - Load class mappings and model config on startup
  - Apply correct preprocessing transformations
  - Return predictions in existing API format
  - Add error handling for model loading failures
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 20. Final checkpoint - Ensure all tests pass






  - Run all property-based tests (12 tests, 100+ iterations each)
  - Run integration tests (end-to-end training, resume, inference)
  - Test training on CPU and GPU (if available)
  - Test all three backbone models
  - Verify MLflow logging works correctly
  - Test Google Colab notebook
  - Verify all artifacts are created correctly
  - Test backend integration with trained model
  - Ensure all tests pass, ask the user if questions arise
