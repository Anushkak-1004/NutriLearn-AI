# Implementation Plan: PyTorch Model Integration

## Overview

This implementation plan breaks down the PyTorch model integration into discrete, manageable coding tasks. Each task builds incrementally on previous steps, ensuring the system remains functional throughout development.

---

## Tasks

- [ ] 1. Create model artifact files
  - Create model_config.json with model configuration parameters
  - Create class_to_idx.json with Food-101 class mappings
  - Ensure these files are in backend/ml-models/ directory
  - _Requirements: 1.2, 1.3_

- [ ] 2. Implement FoodRecognitionModel class
  - Create new FoodRecognitionModel class in backend/app/ml/predictor.py
  - Implement __init__ method with model loading logic
  - Implement _load_model() to load PyTorch model weights
  - Implement _load_config() to parse model_config.json
  - Implement _load_class_mappings() to parse class_to_idx.json
  - Implement _build_transform() to create preprocessing pipeline
  - Add proper error handling and logging for all loading operations
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 5.1, 5.4, 5.5_

- [ ]* 2.1 Write property test for model loading idempotence
  - **Property 1: Model Loading Idempotence**
  - **Validates: Requirements 1.4**

- [ ] 3. Implement image preprocessing
  - Implement preprocess_image() method in FoodRecognitionModel
  - Add RGB conversion for grayscale images
  - Add resize to 224x224 pixels
  - Add normalization with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Add tensor conversion and batch dimension
  - Use config parameters when available, defaults otherwise
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 3.1 Write property test for preprocessing consistency
  - **Property 2: Image Preprocessing Consistency**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [ ]* 3.2 Write property test for RGB conversion idempotence
  - **Property 8: RGB Conversion Idempotence**
  - **Validates: Requirements 2.1**

- [ ] 4. Implement model inference
  - Implement predict() method in FoodRecognitionModel
  - Set model to eval mode before inference
  - Use torch.no_grad() context for inference
  - Apply softmax to convert logits to probabilities
  - Get top-K predictions with confidence scores
  - Map class indices to food names using class mappings
  - Add error handling for inference failures
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.2, 5.5_

- [ ]* 4.1 Write property test for confidence score bounds
  - **Property 3: Confidence Score Bounds**
  - **Validates: Requirements 3.5**

- [ ]* 4.2 Write property test for top-K predictions ordering
  - **Property 4: Top-K Predictions Ordering**
  - **Validates: Requirements 3.3**

- [ ] 5. Implement nutrition mapping
  - Implement map_to_nutrition() method in FoodRecognitionModel
  - Look up predicted food in MOCK_FOOD_DATABASE
  - Implement fallback to generic nutrition by category
  - Add low confidence warning when confidence < 0.6
  - Create and return FoodPrediction object with all required fields
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 5.1 Write property test for nutrition mapping completeness
  - **Property 7: Nutrition Mapping Completeness**
  - **Validates: Requirements 4.1, 4.2, 4.3**

- [ ] 6. Refactor predictor.py module
  - Create module-level singleton instance variable
  - Implement get_model() function to return singleton instance
  - Update load_model() function to create FoodRecognitionModel instance
  - Update simulate_food_recognition() to use FoodRecognitionModel
  - Maintain fallback to mock predictions when model unavailable
  - Ensure all error handling and logging is preserved
  - _Requirements: 1.4, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 6.1 Write property test for fallback activation
  - **Property 5: Fallback Activation**
  - **Validates: Requirements 5.1, 5.2**

- [ ]* 6.2 Write property test for API response format preservation
  - **Property 6: API Response Format Preservation**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

- [ ] 7. Update main.py startup event
  - Update startup_event() in backend/app/main.py
  - Add model loading with proper error handling
  - Add model loading status logging
  - Implement warmup inference with dummy image
  - Add warmup status logging
  - Ensure application continues on model loading failure
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 7.1 Write property test for warmup inference success
  - **Property 10: Warmup Inference Success**
  - **Validates: Requirements 6.3, 6.4**

- [ ]* 7.2 Write property test for error logging completeness
  - **Property 9: Error Logging Completeness**
  - **Validates: Requirements 5.5**

- [ ]* 8. Write unit tests for model loading
  - Test successful model loading with valid files
  - Test graceful failure when model files are missing
  - Test configuration parsing with various formats
  - Test class mapping loading and validation
  - Test PyTorch not available scenario
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 5.4_

- [ ]* 9. Write unit tests for preprocessing
  - Test RGB conversion for grayscale images
  - Test image resizing to 224x224
  - Test normalization with correct mean/std values
  - Test tensor shape validation
  - Test preprocessing with custom config parameters
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 10. Write unit tests for inference
  - Test prediction with valid images
  - Test top-K predictions ordering
  - Test confidence score ranges
  - Test error handling for invalid inputs
  - Test model eval mode activation
  - Test gradient disabling
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.2_

- [ ]* 11. Write unit tests for nutrition mapping
  - Test exact matches in nutrition database
  - Test fallback to generic nutrition
  - Test category-based generic nutrition
  - Test low confidence warning flag
  - Test missing nutrition handling
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 12. Write integration tests
  - Test end-to-end prediction flow with real model
  - Test end-to-end prediction flow with mock fallback
  - Test API endpoint with model available
  - Test API endpoint with model unavailable
  - Test startup event model loading
  - Test warmup inference during startup
  - _Requirements: 1.1, 5.1, 6.1, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Update documentation
  - Add docstrings to all new methods and classes
  - Update API documentation if response format changed
  - Add comments explaining complex logic
  - Document model artifact file formats
  - _Requirements: All_

- [ ] 15. Final validation and cleanup
  - Verify backward compatibility with existing API
  - Test with actual trained model file (if available)
  - Verify all logging statements are appropriate
  - Remove any debug print statements
  - Ensure code follows project style guide
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

---

## Notes

- Tasks marked with `*` are optional testing tasks that can be skipped for faster MVP
- Each task should be completed and tested before moving to the next
- The implementation maintains backward compatibility throughout
- Model artifact files (model_config.json, class_to_idx.json) must be created before testing with real model
- If food_model_v1.pth is not available, the system will gracefully fall back to mock predictions
