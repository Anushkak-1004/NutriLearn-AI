# Requirements Document

## Introduction

This document specifies the requirements for integrating the trained PyTorch food recognition model into the NutriLearn AI backend prediction system. The system must load the trained model, perform inference on uploaded food images, and return predictions with nutrition information while maintaining backward compatibility with the existing API interface.

## Glossary

- **Predictor**: The backend component responsible for loading the ML model and performing food recognition inference
- **Model Artifacts**: The trained model weights file (food_model_v1.pth), class mappings (class_to_idx.json), and configuration (model_config.json)
- **Inference**: The process of using the trained model to make predictions on new food images
- **Preprocessing Pipeline**: The sequence of image transformations applied before feeding images to the model
- **Fallback Mechanism**: The mock prediction system used when the trained model is unavailable
- **Confidence Threshold**: The minimum prediction confidence score (0.0-1.0) required to accept a prediction as valid
- **Top-K Predictions**: The K highest-confidence predictions returned by the model

## Requirements

### Requirement 1

**User Story:** As a backend developer, I want the system to load the trained PyTorch model on startup, so that predictions can be made efficiently without reloading the model for each request.

#### Acceptance Criteria

1. WHEN the application starts THEN the Predictor SHALL attempt to load the model from ml-models/food_model_v1.pth
2. WHEN the application starts THEN the Predictor SHALL load class mappings from ml-models/class_to_idx.json
3. WHEN the application starts THEN the Predictor SHALL load model configuration from ml-models/model_config.json
4. WHEN model files are successfully loaded THEN the Predictor SHALL cache the model in memory for subsequent requests
5. WHEN model files are missing or corrupted THEN the Predictor SHALL log a warning and activate the fallback mechanism without crashing

### Requirement 2

**User Story:** As a backend developer, I want images to be preprocessed correctly before inference, so that the model receives inputs in the expected format.

#### Acceptance Criteria

1. WHEN an image is received for prediction THEN the Predictor SHALL convert the image to RGB format if it is grayscale
2. WHEN preprocessing an image THEN the Predictor SHALL resize the image to 224x224 pixels
3. WHEN preprocessing an image THEN the Predictor SHALL normalize pixel values using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
4. WHEN preprocessing an image THEN the Predictor SHALL convert the image to a PyTorch tensor
5. WHEN preprocessing parameters are specified in model_config.json THEN the Predictor SHALL use those parameters instead of defaults

### Requirement 3

**User Story:** As a backend developer, I want the system to perform efficient inference on food images, so that predictions are returned quickly to users.

#### Acceptance Criteria

1. WHEN performing inference THEN the Predictor SHALL set the model to evaluation mode
2. WHEN performing inference THEN the Predictor SHALL disable gradient computation using torch.no_grad()
3. WHEN inference is complete THEN the Predictor SHALL return the top-3 predictions with their confidence scores
4. WHEN inference is complete THEN the Predictor SHALL map class indices to human-readable food names using the class mappings
5. WHEN the model outputs logits THEN the Predictor SHALL apply softmax to convert them to probabilities

### Requirement 4

**User Story:** As a backend developer, I want predicted foods to be mapped to nutrition information, so that users receive complete nutritional data with their predictions.

#### Acceptance Criteria

1. WHEN a food is predicted THEN the Predictor SHALL look up nutrition information in the nutrition database
2. WHEN a predicted food is not found in the nutrition database THEN the Predictor SHALL use generic nutrition values for the food category
3. WHEN multiple predictions are returned THEN the Predictor SHALL include nutrition information for the top prediction only
4. WHEN the prediction confidence is below 0.6 THEN the Predictor SHALL include a low-confidence warning in the response
5. WHEN nutrition data is unavailable THEN the Predictor SHALL return estimated values with an estimation flag

### Requirement 5

**User Story:** As a backend developer, I want robust error handling and fallback mechanisms, so that the system remains operational even when the model fails.

#### Acceptance Criteria

1. WHEN the model fails to load THEN the Predictor SHALL activate the mock predictor fallback
2. WHEN inference fails due to an error THEN the Predictor SHALL log the error and attempt to use the fallback mechanism
3. WHEN an invalid image is provided THEN the Predictor SHALL raise a ValueError with a descriptive error message
4. WHEN PyTorch is not installed THEN the Predictor SHALL log a warning and use mock predictions
5. WHEN any error occurs THEN the Predictor SHALL log the error with sufficient detail for debugging

### Requirement 6

**User Story:** As a backend developer, I want the application to initialize the model on startup, so that the first prediction request is not delayed by model loading.

#### Acceptance Criteria

1. WHEN the FastAPI application starts THEN the main application SHALL trigger model loading in a startup event
2. WHEN model loading completes THEN the application SHALL log the model loading status
3. WHEN the model is loaded successfully THEN the application SHALL perform a warmup inference with a dummy input
4. WHEN the warmup inference completes THEN the application SHALL log the warmup status
5. WHEN model loading fails THEN the application SHALL continue startup and log the fallback status

### Requirement 7

**User Story:** As a frontend developer, I want the API interface to remain unchanged, so that existing frontend code continues to work without modifications.

#### Acceptance Criteria

1. WHEN the model integration is complete THEN the prediction API endpoint SHALL maintain the same request format
2. WHEN the model integration is complete THEN the prediction API endpoint SHALL maintain the same response format
3. WHEN the model integration is complete THEN the FoodPrediction response model SHALL remain unchanged
4. WHEN the model integration is complete THEN the NutritionInfo response model SHALL remain unchanged
5. WHEN predictions are made THEN the response SHALL include all fields expected by the frontend (food_name, confidence, nutrition, category, cuisine, timestamp)
