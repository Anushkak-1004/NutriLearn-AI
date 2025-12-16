# Design Document: PyTorch Model Integration

## Overview

This design document outlines the technical approach for integrating the trained PyTorch food recognition model into the NutriLearn AI backend. The integration will replace the current mock prediction system with real ML inference while maintaining backward compatibility with the existing API interface. The design emphasizes robustness, performance, and graceful degradation when model files are unavailable.

The system will load a pre-trained MobileNetV2 model (or similar architecture) trained on food images, perform efficient inference on uploaded images, and map predictions to nutrition information. The integration includes proper error handling, fallback mechanisms, and startup optimization through model caching and warmup.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Startup Event Handler                      │ │
│  │  - Load model artifacts                                 │ │
│  │  - Initialize FoodRecognitionModel                      │ │
│  │  - Perform warmup inference                             │ │
│  │  - Log initialization status                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           API Routes (/api/v1/predict)                  │ │
│  │  - Receive image upload                                 │ │
│  │  - Call predictor.simulate_food_recognition()           │ │
│  │  - Return FoodPrediction response                       │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FoodRecognitionModel (Singleton)                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Model        │  │ Config       │  │ Class Mappings   │  │
│  │ (PyTorch)    │  │ (Dict)       │  │ (Dict)           │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Core Methods                               │ │
│  │  - preprocess_image()                                   │ │
│  │  - predict()                                            │ │
│  │  - map_to_nutrition()                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Artifacts (Disk)                     │
│                                                              │
│  - ml-models/food_model_v1.pth      (Model weights)         │
│  - ml-models/model_config.json      (Configuration)         │
│  - ml-models/class_to_idx.json      (Label mappings)        │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Application Startup**: FastAPI startup event triggers model loading
2. **Model Initialization**: FoodRecognitionModel loads artifacts and caches in memory
3. **Warmup**: Dummy inference performed to optimize first real prediction
4. **Request Handling**: API routes receive image uploads and call predictor
5. **Inference**: Model performs prediction and returns top-K results
6. **Nutrition Mapping**: Predicted food mapped to nutrition database
7. **Response**: FoodPrediction object returned to client

### Fallback Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Request                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ PyTorch Available?    │
              └───────────┬───────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
           YES                         NO
            │                           │
            ▼                           ▼
  ┌──────────────────┐        ┌──────────────────┐
  │ Model Files      │        │ Use Mock         │
  │ Exist?           │        │ Predictor        │
  └────────┬─────────┘        └──────────────────┘
           │
    ┌──────┴──────┐
   YES            NO
    │              │
    ▼              ▼
┌─────────┐  ┌─────────┐
│ Real    │  │ Mock    │
│ Model   │  │ Fallback│
└─────────┘  └─────────┘
```

## Components and Interfaces

### 1. FoodRecognitionModel Class

The core class responsible for model management and inference.

```python
class FoodRecognitionModel:
    """
    Singleton class for food recognition using PyTorch model.
    
    Attributes:
        device: torch.device for computation (CPU or CUDA)
        model: Loaded PyTorch model
        config: Model configuration dictionary
        class_to_idx: Mapping from class names to indices
        idx_to_class: Mapping from indices to class names
        transform: Image preprocessing pipeline
        confidence_threshold: Minimum confidence for valid predictions
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        class_mapping_path: str,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the food recognition model.
        
        Args:
            model_path: Path to model weights (.pth file)
            config_path: Path to model configuration (.json)
            class_mapping_path: Path to class mappings (.json)
            confidence_threshold: Minimum confidence threshold
            
        Raises:
            FileNotFoundError: If model files don't exist
            RuntimeError: If model loading fails
        """
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load PyTorch model from disk."""
        
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration."""
        
    def _load_class_mappings(self, class_mapping_path: str) -> Dict:
        """Load class to index mappings."""
        
    def _build_transform(self) -> transforms.Compose:
        """Build image preprocessing pipeline."""
        
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor ready for inference
        """
        
    def predict(
        self,
        image: Image.Image,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Perform inference on image.
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of (food_name, confidence) tuples
            
        Raises:
            ValueError: If image is invalid
            RuntimeError: If inference fails
        """
        
    def map_to_nutrition(
        self,
        food_name: str,
        confidence: float
    ) -> FoodPrediction:
        """
        Map predicted food to nutrition information.
        
        Args:
            food_name: Predicted food name
            confidence: Prediction confidence
            
        Returns:
            FoodPrediction object with nutrition data
        """
```

### 2. Updated predictor.py Module

The module will be refactored to use the FoodRecognitionModel class:

```python
# Module-level singleton instance
_model_instance: Optional[FoodRecognitionModel] = None

def get_model() -> Optional[FoodRecognitionModel]:
    """Get the singleton model instance."""
    
def load_model(
    model_path: str = "./ml-models/food_model_v1.pth",
    config_path: str = "./ml-models/model_config.json",
    class_mapping_path: str = "./ml-models/class_to_idx.json"
) -> Optional[FoodRecognitionModel]:
    """
    Load and cache the food recognition model.
    
    Returns:
        FoodRecognitionModel instance or None if loading fails
    """
    
def simulate_food_recognition(
    image: Image.Image,
    use_real_model: bool = True
) -> FoodPrediction:
    """
    Main prediction function (maintains existing interface).
    
    Args:
        image: PIL Image object
        use_real_model: Whether to use real model (if available)
        
    Returns:
        FoodPrediction object
    """
```

### 3. Updated main.py Startup Event

```python
@app.on_event("startup")
async def startup_event():
    """Initialize services including ML model."""
    
    # Load ML model
    logger.info("Loading ML model...")
    try:
        model = load_model()
        if model:
            logger.info("✓ PyTorch model loaded successfully")
            
            # Warmup inference
            logger.info("Performing model warmup...")
            dummy_image = Image.new('RGB', (224, 224), color='white')
            _ = simulate_food_recognition(dummy_image)
            logger.info("✓ Model warmup complete")
        else:
            logger.info("✓ Using mock predictions (model files not found)")
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {str(e)}")
        logger.info("✓ Falling back to mock predictions")
```

## Data Models

### Model Configuration (model_config.json)

```json
{
  "model_name": "mobilenet_v2",
  "num_classes": 101,
  "input_size": 224,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "version": "1.0",
  "training_date": "2025-12-17",
  "framework": "pytorch",
  "framework_version": "2.0.0"
}
```

### Class Mappings (class_to_idx.json)

```json
{
  "apple_pie": 0,
  "baby_back_ribs": 1,
  "baklava": 2,
  "beef_carpaccio": 3,
  "chicken_biryani": 4,
  "masala_dosa": 5,
  ...
  "waffles": 100
}
```

### Nutrition Database Structure

```python
NUTRITION_DATABASE: Dict[str, Dict] = {
    "chicken_biryani": {
        "name": "Chicken Biryani",
        "category": "main_course",
        "cuisine": "Indian",
        "nutrition": {
            "calories": 450,
            "protein": 25.0,
            "carbs": 55.0,
            "fat": 12.0,
            "fiber": 3.0
        }
    },
    # ... more entries
}

# Generic fallback nutrition by category
GENERIC_NUTRITION: Dict[str, Dict] = {
    "main_course": {"calories": 400, "protein": 20.0, "carbs": 50.0, "fat": 15.0, "fiber": 4.0},
    "snack": {"calories": 200, "protein": 5.0, "carbs": 25.0, "fat": 8.0, "fiber": 2.0},
    "dessert": {"calories": 350, "protein": 5.0, "carbs": 50.0, "fat": 15.0, "fiber": 1.0},
    "unknown": {"calories": 300, "protein": 10.0, "carbs": 40.0, "fat": 10.0, "fiber": 3.0}
}
```

### Prediction Response Structure

The existing FoodPrediction model remains unchanged:

```python
class FoodPrediction(BaseModel):
    food_name: str
    confidence: float  # 0.0 to 1.0
    nutrition: NutritionInfo
    category: str
    cuisine: str
    timestamp: datetime
    
    # Optional fields for enhanced response
    low_confidence_warning: Optional[bool] = None
    top_3_predictions: Optional[List[Dict[str, float]]] = None
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Model Loading Idempotence

*For any* application startup sequence, calling load_model() multiple times should return the same cached model instance without reloading from disk.

**Validates: Requirements 1.4**

### Property 2: Image Preprocessing Consistency

*For any* valid PIL Image, preprocessing should always produce a tensor of shape (1, 3, 224, 224) with values normalized to the expected range.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

### Property 3: Confidence Score Bounds

*For any* model prediction, the confidence score should always be between 0.0 and 1.0 inclusive.

**Validates: Requirements 3.5**

### Property 4: Top-K Predictions Ordering

*For any* prediction with top_k > 1, the returned predictions should be ordered by confidence in descending order.

**Validates: Requirements 3.3**

### Property 5: Fallback Activation

*For any* prediction request when model files are missing, the system should use mock predictions without raising exceptions.

**Validates: Requirements 5.1, 5.2**

### Property 6: API Response Format Preservation

*For any* prediction (real or mock), the response should conform to the FoodPrediction model schema with all required fields present.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### Property 7: Nutrition Mapping Completeness

*For any* predicted food name, the system should return either exact nutrition data from the database or generic nutrition data for the category, never null or missing nutrition information.

**Validates: Requirements 4.1, 4.2, 4.3**

### Property 8: RGB Conversion Idempotence

*For any* image already in RGB format, converting to RGB should return an equivalent image without modification.

**Validates: Requirements 2.1**

### Property 9: Error Logging Completeness

*For any* error that occurs during model loading or inference, the system should log the error with sufficient detail before falling back or raising an exception.

**Validates: Requirements 5.5**

### Property 10: Warmup Inference Success

*For any* successfully loaded model, performing warmup inference with a dummy image should complete without errors.

**Validates: Requirements 6.3, 6.4**

## Error Handling

### Error Categories and Responses

| Error Type | Cause | Response | HTTP Status |
|------------|-------|----------|-------------|
| ModelNotFoundError | Model files missing | Use mock predictor, log warning | 200 (graceful) |
| InvalidImageError | Invalid image format | Raise ValueError | 400 |
| InferenceError | Model inference fails | Try fallback, log error | 500 or 200 (fallback) |
| ConfigurationError | Invalid config file | Use defaults, log warning | 200 (graceful) |
| OutOfMemoryError | Large image/model | Resize image, retry | 500 or 200 (retry) |

### Error Handling Strategy

```python
def simulate_food_recognition(image: Image.Image) -> FoodPrediction:
    try:
        # Validate image
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")
        
        # Try real model
        model = get_model()
        if model:
            try:
                return model.predict(image)
            except Exception as e:
                logger.error(f"Model inference failed: {str(e)}")
                logger.info("Falling back to mock predictions")
        
        # Fallback to mock
        return _mock_prediction(image)
        
    except ValueError as e:
        # Re-raise validation errors
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to process image: {str(e)}")
```

### Logging Strategy

```python
# Model loading
logger.info("Loading model from {path}")
logger.info("✓ Model loaded successfully")
logger.warning("Model file not found, using mock predictions")
logger.error("Failed to load model: {error}")

# Inference
logger.info("Prediction: {food_name} (confidence: {conf:.2%})")
logger.warning("Low confidence prediction: {conf:.2%}")
logger.error("Inference failed: {error}")

# Fallback
logger.info("Using mock predictions")
logger.warning("Falling back to mock predictions due to: {reason}")
```

## Testing Strategy

### Unit Tests

Unit tests will verify specific functionality and edge cases:

1. **Model Loading Tests**
   - Test successful model loading with valid files
   - Test graceful failure when files are missing
   - Test configuration parsing with various formats
   - Test class mapping loading and validation

2. **Preprocessing Tests**
   - Test RGB conversion for grayscale images
   - Test image resizing to 224x224
   - Test normalization with correct mean/std
   - Test tensor shape validation

3. **Inference Tests**
   - Test prediction with valid images
   - Test top-K predictions ordering
   - Test confidence score ranges
   - Test error handling for invalid inputs

4. **Nutrition Mapping Tests**
   - Test exact matches in nutrition database
   - Test fallback to generic nutrition
   - Test category-based generic nutrition
   - Test missing nutrition handling

5. **Integration Tests**
   - Test end-to-end prediction flow
   - Test API endpoint with real/mock models
   - Test startup event model loading
   - Test warmup inference

### Property-Based Tests

Property-based tests will verify universal properties across many inputs using the Hypothesis library:

1. **Property Test: Image Preprocessing Shape**
   - Generate random PIL images of various sizes
   - Verify all preprocessed tensors have shape (1, 3, 224, 224)
   - **Validates: Requirements 2.2, 2.4**

2. **Property Test: Confidence Bounds**
   - Generate random images
   - Verify all confidence scores are in [0.0, 1.0]
   - **Validates: Requirements 3.5**

3. **Property Test: Top-K Ordering**
   - Generate random images
   - Verify top-K predictions are sorted by confidence (descending)
   - **Validates: Requirements 3.3**

4. **Property Test: RGB Conversion Idempotence**
   - Generate random RGB images
   - Verify converting to RGB doesn't change the image
   - **Validates: Requirements 2.1**

5. **Property Test: Nutrition Completeness**
   - Generate random food names (from model classes)
   - Verify all predictions have complete nutrition data
   - **Validates: Requirements 4.1, 4.2, 4.3**

6. **Property Test: API Response Schema**
   - Generate random images
   - Verify all responses conform to FoodPrediction schema
   - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### Testing Configuration

- **Framework**: pytest for unit tests, Hypothesis for property-based tests
- **Coverage Target**: 85%+ code coverage
- **Property Test Iterations**: Minimum 100 iterations per property
- **Mock Strategy**: Use real model when available, mock PyTorch when not installed

### Test File Structure

```
backend/tests/
├── test_model_loading.py          # Unit tests for model loading
├── test_preprocessing.py          # Unit tests for image preprocessing
├── test_inference.py              # Unit tests for inference
├── test_nutrition_mapping.py      # Unit tests for nutrition mapping
├── test_predictor_integration.py  # Integration tests
└── test_predictor_properties.py   # Property-based tests
```

## Performance Considerations

### Model Loading Optimization

- **Lazy Loading**: Model loaded once on startup, not per request
- **Caching**: Model instance cached in memory as module-level singleton
- **Warmup**: Dummy inference performed during startup to optimize first real prediction

### Inference Optimization

- **Batch Size**: Single image inference (batch_size=1) for API requests
- **Device Selection**: Automatic CPU/CUDA selection based on availability
- **Gradient Disabling**: Use `torch.no_grad()` to reduce memory usage
- **Model Mode**: Set model to `eval()` mode to disable dropout/batch norm training behavior

### Memory Management

- **Image Resizing**: Resize large images before preprocessing to reduce memory
- **Tensor Cleanup**: Explicitly delete tensors after inference if memory constrained
- **Model Precision**: Consider using FP16 (half precision) for faster inference on supported hardware

### Expected Performance Metrics

- **Model Loading Time**: < 5 seconds on CPU
- **Warmup Inference Time**: < 2 seconds on CPU
- **Prediction Time**: < 1 second per image on CPU, < 0.3 seconds on GPU
- **Memory Usage**: ~500MB for MobileNetV2 model

## Deployment Considerations

### Environment Variables

```bash
# Model configuration
MODEL_PATH=./ml-models/food_model_v1.pth
MODEL_CONFIG_PATH=./ml-models/model_config.json
CLASS_MAPPING_PATH=./ml-models/class_to_idx.json
CONFIDENCE_THRESHOLD=0.6

# Device selection
TORCH_DEVICE=cpu  # or 'cuda' for GPU
```

### Docker Considerations

- Include PyTorch in requirements.txt
- Copy model files into Docker image or mount as volume
- Set appropriate memory limits for container
- Consider multi-stage builds to reduce image size

### Monitoring and Logging

- Log all predictions with confidence scores
- Track prediction latency metrics
- Monitor model loading failures
- Alert on high fallback usage rates

## Migration Path

### Phase 1: Model Integration (This Spec)

1. Implement FoodRecognitionModel class
2. Update predictor.py to use new class
3. Update main.py startup event
4. Add comprehensive tests
5. Maintain backward compatibility

### Phase 2: Model Artifacts Creation (Future)

1. Create model_config.json file
2. Create class_to_idx.json file
3. Ensure food_model_v1.pth is available
4. Expand nutrition database to cover all Food-101 classes

### Phase 3: Enhanced Features (Future)

1. Add top-3 predictions to API response
2. Implement confidence-based warnings
3. Add prediction caching
4. Implement A/B testing for model versions
5. Add MLflow integration for prediction tracking

## Dependencies

### Required Python Packages

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
```

### Optional Dependencies

```
hypothesis>=6.0.0  # For property-based testing
pytest>=7.0.0      # For unit testing
pytest-cov>=4.0.0  # For coverage reporting
```

## Security Considerations

- **Input Validation**: Validate image format and size before processing
- **Resource Limits**: Limit maximum image size to prevent DoS attacks
- **Model Integrity**: Verify model file checksums on loading (future enhancement)
- **Error Messages**: Don't expose internal paths or sensitive info in error messages

## Backward Compatibility

The integration maintains full backward compatibility:

- ✅ Same API endpoint (`POST /api/v1/predict`)
- ✅ Same request format (multipart/form-data with image file)
- ✅ Same response format (FoodPrediction model)
- ✅ Same error handling behavior
- ✅ Graceful degradation to mock predictions when model unavailable

Frontend code requires **zero changes** to work with the integrated model.
