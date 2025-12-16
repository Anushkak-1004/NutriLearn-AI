"""
Food Recognition ML Predictor
Simulates food recognition with mock data. Ready for PyTorch model integration.
"""

import random
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image
from datetime import datetime

from ..models import FoodPrediction, NutritionInfo

logger = logging.getLogger(__name__)

# Try to import PyTorch (optional for development)
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using mock predictions.")


# Mock food database with accurate nutrition data
MOCK_FOOD_DATABASE: Dict[str, Dict] = {
    # Indian Cuisine
    "chicken_biryani": {
        "name": "Chicken Biryani",
        "category": "main_course",
        "cuisine": "Indian",
        "nutrition": {"calories": 450, "protein": 25.0, "carbs": 55.0, "fat": 12.0, "fiber": 3.0}
    },
    "masala_dosa": {
        "name": "Masala Dosa",
        "category": "main_course",
        "cuisine": "Indian",
        "nutrition": {"calories": 350, "protein": 8.0, "carbs": 48.0, "fat": 14.0, "fiber": 4.0}
    },
    "dal_tadka": {
        "name": "Dal Tadka",
        "category": "main_course",
        "cuisine": "Indian",
        "nutrition": {"calories": 180, "protein": 12.0, "carbs": 28.0, "fat": 4.0, "fiber": 8.0}
    },
    "paneer_tikka": {
        "name": "Paneer Tikka",
        "category": "appetizer",
        "cuisine": "Indian",
        "nutrition": {"calories": 320, "protein": 18.0, "carbs": 12.0, "fat": 22.0, "fiber": 2.0}
    },
    "samosa": {
        "name": "Samosa",
        "category": "snack",
        "cuisine": "Indian",
        "nutrition": {"calories": 262, "protein": 5.0, "carbs": 32.0, "fat": 13.0, "fiber": 3.0}
    },
    "roti": {
        "name": "Roti (Chapati)",
        "category": "bread",
        "cuisine": "Indian",
        "nutrition": {"calories": 120, "protein": 3.5, "carbs": 22.0, "fat": 2.5, "fiber": 2.5}
    },
    "chole_bhature": {
        "name": "Chole Bhature",
        "category": "main_course",
        "cuisine": "Indian",
        "nutrition": {"calories": 550, "protein": 15.0, "carbs": 68.0, "fat": 24.0, "fiber": 10.0}
    },
    
    # Western Cuisine
    "margherita_pizza": {
        "name": "Margherita Pizza",
        "category": "main_course",
        "cuisine": "Italian",
        "nutrition": {"calories": 680, "protein": 28.0, "carbs": 82.0, "fat": 26.0, "fiber": 4.0}
    },
    "cheeseburger": {
        "name": "Cheeseburger",
        "category": "main_course",
        "cuisine": "American",
        "nutrition": {"calories": 540, "protein": 30.0, "carbs": 42.0, "fat": 26.0, "fiber": 2.0}
    },
    "pasta_carbonara": {
        "name": "Pasta Carbonara",
        "category": "main_course",
        "cuisine": "Italian",
        "nutrition": {"calories": 580, "protein": 22.0, "carbs": 65.0, "fat": 24.0, "fiber": 3.0}
    },
    "caesar_salad": {
        "name": "Caesar Salad",
        "category": "salad",
        "cuisine": "American",
        "nutrition": {"calories": 280, "protein": 12.0, "carbs": 18.0, "fat": 18.0, "fiber": 4.0}
    },
    "club_sandwich": {
        "name": "Club Sandwich",
        "category": "sandwich",
        "cuisine": "American",
        "nutrition": {"calories": 480, "protein": 28.0, "carbs": 45.0, "fat": 20.0, "fiber": 3.0}
    },
    "grilled_chicken_salad": {
        "name": "Grilled Chicken Salad",
        "category": "salad",
        "cuisine": "American",
        "nutrition": {"calories": 320, "protein": 35.0, "carbs": 15.0, "fat": 14.0, "fiber": 5.0}
    },
    "french_fries": {
        "name": "French Fries",
        "category": "side",
        "cuisine": "American",
        "nutrition": {"calories": 365, "protein": 4.0, "carbs": 48.0, "fat": 17.0, "fiber": 4.0}
    },
    "spaghetti_bolognese": {
        "name": "Spaghetti Bolognese",
        "category": "main_course",
        "cuisine": "Italian",
        "nutrition": {"calories": 520, "protein": 28.0, "carbs": 62.0, "fat": 16.0, "fiber": 5.0}
    }
}


def simulate_food_recognition(image: Image.Image, use_real_model: bool = True) -> FoodPrediction:
    """
    Recognize food from an image using trained model or mock predictions.
    
    Args:
        image: PIL Image object of the food
        use_real_model: Whether to use real PyTorch model (if available)
        
    Returns:
        FoodPrediction object with recognized food and nutrition info
        
    Raises:
        ValueError: If image is invalid or cannot be processed
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("food.jpg")
        >>> prediction = simulate_food_recognition(img)
        >>> print(f"Detected: {prediction.food_name} ({prediction.confidence:.2%})")
    """
    try:
        # Validate image
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")
        
        # Try to use real model if available
        if use_real_model and PYTORCH_AVAILABLE and model is not None:
            try:
                # Preprocess image
                input_tensor = preprocess_image(image, model_config)
                
                # Move to same device as model
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Run inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Get predicted class name
                idx_to_class = {v: k for k, v in class_mappings.items()}
                predicted_class = idx_to_class[predicted_idx.item()]
                confidence_score = confidence.item()
                
                logger.info(f"Model prediction: {predicted_class} (confidence: {confidence_score:.2%})")
                
                # Map to nutrition database (use mock data for now)
                # TODO: Create comprehensive nutrition database for all Food-101 classes
                food_data = MOCK_FOOD_DATABASE.get(
                    predicted_class.lower().replace(" ", "_"),
                    {
                        "name": predicted_class.replace("_", " ").title(),
                        "category": "unknown",
                        "cuisine": "unknown",
                        "nutrition": {"calories": 300, "protein": 10.0, "carbs": 40.0, "fat": 10.0, "fiber": 3.0}
                    }
                )
                
                # Create prediction object
                prediction = FoodPrediction(
                    food_name=food_data["name"],
                    confidence=round(confidence_score, 3),
                    nutrition=NutritionInfo(**food_data["nutrition"]),
                    category=food_data.get("category", "unknown"),
                    cuisine=food_data.get("cuisine", "unknown"),
                    timestamp=datetime.utcnow()
                )
                
                return prediction
                
            except Exception as e:
                logger.warning(f"Model inference failed: {str(e)}. Falling back to mock predictions.")
        
        # Fallback to mock predictions
        # For now, randomly select a food item
        food_key = random.choice(list(MOCK_FOOD_DATABASE.keys()))
        food_data = MOCK_FOOD_DATABASE[food_key]
        
        # Generate realistic confidence score (85-99%)
        confidence = random.uniform(0.85, 0.99)
        
        logger.info(f"Mock prediction: {food_data['name']} (confidence: {confidence:.2%})")
        
        # Create prediction object
        prediction = FoodPrediction(
            food_name=food_data["name"],
            confidence=round(confidence, 3),
            nutrition=NutritionInfo(**food_data["nutrition"]),
            category=food_data["category"],
            cuisine=food_data["cuisine"],
            timestamp=datetime.utcnow()
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in food recognition: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")


def get_food_by_name(food_name: str) -> Tuple[Dict, float]:
    """
    Get food data by name (for testing and manual entry).
    
    Args:
        food_name: Name of the food to look up
        
    Returns:
        Tuple of (food_data dict, confidence score)
        
    Raises:
        ValueError: If food name not found in database
    """
    # Normalize food name for matching
    normalized_name = food_name.lower().replace(" ", "_")
    
    for key, data in MOCK_FOOD_DATABASE.items():
        if key == normalized_name or data["name"].lower() == food_name.lower():
            return data, 1.0
    
    raise ValueError(f"Food '{food_name}' not found in database")


def load_model(
    model_path: str = "./ml-models/food_model_v1.pth",
    config_path: str = "./ml-models/model_config.json",
    class_mapping_path: str = "./ml-models/class_to_idx.json"
) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[Dict]]:
    """
    Load the PyTorch model for food recognition.
    
    Args:
        model_path: Path to trained model weights
        config_path: Path to model configuration
        class_mapping_path: Path to class mappings
        
    Returns:
        Tuple of (model, config, class_to_idx) or (None, None, None) if loading fails
    """
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch not available. Using mock predictions.")
        return None, None, None
    
    try:
        # Check if model files exist
        model_path = Path(model_path)
        config_path = Path(config_path)
        class_mapping_path = Path(class_mapping_path)
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}. Using mock predictions.")
            return None, None, None
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using mock predictions.")
            return None, None, None
        
        if not class_mapping_path.exists():
            logger.warning(f"Class mapping file not found: {class_mapping_path}. Using mock predictions.")
            return None, None, None
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load class mappings
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model architecture
        model_name = config.get('model_name', 'mobilenet_v2')
        num_classes = config.get('num_classes', 101)
        
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, num_classes)
            )
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
            num_features = 1280
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        else:
            logger.error(f"Unknown model architecture: {model_name}")
            return None, None, None
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded {model_name} model from {model_path}")
        logger.info(f"Model device: {device}")
        logger.info(f"Number of classes: {num_classes}")
        
        return model, config, class_to_idx
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.warning("Falling back to mock predictions")
        return None, None, None


def preprocess_image(image: Image.Image, config: Optional[Dict] = None) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: Input PIL Image
        config: Model configuration with preprocessing parameters
        
    Returns:
        Preprocessed image tensor ready for model
    """
    if not PYTORCH_AVAILABLE:
        return image
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get preprocessing parameters from config
    if config:
        mean = config.get('mean', [0.485, 0.456, 0.406])
        std = config.get('std', [0.229, 0.224, 0.225])
        input_size = config.get('input_size', 224)
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224
    
    # Create preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Apply transformations and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def get_available_foods() -> list:
    """
    Get list of all available foods in the database.
    
    Returns:
        List of food names
    """
    return [data["name"] for data in MOCK_FOOD_DATABASE.values()]


# Model instance (loaded on startup)
model, model_config, class_mappings = load_model()

if model is not None:
    logger.info("✓ Trained PyTorch model loaded successfully")
else:
    logger.info("✓ Using mock predictions (train model with: python train_model.py)")
