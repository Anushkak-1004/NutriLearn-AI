"""
Food Recognition ML Predictor
Simulates food recognition with mock data.

TODO: Integrate trained PyTorch model here
- Load model weights from ml-models/ directory
- Implement preprocessing pipeline
- Add inference logic
"""

import random
import logging
from typing import Dict, Tuple
from PIL import Image
from datetime import datetime

from ..models import FoodPrediction, NutritionInfo

logger = logging.getLogger(__name__)


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


def simulate_food_recognition(image: Image.Image) -> FoodPrediction:
    """
    Simulate food recognition with mock predictions.
    
    This function randomly selects a food item from the database and returns
    a prediction with realistic confidence scores. It's designed for development
    and testing before integrating a real ML model.
    
    Args:
        image: PIL Image object of the food (currently not used for prediction)
        
    Returns:
        FoodPrediction object with recognized food and nutrition info
        
    Raises:
        ValueError: If image is invalid
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("food.jpg")
        >>> prediction = simulate_food_recognition(img)
        >>> print(f"Detected: {prediction.food_name} ({prediction.confidence:.2%})")
    """
    # Validate image input
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    
    # Randomly select a food item from the database
    food_key = random.choice(list(MOCK_FOOD_DATABASE.keys()))
    food_data = MOCK_FOOD_DATABASE[food_key]
    
    # Generate realistic confidence score (85-99%)
    confidence = random.uniform(0.85, 0.99)
    
    logger.info(f"Mock prediction: {food_data['name']} (confidence: {confidence:.2%})")
    
    # Create and return prediction object
    prediction = FoodPrediction(
        food_name=food_data["name"],
        confidence=round(confidence, 3),
        nutrition=NutritionInfo(**food_data["nutrition"]),
        category=food_data["category"],
        cuisine=food_data["cuisine"],
        timestamp=datetime.utcnow()
    )
    
    return prediction


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


def get_available_foods() -> list:
    """
    Get list of all available foods in the database.
    
    Returns:
        List of food names
    """
    return [data["name"] for data in MOCK_FOOD_DATABASE.values()]


def load_model():
    """
    Placeholder for model loading.
    
    TODO: Implement PyTorch model loading here
    - Load model weights from ml-models/ directory
    - Initialize model architecture
    - Return model instance
    
    Returns:
        None (mock mode - no model to load)
    """
    logger.info("✓ Using mock predictions (train model with: python train_model.py)")
    return None
