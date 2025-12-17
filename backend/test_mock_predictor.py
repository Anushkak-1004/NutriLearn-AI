"""
Quick test to verify mock predictor works correctly after revert.
"""

from PIL import Image
from app.ml.predictor import simulate_food_recognition, get_available_foods, get_food_by_name


def test_simulate_food_recognition():
    """Test that mock prediction works."""
    # Create a dummy image
    image = Image.new('RGB', (224, 224), color='white')
    
    # Get prediction
    prediction = simulate_food_recognition(image)
    
    # Verify prediction structure
    assert prediction.food_name is not None
    assert 0.85 <= prediction.confidence <= 0.99
    assert prediction.nutrition is not None
    assert prediction.nutrition.calories > 0
    assert prediction.category is not None
    assert prediction.cuisine is not None
    
    print(f"✓ Mock prediction works: {prediction.food_name} ({prediction.confidence:.2%})")
    print(f"  Calories: {prediction.nutrition.calories}")
    print(f"  Category: {prediction.category}")
    print(f"  Cuisine: {prediction.cuisine}")


def test_get_available_foods():
    """Test getting list of available foods."""
    foods = get_available_foods()
    
    assert len(foods) == 15
    assert "Chicken Biryani" in foods
    assert "Margherita Pizza" in foods
    
    print(f"✓ Available foods: {len(foods)} items")
    for food in foods[:5]:
        print(f"  - {food}")


def test_get_food_by_name():
    """Test looking up food by name."""
    food_data, confidence = get_food_by_name("Chicken Biryani")
    
    assert food_data["name"] == "Chicken Biryani"
    assert confidence == 1.0
    assert food_data["nutrition"]["calories"] == 450
    
    print(f"✓ Food lookup works: {food_data['name']}")
    print(f"  Nutrition: {food_data['nutrition']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Mock Predictor After Revert")
    print("=" * 60)
    
    try:
        test_simulate_food_recognition()
        print()
        test_get_available_foods()
        print()
        test_get_food_by_name()
        print()
        print("=" * 60)
        print("✅ All tests passed! Mock predictor is working correctly.")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
