"""
Quick API Test Script
Tests all NutriLearn AI endpoints to verify functionality.
"""

import requests
import json
from io import BytesIO
from PIL import Image

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("   ✓ Health check passed")

def test_root():
    """Test root endpoint."""
    print("\n2. Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Message: {data['message']}")
    print(f"   Available endpoints: {len(data['endpoints'])}")
    assert response.status_code == 200
    print("   ✓ Root endpoint passed")

def test_predict():
    """Test food prediction endpoint."""
    print("\n3. Testing Food Prediction...")
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(f"{BASE_URL}/api/v1/predict", files=files)
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Predicted: {data['food_name']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print(f"   Calories: {data['nutrition']['calories']}")
        print("   ✓ Prediction passed")
    else:
        print(f"   Error: {response.text}")

def test_log_meal():
    """Test meal logging endpoint."""
    print("\n4. Testing Meal Logging...")
    
    meal_data = {
        "user_id": "test_user_123",
        "food_name": "Chicken Biryani",
        "nutrition": {
            "calories": 450,
            "protein": 25.0,
            "carbs": 55.0,
            "fat": 12.0,
            "fiber": 3.0
        },
        "meal_type": "lunch"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/meals/log", json=meal_data)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Log ID: {data['log_id']}")
        print(f"   Message: {data['message']}")
        print("   ✓ Meal logging passed")
    else:
        print(f"   Error: {response.text}")

def test_user_stats():
    """Test user statistics endpoint."""
    print("\n5. Testing User Statistics...")
    
    response = requests.get(f"{BASE_URL}/api/v1/users/test_user_123/stats")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Total meals: {data['total_meals']}")
        print(f"   Total points: {data['total_points']}")
        print(f"   Current streak: {data['current_streak']}")
        print("   ✓ User stats passed")
    else:
        print(f"   Error: {response.text}")

def test_meal_history():
    """Test meal history endpoint."""
    print("\n6. Testing Meal History...")
    
    response = requests.get(f"{BASE_URL}/api/v1/users/test_user_123/meals?limit=10")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Total meals: {data['pagination']['total']}")
        print(f"   Returned: {len(data['meals'])}")
        print("   ✓ Meal history passed")
    else:
        print(f"   Error: {response.text}")

def test_analysis():
    """Test dietary analysis endpoint."""
    print("\n7. Testing Dietary Analysis...")
    
    # First, log a few more meals for better analysis
    meals = [
        {"food_name": "Pizza", "meal_type": "dinner", "nutrition": {"calories": 680, "protein": 28.0, "carbs": 82.0, "fat": 26.0, "fiber": 4.0}},
        {"food_name": "Pasta", "meal_type": "lunch", "nutrition": {"calories": 580, "protein": 22.0, "carbs": 65.0, "fat": 24.0, "fiber": 3.0}},
        {"food_name": "Burger", "meal_type": "dinner", "nutrition": {"calories": 540, "protein": 30.0, "carbs": 42.0, "fat": 26.0, "fiber": 2.0}},
    ]
    
    for meal in meals:
        meal["user_id"] = "test_user_123"
        requests.post(f"{BASE_URL}/api/v1/meals/log", json=meal)
    
    response = requests.get(f"{BASE_URL}/api/v1/users/test_user_123/analysis?days=7")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Patterns identified: {len(data['patterns'])}")
        for pattern in data['patterns']:
            print(f"   - {pattern['description']} ({pattern['severity']})")
        print(f"   Recommended modules: {len(data['recommended_modules'])}")
        for module in data['recommended_modules']:
            print(f"   - {module['title']}")
        print("   ✓ Analysis passed")
    else:
        print(f"   Error: {response.text}")

def test_complete_module():
    """Test module completion endpoint."""
    print("\n8. Testing Module Completion...")
    
    completion_data = {
        "user_id": "test_user_123",
        "quiz_score": 85
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/modules/balanced_nutrition/complete",
        json=completion_data
    )
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Points earned: {data['points_earned']}")
        print(f"   Total points: {data['total_points']}")
        print("   ✓ Module completion passed")
    else:
        print(f"   Error: {response.text}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("NutriLearn AI - API Test Suite")
    print("=" * 60)
    
    try:
        test_health()
        test_root()
        test_predict()
        test_log_meal()
        test_user_stats()
        test_meal_history()
        test_analysis()
        test_complete_module()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
