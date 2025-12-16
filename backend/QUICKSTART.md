# NutriLearn AI Backend - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.9+ installed
- pip package manager

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Start the Server

```bash
python -m uvicorn app.main:app --reload
```

You should see:
```
============================================================
Starting NutriLearn AI Backend
============================================================
✓ ML model loaded successfully (using mock predictions)
✓ Using in-memory storage for development
============================================================
NutriLearn AI Backend is ready!
API Documentation: http://localhost:8000/api/docs
============================================================
```

### Step 3: Test the API

Open a new terminal and run:

```bash
python test_api.py
```

Expected output:
```
============================================================
NutriLearn AI - API Test Suite
============================================================
✓ Health check passed
✓ Root endpoint passed
✓ Prediction passed
✓ Meal logging passed
✓ User stats passed
✓ Meal history passed
✓ Analysis passed
✓ Module completion passed
============================================================
✓ All tests passed successfully!
============================================================
```

### Step 4: Explore the API

Visit http://localhost:8000/api/docs for interactive API documentation.

---

## 📖 Quick Examples

### Example 1: Predict Food from Image

```python
import requests
from PIL import Image

# Create or load an image
img = Image.open("food.jpg")

# Send to API
with open("food.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        files={"file": f}
    )

prediction = response.json()
print(f"Detected: {prediction['food_name']}")
print(f"Confidence: {prediction['confidence']:.2%}")
print(f"Calories: {prediction['nutrition']['calories']}")
```

### Example 2: Log a Meal

```python
import requests

meal_data = {
    "user_id": "user_123",
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

response = requests.post(
    "http://localhost:8000/api/v1/meals/log",
    json=meal_data
)

result = response.json()
print(f"Meal logged! ID: {result['log_id']}")
```

### Example 3: Get Dietary Analysis

```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/users/user_123/analysis?days=7"
)

analysis = response.json()

print(f"Patterns identified: {len(analysis['patterns'])}")
for pattern in analysis['patterns']:
    print(f"- {pattern['description']} ({pattern['severity']})")

print(f"\nRecommended modules:")
for module in analysis['recommended_modules']:
    print(f"- {module['title']}")
```

---

## 🎯 Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/predict` | Food recognition |
| POST | `/api/v1/meals/log` | Log a meal |
| GET | `/api/v1/users/{user_id}/stats` | User statistics |
| GET | `/api/v1/users/{user_id}/meals` | Meal history |
| GET | `/api/v1/users/{user_id}/analysis` | Dietary analysis |
| POST | `/api/v1/modules/{module_id}/complete` | Complete module |

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database (optional - uses in-memory storage if not set)
DATABASE_URL=postgresql://user:password@localhost:5432/nutrilearn
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000

# ML Model (optional)
MODEL_PATH=../ml-models/food_classifier.pth
```

---

## 🐛 Troubleshooting

### Port Already in Use

If port 8000 is already in use:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

### Import Errors

Make sure you're in the backend directory:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

### Module Not Found

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Next Steps

1. **Read the Documentation**: Check `API_DOCUMENTATION.md` for detailed endpoint info
2. **Explore the Code**: Review `IMPLEMENTATION_SUMMARY.md` for architecture details
3. **Integrate Frontend**: Connect React frontend to these APIs
4. **Train ML Model**: Replace mock predictions with real PyTorch model
5. **Set up Database**: Migrate from in-memory to Supabase

---

## 💡 Tips

- Use the Swagger UI at `/api/docs` for interactive testing
- Check server logs for debugging information
- All data is in-memory and will be lost on restart
- Mock ML model returns random foods with 85-99% confidence

---

## 🎓 Learning Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
- **PyTorch**: https://pytorch.org/tutorials/
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **Supabase**: https://supabase.com/docs

---

**Happy Coding! 🚀**
