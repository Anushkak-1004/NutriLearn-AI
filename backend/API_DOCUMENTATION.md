# NutriLearn AI - API Documentation

## Overview

NutriLearn AI Backend provides RESTful APIs for food recognition, meal logging, dietary analysis, and personalized nutrition education.

**Base URL:** `http://localhost:8000`

**API Version:** v1

**Documentation:** 
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

---

## Authentication

The API uses JWT (JSON Web Token) based authentication to secure user-specific endpoints.

### Authentication Flow

1. **Sign Up:** Create a new account with email and password
2. **Login:** Authenticate and receive a JWT access token
3. **Access Protected Endpoints:** Include the token in the Authorization header

### Token Format

All authenticated requests must include the JWT token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### Token Expiration

Access tokens expire after **7 days**. After expiration, users must log in again to receive a new token.

### Public Endpoints (No Authentication Required)

- `POST /api/v1/predict` - Food recognition
- `GET /health` - Health check
- `GET /` - API information
- `POST /api/v1/auth/signup` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/docs` - API documentation

### Protected Endpoints (Authentication Required)

- `POST /api/v1/meals/log` - Log meals
- `GET /api/v1/users/{user_id}/stats` - User statistics
- `GET /api/v1/users/{user_id}/meals` - Meal history
- `GET /api/v1/users/{user_id}/analysis` - Dietary analysis
- `POST /api/v1/modules/{module_id}/complete` - Complete learning modules
- `GET /api/v1/auth/me` - Get current user profile

### Authorization

Protected endpoints enforce authorization - users can only access their own data. Attempting to access another user's data will result in a `403 Forbidden` error.

---

## Endpoints

### Authentication Endpoints

#### 1. Sign Up

**POST** `/api/v1/auth/signup`

Create a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Validation:**
- Email must be a valid email format
- Password must be at least 8 characters

**Response (201 Created):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `400 Bad Request`: Email already registered
- `422 Unprocessable Entity`: Validation error (invalid email or short password)

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/auth/signup",
    json={
        "email": "user@example.com",
        "password": "SecurePass123!"
    }
)
token = response.json()["access_token"]
```

---

#### 2. Login

**POST** `/api/v1/auth/login`

Authenticate and receive an access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid email or password

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    json={
        "email": "user@example.com",
        "password": "SecurePass123!"
    }
)
token = response.json()["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {token}"}
```

---

#### 3. Get Current User Profile

**GET** `/api/v1/auth/me`

Get the authenticated user's profile information.

**Authentication:** Required

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Response (200 OK):**
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "created_at": "2025-12-17T10:30:00Z"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing, invalid, or expired token

**Example (Python):**
```python
import requests

headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "http://localhost:8000/api/v1/auth/me",
    headers=headers
)
user_info = response.json()
```

---

### Public Endpoints

#### 4. Health Check

**GET** `/health`

Check if the API is running and get system status.

**Response:**
```json
{
  "status": "healthy",
  "service": "NutriLearn AI Backend",
  "version": "1.0.0",
  "timestamp": "2025-12-09T17:48:40.104720",
  "system": {
    "python_version": "3.12.1",
    "platform": "Windows"
  },
  "components": {
    "api": "operational",
    "ml_model": "operational (mock)",
    "database": "operational (in-memory)"
  }
}
```

---

#### 5. Food Prediction

**POST** `/api/v1/predict`

Upload a food image and get AI-powered recognition with nutrition info.

**Authentication:** Not required (public endpoint)

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file - JPEG, PNG)

**Example (Python):**
```python
import requests

with open("food.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        files={"file": f}
    )
prediction = response.json()
```

**Response:**
```json
{
  "food_name": "Chicken Biryani",
  "confidence": 0.867,
  "nutrition": {
    "calories": 450,
    "protein": 25.0,
    "carbs": 55.0,
    "fat": 12.0,
    "fiber": 3.0
  },
  "category": "main_course",
  "cuisine": "Indian",
  "timestamp": "2025-12-09T17:48:40.104720"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid image file
- `500`: Processing error

---

### Protected Endpoints

#### 6. Log Meal

**POST** `/api/v1/meals/log`

Log a consumed meal for tracking and analysis.

**Authentication:** Required

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Request Body:**
```json
{
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
```

**Note:** The `user_id` is automatically extracted from the JWT token. If provided in the request body, it will be ignored.

**Meal Types:** `breakfast`, `lunch`, `dinner`, `snack`

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Meal logged successfully",
  "log_id": "log_user_123_1_1765282794",
  "meal": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "food_name": "Chicken Biryani",
    "nutrition": {...},
    "meal_type": "lunch",
    "timestamp": "2025-12-09T17:48:40.104720",
    "log_id": "log_user_123_1_1765282794"
  }
}
```

**Error Responses:**
- `401 Unauthorized`: Missing, invalid, or expired token
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

**Example (Python):**
```python
import requests

headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/api/v1/meals/log",
    json={
        "food_name": "Chicken Biryani",
        "nutrition": {
            "calories": 450,
            "protein": 25.0,
            "carbs": 55.0,
            "fat": 12.0,
            "fiber": 3.0
        },
        "meal_type": "lunch"
    },
    headers=headers
)
```

---

#### 7. Get User Statistics

**GET** `/api/v1/users/{user_id}/stats`

Get user's overall statistics and progress.

**Authentication:** Required

**Authorization:** Users can only access their own statistics

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Parameters:**
- `user_id` (path): User identifier (must match authenticated user)

**Response (200 OK):**
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_meals": 45,
  "total_points": 350,
  "completed_modules": ["nutrition_basics", "portion_control"],
  "current_streak": 7,
  "last_updated": "2025-12-09T17:48:40.104720"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing, invalid, or expired token
- `403 Forbidden`: Attempting to access another user's statistics

**Example (Python):**
```python
import requests

headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    f"http://localhost:8000/api/v1/users/{user_id}/stats",
    headers=headers
)
```

---

#### 8. Get Meal History

**GET** `/api/v1/users/{user_id}/meals`

Retrieve user's meal history with pagination and filtering.

**Authentication:** Required

**Authorization:** Users can only access their own meal history

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Parameters:**
- `user_id` (path): User identifier (must match authenticated user)
- `limit` (query, optional): Number of meals to return (1-100, default: 50)
- `offset` (query, optional): Number of meals to skip (default: 0)
- `start_date` (query, optional): Filter meals after this date (ISO format)
- `end_date` (query, optional): Filter meals before this date (ISO format)

**Example:**
```
GET /api/v1/users/{user_id}/meals?limit=20&offset=0
GET /api/v1/users/{user_id}/meals?start_date=2025-12-01T00:00:00&end_date=2025-12-07T23:59:59
```

**Response (200 OK):**
```json
{
  "status": "success",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "meals": [
    {
      "user_id": "550e8400-e29b-41d4-a716-446655440000",
      "food_name": "Chicken Biryani",
      "nutrition": {...},
      "meal_type": "lunch",
      "timestamp": "2025-12-09T12:30:00",
      "log_id": "log_user_123_1_1765282794"
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 45,
    "has_more": true
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid date format
- `401 Unauthorized`: Missing, invalid, or expired token
- `403 Forbidden`: Attempting to access another user's meal history
- `500 Internal Server Error`: Server error

---

#### 9. Dietary Analysis

**GET** `/api/v1/users/{user_id}/analysis`

Analyze user's dietary patterns and get personalized recommendations.

**Authentication:** Required

**Authorization:** Users can only access their own dietary analysis

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Parameters:**
- `user_id` (path): User identifier (must match authenticated user)
- `days` (query, optional): Number of recent days to analyze (1-90, default: 7)

**Example:**
```
GET /api/v1/users/{user_id}/analysis?days=14
```

**Response (200 OK):**
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "analysis_date": "2025-12-09T17:48:40.104720",
  "patterns": [
    {
      "pattern_id": "high_carb_intake",
      "description": "Your carbohydrate intake is 30% above recommended levels",
      "severity": "medium",
      "recommendation": "Try incorporating more protein-rich foods...",
      "affected_meals": 12
    }
  ],
  "recommended_modules": [
    {
      "module_id": "carb_smart",
      "title": "Smart Carbohydrate Choices",
      "reason": "Based on your high carbohydrate intake",
      "content": [...],
      "quiz": {...},
      "points": 50,
      "estimated_time": 10
    }
  ],
  "summary": {
    "total_meals": 45,
    "date_range": {
      "start": "2025-12-02T08:30:00",
      "end": "2025-12-09T20:15:00"
    },
    "total_nutrition": {...},
    "daily_average": {
      "calories": 2150.5,
      "protein": 85.2,
      "carbs": 280.5,
      "fat": 75.8,
      "fiber": 18.5
    },
    "meal_type_distribution": {
      "breakfast": 14,
      "lunch": 14,
      "dinner": 14,
      "snack": 3
    },
    "days_tracked": 7
  }
}
```

**Error Responses:**
- `401 Unauthorized`: Missing, invalid, or expired token
- `403 Forbidden`: Attempting to access another user's analysis
- `404 Not Found`: No meal data found for user
- `500 Internal Server Error`: Server error

**Identified Patterns:**
- `high_carb_intake`: Carbohydrate intake above recommendations
- `low_protein_intake`: Protein intake below recommendations
- `high_fat_intake`: Fat intake above recommendations
- `low_fiber_intake`: Fiber intake below recommendations
- `excessive_calories`: Daily calories exceed recommendations
- `irregular_meal_timing`: Inconsistent meal times

**Severity Levels:** `low`, `medium`, `high`

---

#### 10. Complete Learning Module

**POST** `/api/v1/modules/{module_id}/complete`

Mark a learning module as completed and earn points.

**Authentication:** Required

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Parameters:**
- `module_id` (path): Module identifier

**Request Body:**
```json
{
  "quiz_score": 85
}
```

**Note:** The `user_id` is automatically extracted from the JWT token. If provided in the request body, it will be ignored.

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Module completed! You earned 65 points.",
  "points_earned": 65,
  "total_points": 415,
  "new_modules_unlocked": ["advanced_nutrition"]
}
```

**Error Responses:**
- `401 Unauthorized`: Missing, invalid, or expired token
- `422 Unprocessable Entity`: Validation error (invalid quiz score)
- `500 Internal Server Error`: Server error

**Points Calculation:**
- Base points: 50
- Bonus: +10 points for every 10% above 70% quiz score
- Example: 85% score = 50 + 15 = 65 points

**Available Modules:**
- `balanced_nutrition`: Understanding Balanced Nutrition
- `protein_power`: The Power of Protein
- `carb_smart`: Smart Carbohydrate Choices
- `healthy_fats`: Understanding Healthy Fats
- `fiber_focus`: The Importance of Fiber
- `portion_control`: Mastering Portion Control
- `meal_timing`: Optimal Meal Timing

---

## Data Models

### NutritionInfo
```json
{
  "calories": 450,
  "protein": 25.0,
  "carbs": 55.0,
  "fat": 12.0,
  "fiber": 3.0
}
```

### MealType Enum
- `breakfast`
- `lunch`
- `dinner`
- `snack`

### Severity Enum
- `low`
- `medium`
- `high`

---

## Error Handling

All endpoints return consistent error responses:

**Authentication Error (401 Unauthorized):**
```json
{
  "detail": "Could not validate credentials"
}
```

Common causes:
- Missing Authorization header
- Invalid token format
- Expired token
- Invalid token signature

**Authorization Error (403 Forbidden):**
```json
{
  "detail": "You can only access your own statistics"
}
```

Occurs when attempting to access another user's protected resources.

**Validation Error (422 Unprocessable Entity):**
```json
{
  "status": "error",
  "message": "Validation error",
  "errors": [
    {
      "loc": ["body", "nutrition", "calories"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error"
    }
  ],
  "timestamp": "2025-12-09T17:48:40.104720"
}
```

**Server Error (500 Internal Server Error):**
```json
{
  "status": "error",
  "message": "Internal server error",
  "detail": "Failed to process request",
  "timestamp": "2025-12-09T17:48:40.104720"
}
```

---

## Testing

Run the test suite:

```bash
cd backend
python test_api.py
```

This will test all endpoints and verify functionality.

---

## Development Notes

### Current Implementation
- **ML Model:** Mock predictions (random food selection)
- **Database:** In-memory storage (data lost on restart)
- **Authentication:** Simple user_id (no security)

### Completed Features
1. **Authentication:** ✅
   - JWT token-based authentication
   - User registration and login
   - Secure endpoints with auth middleware
   - Password hashing with bcrypt
   - 7-day token expiration

2. **Database:** ✅
   - Supabase integration
   - User authentication schema
   - Database migrations

3. **MLOps:** ✅
   - MLflow experiment tracking
   - Prediction logging and monitoring

### Production TODO
1. **ML Model Integration:**
   - Train PyTorch model on Food-101 dataset
   - Implement proper image preprocessing
   - Add model versioning with MLflow
   - Deploy model with TorchServe

2. **Enhanced Security:**
   - Add refresh tokens for better UX
   - Implement rate limiting
   - Add password reset functionality
   - Add email verification

3. **Performance:**
   - Add Redis caching
   - Implement async database queries
   - Optimize image processing
   - Add CDN for static assets

4. **Monitoring:**
   - Implement error tracking (Sentry)
   - Add performance monitoring
   - Set up alerting for critical errors

---

## Support

For issues or questions, please refer to the main README.md or contact the development team.
