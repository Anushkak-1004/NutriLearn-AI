"""
Data Models for NutriLearn AI
Pydantic models for request/response validation and data structures.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict, EmailStr


class MealType(str, Enum):
    """Enumeration for meal types."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class Severity(str, Enum):
    """Enumeration for dietary pattern severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NutritionInfo(BaseModel):
    """
    Nutrition information for a food item.
    
    Attributes:
        calories: Total calories in kcal
        protein: Protein content in grams
        carbs: Carbohydrate content in grams
        fat: Fat content in grams
        fiber: Fiber content in grams
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "calories": 450,
            "protein": 12.5,
            "carbs": 58.0,
            "fat": 18.5,
            "fiber": 3.2
        }
    })
    
    calories: int = Field(..., ge=0, description="Total calories in kcal")
    protein: float = Field(..., ge=0.0, description="Protein content in grams")
    carbs: float = Field(..., ge=0.0, description="Carbohydrate content in grams")
    fat: float = Field(..., ge=0.0, description="Fat content in grams")
    fiber: float = Field(..., ge=0.0, description="Fiber content in grams")
    
    @field_validator('calories')
    @classmethod
    def validate_calories(cls, v: int) -> int:
        """Validate calories are within reasonable range."""
        if v > 5000:
            raise ValueError("Calories cannot exceed 5000 kcal per serving")
        return v


class FoodPrediction(BaseModel):
    """
    Food recognition prediction result.
    
    Attributes:
        food_name: Name of the recognized food item
        confidence: Prediction confidence score (0-1)
        nutrition: Nutritional information
        category: Food category (e.g., 'main_course', 'snack')
        cuisine: Cuisine type (e.g., 'Indian', 'Western')
        timestamp: When the prediction was made
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "food_name": "Chicken Biryani",
            "confidence": 0.95,
            "nutrition": {
                "calories": 450,
                "protein": 25.0,
                "carbs": 55.0,
                "fat": 12.0,
                "fiber": 3.0
            },
            "category": "main_course",
            "cuisine": "Indian",
            "timestamp": "2025-12-06T10:30:00"
        }
    })
    
    food_name: str = Field(..., min_length=1, description="Name of the recognized food")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    nutrition: NutritionInfo = Field(..., description="Nutritional information")
    category: str = Field(..., description="Food category")
    cuisine: str = Field(..., description="Cuisine type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class MealLog(BaseModel):
    """
    Log entry for a consumed meal.
    
    Attributes:
        user_id: Unique identifier for the user
        food_name: Name of the food consumed
        nutrition: Nutritional information
        meal_type: Type of meal (breakfast/lunch/dinner/snack)
        timestamp: When the meal was consumed
        log_id: Unique identifier for this log entry
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_id": "user_123",
            "food_name": "Masala Dosa",
            "nutrition": {
                "calories": 350,
                "protein": 8.0,
                "carbs": 48.0,
                "fat": 14.0,
                "fiber": 4.0
            },
            "meal_type": "breakfast",
            "timestamp": "2025-12-06T08:30:00"
        }
    })
    
    user_id: str = Field(..., min_length=1, description="User identifier")
    food_name: str = Field(..., min_length=1, description="Name of the food")
    nutrition: NutritionInfo = Field(..., description="Nutritional information")
    meal_type: MealType = Field(..., description="Type of meal")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Meal timestamp")
    log_id: Optional[str] = Field(None, description="Unique log entry identifier")


class UserStats(BaseModel):
    """
    User statistics and progress tracking.
    
    Attributes:
        user_id: Unique identifier for the user
        total_meals: Total number of meals logged
        total_points: Total learning points earned
        completed_modules: List of completed learning module IDs
        current_streak: Current daily logging streak
        last_updated: When stats were last updated
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_id": "user_123",
            "total_meals": 45,
            "total_points": 350,
            "completed_modules": ["nutrition_basics", "portion_control"],
            "current_streak": 7,
            "last_updated": "2025-12-06T10:30:00"
        }
    })
    
    user_id: str = Field(..., min_length=1, description="User identifier")
    total_meals: int = Field(default=0, ge=0, description="Total meals logged")
    total_points: int = Field(default=0, ge=0, description="Total points earned")
    completed_modules: List[str] = Field(default_factory=list, description="Completed module IDs")
    current_streak: int = Field(default=0, ge=0, description="Current daily streak")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class DietaryPattern(BaseModel):
    """
    Identified dietary pattern or issue.
    
    Attributes:
        pattern_id: Unique identifier for the pattern
        description: Human-readable description of the pattern
        severity: Severity level (low/medium/high)
        recommendation: Actionable recommendation
        affected_meals: Number of meals showing this pattern
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "pattern_id": "high_carb_intake",
            "description": "Your carbohydrate intake is consistently above recommended levels",
            "severity": "medium",
            "recommendation": "Try incorporating more protein-rich foods and vegetables",
            "affected_meals": 12
        }
    })
    
    pattern_id: str = Field(..., description="Pattern identifier")
    description: str = Field(..., min_length=1, description="Pattern description")
    severity: Severity = Field(..., description="Severity level")
    recommendation: str = Field(..., min_length=1, description="Actionable recommendation")
    affected_meals: int = Field(default=0, ge=0, description="Number of affected meals")


class LearningModule(BaseModel):
    """
    Educational learning module.
    
    Attributes:
        module_id: Unique identifier for the module
        title: Module title
        reason: Why this module is recommended
        content: List of content sections
        quiz: Quiz questions and answers
        points: Points awarded for completion
        estimated_time: Estimated completion time in minutes
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "module_id": "balanced_nutrition",
            "title": "Understanding Balanced Nutrition",
            "reason": "Based on your high carb intake pattern",
            "content": [
                {
                    "type": "text",
                    "data": "A balanced diet includes proteins, carbs, and fats..."
                }
            ],
            "quiz": {
                "questions": [
                    {
                        "question": "What percentage of daily calories should come from protein?",
                        "options": ["10-15%", "20-35%", "40-50%"],
                        "correct": 1
                    }
                ]
            },
            "points": 50,
            "estimated_time": 10
        }
    })
    
    module_id: str = Field(..., description="Module identifier")
    title: str = Field(..., min_length=1, description="Module title")
    reason: str = Field(..., description="Why this module is recommended")
    content: List[Dict[str, Any]] = Field(..., description="Content sections")
    quiz: Dict[str, Any] = Field(..., description="Quiz data")
    points: int = Field(..., ge=0, description="Points for completion")
    estimated_time: int = Field(default=10, ge=1, description="Estimated time in minutes")


class MealLogRequest(BaseModel):
    """
    Request model for logging a meal.
    
    Note: user_id is optional and will be ignored if provided.
    The authenticated user_id from the JWT token is used instead.
    """
    user_id: Optional[str] = Field(None, min_length=1, description="User ID (ignored, extracted from token)")
    food_name: str = Field(..., min_length=1)
    nutrition: NutritionInfo
    meal_type: MealType


class MealLogResponse(BaseModel):
    """Response model for meal logging."""
    status: str
    message: str
    log_id: str
    meal: MealLog


class AnalysisResponse(BaseModel):
    """Response model for dietary pattern analysis."""
    user_id: str
    analysis_date: datetime
    patterns: List[DietaryPattern]
    recommended_modules: List[LearningModule]
    summary: Dict[str, Any]


class ModuleCompletionRequest(BaseModel):
    """
    Request model for completing a learning module.
    
    Note: user_id is optional and will be ignored if provided.
    The authenticated user_id from the JWT token is used instead.
    """
    user_id: Optional[str] = Field(None, min_length=1, description="User ID (ignored, extracted from token)")
    quiz_score: int = Field(..., ge=0, le=100, description="Quiz score percentage")


class ModuleCompletionResponse(BaseModel):
    """Response model for module completion."""
    status: str
    message: str
    points_earned: int
    total_points: int
    new_modules_unlocked: List[str]


# Authentication Models

class UserCreate(BaseModel):
    """
    Request model for user registration.
    
    Attributes:
        email: User's email address (must be valid email format)
        password: User's password (minimum 8 characters)
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "email": "user@example.com",
            "password": "SecurePass123!"
        }
    })
    
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")


class UserLogin(BaseModel):
    """
    Request model for user login.
    
    Attributes:
        email: User's email address
        password: User's password
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "email": "user@example.com",
            "password": "SecurePass123!"
        }
    })
    
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class Token(BaseModel):
    """
    Response model for authentication tokens.
    
    Attributes:
        access_token: JWT access token string
        token_type: Token type (always "bearer")
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer"
        }
    })
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")


class UserResponse(BaseModel):
    """
    Response model for user information.
    
    Attributes:
        user_id: Unique identifier for the user
        email: User's email address
        created_at: Timestamp when the user account was created
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "user@example.com",
            "created_at": "2025-12-17T10:30:00Z"
        }
    })
    
    user_id: str = Field(..., description="User's unique identifier")
    email: str = Field(..., description="User's email address")
    created_at: datetime = Field(..., description="Account creation timestamp")
