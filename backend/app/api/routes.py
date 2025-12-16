"""
API Routes for NutriLearn AI
RESTful endpoints for food recognition, meal logging, and user analytics.
"""

import logging
from typing import Optional
from datetime import datetime
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from PIL import Image

from ..models import (
    FoodPrediction, MealLogRequest, MealLogResponse, MealLog,
    AnalysisResponse, UserStats, ModuleCompletionRequest,
    ModuleCompletionResponse
)
from ..ml.predictor import simulate_food_recognition
from ..database import (
    add_meal_log, get_user_meals, get_user_stats, update_user_stats,
    mark_module_completed, get_completed_modules, calculate_streak
)
from ..utils import (
    analyze_dietary_patterns, generate_learning_recommendations,
    get_nutrition_summary
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["api"])


@router.post("/predict", response_model=FoodPrediction)
async def predict_food(file: UploadFile = File(...)):
    """
    Predict food item from uploaded image with MLflow tracking.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        FoodPrediction with recognized food and nutrition info
        
    Raises:
        HTTPException: If image is invalid or processing fails
        
    MLOps Note:
    This endpoint logs predictions to MLflow for monitoring and analysis.
        
    Example:
        ```python
        import requests
        
        with open("food.jpg", "rb") as f:
            response = requests.post(
                "http://localhost:8000/api/v1/predict",
                files={"file": f}
            )
        prediction = response.json()
        print(f"Detected: {prediction['food_name']}")
        ```
    """
    import time
    from ..mlops.tracker import log_prediction
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Get image size for logging
        image_size = image.size
        
        # Verify image is valid
        image.verify()
        
        # Reopen image after verify (verify closes the file)
        image = Image.open(BytesIO(contents))
        
        # Run prediction
        prediction = simulate_food_recognition(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log prediction to MLflow (async, don't block response)
        try:
            log_prediction(
                food_name=prediction.food_name,
                confidence=prediction.confidence,
                user_id="anonymous",  # Will be updated when auth is implemented
                processing_time=processing_time,
                image_size=image_size,
                timestamp=prediction.timestamp
            )
        except Exception as e:
            # Don't fail the request if logging fails
            logger.warning(f"Failed to log prediction to MLflow: {str(e)}")
        
        logger.info(f"Prediction successful: {prediction.food_name} ({prediction.confidence:.2%}) in {processing_time:.3f}s")
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )


@router.post("/meals/log", response_model=MealLogResponse)
async def log_meal(
    request: MealLogRequest,
    current_user_id: str = Depends(get_current_user)
):
    """
    Log a consumed meal for the authenticated user.
    
    Args:
        request: MealLogRequest with food details and nutrition
        current_user_id: Authenticated user ID from JWT token (injected by dependency)
        
    Returns:
        MealLogResponse with confirmation and log ID
        
    Note:
        This endpoint requires authentication. The user_id is extracted from the JWT token,
        not from the request body, ensuring users can only log meals for themselves.
        
    Example:
        ```python
        import requests
        
        headers = {"Authorization": "Bearer <your_jwt_token>"}
        meal_data = {
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
            json=meal_data,
            headers=headers
        )
        ```
    """
    try:
        # Create meal log entry using authenticated user_id
        meal = MealLog(
            user_id=current_user_id,
            food_name=request.food_name,
            nutrition=request.nutrition,
            meal_type=request.meal_type,
            timestamp=datetime.utcnow()
        )
        
        # Save to database
        log_id = add_meal_log(meal)
        
        # Update user stats
        stats = get_user_stats(current_user_id)
        stats.total_meals += 1
        stats.current_streak = calculate_streak(current_user_id)
        update_user_stats(stats)
        
        logger.info(f"Meal logged: {request.food_name} for user {current_user_id}")
        
        return MealLogResponse(
            status="success",
            message="Meal logged successfully",
            log_id=log_id,
            meal=meal
        )
        
    except Exception as e:
        logger.error(f"Error logging meal: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log meal: {str(e)}"
        )


@router.get("/users/{user_id}/analysis", response_model=AnalysisResponse)
async def get_dietary_analysis(
    user_id: str,
    days: int = Query(default=7, ge=1, le=90, description="Number of days to analyze"),
    current_user_id: str = Depends(get_current_user)
):
    """
    Analyze user's dietary patterns and generate recommendations.
    
    Args:
        user_id: User identifier from URL path
        days: Number of recent days to analyze (default: 7)
        current_user_id: Authenticated user ID from JWT token (injected by dependency)
        
    Returns:
        AnalysisResponse with patterns, recommendations, and summary
        
    Raises:
        HTTPException 403: If user attempts to access another user's analysis
        
    Note:
        This endpoint requires authentication. Users can only access their own analysis.
        
    Example:
        ```python
        import requests
        
        headers = {"Authorization": "Bearer <your_jwt_token>"}
        response = requests.get(
            "http://localhost:8000/api/v1/users/user_123/analysis?days=14",
            headers=headers
        )
        analysis = response.json()
        
        for pattern in analysis["patterns"]:
            print(f"{pattern['description']} - {pattern['severity']}")
        ```
    """
    try:
        # Authorization check: verify user can only access their own analysis
        if user_id != current_user_id:
            logger.warning(f"User {current_user_id} attempted to access analysis for user {user_id}")
            raise HTTPException(
                status_code=403,
                detail="You can only access your own dietary analysis"
            )
        
        # Get recent meals
        start_date = datetime.utcnow() - timedelta(days=days)
        meals = get_user_meals(user_id, limit=1000, start_date=start_date)
        
        if not meals:
            raise HTTPException(
                status_code=404,
                detail=f"No meal data found for user {user_id}"
            )
        
        # Analyze dietary patterns
        patterns = analyze_dietary_patterns(meals)
        
        # Get completed modules
        completed = get_completed_modules(user_id)
        
        # Generate learning recommendations
        recommended_modules = generate_learning_recommendations(patterns, completed)
        
        # Generate summary
        summary = get_nutrition_summary(meals)
        
        logger.info(f"Analysis complete for user {user_id}: {len(patterns)} patterns identified")
        
        return AnalysisResponse(
            user_id=user_id,
            analysis_date=datetime.utcnow(),
            patterns=patterns,
            recommended_modules=recommended_modules,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing dietary patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze dietary patterns: {str(e)}"
        )


@router.get("/users/{user_id}/stats", response_model=UserStats)
async def get_user_statistics(
    user_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    Get user statistics and progress.
    
    Args:
        user_id: User identifier from URL path
        current_user_id: Authenticated user ID from JWT token (injected by dependency)
        
    Returns:
        UserStats with meals, points, modules, and streak
        
    Raises:
        HTTPException 403: If user attempts to access another user's stats
        
    Note:
        This endpoint requires authentication. Users can only access their own statistics.
        
    Example:
        ```python
        import requests
        
        headers = {"Authorization": "Bearer <your_jwt_token>"}
        response = requests.get(
            "http://localhost:8000/api/v1/users/user_123/stats",
            headers=headers
        )
        stats = response.json()
        print(f"Total meals: {stats['total_meals']}")
        print(f"Current streak: {stats['current_streak']} days")
        ```
    """
    try:
        # Authorization check: verify user can only access their own stats
        if user_id != current_user_id:
            logger.warning(f"User {current_user_id} attempted to access stats for user {user_id}")
            raise HTTPException(
                status_code=403,
                detail="You can only access your own statistics"
            )
        
        stats = get_user_stats(user_id)
        
        # Update streak
        stats.current_streak = calculate_streak(user_id)
        
        # Update completed modules
        stats.completed_modules = get_completed_modules(user_id)
        
        update_user_stats(stats)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user statistics: {str(e)}"
        )


@router.get("/users/{user_id}/meals")
async def get_meal_history(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=100, description="Number of meals to return"),
    offset: int = Query(default=0, ge=0, description="Number of meals to skip"),
    start_date: Optional[str] = Query(None, description="Filter meals after this date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter meals before this date (ISO format)"),
    current_user_id: str = Depends(get_current_user)
):
    """
    Get user's meal history with pagination and filtering.
    
    Args:
        user_id: User identifier from URL path
        limit: Maximum number of meals to return (1-100)
        offset: Number of meals to skip for pagination
        start_date: Filter meals after this date (ISO format)
        end_date: Filter meals before this date (ISO format)
        current_user_id: Authenticated user ID from JWT token (injected by dependency)
        
    Returns:
        JSON response with meals array and pagination info
        
    Raises:
        HTTPException 403: If user attempts to access another user's meals
        
    Note:
        This endpoint requires authentication. Users can only access their own meal history.
        
    Example:
        ```python
        import requests
        
        headers = {"Authorization": "Bearer <your_jwt_token>"}
        
        # Get first 20 meals
        response = requests.get(
            "http://localhost:8000/api/v1/users/user_123/meals?limit=20&offset=0",
            headers=headers
        )
        
        # Get meals from specific date range
        response = requests.get(
            "http://localhost:8000/api/v1/users/user_123/meals",
            params={
                "start_date": "2025-12-01T00:00:00",
                "end_date": "2025-12-07T23:59:59"
            },
            headers=headers
        )
        ```
    """
    try:
        # Authorization check: verify user can only access their own meals
        if user_id != current_user_id:
            logger.warning(f"User {current_user_id} attempted to access meals for user {user_id}")
            raise HTTPException(
                status_code=403,
                detail="You can only access your own meal history"
            )
        
        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Get meals
        meals = get_user_meals(
            user_id=user_id,
            limit=limit,
            offset=offset,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # Get total count (for pagination)
        all_meals = get_user_meals(user_id, limit=10000, start_date=start_dt, end_date=end_dt)
        total_count = len(all_meals)
        
        # Serialize meals with datetime handling
        serialized_meals = []
        for meal in meals:
            meal_dict = meal.model_dump()
            meal_dict['timestamp'] = meal.timestamp.isoformat()
            serialized_meals.append(meal_dict)
        
        return JSONResponse(content={
            "status": "success",
            "user_id": user_id,
            "meals": serialized_meals,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": offset + limit < total_count
            }
        })
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error fetching meal history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch meal history: {str(e)}"
        )


@router.post("/modules/{module_id}/complete", response_model=ModuleCompletionResponse)
async def complete_learning_module(
    module_id: str,
    request: ModuleCompletionRequest,
    current_user_id: str = Depends(get_current_user)
):
    """
    Mark a learning module as completed and award points.
    
    Args:
        module_id: Module identifier
        request: ModuleCompletionRequest with quiz_score
        current_user_id: Authenticated user ID from JWT token (injected by dependency)
        
    Returns:
        ModuleCompletionResponse with points earned and new unlocks
        
    Note:
        This endpoint requires authentication. The user_id is extracted from the JWT token,
        not from the request body, ensuring users can only complete modules for themselves.
        
    Example:
        ```python
        import requests
        
        headers = {"Authorization": "Bearer <your_jwt_token>"}
        completion_data = {
            "quiz_score": 85
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/modules/balanced_nutrition/complete",
            json=completion_data,
            headers=headers
        )
        result = response.json()
        print(f"Points earned: {result['points_earned']}")
        ```
    """
    try:
        # Check if already completed
        completed = get_completed_modules(current_user_id)
        
        if module_id in completed:
            return ModuleCompletionResponse(
                status="already_completed",
                message="Module already completed",
                points_earned=0,
                total_points=get_user_stats(current_user_id).total_points,
                new_modules_unlocked=[]
            )
        
        # Calculate points based on quiz score
        base_points = 50
        bonus_points = int((request.quiz_score - 70) / 10 * 10) if request.quiz_score >= 70 else 0
        points_earned = base_points + bonus_points
        
        # Mark module as completed
        mark_module_completed(current_user_id, module_id, request.quiz_score, points_earned)
        
        # Update user stats
        stats = get_user_stats(current_user_id)
        stats.total_points += points_earned
        stats.completed_modules.append(module_id)
        update_user_stats(stats)
        
        # Determine newly unlocked modules (simple logic for now)
        new_unlocks = []
        if len(stats.completed_modules) >= 3:
            new_unlocks.append("advanced_nutrition")
        
        logger.info(f"Module {module_id} completed by user {current_user_id} - {points_earned} points earned")
        
        return ModuleCompletionResponse(
            status="success",
            message=f"Module completed! You earned {points_earned} points.",
            points_earned=points_earned,
            total_points=stats.total_points,
            new_modules_unlocked=new_unlocks
        )
        
    except Exception as e:
        logger.error(f"Error completing module: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete module: {str(e)}"
        )


# Import timedelta for date calculations
from datetime import timedelta
