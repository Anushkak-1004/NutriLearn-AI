"""
Utility Functions for NutriLearn AI
Helper functions for analysis, recommendations, and calculations.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .models import (
    MealLog, NutritionInfo, DietaryPattern, LearningModule,
    Severity, MealType
)

logger = logging.getLogger(__name__)


def calculate_nutrition_totals(meals: List[MealLog]) -> NutritionInfo:
    """
    Calculate total nutrition across multiple meals.
    
    Args:
        meals: List of MealLog objects
        
    Returns:
        NutritionInfo object with summed values
        
    Example:
        >>> meals = [meal1, meal2, meal3]
        >>> totals = calculate_nutrition_totals(meals)
        >>> print(f"Total calories: {totals.calories}")
    """
    if not meals:
        return NutritionInfo(calories=0, protein=0.0, carbs=0.0, fat=0.0, fiber=0.0)
    
    total_calories = sum(meal.nutrition.calories for meal in meals)
    total_protein = sum(meal.nutrition.protein for meal in meals)
    total_carbs = sum(meal.nutrition.carbs for meal in meals)
    total_fat = sum(meal.nutrition.fat for meal in meals)
    total_fiber = sum(meal.nutrition.fiber for meal in meals)
    
    return NutritionInfo(
        calories=total_calories,
        protein=round(total_protein, 1),
        carbs=round(total_carbs, 1),
        fat=round(total_fat, 1),
        fiber=round(total_fiber, 1)
    )


def analyze_dietary_patterns(meals: List[MealLog]) -> List[DietaryPattern]:
    """
    Analyze meal logs to identify dietary patterns and issues.
    
    This function examines nutrition data across meals to detect:
    - High carbohydrate intake
    - Low protein intake
    - High fat intake
    - Low fiber intake
    - Irregular meal timing
    - Excessive calorie intake
    
    Args:
        meals: List of MealLog objects to analyze
        
    Returns:
        List of DietaryPattern objects with identified issues
        
    Example:
        >>> patterns = analyze_dietary_patterns(user_meals)
        >>> for pattern in patterns:
        ...     print(f"{pattern.description} - {pattern.severity}")
    """
    if not meals:
        logger.info("No meals to analyze")
        return []
    
    patterns = []
    
    # Calculate daily averages
    days = max(1, len(set(meal.timestamp.date() for meal in meals)))
    totals = calculate_nutrition_totals(meals)
    
    daily_avg_calories = totals.calories / days
    daily_avg_protein = totals.protein / days
    daily_avg_carbs = totals.carbs / days
    daily_avg_fat = totals.fat / days
    daily_avg_fiber = totals.fiber / days
    
    # Recommended daily values (approximate for average adult)
    RECOMMENDED_CALORIES = 2000
    RECOMMENDED_PROTEIN = 50  # grams
    RECOMMENDED_CARBS = 275  # grams
    RECOMMENDED_FAT = 78  # grams
    RECOMMENDED_FIBER = 25  # grams
    
    # Pattern 1: High Carbohydrate Intake
    if daily_avg_carbs > RECOMMENDED_CARBS * 1.3:
        carb_meals = sum(1 for meal in meals if meal.nutrition.carbs > 60)
        severity = Severity.HIGH if daily_avg_carbs > RECOMMENDED_CARBS * 1.5 else Severity.MEDIUM
        
        patterns.append(DietaryPattern(
            pattern_id="high_carb_intake",
            description=f"Your carbohydrate intake is {int((daily_avg_carbs/RECOMMENDED_CARBS - 1) * 100)}% above recommended levels",
            severity=severity,
            recommendation="Try incorporating more protein-rich foods like lean meats, fish, eggs, and legumes. Replace refined carbs with whole grains.",
            affected_meals=carb_meals
        ))
    
    # Pattern 2: Low Protein Intake
    if daily_avg_protein < RECOMMENDED_PROTEIN * 0.7:
        protein_deficit = int((1 - daily_avg_protein/RECOMMENDED_PROTEIN) * 100)
        severity = Severity.HIGH if daily_avg_protein < RECOMMENDED_PROTEIN * 0.5 else Severity.MEDIUM
        
        patterns.append(DietaryPattern(
            pattern_id="low_protein_intake",
            description=f"Your protein intake is {protein_deficit}% below recommended levels",
            severity=severity,
            recommendation="Add protein sources to each meal: chicken, fish, tofu, lentils, Greek yogurt, or protein shakes.",
            affected_meals=len(meals)
        ))
    
    # Pattern 3: High Fat Intake
    if daily_avg_fat > RECOMMENDED_FAT * 1.3:
        fat_meals = sum(1 for meal in meals if meal.nutrition.fat > 25)
        severity = Severity.MEDIUM if daily_avg_fat < RECOMMENDED_FAT * 1.5 else Severity.HIGH
        
        patterns.append(DietaryPattern(
            pattern_id="high_fat_intake",
            description=f"Your fat intake is {int((daily_avg_fat/RECOMMENDED_FAT - 1) * 100)}% above recommended levels",
            severity=severity,
            recommendation="Choose lean proteins, use cooking methods like grilling or baking, and limit fried foods. Focus on healthy fats from nuts and fish.",
            affected_meals=fat_meals
        ))
    
    # Pattern 4: Low Fiber Intake
    if daily_avg_fiber < RECOMMENDED_FIBER * 0.6:
        fiber_deficit = int((1 - daily_avg_fiber/RECOMMENDED_FIBER) * 100)
        severity = Severity.MEDIUM
        
        patterns.append(DietaryPattern(
            pattern_id="low_fiber_intake",
            description=f"Your fiber intake is {fiber_deficit}% below recommended levels",
            severity=severity,
            recommendation="Increase vegetables, fruits, whole grains, and legumes in your diet. Aim for at least 5 servings of fruits and vegetables daily.",
            affected_meals=len(meals)
        ))
    
    # Pattern 5: Excessive Calorie Intake
    if daily_avg_calories > RECOMMENDED_CALORIES * 1.2:
        calorie_excess = int(daily_avg_calories - RECOMMENDED_CALORIES)
        severity = Severity.HIGH if daily_avg_calories > RECOMMENDED_CALORIES * 1.4 else Severity.MEDIUM
        
        patterns.append(DietaryPattern(
            pattern_id="excessive_calories",
            description=f"Your daily calorie intake exceeds recommendations by {calorie_excess} calories",
            severity=severity,
            recommendation="Focus on portion control, choose lower-calorie alternatives, and increase physical activity. Consider tracking portion sizes.",
            affected_meals=len(meals)
        ))
    
    # Pattern 6: Irregular Meal Timing
    meal_times_by_type = defaultdict(list)
    for meal in meals:
        meal_times_by_type[meal.meal_type].append(meal.timestamp.hour)
    
    irregular_count = 0
    for meal_type, times in meal_times_by_type.items():
        if len(times) > 1:
            time_variance = max(times) - min(times)
            if time_variance > 3:  # More than 3 hours variance
                irregular_count += 1
    
    if irregular_count >= 2:
        patterns.append(DietaryPattern(
            pattern_id="irregular_meal_timing",
            description="Your meal times vary significantly from day to day",
            severity=Severity.LOW,
            recommendation="Try to maintain consistent meal times. Regular eating patterns help regulate metabolism and energy levels.",
            affected_meals=len(meals)
        ))
    
    logger.info(f"Identified {len(patterns)} dietary patterns")
    return patterns


def generate_learning_recommendations(
    patterns: List[DietaryPattern],
    completed_modules: List[str]
) -> List[LearningModule]:
    """
    Generate personalized learning module recommendations based on dietary patterns.
    
    Args:
        patterns: List of identified dietary patterns
        completed_modules: List of module IDs already completed by user
        
    Returns:
        List of recommended LearningModule objects
        
    Example:
        >>> patterns = analyze_dietary_patterns(meals)
        >>> modules = generate_learning_recommendations(patterns, ["nutrition_basics"])
        >>> for module in modules:
        ...     print(f"Recommended: {module.title}")
    """
    # Module database
    ALL_MODULES = {
        "balanced_nutrition": LearningModule(
            module_id="balanced_nutrition",
            title="Understanding Balanced Nutrition",
            reason="Learn the fundamentals of a balanced diet",
            content=[
                {
                    "type": "text",
                    "data": "A balanced diet includes the right proportions of proteins, carbohydrates, and fats..."
                },
                {
                    "type": "infographic",
                    "data": "Macronutrient distribution: 45-65% carbs, 20-35% fat, 10-35% protein"
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "What percentage of daily calories should come from protein?",
                        "options": ["5-10%", "10-35%", "40-50%", "60-70%"],
                        "correct": 1
                    },
                    {
                        "question": "Which macronutrient provides 4 calories per gram?",
                        "options": ["Fat", "Protein and Carbs", "Fiber", "Water"],
                        "correct": 1
                    }
                ]
            },
            points=50,
            estimated_time=10
        ),
        "protein_power": LearningModule(
            module_id="protein_power",
            title="The Power of Protein",
            reason="Increase your protein intake for better health",
            content=[
                {
                    "type": "text",
                    "data": "Protein is essential for muscle building, repair, and overall health..."
                },
                {
                    "type": "list",
                    "data": ["Lean meats", "Fish", "Eggs", "Legumes", "Dairy", "Tofu"]
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "How much protein should an average adult consume daily?",
                        "options": ["20-30g", "50-60g", "100-120g", "150-200g"],
                        "correct": 1
                    }
                ]
            },
            points=50,
            estimated_time=8
        ),
        "carb_smart": LearningModule(
            module_id="carb_smart",
            title="Smart Carbohydrate Choices",
            reason="Learn to choose healthier carbohydrate sources",
            content=[
                {
                    "type": "text",
                    "data": "Not all carbs are created equal. Complex carbs provide sustained energy..."
                },
                {
                    "type": "comparison",
                    "data": {
                        "good": ["Whole grains", "Vegetables", "Legumes"],
                        "limit": ["White bread", "Sugary drinks", "Pastries"]
                    }
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "Which is a complex carbohydrate?",
                        "options": ["White sugar", "Brown rice", "Candy", "Soda"],
                        "correct": 1
                    }
                ]
            },
            points=50,
            estimated_time=10
        ),
        "healthy_fats": LearningModule(
            module_id="healthy_fats",
            title="Understanding Healthy Fats",
            reason="Learn about good vs. bad fats",
            content=[
                {
                    "type": "text",
                    "data": "Fats are essential for health, but choosing the right types matters..."
                },
                {
                    "type": "list",
                    "data": ["Omega-3 from fish", "Nuts and seeds", "Avocados", "Olive oil"]
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "Which fat is considered healthiest?",
                        "options": ["Trans fat", "Saturated fat", "Unsaturated fat", "Hydrogenated fat"],
                        "correct": 2
                    }
                ]
            },
            points=50,
            estimated_time=9
        ),
        "fiber_focus": LearningModule(
            module_id="fiber_focus",
            title="The Importance of Fiber",
            reason="Boost your fiber intake for digestive health",
            content=[
                {
                    "type": "text",
                    "data": "Fiber aids digestion, helps control blood sugar, and promotes satiety..."
                },
                {
                    "type": "list",
                    "data": ["Vegetables", "Fruits", "Whole grains", "Legumes", "Nuts"]
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "What is the recommended daily fiber intake?",
                        "options": ["10-15g", "25-30g", "50-60g", "80-100g"],
                        "correct": 1
                    }
                ]
            },
            points=50,
            estimated_time=8
        ),
        "portion_control": LearningModule(
            module_id="portion_control",
            title="Mastering Portion Control",
            reason="Learn to manage portion sizes effectively",
            content=[
                {
                    "type": "text",
                    "data": "Portion control is key to maintaining a healthy weight..."
                },
                {
                    "type": "tips",
                    "data": ["Use smaller plates", "Measure servings", "Eat slowly", "Stop when 80% full"]
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "A serving of protein should be about the size of:",
                        "options": ["Your whole hand", "Your palm", "Your fist", "Two fists"],
                        "correct": 1
                    }
                ]
            },
            points=50,
            estimated_time=10
        ),
        "meal_timing": LearningModule(
            module_id="meal_timing",
            title="Optimal Meal Timing",
            reason="Establish regular eating patterns",
            content=[
                {
                    "type": "text",
                    "data": "Regular meal timing helps regulate metabolism and energy levels..."
                },
                {
                    "type": "schedule",
                    "data": {
                        "breakfast": "7-9 AM",
                        "lunch": "12-2 PM",
                        "dinner": "6-8 PM"
                    }
                }
            ],
            quiz={
                "questions": [
                    {
                        "question": "How many hours should you wait between meals?",
                        "options": ["1-2 hours", "3-4 hours", "6-7 hours", "8-10 hours"],
                        "correct": 1
                    }
                ]
            },
            points=50,
            estimated_time=7
        )
    }
    
    # Map patterns to relevant modules
    pattern_to_modules = {
        "high_carb_intake": ["carb_smart", "balanced_nutrition"],
        "low_protein_intake": ["protein_power", "balanced_nutrition"],
        "high_fat_intake": ["healthy_fats", "balanced_nutrition"],
        "low_fiber_intake": ["fiber_focus", "balanced_nutrition"],
        "excessive_calories": ["portion_control", "balanced_nutrition"],
        "irregular_meal_timing": ["meal_timing"]
    }
    
    # Collect recommended module IDs
    recommended_ids = set()
    for pattern in patterns:
        if pattern.pattern_id in pattern_to_modules:
            for module_id in pattern_to_modules[pattern.pattern_id]:
                if module_id not in completed_modules:
                    recommended_ids.add(module_id)
    
    # Always recommend balanced_nutrition if not completed
    if "balanced_nutrition" not in completed_modules and not patterns:
        recommended_ids.add("balanced_nutrition")
    
    # Build module list with reasons
    recommended_modules = []
    for module_id in recommended_ids:
        if module_id in ALL_MODULES:
            module = ALL_MODULES[module_id]
            # Customize reason based on patterns
            for pattern in patterns:
                if pattern.pattern_id in pattern_to_modules and module_id in pattern_to_modules[pattern.pattern_id]:
                    module.reason = f"Based on your {pattern.description.lower()}"
                    break
            recommended_modules.append(module)
    
    # Limit to top 3 recommendations
    recommended_modules = recommended_modules[:3]
    
    logger.info(f"Generated {len(recommended_modules)} learning recommendations")
    return recommended_modules


def get_nutrition_summary(meals: List[MealLog]) -> Dict[str, Any]:
    """
    Generate a comprehensive nutrition summary from meal logs.
    
    Args:
        meals: List of MealLog objects
        
    Returns:
        Dictionary with nutrition summary statistics
    """
    if not meals:
        return {
            "total_meals": 0,
            "date_range": None,
            "daily_average": None,
            "meal_type_distribution": {}
        }
    
    totals = calculate_nutrition_totals(meals)
    days = max(1, len(set(meal.timestamp.date() for meal in meals)))
    
    # Meal type distribution
    meal_type_counts = defaultdict(int)
    for meal in meals:
        meal_type_counts[meal.meal_type.value] += 1
    
    # Date range
    sorted_meals = sorted(meals, key=lambda x: x.timestamp)
    date_range = {
        "start": sorted_meals[0].timestamp.isoformat(),
        "end": sorted_meals[-1].timestamp.isoformat()
    }
    
    return {
        "total_meals": len(meals),
        "date_range": date_range,
        "total_nutrition": totals.model_dump(),
        "daily_average": {
            "calories": round(totals.calories / days, 1),
            "protein": round(totals.protein / days, 1),
            "carbs": round(totals.carbs / days, 1),
            "fat": round(totals.fat / days, 1),
            "fiber": round(totals.fiber / days, 1)
        },
        "meal_type_distribution": dict(meal_type_counts),
        "days_tracked": days
    }
