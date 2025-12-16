"""
Test script for Supabase integration.
Run this to verify database connection and basic operations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Verify environment variables are loaded
print(f"Loading .env from: {env_path}")
print(f"SUPABASE_URL loaded: {bool(os.getenv('SUPABASE_URL'))}")
print(f"SUPABASE_KEY loaded: {bool(os.getenv('SUPABASE_KEY'))}")
print()

from app.database import (
    test_connection,
    get_database_info,
    add_meal_log,
    get_user_meals,
    get_user_stats,
    update_user_stats,
    mark_module_completed,
    get_completed_modules
)
from app.models import MealLog, NutritionInfo, MealType
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_database_connection():
    """Test 1: Database connection."""
    print_section("TEST 1: Database Connection")
    
    success = test_connection()
    info = get_database_info()
    
    print(f"\nConnection Status: {'✅ SUCCESS' if success else '⚠️  FALLBACK'}")
    print(f"Using Supabase: {info['using_supabase']}")
    print(f"Supabase URL: {info['supabase_url']}")
    print(f"Fallback Storage: {info['fallback_storage']}")
    
    return success


def test_meal_logging():
    """Test 2: Meal logging."""
    print_section("TEST 2: Meal Logging")
    
    try:
        # Create test meal
        nutrition = NutritionInfo(
            calories=450,
            protein=25.0,
            carbs=55.0,
            fat=12.0,
            fiber=3.0
        )
        
        meal = MealLog(
            user_id="test-user-integration",
            food_name="Test Chicken Biryani",
            nutrition=nutrition,
            meal_type=MealType.LUNCH,
            timestamp=datetime.utcnow()
        )
        
        # Log meal
        log_id = add_meal_log(meal)
        print(f"\n✅ Meal logged successfully")
        print(f"   Log ID: {log_id}")
        print(f"   Food: {meal.food_name}")
        print(f"   Calories: {meal.nutrition.calories}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Meal logging failed: {str(e)}")
        return False


def test_meal_retrieval():
    """Test 3: Meal retrieval."""
    print_section("TEST 3: Meal Retrieval")
    
    try:
        # Get meals
        meals = get_user_meals("test-user-integration", limit=10)
        
        print(f"\n✅ Retrieved {len(meals)} meals")
        
        if meals:
            print("\nMost recent meal:")
            meal = meals[0]
            print(f"   Food: {meal.food_name}")
            print(f"   Calories: {meal.nutrition.calories}")
            print(f"   Type: {meal.meal_type}")
            print(f"   Logged: {meal.timestamp}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Meal retrieval failed: {str(e)}")
        return False


def test_user_stats():
    """Test 4: User statistics."""
    print_section("TEST 4: User Statistics")
    
    try:
        # Get stats
        stats = get_user_stats("test-user-integration")
        
        print(f"\n✅ User stats retrieved")
        print(f"   User ID: {stats.user_id}")
        print(f"   Total Meals: {stats.total_meals}")
        print(f"   Total Points: {stats.total_points}")
        print(f"   Current Streak: {stats.current_streak}")
        print(f"   Completed Modules: {len(stats.completed_modules)}")
        
        # Update stats
        stats.total_points += 10
        updated_stats = update_user_stats(stats)
        
        print(f"\n✅ Stats updated")
        print(f"   New Total Points: {updated_stats.total_points}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ User stats test failed: {str(e)}")
        return False


def test_module_completion():
    """Test 5: Module completion."""
    print_section("TEST 5: Module Completion")
    
    try:
        # Mark module as completed
        success = mark_module_completed(
            user_id="test-user-integration",
            module_id="test-module-nutrition",
            quiz_score=85,
            points_earned=60
        )
        
        if success:
            print(f"\n✅ Module marked as completed")
            print(f"   Module ID: test-module-nutrition")
            print(f"   Quiz Score: 85")
            print(f"   Points Earned: 60")
        else:
            print(f"\n⚠️  Module already completed")
        
        # Get completed modules
        completed = get_completed_modules("test-user-integration")
        print(f"\n✅ Retrieved completed modules: {len(completed)}")
        for module_id in completed:
            print(f"   - {module_id}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Module completion test failed: {str(e)}")
        return False


def test_pagination():
    """Test 6: Pagination."""
    print_section("TEST 6: Pagination")
    
    try:
        # Get first page
        page1 = get_user_meals("test-user-integration", limit=2, offset=0)
        print(f"\n✅ Page 1: {len(page1)} meals")
        
        # Get second page
        page2 = get_user_meals("test-user-integration", limit=2, offset=2)
        print(f"✅ Page 2: {len(page2)} meals")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pagination test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  SUPABASE INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Meal Logging", test_meal_logging),
        ("Meal Retrieval", test_meal_retrieval),
        ("User Statistics", test_user_stats),
        ("Module Completion", test_module_completion),
        ("Pagination", test_pagination),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    print()
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Supabase integration is working correctly.")
        print("\nNext steps:")
        print("  1. Check Supabase Table Editor to see the data")
        print("  2. Start the backend: uvicorn app.main:app --reload")
        print("  3. Test the API endpoints")
    else:
        print("\n[WARNING] Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("  1. Verify SUPABASE_URL and SUPABASE_KEY in .env")
        print("  2. Ensure migration SQL has been run")
        print("  3. Check Supabase project is active")
        print("  4. Review logs for error details")
    
    print("\n" + "=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
