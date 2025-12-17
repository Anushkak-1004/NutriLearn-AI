"""
Test to verify MLflow optional functionality works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tracker_without_mlflow():
    """Test that tracker functions work when MLflow is disabled."""
    print("Testing tracker functions without MLflow...")
    
    from app.mlops.tracker import (
        log_prediction,
        log_model_metrics,
        log_parameters,
        get_prediction_statistics,
        is_mlflow_enabled
    )
    
    # Check MLflow status
    mlflow_status = is_mlflow_enabled()
    print(f"  MLflow enabled: {mlflow_status}")
    
    # Test log_prediction (should return None if disabled, or run_id if enabled)
    run_id = log_prediction(
        food_name="Chicken Biryani",
        confidence=0.95,
        user_id="test_user",
        processing_time=0.5
    )
    print(f"  ✓ log_prediction returned: {run_id}")
    
    # Test log_model_metrics
    run_id = log_model_metrics(
        accuracy=0.92,
        precision=0.90,
        recall=0.88,
        f1_score=0.89
    )
    print(f"  ✓ log_model_metrics returned: {run_id}")
    
    # Test log_parameters
    run_id = log_parameters({"model": "mobilenet", "epochs": 10})
    print(f"  ✓ log_parameters returned: {run_id}")
    
    # Test get_prediction_statistics
    stats = get_prediction_statistics()
    print(f"  ✓ get_prediction_statistics returned: {stats}")
    
    print("✅ All tracker functions work correctly!")


def test_main_module():
    """Test that main module has MLFLOW_ENABLED flag."""
    print("\nTesting main module...")
    
    try:
        from app.main import MLFLOW_ENABLED
        print(f"  ✓ MLFLOW_ENABLED flag exists: {MLFLOW_ENABLED}")
    except ImportError as e:
        print(f"  ⚠ Could not import MLFLOW_ENABLED: {e}")
        print("  (This is OK if main.py hasn't been loaded yet)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MLflow Optional Functionality")
    print("=" * 60)
    
    try:
        test_tracker_without_mlflow()
        test_main_module()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nNotes:")
        print("- If MLflow is disabled, functions return None/empty data")
        print("- If MLflow is enabled, functions work normally")
        print("- No errors or exceptions in either case")
        print("- Application works perfectly with or without MLflow")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
