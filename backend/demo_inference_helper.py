"""
Demonstration script for InferenceHelper component.

This script shows how to use the InferenceHelper class for food image classification.
It demonstrates initialization, model loading, and making predictions.

Note: This is a demonstration script. To actually run it, you need:
1. A trained model file (food_model_v1.pth)
2. Model configuration (model_config.json)
3. Class mapping (class_to_idx.json)
4. A test image

Usage:
    python demo_inference_helper.py --image path/to/food_image.jpg
"""

import argparse
import sys
from pathlib import Path

# Import InferenceHelper
from train_model import InferenceHelper, InvalidImageError, ModelNotFoundError


def demo_basic_usage():
    """Demonstrate basic usage of InferenceHelper."""
    
    print("=" * 80)
    print("InferenceHelper Basic Usage Demo")
    print("=" * 80)
    
    # Step 1: Initialize InferenceHelper
    print("\n1. Initializing InferenceHelper...")
    print("   This validates that all required files exist.")
    
    try:
        helper = InferenceHelper(
            model_path="ml-models/food_model_v1.pth",
            config_path="ml-models/model_config.json",
            class_mapping_path="ml-models/class_to_idx.json"
        )
        print("   ✓ InferenceHelper initialized successfully")
    except ModelNotFoundError as e:
        print(f"   ✗ Model file not found: {e}")
        print("   → Train a model first using train_model.py")
        return
    except FileNotFoundError as e:
        print(f"   ✗ Configuration file not found: {e}")
        print("   → Ensure all model artifacts are present")
        return
    
    # Step 2: Load model
    print("\n2. Loading model and configuration...")
    print("   This loads the model weights, config, and class mappings.")
    
    try:
        helper.load_model()
        print("   ✓ Model loaded successfully")
        print(f"   → Device: {helper.device}")
        print(f"   → Number of classes: {len(helper.idx_to_class)}")
        print(f"   → Model architecture: {helper.config['model_name']}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return
    
    # Step 3: Make prediction
    print("\n3. Making prediction on test image...")
    
    # Check if test image exists
    test_image = "test_food_image.jpg"
    if not Path(test_image).exists():
        print(f"   ✗ Test image not found: {test_image}")
        print("   → Place a food image in the current directory")
        print("   → Or specify image path with --image argument")
        return
    
    try:
        predictions = helper.predict(test_image, top_k=3)
        
        print("   ✓ Prediction successful!")
        print("\n   Top 3 Predictions:")
        print("   " + "-" * 60)
        
        for i, (class_name, prob) in enumerate(predictions, 1):
            # Format class name (replace underscores with spaces, title case)
            formatted_name = class_name.replace('_', ' ').title()
            
            # Create confidence bar
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print(f"   {i}. {formatted_name:30s} {prob*100:5.2f}% {bar}")
        
        print("   " + "-" * 60)
        
    except InvalidImageError as e:
        print(f"   ✗ Invalid image: {e}")
        print("   → Ensure the image is a valid format (JPG, PNG)")
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
    
    print("\n" + "=" * 80)


def demo_batch_prediction():
    """Demonstrate batch prediction on multiple images."""
    
    print("=" * 80)
    print("InferenceHelper Batch Prediction Demo")
    print("=" * 80)
    
    # Initialize and load model
    try:
        helper = InferenceHelper(
            model_path="ml-models/food_model_v1.pth",
            config_path="ml-models/model_config.json",
            class_mapping_path="ml-models/class_to_idx.json"
        )
        helper.load_model()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # List of test images
    test_images = [
        "test_image_1.jpg",
        "test_image_2.jpg",
        "test_image_3.jpg"
    ]
    
    # Filter to only existing images
    existing_images = [img for img in test_images if Path(img).exists()]
    
    if not existing_images:
        print("No test images found. Place some food images in the current directory.")
        return
    
    print(f"\nPredicting for {len(existing_images)} images...\n")
    
    # Make batch predictions
    all_predictions = helper.predict_batch(existing_images, top_k=3)
    
    # Display results
    for image_path, predictions in zip(existing_images, all_predictions):
        print(f"\n{image_path}:")
        if predictions:
            for i, (class_name, prob) in enumerate(predictions, 1):
                formatted_name = class_name.replace('_', ' ').title()
                print(f"  {i}. {formatted_name}: {prob*100:.2f}%")
        else:
            print("  (prediction failed)")
    
    print("\n" + "=" * 80)


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    
    print("=" * 80)
    print("InferenceHelper Error Handling Demo")
    print("=" * 80)
    
    # Test 1: Missing model file
    print("\n1. Testing with missing model file...")
    try:
        helper = InferenceHelper(
            model_path="nonexistent_model.pth",
            config_path="ml-models/model_config.json",
            class_mapping_path="ml-models/class_to_idx.json"
        )
        print("   ✗ Should have raised ModelNotFoundError")
    except ModelNotFoundError as e:
        print(f"   ✓ Correctly raised ModelNotFoundError: {e}")
    
    # Test 2: Missing config file
    print("\n2. Testing with missing config file...")
    try:
        # Create a dummy model file
        Path("temp_model.pth").touch()
        
        helper = InferenceHelper(
            model_path="temp_model.pth",
            config_path="nonexistent_config.json",
            class_mapping_path="ml-models/class_to_idx.json"
        )
        print("   ✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"   ✓ Correctly raised FileNotFoundError: {e}")
    finally:
        # Clean up
        if Path("temp_model.pth").exists():
            Path("temp_model.pth").unlink()
    
    # Test 3: Invalid image
    print("\n3. Testing with invalid image...")
    try:
        helper = InferenceHelper(
            model_path="ml-models/food_model_v1.pth",
            config_path="ml-models/model_config.json",
            class_mapping_path="ml-models/class_to_idx.json"
        )
        helper.load_model()
        helper.predict("nonexistent_image.jpg")
        print("   ✗ Should have raised InvalidImageError")
    except InvalidImageError as e:
        print(f"   ✓ Correctly raised InvalidImageError: {e}")
    except Exception as e:
        print(f"   (Skipped - model not available: {e})")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point for demo script."""
    
    parser = argparse.ArgumentParser(
        description="Demonstrate InferenceHelper functionality"
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to test image for prediction'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['basic', 'batch', 'errors', 'all'],
        default='basic',
        help='Which demo to run (default: basic)'
    )
    
    args = parser.parse_args()
    
    # Update test image if provided
    if args.image:
        global test_image
        test_image = args.image
    
    # Run selected demo
    if args.demo == 'basic' or args.demo == 'all':
        demo_basic_usage()
    
    if args.demo == 'batch' or args.demo == 'all':
        print("\n")
        demo_batch_prediction()
    
    if args.demo == 'errors' or args.demo == 'all':
        print("\n")
        demo_error_handling()


if __name__ == "__main__":
    main()
