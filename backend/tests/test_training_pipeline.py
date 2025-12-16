"""
Property-Based Tests for Model Training Pipeline

This module contains property-based tests using Hypothesis to verify
correctness properties of the training pipeline components.

Each test is tagged with the corresponding property from the design document.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from hypothesis import given, strategies as st, settings
from torchvision import transforms
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_model import TrainingConfig, FoodClassificationTrainer


# ============================================================================
# Property Test 1: Data Augmentation Preservation
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    width=st.integers(min_value=50, max_value=1000),
    height=st.integers(min_value=50, max_value=1000),
    channels=st.just(3)  # RGB images only
)
def test_data_augmentation_preserves_dimensions(width, height, channels):
    """
    **Feature: model-training-pipeline, Property 2: Data augmentation preservation**
    **Validates: Requirements 1.4**
    
    Property: For any training image, applying data augmentation should preserve 
    the image dimensions at 224x224 and maintain valid pixel value ranges [0, 1] 
    after normalization.
    
    This test verifies that:
    1. Output dimensions are always 224x224 regardless of input size
    2. Output has 3 channels (RGB)
    3. Pixel values are in valid range after normalization
    """
    # Create a random image with the given dimensions
    image_array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    image = Image.fromarray(image_array, mode='RGB')
    
    # Get training transforms (with augmentation)
    config = TrainingConfig()
    trainer = FoodClassificationTrainer(config)
    train_transform, _ = trainer.get_data_transforms()
    
    # Apply augmentation
    augmented_tensor = train_transform(image)
    
    # Property 1: Output should be a tensor
    assert isinstance(augmented_tensor, torch.Tensor), \
        "Augmented output should be a PyTorch tensor"
    
    # Property 2: Output dimensions should be [3, 224, 224]
    assert augmented_tensor.shape == (3, 224, 224), \
        f"Expected shape (3, 224, 224), got {augmented_tensor.shape}"
    
    # Property 3: Pixel values should be finite (no NaN or Inf)
    assert torch.isfinite(augmented_tensor).all(), \
        "Augmented tensor contains NaN or Inf values"
    
    # Property 4: After normalization, values should be in a reasonable range
    # ImageNet normalization can produce negative values, but they should be bounded
    # Typical range after normalization is approximately [-3, 3]
    assert augmented_tensor.min() >= -10.0, \
        f"Pixel values too low: {augmented_tensor.min()}"
    assert augmented_tensor.max() <= 10.0, \
        f"Pixel values too high: {augmented_tensor.max()}"


@settings(max_examples=100, deadline=None)
@given(
    width=st.integers(min_value=50, max_value=1000),
    height=st.integers(min_value=50, max_value=1000)
)
def test_validation_preprocessing_preserves_dimensions(width, height):
    """
    **Feature: model-training-pipeline, Property 2: Data augmentation preservation**
    **Validates: Requirements 1.4**
    
    Property: For any validation image, applying preprocessing should preserve 
    the image dimensions at 224x224 and maintain valid pixel value ranges.
    
    This test verifies validation preprocessing (without augmentation) also
    produces consistent output dimensions.
    """
    # Create a random RGB image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode='RGB')
    
    # Get validation transforms (no augmentation)
    config = TrainingConfig()
    trainer = FoodClassificationTrainer(config)
    _, val_transform = trainer.get_data_transforms()
    
    # Apply preprocessing
    preprocessed_tensor = val_transform(image)
    
    # Property 1: Output should be a tensor
    assert isinstance(preprocessed_tensor, torch.Tensor), \
        "Preprocessed output should be a PyTorch tensor"
    
    # Property 2: Output dimensions should be [3, 224, 224]
    assert preprocessed_tensor.shape == (3, 224, 224), \
        f"Expected shape (3, 224, 224), got {preprocessed_tensor.shape}"
    
    # Property 3: Pixel values should be finite
    assert torch.isfinite(preprocessed_tensor).all(), \
        "Preprocessed tensor contains NaN or Inf values"
    
    # Property 4: Values should be in reasonable range after normalization
    assert preprocessed_tensor.min() >= -10.0, \
        f"Pixel values too low: {preprocessed_tensor.min()}"
    assert preprocessed_tensor.max() <= 10.0, \
        f"Pixel values too high: {preprocessed_tensor.max()}"


@settings(max_examples=50, deadline=None)
@given(
    width=st.integers(min_value=100, max_value=500),
    height=st.integers(min_value=100, max_value=500)
)
def test_augmentation_determinism_with_seed(width, height):
    """
    **Feature: model-training-pipeline, Property 2: Data augmentation preservation**
    **Validates: Requirements 1.4**
    
    Property: For any image, applying the same random seed should produce
    deterministic augmentation results.
    
    This verifies that the augmentation pipeline is reproducible when needed.
    """
    # Create a test image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode='RGB')
    
    # Get training transforms
    config = TrainingConfig()
    trainer = FoodClassificationTrainer(config)
    train_transform, _ = trainer.get_data_transforms()
    
    # Apply augmentation twice with same seed
    torch.manual_seed(42)
    result1 = train_transform(image)
    
    torch.manual_seed(42)
    result2 = train_transform(image)
    
    # Property: Results should be identical with same seed
    assert torch.allclose(result1, result2, rtol=1e-5, atol=1e-5), \
        "Augmentation should be deterministic with same random seed"


def test_augmentation_produces_variation():
    """
    **Feature: model-training-pipeline, Property 2: Data augmentation preservation**
    **Validates: Requirements 1.4**
    
    Property: For any image, applying augmentation multiple times without
    fixing the seed should produce different results (verifying randomness).
    
    This ensures augmentation is actually introducing variation.
    """
    # Create a test image
    image_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode='RGB')
    
    # Get training transforms
    config = TrainingConfig()
    trainer = FoodClassificationTrainer(config)
    train_transform, _ = trainer.get_data_transforms()
    
    # Apply augmentation multiple times
    results = []
    for _ in range(5):
        result = train_transform(image)
        results.append(result)
    
    # Property: At least some results should be different
    # (with high probability, random augmentation will produce different outputs)
    all_same = all(torch.allclose(results[0], r, rtol=1e-5, atol=1e-5) for r in results[1:])
    
    assert not all_same, \
        "Augmentation should produce variation across multiple applications"


@settings(max_examples=100, deadline=None)
@given(
    width=st.integers(min_value=50, max_value=1000),
    height=st.integers(min_value=50, max_value=1000)
)
def test_normalization_consistency(width, height):
    """
    **Feature: model-training-pipeline, Property 2: Data augmentation preservation**
    **Validates: Requirements 1.4**
    
    Property: For any image, the normalization parameters should be consistent
    between training and validation transforms (same mean and std).
    
    This ensures models see consistently normalized data.
    """
    # Create a test image
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, mode='RGB')
    
    # Get both transforms
    config = TrainingConfig()
    trainer = FoodClassificationTrainer(config)
    train_transform, val_transform = trainer.get_data_transforms()
    
    # Extract normalization parameters from transforms
    # Both should use ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    # Find Normalize transform in train pipeline
    train_normalize = None
    for t in train_transform.transforms:
        if isinstance(t, transforms.Normalize):
            train_normalize = t
            break
    
    # Find Normalize transform in val pipeline
    val_normalize = None
    for t in val_transform.transforms:
        if isinstance(t, transforms.Normalize):
            val_normalize = t
            break
    
    # Property: Both should have Normalize transform
    assert train_normalize is not None, "Training transform should include Normalize"
    assert val_normalize is not None, "Validation transform should include Normalize"
    
    # Property: Normalization parameters should be identical
    assert train_normalize.mean == val_normalize.mean, \
        "Training and validation should use same normalization mean"
    assert train_normalize.std == val_normalize.std, \
        "Training and validation should use same normalization std"
    
    # Property: Should use ImageNet normalization
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    
    assert list(train_normalize.mean) == expected_mean, \
        f"Expected ImageNet mean {expected_mean}, got {list(train_normalize.mean)}"
    assert list(train_normalize.std) == expected_std, \
        f"Expected ImageNet std {expected_std}, got {list(train_normalize.std)}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
