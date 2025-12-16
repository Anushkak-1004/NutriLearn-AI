"""
Simple test to verify data augmentation works correctly.
This is a standalone test that doesn't require pytest.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

def test_augmentation_basic():
    """Test that augmentation produces correct output dimensions."""
    print("Testing data augmentation...")
    
    # Create ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test with various image sizes
    test_sizes = [(100, 100), (300, 200), (500, 500), (800, 600)]
    
    for width, height in test_sizes:
        # Create random image
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='RGB')
        
        # Apply augmentation
        augmented = train_transform(image)
        
        # Verify properties
        assert isinstance(augmented, torch.Tensor), "Output should be tensor"
        assert augmented.shape == (3, 224, 224), f"Expected (3, 224, 224), got {augmented.shape}"
        assert torch.isfinite(augmented).all(), "Tensor contains NaN or Inf"
        
        print(f"✓ Image size {width}x{height} -> {augmented.shape} [OK]")
    
    print("\n✅ All augmentation tests passed!")
    return True

def test_validation_preprocessing():
    """Test that validation preprocessing works correctly."""
    print("\nTesting validation preprocessing...")
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test with various image sizes
    test_sizes = [(100, 100), (300, 200), (500, 500), (800, 600)]
    
    for width, height in test_sizes:
        # Create random image
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='RGB')
        
        # Apply preprocessing
        preprocessed = val_transform(image)
        
        # Verify properties
        assert isinstance(preprocessed, torch.Tensor), "Output should be tensor"
        assert preprocessed.shape == (3, 224, 224), f"Expected (3, 224, 224), got {preprocessed.shape}"
        assert torch.isfinite(preprocessed).all(), "Tensor contains NaN or Inf"
        
        print(f"✓ Image size {width}x{height} -> {preprocessed.shape} [OK]")
    
    print("\n✅ All validation preprocessing tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_augmentation_basic()
        test_validation_preprocessing()
        print("\n" + "="*60)
        print("SUCCESS: All data augmentation tests passed!")
        print("="*60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
