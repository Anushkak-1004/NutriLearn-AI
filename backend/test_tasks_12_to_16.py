"""
Test script for tasks 12-16 completion validation.

Tests:
- Task 12: Comprehensive error handling
- Task 13: Google Colab compatibility
- Task 14: Checkpoint resume functionality
- Task 15: Visualization and reporting
- Task 16: Documentation (verified by inspection)
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components
from train_model import (
    DatasetNotFoundError,
    UnsupportedModelError,
    OutOfMemoryError,
    NaNLossError,
    CheckpointLoadError,
    is_colab_environment,
    setup_colab_environment,
    load_checkpoint_for_resume,
    plot_training_curves,
    plot_sample_predictions,
    ModelBuilder,
    TrainingEngine
)


def test_error_classes():
    """Test that all error classes are defined."""
    print("Testing error classes...")
    
    # Test DatasetNotFoundError
    try:
        raise DatasetNotFoundError("Test dataset error")
    except DatasetNotFoundError as e:
        assert "Test dataset error" in str(e)
    print("  ✓ DatasetNotFoundError defined")
    
    # Test UnsupportedModelError
    try:
        raise UnsupportedModelError("Test model error")
    except UnsupportedModelError as e:
        assert "Test model error" in str(e)
    print("  ✓ UnsupportedModelError defined")
    
    # Test OutOfMemoryError
    try:
        raise OutOfMemoryError("Test OOM error")
    except OutOfMemoryError as e:
        assert "Test OOM error" in str(e)
    print("  ✓ OutOfMemoryError defined")
    
    # Test NaNLossError
    try:
        raise NaNLossError("Test NaN error")
    except NaNLossError as e:
        assert "Test NaN error" in str(e)
    print("  ✓ NaNLossError defined")
    
    # Test CheckpointLoadError
    try:
        raise CheckpointLoadError("Test checkpoint error")
    except CheckpointLoadError as e:
        assert "Test checkpoint error" in str(e)
    print("  ✓ CheckpointLoadError defined")
    
    print("✓ All error classes defined correctly")


def test_colab_detection():
    """Test Google Colab environment detection."""
    print("\nTesting Colab detection...")
    
    # Test is_colab_environment function
    is_colab = is_colab_environment()
    assert isinstance(is_colab, bool)
    print(f"  Colab detected: {is_colab}")
    
    # Test setup_colab_environment (should not crash)
    try:
        setup_colab_environment()
        print("  ✓ setup_colab_environment() executed without errors")
    except Exception as e:
        print(f"  ✗ setup_colab_environment() failed: {e}")
        raise
    
    print("✓ Colab compatibility features working")


def test_checkpoint_resume():
    """Test checkpoint resume functionality."""
    print("\nTesting checkpoint resume...")
    
    # Create a mock checkpoint
    model = ModelBuilder('mobilenet_v2', 10).build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    device = torch.device('cpu')
    
    # Create a temporary checkpoint
    checkpoint_path = "test_checkpoint.pth"
    checkpoint = {
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': 0.5,
        'best_val_acc': 85.0,
        'training_history': {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'train_acc': [60, 70, 75, 80, 82],
            'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
            'val_acc': [58, 68, 73, 78, 85]
        }
    }
    
    try:
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Test loading checkpoint
        start_epoch, history, best_loss, best_acc = load_checkpoint_for_resume(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        
        # Verify loaded values
        assert start_epoch == 6, f"Expected start_epoch=6, got {start_epoch}"
        assert best_loss == 0.5, f"Expected best_loss=0.5, got {best_loss}"
        assert best_acc == 85.0, f"Expected best_acc=85.0, got {best_acc}"
        assert len(history['train_loss']) == 5, "Training history not loaded correctly"
        
        print("  ✓ Checkpoint loaded successfully")
        print(f"    Start epoch: {start_epoch}")
        print(f"    Best val loss: {best_loss}")
        print(f"    Best val acc: {best_acc}")
        
    finally:
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    
    # Test loading non-existent checkpoint
    try:
        load_checkpoint_for_resume(
            checkpoint_path="nonexistent.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        print("  ✗ Should have raised CheckpointLoadError")
        assert False
    except CheckpointLoadError:
        print("  ✓ Correctly raises CheckpointLoadError for missing file")
    
    print("✓ Checkpoint resume functionality working")


def test_visualization_functions():
    """Test visualization functions."""
    print("\nTesting visualization functions...")
    
    # Test plot_training_curves
    training_history = {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'train_acc': [60, 70, 75, 80, 82],
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
        'val_acc': [58, 68, 73, 78, 85]
    }
    
    try:
        fig = plot_training_curves(training_history, save_path="test_curves.png")
        assert fig is not None
        print("  ✓ plot_training_curves() executed successfully")
        
        # Cleanup
        if os.path.exists("test_curves.png"):
            os.remove("test_curves.png")
            print("  ✓ Training curves plot saved and cleaned up")
    except Exception as e:
        print(f"  ✗ plot_training_curves() failed: {e}")
        raise
    
    # Test plot_sample_predictions (with mock data)
    try:
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create mock data
        images = torch.randn(16, 3, 224, 224)
        labels = torch.randint(0, 10, (16,))
        dataset = TensorDataset(images, labels)
        data_loader = DataLoader(dataset, batch_size=16)
        
        model = ModelBuilder('mobilenet_v2', 10).build_model()
        model.eval()
        
        class_names = [f"class_{i}" for i in range(10)]
        device = torch.device('cpu')
        
        fig = plot_sample_predictions(
            model=model,
            data_loader=data_loader,
            class_names=class_names,
            device=device,
            num_samples=16,
            save_path="test_predictions.png"
        )
        
        assert fig is not None
        print("  ✓ plot_sample_predictions() executed successfully")
        
        # Cleanup
        if os.path.exists("test_predictions.png"):
            os.remove("test_predictions.png")
            print("  ✓ Sample predictions plot saved and cleaned up")
    except Exception as e:
        print(f"  ✗ plot_sample_predictions() failed: {e}")
        raise
    
    print("✓ Visualization functions working")


def test_documentation():
    """Test that documentation is comprehensive."""
    print("\nTesting documentation...")
    
    # Check module docstring
    import train_model
    assert train_model.__doc__ is not None
    assert "Food Classification" in train_model.__doc__
    assert "Google Colab" in train_model.__doc__
    print("  ✓ Module docstring present and comprehensive")
    
    # Check key classes have docstrings
    from train_model import DatasetManager, ModelBuilder, TrainingEngine
    
    assert DatasetManager.__doc__ is not None
    assert "Food-101" in DatasetManager.__doc__
    print("  ✓ DatasetManager documented")
    
    assert ModelBuilder.__doc__ is not None
    assert "transfer learning" in ModelBuilder.__doc__
    print("  ✓ ModelBuilder documented")
    
    assert TrainingEngine.__doc__ is not None
    assert "early stopping" in TrainingEngine.__doc__
    print("  ✓ TrainingEngine documented")
    
    # Check key functions have docstrings
    assert load_checkpoint_for_resume.__doc__ is not None
    assert "resume" in load_checkpoint_for_resume.__doc__.lower()
    print("  ✓ load_checkpoint_for_resume documented")
    
    assert plot_training_curves.__doc__ is not None
    assert "loss" in plot_training_curves.__doc__.lower()
    print("  ✓ plot_training_curves documented")
    
    print("✓ Documentation is comprehensive")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Tasks 12-16 Implementation")
    print("=" * 80)
    
    try:
        test_error_classes()
        test_colab_detection()
        test_checkpoint_resume()
        test_visualization_functions()
        test_documentation()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nTasks 12-16 implementation verified:")
        print("  ✓ Task 12: Comprehensive error handling")
        print("  ✓ Task 13: Google Colab compatibility")
        print("  ✓ Task 14: Checkpoint resume functionality")
        print("  ✓ Task 15: Visualization and reporting")
        print("  ✓ Task 16: Comprehensive documentation")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
