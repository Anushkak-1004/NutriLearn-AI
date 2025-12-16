"""
Test script for configuration management.

This script tests the TrainingConfig dataclass and argument parsing.
"""

import sys
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from train_model import TrainingConfig, parse_arguments


def test_default_config():
    """Test creating config with default values."""
    print("Testing default configuration...")
    config = TrainingConfig()
    print("✓ Default config created successfully")
    print(f"  Device: {config.device}")
    print(f"  Model: {config.model_name}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    return config


def test_custom_config():
    """Test creating config with custom values."""
    print("\nTesting custom configuration...")
    config = TrainingConfig(
        model_name="efficientnet_b0",
        epochs=10,
        batch_size=64,
        learning_rate=0.0001,
        device="cpu"
    )
    print("✓ Custom config created successfully")
    print(f"  Device: {config.device}")
    print(f"  Model: {config.model_name}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    return config


def test_device_detection():
    """Test automatic device detection."""
    print("\nTesting device detection...")
    
    # Test auto detection
    config_auto = TrainingConfig(device="auto")
    print(f"✓ Auto device detection: {config_auto.device}")
    
    # Test CPU
    config_cpu = TrainingConfig(device="cpu")
    print(f"✓ CPU device: {config_cpu.device}")
    
    # Test CUDA (will fallback to CPU if not available)
    config_cuda = TrainingConfig(device="cuda")
    print(f"✓ CUDA device (or fallback): {config_cuda.device}")


def test_validation():
    """Test input validation."""
    print("\nTesting input validation...")
    
    # Test invalid model name
    try:
        config = TrainingConfig(model_name="invalid_model")
        print("✗ Should have raised ValueError for invalid model")
    except ValueError as e:
        print(f"✓ Caught invalid model name: {e}")
    
    # Test invalid epochs
    try:
        config = TrainingConfig(epochs=-5)
        print("✗ Should have raised ValueError for negative epochs")
    except ValueError as e:
        print(f"✓ Caught invalid epochs: {e}")
    
    # Test invalid batch size
    try:
        config = TrainingConfig(batch_size=0)
        print("✗ Should have raised ValueError for zero batch size")
    except ValueError as e:
        print(f"✓ Caught invalid batch size: {e}")
    
    # Test invalid learning rate
    try:
        config = TrainingConfig(learning_rate=-0.001)
        print("✗ Should have raised ValueError for negative learning rate")
    except ValueError as e:
        print(f"✓ Caught invalid learning rate: {e}")
    
    # Test invalid train split
    try:
        config = TrainingConfig(train_split=1.5)
        print("✗ Should have raised ValueError for invalid train split")
    except ValueError as e:
        print(f"✓ Caught invalid train split: {e}")
    
    # Test invalid device
    try:
        config = TrainingConfig(device="gpu")
        print("✗ Should have raised ValueError for invalid device")
    except ValueError as e:
        print(f"✓ Caught invalid device: {e}")


def test_save_load():
    """Test saving and loading configuration."""
    print("\nTesting save/load functionality...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config1 = TrainingConfig(
            model_name="resnet50",
            epochs=15,
            batch_size=16,
            learning_rate=0.0005
        )
        
        # Save config
        config_path = Path(tmpdir) / "test_config.json"
        config1.save(str(config_path))
        print(f"✓ Saved config to {config_path}")
        
        # Load config
        config2 = TrainingConfig.from_file(str(config_path))
        print("✓ Loaded config from file")
        
        # Verify values match
        assert config2.model_name == "resnet50"
        assert config2.epochs == 15
        assert config2.batch_size == 16
        assert config2.learning_rate == 0.0005
        print("✓ Loaded config matches saved config")


def test_to_dict():
    """Test converting config to dictionary."""
    print("\nTesting to_dict functionality...")
    config = TrainingConfig(
        model_name="mobilenet_v2",
        epochs=20,
        batch_size=32
    )
    
    config_dict = config.to_dict()
    print("✓ Converted config to dictionary")
    print(f"  Keys: {len(config_dict)}")
    assert "model_name" in config_dict
    assert "epochs" in config_dict
    assert "batch_size" in config_dict
    print("✓ Dictionary contains expected keys")


def test_get_device():
    """Test get_device method."""
    print("\nTesting get_device method...")
    config = TrainingConfig(device="cpu")
    device = config.get_device()
    print(f"✓ get_device() returned: {device}")
    assert str(device) == "cpu"
    print("✓ Device object is correct")


def test_directory_creation():
    """Test automatic directory creation."""
    print("\nTesting directory creation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "models"
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        
        config = TrainingConfig(
            output_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir)
        )
        
        assert output_dir.exists()
        assert checkpoint_dir.exists()
        print("✓ Directories created automatically")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Configuration Management Tests")
    print("=" * 60)
    
    try:
        test_default_config()
        test_custom_config()
        test_device_detection()
        test_validation()
        test_save_load()
        test_to_dict()
        test_get_device()
        test_directory_creation()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
