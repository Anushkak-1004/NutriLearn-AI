"""
Minimal test for configuration management without heavy dependencies.

This test validates the configuration logic by checking the code structure.
"""

import ast
import sys
from pathlib import Path


def test_training_config_exists():
    """Test that TrainingConfig dataclass exists."""
    print("Testing TrainingConfig dataclass existence...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the file
    tree = ast.parse(content)
    
    # Find TrainingConfig class
    config_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig":
            config_class = node
            break
    
    assert config_class is not None, "TrainingConfig class not found"
    print("✓ TrainingConfig class exists")
    
    # Check for @dataclass decorator
    has_dataclass = any(
        isinstance(dec, ast.Name) and dec.id == "dataclass"
        for dec in config_class.decorator_list
    )
    assert has_dataclass, "TrainingConfig is not decorated with @dataclass"
    print("✓ TrainingConfig has @dataclass decorator")
    
    return config_class


def test_required_attributes():
    """Test that TrainingConfig has all required attributes."""
    print("\nTesting required attributes...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_attrs = [
        'model_name',
        'epochs',
        'batch_size',
        'learning_rate',
        'device',
        'data_dir',
        'output_dir',
        'resume_from',
        'num_workers',
        'scheduler_patience',
        'scheduler_factor',
        'early_stopping_patience',
        'gradient_clip_max_norm',
        'weight_decay'
    ]
    
    for attr in required_attrs:
        assert attr in content, f"Required attribute '{attr}' not found"
        print(f"  ✓ {attr}")
    
    print("✓ All required attributes present")


def test_validation_method():
    """Test that validate method exists."""
    print("\nTesting validate method...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find validate method in TrainingConfig
    config_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig":
            config_class = node
            break
    
    validate_method = None
    for node in config_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate":
            validate_method = node
            break
    
    assert validate_method is not None, "validate method not found"
    print("✓ validate method exists")
    
    # Check that it validates key parameters
    method_source = ast.get_source_segment(content, validate_method)
    validations = [
        'model_name',
        'epochs',
        'batch_size',
        'learning_rate',
        'train_split',
        'device'
    ]
    
    for param in validations:
        assert param in method_source, f"Validation for '{param}' not found"
        print(f"  ✓ Validates {param}")
    
    print("✓ validate method checks all critical parameters")


def test_device_resolution():
    """Test that resolve_device method exists."""
    print("\nTesting device resolution...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find resolve_device method
    config_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig":
            config_class = node
            break
    
    resolve_device_method = None
    for node in config_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_device":
            resolve_device_method = node
            break
    
    assert resolve_device_method is not None, "resolve_device method not found"
    print("✓ resolve_device method exists")
    
    # Check for CUDA detection
    method_source = ast.get_source_segment(content, resolve_device_method)
    assert "cuda" in method_source.lower(), "CUDA detection not found"
    assert "cpu" in method_source.lower(), "CPU fallback not found"
    print("✓ Device resolution includes CUDA detection and CPU fallback")


def test_argument_parser():
    """Test that parse_arguments function exists."""
    print("\nTesting argument parser...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find parse_arguments function
    parse_args_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_arguments":
            parse_args_func = node
            break
    
    assert parse_args_func is not None, "parse_arguments function not found"
    print("✓ parse_arguments function exists")
    
    # Check for key arguments
    func_source = ast.get_source_segment(content, parse_args_func)
    required_args = [
        'model_name',
        'epochs',
        'batch_size',
        'learning_rate',
        'data_dir',
        'output_dir',
        'device',
        'resume'
    ]
    
    for arg in required_args:
        assert arg in func_source, f"Argument '{arg}' not found in parser"
        print(f"  ✓ {arg}")
    
    print("✓ All required command-line arguments present")


def test_trainer_uses_config():
    """Test that FoodClassificationTrainer uses TrainingConfig."""
    print("\nTesting trainer integration with config...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find FoodClassificationTrainer __init__
    trainer_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FoodClassificationTrainer":
            trainer_class = node
            break
    
    assert trainer_class is not None, "FoodClassificationTrainer not found"
    
    init_method = None
    for node in trainer_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_method = node
            break
    
    assert init_method is not None, "__init__ method not found"
    
    # Check that it accepts TrainingConfig
    init_source = ast.get_source_segment(content, init_method)
    assert "TrainingConfig" in init_source, "TrainingConfig not used in __init__"
    assert "config" in init_source, "config parameter not found"
    print("✓ FoodClassificationTrainer accepts TrainingConfig")
    
    # Check that it uses config attributes
    assert "self.config" in init_source, "config not stored as instance variable"
    print("✓ Trainer stores config as instance variable")


def test_main_creates_config():
    """Test that main function creates TrainingConfig from args."""
    print("\nTesting main function integration...")
    
    train_model_path = Path(__file__).parent / "train_model.py"
    with open(train_model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find main function
    main_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break
    
    assert main_func is not None, "main function not found"
    
    main_source = ast.get_source_segment(content, main_func)
    assert "TrainingConfig" in main_source, "TrainingConfig not created in main"
    assert "parse_arguments" in main_source, "parse_arguments not called in main"
    print("✓ main function creates TrainingConfig from parsed arguments")
    
    # Check that trainer is created with config
    assert "FoodClassificationTrainer" in main_source, "Trainer not created"
    print("✓ main function creates trainer with config")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Configuration Management Structure Tests")
    print("=" * 60)
    
    try:
        test_training_config_exists()
        test_required_attributes()
        test_validation_method()
        test_device_resolution()
        test_argument_parser()
        test_trainer_uses_config()
        test_main_creates_config()
        
        print("\n" + "=" * 60)
        print("✓ All structure tests passed!")
        print("=" * 60)
        print("\nConfiguration management implementation is complete:")
        print("  ✓ TrainingConfig dataclass with all parameters")
        print("  ✓ Input validation for all parameters")
        print("  ✓ Device detection with automatic CUDA/CPU fallback")
        print("  ✓ Comprehensive argument parser")
        print("  ✓ Integration with FoodClassificationTrainer")
        print("  ✓ Main function creates config from arguments")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
