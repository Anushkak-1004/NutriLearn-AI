"""
Tests for InferenceHelper component.

Tests model loading, image preprocessing, and prediction functionality.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

# Import from train_model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_model import InferenceHelper, InvalidImageError, ModelNotFoundError


class TestInferenceHelper:
    """Test suite for InferenceHelper class."""
    
    @pytest.fixture
    def mock_model_files(self, tmp_path):
        """Create mock model files for testing."""
        # Create mock model config
        config = {
            'model_name': 'mobilenet_v2',
            'num_classes': 10,
            'freeze_layers': 5,
            'dropout_rate': 0.2,
            'input_size': [224, 224],
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        config_path = tmp_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Create mock class mapping
        class_mapping = {
            f"class_{i}": i for i in range(10)
        }
        
        mapping_path = tmp_path / "class_to_idx.json"
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f)
        
        # Create mock model file (empty for now, will be created by test)
        model_path = tmp_path / "model.pth"
        
        return {
            'model_path': str(model_path),
            'config_path': str(config_path),
            'mapping_path': str(mapping_path),
            'tmp_path': tmp_path
        }
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        # Create a simple RGB image
        img = Image.new('RGB', (256, 256), color='red')
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)
        return str(img_path)
    
    def test_initialization_with_missing_model(self, mock_model_files):
        """Test that ModelNotFoundError is raised when model file is missing."""
        with pytest.raises(ModelNotFoundError):
            InferenceHelper(
                model_path=mock_model_files['model_path'],
                config_path=mock_model_files['config_path'],
                class_mapping_path=mock_model_files['mapping_path']
            )
    
    def test_initialization_with_missing_config(self, mock_model_files):
        """Test that FileNotFoundError is raised when config file is missing."""
        # Create empty model file
        Path(mock_model_files['model_path']).touch()
        
        with pytest.raises(FileNotFoundError):
            InferenceHelper(
                model_path=mock_model_files['model_path'],
                config_path=str(Path(mock_model_files['config_path']).parent / "nonexistent.json"),
                class_mapping_path=mock_model_files['mapping_path']
            )
    
    def test_initialization_with_missing_class_mapping(self, mock_model_files):
        """Test that FileNotFoundError is raised when class mapping is missing."""
        # Create empty model file
        Path(mock_model_files['model_path']).touch()
        
        with pytest.raises(FileNotFoundError):
            InferenceHelper(
                model_path=mock_model_files['model_path'],
                config_path=mock_model_files['config_path'],
                class_mapping_path=str(Path(mock_model_files['mapping_path']).parent / "nonexistent.json")
            )
    
    def test_preprocess_image_invalid_path(self, mock_model_files):
        """Test that InvalidImageError is raised for invalid image path."""
        # Create a minimal model file
        from train_model import ModelBuilder
        
        builder = ModelBuilder(
            model_name='mobilenet_v2',
            num_classes=10,
            freeze_layers=5,
            dropout_rate=0.2
        )
        model = builder.build_model()
        torch.save(model.state_dict(), mock_model_files['model_path'])
        
        # Initialize helper
        helper = InferenceHelper(
            model_path=mock_model_files['model_path'],
            config_path=mock_model_files['config_path'],
            class_mapping_path=mock_model_files['mapping_path']
        )
        
        # Load model
        helper.load_model()
        
        # Test with invalid image path
        with pytest.raises(InvalidImageError):
            helper.preprocess_image("nonexistent_image.jpg")
    
    def test_preprocess_image_output_shape(self, mock_model_files, sample_image):
        """
        Test that preprocessed image has correct shape [1, 3, 224, 224].
        
        **Validates: Requirements 7.2**
        """
        # Create a minimal model file
        from train_model import ModelBuilder
        
        builder = ModelBuilder(
            model_name='mobilenet_v2',
            num_classes=10,
            freeze_layers=5,
            dropout_rate=0.2
        )
        model = builder.build_model()
        torch.save(model.state_dict(), mock_model_files['model_path'])
        
        # Initialize helper
        helper = InferenceHelper(
            model_path=mock_model_files['model_path'],
            config_path=mock_model_files['config_path'],
            class_mapping_path=mock_model_files['mapping_path']
        )
        
        # Load model
        helper.load_model()
        
        # Preprocess image
        tensor = helper.preprocess_image(sample_image)
        
        # Verify shape is [1, 3, 224, 224]
        assert tensor.shape == torch.Size([1, 3, 224, 224]), \
            f"Expected shape [1, 3, 224, 224], got {tensor.shape}"
        
        # Verify tensor is normalized (values should be roughly in range [-3, 3])
        assert tensor.min() >= -5 and tensor.max() <= 5, \
            "Tensor values should be normalized"
    
    def test_predict_returns_top_k_predictions(self, mock_model_files, sample_image):
        """
        Test that predict returns exactly top-k predictions with valid probabilities.
        
        **Validates: Requirements 7.3**
        """
        # Create a minimal model file
        from train_model import ModelBuilder
        
        builder = ModelBuilder(
            model_name='mobilenet_v2',
            num_classes=10,
            freeze_layers=5,
            dropout_rate=0.2
        )
        model = builder.build_model()
        torch.save(model.state_dict(), mock_model_files['model_path'])
        
        # Initialize helper
        helper = InferenceHelper(
            model_path=mock_model_files['model_path'],
            config_path=mock_model_files['config_path'],
            class_mapping_path=mock_model_files['mapping_path']
        )
        
        # Load model
        helper.load_model()
        
        # Make prediction
        predictions = helper.predict(sample_image, top_k=3)
        
        # Verify we get exactly 3 predictions
        assert len(predictions) == 3, f"Expected 3 predictions, got {len(predictions)}"
        
        # Verify each prediction is a tuple of (class_name, probability)
        for class_name, prob in predictions:
            assert isinstance(class_name, str), "Class name should be a string"
            assert isinstance(prob, float), "Probability should be a float"
            assert 0.0 <= prob <= 1.0, f"Probability {prob} should be in [0, 1]"
        
        # Verify probabilities sum to <= 1.0 (they're top-3, not all classes)
        total_prob = sum(prob for _, prob in predictions)
        assert total_prob <= 1.0, f"Sum of top-3 probabilities {total_prob} should be <= 1.0"
        
        # Verify predictions are sorted by probability (descending)
        probs = [prob for _, prob in predictions]
        assert probs == sorted(probs, reverse=True), "Predictions should be sorted by probability"
    
    def test_predict_without_loading_model(self, mock_model_files, sample_image):
        """Test that RuntimeError is raised when predicting without loading model."""
        # Create a minimal model file
        from train_model import ModelBuilder
        
        builder = ModelBuilder(
            model_name='mobilenet_v2',
            num_classes=10,
            freeze_layers=5,
            dropout_rate=0.2
        )
        model = builder.build_model()
        torch.save(model.state_dict(), mock_model_files['model_path'])
        
        # Initialize helper but don't load model
        helper = InferenceHelper(
            model_path=mock_model_files['model_path'],
            config_path=mock_model_files['config_path'],
            class_mapping_path=mock_model_files['mapping_path']
        )
        
        # Try to predict without loading model
        with pytest.raises(RuntimeError, match="Model not loaded"):
            helper.predict(sample_image)
    
    def test_load_model_creates_correct_architecture(self, mock_model_files):
        """Test that load_model correctly loads model architecture and weights."""
        # Create a minimal model file
        from train_model import ModelBuilder
        
        builder = ModelBuilder(
            model_name='mobilenet_v2',
            num_classes=10,
            freeze_layers=5,
            dropout_rate=0.2
        )
        model = builder.build_model()
        torch.save(model.state_dict(), mock_model_files['model_path'])
        
        # Initialize helper
        helper = InferenceHelper(
            model_path=mock_model_files['model_path'],
            config_path=mock_model_files['config_path'],
            class_mapping_path=mock_model_files['mapping_path']
        )
        
        # Load model
        helper.load_model()
        
        # Verify model is loaded
        assert helper.model is not None, "Model should be loaded"
        assert helper.config is not None, "Config should be loaded"
        assert helper.idx_to_class is not None, "Class mapping should be loaded"
        assert helper.transform is not None, "Transform should be created"
        
        # Verify model is in eval mode
        assert not helper.model.training, "Model should be in eval mode"
        
        # Verify class mapping has correct number of classes
        assert len(helper.idx_to_class) == 10, "Should have 10 classes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
