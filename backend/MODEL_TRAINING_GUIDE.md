# 🎓 Model Training Guide - NutriLearn AI

Complete guide for training the food classification model with PyTorch and MLflow.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Training the Model](#training-the-model)
4. [Model Evaluation](#model-evaluation)
5. [Testing Inference](#testing-inference)
6. [Google Colab Setup](#google-colab-setup)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Packages

```bash
pip install torch torchvision torchaudio
pip install mlflow
pip install scikit-learn matplotlib seaborn
pip install tqdm Pillow
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow training)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Optimal**: Google Colab with GPU (free)

## Dataset Setup

### Option 1: Automatic Download (Recommended)

The script automatically downloads Food-101 dataset:

```bash
python train_model.py --epochs 20
```

The dataset will be downloaded to `./data/food-101/` (3GB download).

### Option 2: Manual Download

1. Download Food-101 dataset:
   ```bash
   wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```

2. Organize directory structure:
   ```
   data/
   └── food-101/
       ├── images/
       │   ├── apple_pie/
       │   ├── baby_back_ribs/
       │   └── ... (101 classes)
       └── meta/
           ├── train.txt
           └── test.txt
   ```

### Dataset Information

- **Classes**: 101 food categories
- **Images**: 101,000 images (1,000 per class)
- **Split**: 75,750 training + 25,250 test images
- **Resolution**: Variable (resized to 224x224 during training)

**Food Categories Include**:
- Indian: samosa, biryani, gulab_jamun, etc.
- Western: pizza, burger, pasta, etc.
- Asian: sushi, ramen, pad_thai, etc.
- Desserts: cheesecake, ice_cream, etc.

## Training the Model

### Basic Training

```bash
python train_model.py --epochs 20 --batch_size 32 --lr 0.001
```

### Advanced Options

```bash
# Use EfficientNet (better accuracy)
python train_model.py --model efficientnet_b0 --epochs 25

# Use ResNet50 (more parameters)
python train_model.py --model resnet50 --epochs 20

# Larger batch size (requires more GPU memory)
python train_model.py --batch_size 64

# Custom learning rate
python train_model.py --lr 0.0005

# Resume from checkpoint
python train_model.py --resume ./ml-models/checkpoint_latest.pth

# Force CPU training
python train_model.py --device cpu
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | mobilenet_v2 | Model architecture (mobilenet_v2, efficientnet_b0, resnet50) |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Batch size (reduce if out of memory) |
| `--lr` | 0.001 | Initial learning rate |
| `--data_dir` | ./data/food-101 | Dataset directory |
| `--output_dir` | ./ml-models | Output directory for models |
| `--device` | auto | Device (auto, cuda, cpu) |
| `--resume` | None | Checkpoint path to resume training |

### Expected Training Time

| Hardware | Model | Time per Epoch | Total Time (20 epochs) |
|----------|-------|----------------|------------------------|
| CPU (8 cores) | MobileNetV2 | ~45 min | ~15 hours |
| GPU (T4) | MobileNetV2 | ~3 min | ~1 hour |
| GPU (V100) | MobileNetV2 | ~1.5 min | ~30 min |
| GPU (T4) | EfficientNet-B0 | ~5 min | ~1.7 hours |

### Training Output

During training, you'll see:

```
INFO - Using device: cuda
INFO - Loading Food-101 dataset...
INFO - Loaded Food-101: 75750 train, 25250 test images
INFO - Number of classes: 101
INFO - Train: 60600, Val: 15150
INFO - Building mobilenet_v2 model...
INFO - Trainable parameters: 2,296,965 / 3,504,872
INFO - MLflow experiment tracking enabled
INFO - Starting training...

Epoch 1/20 [Train]: 100%|████████| 1894/1894 [03:12<00:00, loss: 2.456, acc: 45.2%]
Epoch 1/20 [Val]: 100%|████████| 474/474 [00:42<00:00, loss: 1.987]
INFO - New best validation accuracy: 52.3%
INFO - Saved checkpoint: ./ml-models/checkpoint_latest.pth
INFO - Saved best model: ./ml-models/food_model_mobilenet_v2_best.pth

...

INFO - Training completed in 3847.23 seconds
INFO - Best validation accuracy: 78.45%
```

### Output Files

After training, you'll find:

```
ml-models/
├── food_model_v1.pth                    # Best model (for inference)
├── food_model_mobilenet_v2_best.pth     # Best model with optimizer
├── checkpoint_latest.pth                # Latest checkpoint
├── class_to_idx.json                    # Class name to index mapping
├── model_config.json                    # Model configuration
└── confusion_matrix.png                 # Confusion matrix visualization
```

## Model Evaluation

### Metrics Tracked

The training script tracks:

1. **Training Metrics** (per epoch):
   - Loss
   - Accuracy

2. **Validation Metrics** (per epoch):
   - Loss
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1 Score (weighted)

3. **Per-Class Metrics** (final):
   - Precision per class
   - Recall per class
   - F1 score per class
   - Support (number of samples)

### Viewing Results

#### MLflow UI

```bash
cd backend
mlflow ui
```

Visit http://localhost:5000 to see:
- All training runs
- Metric charts (loss, accuracy over epochs)
- Parameter comparison
- Model artifacts

#### Confusion Matrix

Open `ml-models/confusion_matrix.png` to see:
- Which classes are confused with each other
- Model performance per class
- Identify classes that need more training data

### Expected Performance

| Model | Validation Accuracy | Top-5 Accuracy | Inference Time (CPU) |
|-------|---------------------|----------------|----------------------|
| MobileNetV2 | 75-80% | 92-95% | ~50ms |
| EfficientNet-B0 | 78-83% | 94-96% | ~80ms |
| ResNet50 | 80-85% | 95-97% | ~120ms |

## Testing Inference

### Test Single Image

```bash
python train_model.py --test \
    --test_model ./ml-models/food_model_v1.pth \
    --test_config ./ml-models/model_config.json \
    --test_image ./test_images/pizza.jpg
```

**Output**:
```
Top 3 predictions for ./test_images/pizza.jpg:
1. pizza: 94.23%
2. flatbread: 3.45%
3. bruschetta: 1.12%
```

### Integration with Backend

Update `backend/app/ml/predictor.py` to use the trained model:

```python
import torch
from torchvision import transforms, models
from PIL import Image

class FoodPredictor:
    def __init__(self, model_path: str, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load class mappings
        with open("ml-models/class_to_idx.json", 'r') as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['mean'],
                std=self.config['std']
            )
        ])
    
    def predict(self, image_path: str) -> dict:
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get food name
        food_name = self.idx_to_class[predicted.item()]
        
        return {
            'food_name': food_name,
            'confidence': confidence.item()
        }
```

## Google Colab Setup

### Step 1: Create New Notebook

Go to https://colab.research.google.com/ and create a new notebook.

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4 or V100)
3. Click **Save**

### Step 3: Install Dependencies

```python
!pip install torch torchvision mlflow scikit-learn matplotlib seaborn tqdm
```

### Step 4: Clone Repository

```python
!git clone https://github.com/yourusername/nutrilearn-ai.git
%cd nutrilearn-ai/backend
```

### Step 5: Train Model

```python
!python train_model.py --epochs 20 --batch_size 64
```

### Step 6: Download Trained Model

```python
from google.colab import files

# Download model
files.download('ml-models/food_model_v1.pth')
files.download('ml-models/class_to_idx.json')
files.download('ml-models/model_config.json')
```

### Complete Colab Notebook

```python
# Install dependencies
!pip install -q torch torchvision mlflow scikit-learn matplotlib seaborn tqdm

# Clone repository
!git clone https://github.com/yourusername/nutrilearn-ai.git
%cd nutrilearn-ai/backend

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Train model
!python train_model.py \
    --model mobilenet_v2 \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001

# View MLflow results
!mlflow ui --port 5000 &

# Download trained model
from google.colab import files
files.download('ml-models/food_model_v1.pth')
files.download('ml-models/class_to_idx.json')
files.download('ml-models/model_config.json')
files.download('ml-models/confusion_matrix.png')
```

## Troubleshooting

### Out of Memory Error

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch_size 16`
2. Use smaller model: `--model mobilenet_v2`
3. Use CPU: `--device cpu`
4. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Slow Training on CPU

**Problem**: Training takes too long on CPU

**Solutions**:
1. Use Google Colab with GPU (free)
2. Use smaller model: MobileNetV2
3. Reduce epochs: `--epochs 10`
4. Use smaller dataset subset

### Dataset Download Fails

**Problem**: Food-101 download fails or is slow

**Solutions**:
1. Manual download:
   ```bash
   wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```
2. Use alternative mirror
3. Download on faster connection and transfer

### Low Accuracy

**Problem**: Model accuracy is below 70%

**Solutions**:
1. Train for more epochs: `--epochs 30`
2. Use better model: `--model efficientnet_b0`
3. Adjust learning rate: `--lr 0.0005`
4. Check data augmentation
5. Ensure dataset is complete

### MLflow Not Working

**Problem**: MLflow tracking not working

**Solutions**:
1. Install MLflow: `pip install mlflow`
2. Check MLflow UI: `mlflow ui`
3. Training continues without MLflow (non-blocking)

## Best Practices

### 1. Start Small
- Train for 5 epochs first to verify everything works
- Use small batch size (16) to test
- Then scale up

### 2. Monitor Training
- Watch for overfitting (train acc >> val acc)
- Check if loss is decreasing
- Use early stopping (automatic)

### 3. Experiment Tracking
- Always use MLflow to track experiments
- Compare different models and hyperparameters
- Document what works and what doesn't

### 4. Model Selection
- **MobileNetV2**: Fast inference, good for mobile/edge
- **EfficientNet-B0**: Best accuracy/speed trade-off
- **ResNet50**: Highest accuracy, slower inference

### 5. Data Augmentation
- Already included: rotation, flip, color jitter
- Helps prevent overfitting
- Improves generalization

## Next Steps

After training:

1. **Integrate with Backend**:
   - Update `backend/app/ml/predictor.py`
   - Replace mock predictor with real model
   - Test API endpoints

2. **Optimize for Production**:
   - Convert to TorchScript for faster inference
   - Quantize model for smaller size
   - Add model caching

3. **Add Custom Foods**:
   - Collect images of Indian foods
   - Fine-tune on custom dataset
   - Improve regional cuisine recognition

4. **Deploy**:
   - Upload model to Hugging Face
   - Deploy backend with model
   - Monitor predictions with MLflow

## Resources

- **Food-101 Dataset**: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- **PyTorch Docs**: https://pytorch.org/docs/
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Transfer Learning Guide**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

**Happy Training! 🚀**

For questions or issues, check the troubleshooting section or open an issue on GitHub.
