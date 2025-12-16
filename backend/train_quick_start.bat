@echo off
REM Quick Start Script for Model Training (Windows)
REM NutriLearn AI - Food Classification

echo ========================================
echo NutriLearn AI - Model Training Quick Start
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.

REM Create directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "ml-models" mkdir ml-models
echo.

REM Start training
echo Starting model training...
echo This will download Food-101 dataset (~3GB) and train for 20 epochs
echo Estimated time: 1 hour on GPU, 15 hours on CPU
echo.
set /p CONTINUE="Continue? (y/n): "

if /i "%CONTINUE%"=="y" (
    python train_model.py --model mobilenet_v2 --epochs 20 --batch_size 32 --lr 0.001
    
    echo.
    echo ========================================
    echo Training completed!
    echo ========================================
    echo.
    echo Output files:
    echo   - ml-models\food_model_v1.pth (best model)
    echo   - ml-models\class_to_idx.json (class mappings)
    echo   - ml-models\model_config.json (configuration)
    echo   - ml-models\confusion_matrix.png (evaluation)
    echo.
    echo View MLflow results:
    echo   mlflow ui
    echo   Visit: http://localhost:5000
    echo.
    echo Test inference:
    echo   python train_model.py --test --test_model .\ml-models\food_model_v1.pth --test_config .\ml-models\model_config.json --test_image .\path\to\image.jpg
) else (
    echo Training cancelled.
)

pause
