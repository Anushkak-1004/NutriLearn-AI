@echo off
REM Project Setup Validation Script for Windows
REM Runs the validation script to check project structure

echo NutriLearn AI - Project Setup Validation
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    exit /b 1
)

REM Run validation script
echo Running validation checks...
echo.
python backend\validate_setup.py %*

REM Check exit code
if errorlevel 1 (
    echo.
    echo Validation FAILED - Please fix the errors above
    exit /b 1
) else (
    echo.
    echo Validation PASSED - Project setup is correct
    exit /b 0
)
