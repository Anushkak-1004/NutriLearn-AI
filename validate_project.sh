#!/bin/bash
# Project Setup Validation Script for Linux/Mac
# Runs the validation script to check project structure

echo "NutriLearn AI - Project Setup Validation"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Run validation script
echo "Running validation checks..."
echo ""
python3 backend/validate_setup.py "$@"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "Validation PASSED - Project setup is correct"
    exit 0
else
    echo ""
    echo "Validation FAILED - Please fix the errors above"
    exit 1
fi
