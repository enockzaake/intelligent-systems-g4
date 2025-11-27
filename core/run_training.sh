#!/bin/bash

echo "========================================"
echo "AI Route Optimization - Training Pipeline"
echo "========================================"
echo ""

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null
then
    echo "Error: Python3 not found. Please install Python 3.13 or higher."
    exit 1
fi

python3 --version

echo ""
echo "Installing/Updating dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

echo ""
echo "Starting training pipeline..."
python3 main.py

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo ""
echo "Check the outputs/ directory for:"
echo "  - Trained models"
echo "  - Evaluation reports"
echo "  - Feature importance plots"
echo ""

