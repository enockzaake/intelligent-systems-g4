@echo off
echo ========================================
echo AI Route Optimization - Training Pipeline
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.13 or higher.
    pause
    exit /b 1
)

echo.
echo Installing/Updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting training pipeline...
python main.py

echo.
echo ========================================
echo Training complete!
echo ========================================
echo.
echo Check the outputs/ directory for:
echo   - Trained models
echo   - Evaluation reports
echo   - Feature importance plots
echo.

pause

