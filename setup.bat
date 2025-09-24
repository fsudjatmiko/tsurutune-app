@echo off
REM TsuruTune Setup Script for Windows
REM This script sets up the Python environment for TsuruTune

echo Setting up TsuruTune Python backend...

REM Check if Python 3 is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python 3 is required but not installed.
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

echo Python found ✓

REM Navigate to python directory
cd /d "%~dp0python"

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python dependencies...
echo Choose installation type:
echo 1) CPU-only optimization (default)
echo 2) GPU/CUDA optimization with TensorRT
set /p choice="Enter choice (1 or 2, default is 1): "

if "%choice%"=="2" (
    echo Installing CUDA/GPU requirements...
    echo Note: Make sure you have NVIDIA GPU, CUDA toolkit, and TensorRT installed
    pip install -r requirements-cuda.txt
) else (
    echo Installing CPU-only requirements...
    pip install -r requirements-cpu.txt
)

REM Check for CUDA availability
echo Checking for CUDA availability...
python -c "try: import torch; print('CUDA is available ✓' if torch.cuda.is_available() else 'CUDA is not available - CPU optimization only'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else ''); print(f'Available GPUs: {torch.cuda.device_count()}' if torch.cuda.is_available() else '') except ImportError: print('PyTorch not installed - CUDA check skipped'); print('Install PyTorch with CUDA support for GPU optimization')"

echo.
echo Setup complete! ✓
echo.
echo To run TsuruTune:
echo 1. Start the Electron app: npm start
echo 2. The app will automatically use the Python backend
echo.
echo For manual Python backend testing:
echo cd python && venv\Scripts\activate.bat && python main.py --help
pause
