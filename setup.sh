#!/bin/bash

# TsuruTune Setup Script
# This script sets up the Python environment for TsuruTune

echo "Setting up TsuruTune Python backend..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3.8 or later from https://python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or later is required. Found Python $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION found ✓"

# Navigate to python directory
cd "$(dirname "$0")/python" || exit 1

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
echo "Choose installation type:"
echo "1) CPU-only optimization (default)"
echo "2) GPU/CUDA optimization with TensorRT"
read -p "Enter choice (1 or 2, default is 1): " choice

case $choice in
    2)
        echo "Installing CUDA/GPU requirements..."
        echo "Note: Make sure you have NVIDIA GPU, CUDA toolkit, and TensorRT installed"
        pip install -r requirements-cuda.txt
        ;;
    *)
        echo "Installing CPU-only requirements..."
        pip install -r requirements-cpu.txt
        ;;
esac

# Check for CUDA availability
echo "Checking for CUDA availability..."
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print('CUDA is available ✓')
        print(f'CUDA version: {torch.version.cuda}')
        print(f'Available GPUs: {torch.cuda.device_count()}')
    else:
        print('CUDA is not available - CPU optimization only')
except ImportError:
    print('PyTorch not installed - CUDA check skipped')
    print('Install PyTorch with CUDA support for GPU optimization')
"

echo ""
echo "Setup complete! ✓"
echo ""
echo "To run TsuruTune:"
echo "1. Start the Electron app: npm start"
echo "2. The app will automatically use the Python backend"
echo ""
echo "For manual Python backend testing:"
echo "cd python && source venv/bin/activate && python main.py --help"
