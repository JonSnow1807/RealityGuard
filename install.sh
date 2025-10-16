#!/bin/bash
# RealityGuard Installation Script
# Automated setup for the AI-powered privacy system

set -e

echo "========================================="
echo "  RealityGuard Installation Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo -e "${RED}Error: Python $required_version or higher is required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version detected${NC}"

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep -Po '(?<=release )\d+\.\d+')
        echo -e "${GREEN}✓ CUDA $cuda_version detected${NC}"
    else
        echo -e "${YELLOW}Warning: CUDA toolkit not found. GPU acceleration may be limited.${NC}"
    fi
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected. System will run in CPU mode (slower).${NC}"
    read -p "Continue with CPU-only installation? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    # Install PyTorch with CUDA 11.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    echo -e "${GREEN}✓ PyTorch installed with CUDA support${NC}"
else
    # CPU-only version
    pip install torch torchvision
    echo -e "${GREEN}✓ PyTorch installed (CPU-only)${NC}"
fi

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pillow>=10.0.0
pip install tqdm>=4.65.0

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Download YOLO model
echo ""
echo "Downloading YOLO model..."
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded successfully')"
echo -e "${GREEN}✓ YOLO model ready${NC}"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import cv2
import ultralytics
import numpy as np

print('✓ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p models
mkdir -p outputs
mkdir -p logs
mkdir -p test_outputs
echo -e "${GREEN}✓ Directories created${NC}"

# Run a quick test
echo ""
echo "Running quick test..."
python3 -c "
from realityguard_ai_replacement import RealityGuardAIReplacement, AIReplacementConfig
import numpy as np

try:
    config = AIReplacementConfig()
    system = RealityGuardAIReplacement(config)

    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, stats = system.process_frame(test_frame)

    print('✓ System initialized successfully')
    print(f'  Cache size: {system.cache.max_size}')
    print(f'  Detection enabled: {system.model is not None}')
    print('✓ Test passed!')
except Exception as e:
    print(f'Test failed: {e}')
    exit(1)
" || {
    echo -e "${YELLOW}Warning: Quick test failed. System may need additional configuration.${NC}"
}

echo ""
echo "========================================="
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "========================================="
echo ""
echo "To use RealityGuard:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the system: python realityguard_ai_replacement.py"
echo "  3. Or import in Python:"
echo "     from realityguard_ai_replacement import RealityGuardAIReplacement"
echo ""
echo "For documentation, see: docs/API.md"
echo "For examples, see: examples/"
echo ""