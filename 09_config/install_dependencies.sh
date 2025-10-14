#!/bin/bash

# Install dependencies for production-ready SAM2 + Diffusion system
echo "Installing Production Dependencies..."

# Update pip
pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless numpy scipy Pillow

# Segmentation models
pip install ultralytics>=8.3.189

# SAM2 (Segment Anything)
echo "Installing SAM2..."
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download SAM weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth

# Diffusion models
echo "Installing Diffusion models..."
pip install diffusers>=0.24.0 transformers accelerate safetensors
pip install xformers  # For memory efficient attention

# Optimization libraries
pip install onnx onnxruntime-gpu
pip install tensorrt  # If available

# Additional optimizations
pip install numba joblib

echo "✓ Dependencies installed!"
echo "Note: For best performance, ensure CUDA 11.8+ is installed"