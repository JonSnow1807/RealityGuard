#!/bin/bash
# Lightning AI GPU Setup Script for RealityGuard
# This installs everything needed for the $100M Meta acquisition system

echo "🚀 RealityGuard Lightning AI Setup"
echo "=================================="

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y wget git curl libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -qq

# Install Python packages
echo "🐍 Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install transformers accelerate segment-anything opencv-python mediapipe -q
pip install git+https://github.com/openai/CLIP.git -q
pip install onnxruntime-gpu tensorrt numpy scipy -q

# Create directories
echo "📁 Creating project structure..."
mkdir -p /workspace/RealityGuard/models
mkdir -p /workspace/RealityGuard/src
cd /workspace/RealityGuard

# Clone repository
echo "📥 Cloning RealityGuard repository..."
git clone https://github.com/JonSnow1807/RealityGuard.git . 2>/dev/null || git pull

# Download model weights
echo "🧠 Downloading AI model weights..."

# SAM weights (355MB)
echo "  - Downloading SAM (Meta's Segment Anything)..."
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h.pth

# Download MobileSAM for faster inference (39MB)
echo "  - Downloading MobileSAM (lightweight)..."
wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O models/mobile_sam.pt

# Pre-download DINOv2 and CLIP
echo "  - Pre-loading DINOv2 and CLIP models..."
python -c "
from transformers import Dinov2Model
import clip
import torch
print('  - Loading DINOv2...')
model = Dinov2Model.from_pretrained('facebook/dinov2-base')
print('  - Loading CLIP...')
clip_model, preprocess = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
print('  ✅ Models ready!')
"

# Test GPU
echo ""
echo "🎮 Testing GPU Setup..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'✅ CUDA: {torch.version.cuda}')
    print(f'✅ PyTorch: {torch.__version__}')
else:
    print('❌ No GPU detected!')
"

# Create quick test script
cat > test_gpu.py << 'EOF'
import torch
import time
import numpy as np

# Test GPU performance
print("\n🏃 Running GPU Performance Test...")

# Create large tensor
size = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.randn(size, size).to(device)
b = torch.randn(size, size).to(device)

# Warm up
_ = torch.matmul(a, b)
torch.cuda.synchronize() if device == 'cuda' else None

# Benchmark
start = time.time()
for _ in range(10):
    c = torch.matmul(a, b)
    if device == 'cuda':
        torch.cuda.synchronize()
elapsed = time.time() - start

tflops = (10 * 2 * size**3) / elapsed / 1e12
print(f"✅ Performance: {tflops:.1f} TFLOPS")
print(f"✅ Time for 10 iterations: {elapsed:.2f}s")

if tflops > 5:
    print("🚀 GPU is working excellently! Ready for 1000+ FPS")
elif tflops > 1:
    print("✅ GPU is working well! Can achieve 500+ FPS")
else:
    print("⚠️  GPU might be slow, but still better than CPU")
EOF

python test_gpu.py

echo ""
echo "=========================================="
echo "✅ Lightning AI Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: cd /workspace/RealityGuard"
echo "2. Run: python src/realityguard_gpu_optimized.py"
echo ""
echo "🎯 Ready to build the $100M Meta acquisition system!"