import torch
import time
import numpy as np
from transformers import Dinov2Model, AutoImageProcessor
import clip

print("="*60)
print("🎮 GPU SETUP TEST")
print("="*60)

# Test GPU availability
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"✅ CUDA: {torch.version.cuda}")
else:
    print("❌ No GPU detected!")
    exit(1)

# Test model loading
print("\n📦 Loading AI Models...")

try:
    # Load DINOv2
    print("Loading DINOv2...")
    dinov2 = Dinov2Model.from_pretrained('facebook/dinov2-small').cuda()
    print("✅ DINOv2 loaded")

    # Load CLIP
    print("Loading CLIP...")
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    print("✅ CLIP loaded")

    # Test performance
    print("\n🏃 Performance Test...")
    test_tensor = torch.randn(1, 3, 224, 224).cuda()

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = dinov2(test_tensor).last_hidden_state

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = dinov2(test_tensor).last_hidden_state
    torch.cuda.synchronize()
    elapsed = time.time() - start

    fps = 100 / elapsed
    print(f"✅ DINOv2 Performance: {fps:.1f} FPS")

    if fps > 100:
        print("🚀 Excellent! Ready for 1000+ FPS pipeline")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("Setup test complete!")
print("="*60)