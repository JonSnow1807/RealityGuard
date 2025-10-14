#!/usr/bin/env python3
"""
Verify CUDA usage and identify bottlenecks
"""

import torch
import numpy as np
from ultralytics import YOLO
import time
import psutil
import os

print("=" * 70)
print("CUDA VERIFICATION TEST")
print("=" * 70)

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("ERROR: CUDA not available!")
    exit(1)

print("\n1. Testing Simple CUDA Operations")
print("-" * 50)

# Test raw CUDA performance
x = torch.randn(1000, 1000).cuda()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    y = torch.matmul(x, x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"Raw CUDA matmul: {1000/elapsed:.0f} ops/sec")

print("\n2. Testing YOLOv8 Model")
print("-" * 50)

# Load model
model = YOLO('yolov8n-seg.pt')
print(f"Model loaded: {model.model.__class__.__name__}")
print(f"Model device: {next(model.model.parameters()).device}")

# Force model to CUDA
model.to('cuda')
print(f"Model moved to CUDA")
print(f"Model device after move: {next(model.model.parameters()).device}")

print("\n3. Testing Inference Speed")
print("-" * 50)

# Single image test
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
print("Warming up...")
for _ in range(5):
    _ = model(img, verbose=False, device='cuda')

# Test single image
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    results = model(img, verbose=False, device='cuda')
torch.cuda.synchronize()
single_time = time.perf_counter() - start
single_fps = 10 / single_time

print(f"Single image: {single_fps:.1f} FPS ({single_time/10*1000:.1f} ms/img)")

# Test batch
batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(8)]

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    results = model(batch, verbose=False, device='cuda')
torch.cuda.synchronize()
batch_time = time.perf_counter() - start
batch_fps = (10 * 8) / batch_time

print(f"Batch 8: {batch_fps:.1f} FPS ({batch_time/10*1000:.1f} ms/batch)")

print("\n4. Checking GPU Memory")
print("-" * 50)
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n5. System Resources")
print("-" * 50)
print(f"CPU Usage: {psutil.cpu_percent()}%")
print(f"RAM Usage: {psutil.virtual_memory().percent}%")

# Check if we're in a container or limited environment
if os.path.exists('/proc/self/cgroup'):
    with open('/proc/self/cgroup', 'r') as f:
        cgroup = f.read()
        if 'docker' in cgroup or 'containerd' in cgroup:
            print("⚠️  Running in container - may have resource limits")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if single_fps < 50:
    print("⚠️  Performance is lower than expected for L4 GPU")
    print("Possible causes:")
    print("  1. CPU bottleneck in preprocessing")
    print("  2. Model not fully utilizing GPU")
    print("  3. Resource limits in environment")
    print("  4. Thermal throttling")
else:
    print("✓ Performance is as expected")

print(f"\nBatch speedup: {batch_fps/single_fps:.1f}x")
if batch_fps/single_fps < 3:
    print("⚠️  Batch processing not providing expected speedup")
    print("  This suggests CPU preprocessing bottleneck")