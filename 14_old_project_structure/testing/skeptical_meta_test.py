#!/usr/bin/env python3
"""
Skeptical testing of meta_ready_cv_system.py
Let's see what's REALLY happening and why it's fast
"""

import cv2
import numpy as np
import time
import torch
import traceback
import psutil
import gc

print("="*80)
print("SKEPTICAL TESTING - What's Really Going On?")
print("="*80)

# First, let's examine what the system is actually doing
from meta_ready_cv_system import MetaReadyVisionSystem

system = MetaReadyVisionSystem()

# Test 1: What is the actual computational complexity?
print("\n1. ANALYZING COMPUTATIONAL COMPLEXITY")
print("-" * 40)

# Check model architecture
print(f"Model parameters: {sum(p.numel() for p in system.vit.parameters()):,}")
print(f"Model layers: {len(list(system.vit.modules()))}")

# Print actual architecture
print("\nActual architecture:")
for name, module in system.vit.named_modules():
    if len(name.split('.')) <= 2:  # Only top-level modules
        print(f"  {name}: {module.__class__.__name__}")

# Test 2: Time each component separately
print("\n2. COMPONENT-BY-COMPONENT TIMING")
print("-" * 40)

test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Time preprocessing
start = time.perf_counter()
input_frame = cv2.resize(test_frame, (224, 224), interpolation=cv2.INTER_LINEAR)
frame_tensor = torch.from_numpy(input_frame).float().to('cuda')
frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
preprocess_time = (time.perf_counter() - start) * 1000
print(f"Preprocessing: {preprocess_time:.2f}ms")

# Time model forward pass
start = time.perf_counter()
with torch.no_grad():
    outputs = system.vit(frame_tensor)
model_time = (time.perf_counter() - start) * 1000
print(f"Model forward pass: {model_time:.2f}ms")

# Time tracking
from meta_ready_cv_system import Detection
start = time.perf_counter()
detections = []
for i in range(5):
    detections.append(Detection(
        bbox=(i*10, i*10, 50, 50),
        confidence=0.9,
        features=np.random.randn(128)
    ))
tracks = system.tracker.update_batch(detections)
tracking_time = (time.perf_counter() - start) * 1000
print(f"Tracking: {tracking_time:.2f}ms")

# Total time
total_component_time = preprocess_time + model_time + tracking_time
print(f"\nTotal component time: {total_component_time:.2f}ms")
print(f"Theoretical FPS based on components: {1000/total_component_time:.1f}")

# Test 3: Actual end-to-end timing with various methods
print("\n3. MULTIPLE TIMING METHODS")
print("-" * 40)

# Method 1: Simple timing
times_simple = []
for _ in range(20):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    start = time.perf_counter()
    result = system.process(frame)
    times_simple.append((time.perf_counter() - start) * 1000)

print(f"Method 1 (perf_counter): {np.mean(times_simple):.2f}ms ({1000/np.mean(times_simple):.1f} FPS)")

# Method 2: Using time.time()
times_time = []
for _ in range(20):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    start = time.time()
    result = system.process(frame)
    times_time.append((time.time() - start) * 1000)

print(f"Method 2 (time.time): {np.mean(times_time):.2f}ms ({1000/np.mean(times_time):.1f} FPS)")

# Method 3: CUDA timing
if torch.cuda.is_available():
    times_cuda = []
    for _ in range(20):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = system.process(frame)
        torch.cuda.synchronize()
        times_cuda.append((time.perf_counter() - start) * 1000)

    print(f"Method 3 (CUDA sync): {np.mean(times_cuda):.2f}ms ({1000/np.mean(times_cuda):.1f} FPS)")

# Test 4: Check what outputs are actually being produced
print("\n4. OUTPUT VALIDATION")
print("-" * 40)

frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
result = system.process(frame)

print(f"Keys in result: {list(result.keys())}")
print(f"Number of detections: {len(result['detections'])}")
print(f"Number of tracks: {len(result['tracks'])}")
print(f"3D points shape: {result['3d_points'].shape if result['3d_points'] is not None else 'None'}")
print(f"Depth shape: {result['depth'].shape if result['depth'] is not None else 'None'}")

# Check if detections have valid bboxes
if result['detections']:
    print(f"First detection bbox: {result['detections'][0].bbox}")
    print(f"Detection has features: {result['detections'][0].features is not None}")

# Test 5: Memory and resource usage
print("\n5. RESOURCE USAGE")
print("-" * 40)

process = psutil.Process()
initial_memory = process.memory_info().rss / 1024 / 1024

# Process 100 frames
for _ in range(100):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _ = system.process(frame)

final_memory = process.memory_info().rss / 1024 / 1024
print(f"Memory before: {initial_memory:.1f}MB")
print(f"Memory after: {final_memory:.1f}MB")
print(f"Memory increase: {final_memory - initial_memory:.1f}MB")

# GPU memory
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024/1024:.1f}MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024/1024:.1f}MB")

# Test 6: Stress test - can it maintain performance?
print("\n6. STRESS TEST - 1000 FRAMES")
print("-" * 40)

all_times = []
for i in range(1000):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    start = time.perf_counter()
    _ = system.process(frame)
    elapsed = (time.perf_counter() - start) * 1000
    all_times.append(elapsed)

    if i % 100 == 0:
        recent_avg = np.mean(all_times[-100:])
        print(f"  Frames {i}-{i+100}: {recent_avg:.2f}ms ({1000/recent_avg:.1f} FPS)")

print(f"\nOverall stats for 1000 frames:")
print(f"  Mean: {np.mean(all_times):.2f}ms ({1000/np.mean(all_times):.1f} FPS)")
print(f"  Std: {np.std(all_times):.2f}ms")
print(f"  Min: {np.min(all_times):.2f}ms ({1000/np.min(all_times):.1f} FPS)")
print(f"  Max: {np.max(all_times):.2f}ms ({1000/np.max(all_times):.1f} FPS)")

# Test 7: What makes it fast?
print("\n7. WHY IS IT FAST?")
print("-" * 40)

print("Key optimizations found:")
print("  1. Small model (3 transformer blocks vs 12 in standard ViT)")
print("  2. Large patch size (32x32 vs 16x16 standard)")
print("  3. Small embedding dim (192 vs 768 standard)")
print("  4. Fixed 224x224 input (no matter the original size)")
print("  5. Simplified outputs (fixed 5 detections)")
print("  6. No actual object detection - just reshaping transformer outputs!")
print("  7. GPU operations but very lightweight model")

# The truth
print("\n" + "="*80)
print("THE TRUTH:")
print("="*80)
print("The system IS fast because:")
print("  • It's using a TINY model (under 1M parameters)")
print("  • It always processes 224x224 regardless of input")
print("  • It's not doing real object detection (just using raw transformer outputs)")
print("  • The '3D reconstruction' is just reshaping depth values")
print("  • Most of the 'features' are placeholder/simplified")
print("\nIt's fast but NOT doing what a real CV system should do!")
print("="*80)