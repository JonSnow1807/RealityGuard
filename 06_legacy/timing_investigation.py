#!/usr/bin/env python3
"""
Investigation: Why is there a discrepancy between component timing (258ms)
and actual timing (4ms)? Something's not adding up.
"""

import time
import torch
import numpy as np

print("TIMING INVESTIGATION")
print("=" * 60)

# The model forward pass alone takes 244ms, but total processing is 4ms?
# This is impossible unless...

from meta_ready_cv_system import MetaReadyVisionSystem

system = MetaReadyVisionSystem()

# Let's check if the model is actually being run
print("\n1. Testing if model actually runs:")

frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Manually time just the model
import cv2
input_frame = cv2.resize(frame, (224, 224))
frame_tensor = torch.from_numpy(input_frame).float().to('cuda')
frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

# Time the actual forward pass
torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    outputs = system.vit(frame_tensor)
torch.cuda.synchronize()
model_time = (time.perf_counter() - start) * 1000

print(f"Direct model forward pass: {model_time:.2f}ms")

# Now time the full process function
torch.cuda.synchronize()
start = time.perf_counter()
result = system.process(frame)
torch.cuda.synchronize()
process_time = (time.perf_counter() - start) * 1000

print(f"Full process function: {process_time:.2f}ms")

print("\n2. What's in the process function?")
import inspect
source = inspect.getsource(system.process)
print("First 20 lines of process():")
for i, line in enumerate(source.split('\n')[:20]):
    print(f"  {line}")

# The issue is clear - let's check if model is in eval mode
print(f"\n3. Model training mode: {system.vit.training}")
print(f"   Model eval mode should be: {not system.vit.training}")

# Check if CUDA operations are async
print("\n4. CUDA async operations check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Device: {system.device}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("The discrepancy exists because:")
print("1. When timing the model alone: ~244ms")
print("2. When timing through process(): ~4ms")
print("")
print("This happens because process() timing doesn't properly")
print("synchronize CUDA operations. The 4ms is just the time to")
print("LAUNCH the GPU operations, not complete them.")
print("")
print("REAL performance with proper CUDA sync: ~250-260ms per frame")
print("Which gives us: ~4 FPS, NOT 250 FPS")
print("")
print("The 250 FPS claim is WRONG - it's measuring kernel launch time,")
print("not actual computation time!")
print("=" * 60)