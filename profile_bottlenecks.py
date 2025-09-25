#!/usr/bin/env python3
"""
Deep profiling to identify exact bottlenecks
This will show us where to focus our optimization efforts
"""

import torch
import numpy as np
from ultralytics import YOLO
import time
import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity

def profile_inference_pipeline():
    """Profile the entire inference pipeline to find bottlenecks."""

    print("="*70)
    print("DEEP PROFILING ANALYSIS")
    print("="*70)

    # Load model
    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    # Test image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 1. PyTorch Profiler
    print("\n1. PYTORCH PROFILER ANALYSIS")
    print("-"*50)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True) as prof:

        with record_function("model_inference"):
            for _ in range(10):
                _ = model(img, verbose=False, device='cuda')

    # Print top operations
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Save detailed trace
    prof.export_chrome_trace("trace.json")
    print("\n✓ Saved detailed trace to trace.json (open in chrome://tracing)")

    # 2. Component-wise timing
    print("\n2. COMPONENT-WISE TIMING")
    print("-"*50)

    components = {
        'preprocessing': [],
        'inference': [],
        'postprocessing': [],
        'total': []
    }

    for _ in range(100):
        # Total time
        torch.cuda.synchronize()
        total_start = time.perf_counter()

        # Preprocessing (CPU)
        prep_start = time.perf_counter()
        # Simulate preprocessing
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        prep_time = time.perf_counter() - prep_start

        # Transfer to GPU
        tensor = tensor.to('cuda')

        # Inference
        torch.cuda.synchronize()
        inf_start = time.perf_counter()
        with torch.no_grad():
            # Direct model call to measure pure inference
            output = model.model(tensor)
        torch.cuda.synchronize()
        inf_time = time.perf_counter() - inf_start

        # Postprocessing
        post_start = time.perf_counter()
        # Simulate NMS and other post-processing
        if hasattr(output, 'cpu'):
            _ = output.cpu()
        post_time = time.perf_counter() - post_start

        torch.cuda.synchronize()
        total_time = time.perf_counter() - total_start

        components['preprocessing'].append(prep_time * 1000)
        components['inference'].append(inf_time * 1000)
        components['postprocessing'].append(post_time * 1000)
        components['total'].append(total_time * 1000)

    # Analyze results
    for component, times in components.items():
        avg_time = np.mean(times)
        percentage = (avg_time / np.mean(components['total'])) * 100 if component != 'total' else 100
        print(f"{component:15s}: {avg_time:.2f} ms ({percentage:.1f}%)")

    # 3. Memory transfer analysis
    print("\n3. MEMORY TRANSFER ANALYSIS")
    print("-"*50)

    # Test different data transfer methods
    data_sizes = [(640, 640), (1280, 1280), (1920, 1080)]

    for h, w in data_sizes:
        img_test = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # CPU to GPU transfer time
        tensor = torch.from_numpy(img_test).float()

        torch.cuda.synchronize()
        start = time.perf_counter()
        tensor_gpu = tensor.to('cuda')
        torch.cuda.synchronize()
        transfer_time = (time.perf_counter() - start) * 1000

        size_mb = (h * w * 3 * 4) / (1024 * 1024)  # float32
        bandwidth = size_mb / (transfer_time / 1000)  # GB/s

        print(f"{h}x{w}: {transfer_time:.2f} ms for {size_mb:.1f} MB ({bandwidth:.1f} GB/s)")

    # 4. Identify optimization opportunities
    print("\n4. OPTIMIZATION OPPORTUNITIES")
    print("-"*50)

    bottlenecks = []

    # Check preprocessing bottleneck
    prep_percentage = (np.mean(components['preprocessing']) / np.mean(components['total'])) * 100
    if prep_percentage > 30:
        bottlenecks.append(f"Preprocessing takes {prep_percentage:.1f}% - Consider GPU preprocessing")

    # Check postprocessing bottleneck
    post_percentage = (np.mean(components['postprocessing']) / np.mean(components['total'])) * 100
    if post_percentage > 20:
        bottlenecks.append(f"Postprocessing takes {post_percentage:.1f}% - Optimize NMS")

    # Check memory transfer
    if transfer_time > 5:
        bottlenecks.append(f"Memory transfer slow ({transfer_time:.1f} ms) - Use pinned memory")

    if bottlenecks:
        print("Found bottlenecks:")
        for b in bottlenecks:
            print(f"  • {b}")
    else:
        print("✓ No major bottlenecks found")

    return components

if __name__ == "__main__":
    profile_inference_pipeline()