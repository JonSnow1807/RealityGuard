#!/usr/bin/env python3
"""
Comprehensive test of current performance claims.
Testing what actually works vs what doesn't.
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def test_baseline():
    """Test baseline YOLOv8 performance."""
    print("\n" + "="*80)
    print("BASELINE TEST: Standard YOLOv8n")
    print("="*80)

    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    # Test single image
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = model(test_img, verbose=False, device='cuda')

    # Single image test
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 100
    for _ in range(iterations):
        _ = model(test_img, verbose=False, device='cuda')

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    single_fps = iterations / elapsed
    print(f"Single image: {single_fps:.1f} FPS")

    return single_fps


def test_batch_processing():
    """Test batch processing optimization."""
    print("\n" + "="*80)
    print("BATCH PROCESSING TEST")
    print("="*80)

    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    batch_sizes = [1, 4, 8, 16, 32]
    results = {}

    for batch_size in batch_sizes:
        # Create batch
        batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                 for _ in range(batch_size)]

        # Warmup
        for _ in range(5):
            _ = model(batch, verbose=False, device='cuda')

        # Test
        torch.cuda.synchronize()
        start = time.perf_counter()

        iterations = 50
        for _ in range(iterations):
            _ = model(batch, verbose=False, device='cuda')

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        fps = (iterations * batch_size) / elapsed
        avg_time_per_image = (elapsed / iterations) / batch_size * 1000

        print(f"Batch {batch_size:2d}: {fps:6.1f} FPS ({avg_time_per_image:.2f} ms/img)")
        results[batch_size] = fps

    best_batch = max(results.items(), key=lambda x: x[1])
    print(f"\nBest: Batch {best_batch[0]} = {best_batch[1]:.1f} FPS")

    return results


def test_cuda_optimizations():
    """Test with all CUDA optimizations enabled."""
    print("\n" + "="*80)
    print("CUDA OPTIMIZATIONS TEST")
    print("="*80)

    # Enable all optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Enabled:")
    print("  ✓ cuDNN benchmark mode")
    print("  ✓ TF32 for Tensor Cores")
    print("  ✓ Mixed precision")

    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    # Test with optimal batch size (32)
    batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
             for _ in range(32)]

    # Warmup
    for _ in range(5):
        with torch.cuda.amp.autocast():
            _ = model(batch, verbose=False, device='cuda')

    # Test
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 50
    for _ in range(iterations):
        with torch.cuda.amp.autocast():
            _ = model(batch, verbose=False, device='cuda')

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = (iterations * 32) / elapsed
    print(f"\nOptimized batch 32: {fps:.1f} FPS")

    return fps


def test_mixed_precision():
    """Test mixed precision impact."""
    print("\n" + "="*80)
    print("MIXED PRECISION COMPARISON")
    print("="*80)

    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Test without AMP
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        _ = model(test_img, verbose=False, device='cuda')

    torch.cuda.synchronize()
    no_amp_time = time.perf_counter() - start
    no_amp_fps = 100 / no_amp_time

    # Test with AMP
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        with torch.cuda.amp.autocast():
            _ = model(test_img, verbose=False, device='cuda')

    torch.cuda.synchronize()
    amp_time = time.perf_counter() - start
    amp_fps = 100 / amp_time

    print(f"Without AMP: {no_amp_fps:.1f} FPS")
    print(f"With AMP:    {amp_fps:.1f} FPS")
    print(f"Difference:  {(amp_fps/no_amp_fps - 1)*100:+.1f}%")

    return no_amp_fps, amp_fps


def test_memory_usage():
    """Test GPU memory usage."""
    print("\n" + "="*80)
    print("MEMORY USAGE TEST")
    print("="*80)

    # Clear cache
    torch.cuda.empty_cache()

    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    # Check memory after loading model
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"After model load:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")

    # Test with large batch
    batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
             for _ in range(64)]

    try:
        _ = model(batch, verbose=False, device='cuda')

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"\nAfter batch 64:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  ✅ Large batch successful")
    except torch.cuda.OutOfMemoryError:
        print(f"  ❌ Out of memory with batch 64")

    return allocated, reserved


def test_real_video():
    """Test on simulated video stream."""
    print("\n" + "="*80)
    print("REAL VIDEO SIMULATION TEST")
    print("="*80)

    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Simulate 100 frames
    frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
              for _ in range(100)]

    print("Processing 100 frames (640x640)...")

    torch.cuda.synchronize()
    start = time.perf_counter()

    for frame in frames:
        with torch.cuda.amp.autocast():
            _ = model(frame, verbose=False, device='cuda')

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = len(frames) / elapsed
    print(f"Average FPS: {fps:.1f}")
    print(f"Real-time factor: {fps/30:.1f}x (compared to 30 FPS video)")

    return fps


def main():
    """Run comprehensive performance tests."""
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE TEST")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print("="*80)

    results = {}

    # 1. Baseline test
    results['baseline'] = test_baseline()

    # 2. Batch processing
    results['batch'] = test_batch_processing()

    # 3. CUDA optimizations
    results['cuda_optimized'] = test_cuda_optimizations()

    # 4. Mixed precision comparison
    no_amp, amp = test_mixed_precision()
    results['mixed_precision'] = {'no_amp': no_amp, 'amp': amp}

    # 5. Memory usage
    allocated, reserved = test_memory_usage()
    results['memory'] = {'allocated_gb': allocated, 'reserved_gb': reserved}

    # 6. Real video test
    results['video'] = test_real_video()

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    print(f"Baseline (single):     {results['baseline']:.1f} FPS")
    print(f"Best batch:            {max(results['batch'].values()):.1f} FPS")
    print(f"CUDA optimized:        {results['cuda_optimized']:.1f} FPS")
    print(f"Video stream:          {results['video']:.1f} FPS")

    # Check 283 FPS claim
    print("\n" + "="*80)
    print("283 FPS CLAIM VERIFICATION")
    print("="*80)

    max_fps = max(results['cuda_optimized'], max(results['batch'].values()))

    if max_fps >= 283:
        print(f"✅ VERIFIED: {max_fps:.1f} FPS achieved!")
    else:
        print(f"❌ NOT ACHIEVED: Only {max_fps:.1f} FPS")
        print(f"   Gap to 283 FPS: {283 - max_fps:.1f} FPS ({max_fps/283*100:.1f}%)")

    # Save results
    with open('current_performance_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to current_performance_test.json")

    return results


if __name__ == "__main__":
    results = main()