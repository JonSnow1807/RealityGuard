#!/usr/bin/env python3
"""
Comprehensive Testing Suite for RealityGuard
Tests all implementations and documents performance.
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time
import json
from pathlib import Path
import psutil
import GPUtil


def test_system_info():
    """Get system information."""
    print("="*80)
    print("SYSTEM INFORMATION")
    print("="*80)

    # GPU Info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU: Not available")

    # CPU Info
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    # PyTorch Info
    print(f"PyTorch: {torch.__version__}")

    return {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'cuda': torch.version.cuda if torch.cuda.is_available() else 'None',
        'pytorch': torch.__version__
    }


def test_baseline_performance():
    """Test baseline YOLOv8 performance."""
    print("\n" + "="*80)
    print("TEST 1: BASELINE YOLO PERFORMANCE")
    print("="*80)

    model = YOLO('yolov8n.pt')
    model.to('cuda')

    # Test different image sizes
    sizes = [(640, 640), (1280, 720), (1920, 1080)]
    results = {}

    for size in sizes:
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            _ = model(img, verbose=False)

        # Test
        torch.cuda.synchronize()
        start = time.perf_counter()

        iterations = 100
        for _ in range(iterations):
            _ = model(img, verbose=False)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        fps = iterations / elapsed
        ms_per_frame = elapsed / iterations * 1000

        print(f"{size[0]}x{size[1]}: {fps:.1f} FPS ({ms_per_frame:.2f} ms/frame)")
        results[f"{size[0]}x{size[1]}"] = {
            'fps': fps,
            'ms_per_frame': ms_per_frame
        }

    return results


def test_batch_optimization():
    """Test batch processing optimization."""
    print("\n" + "="*80)
    print("TEST 2: BATCH PROCESSING OPTIMIZATION")
    print("="*80)

    model = YOLO('yolov8n.pt')
    model.to('cuda')

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = {}

    for batch_size in batch_sizes:
        try:
            batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    for _ in range(batch_size)]

            # Warmup
            for _ in range(5):
                _ = model(batch, verbose=False)

            # Test
            torch.cuda.synchronize()
            start = time.perf_counter()

            iterations = 20
            for _ in range(iterations):
                _ = model(batch, verbose=False)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            total_images = iterations * batch_size
            fps = total_images / elapsed

            print(f"Batch {batch_size:3d}: {fps:7.1f} FPS")
            results[batch_size] = fps

        except torch.cuda.OutOfMemoryError:
            print(f"Batch {batch_size:3d}: Out of memory")
            results[batch_size] = 0
            torch.cuda.empty_cache()

    return results


def test_realtime_blur():
    """Test real-time blur performance."""
    print("\n" + "="*80)
    print("TEST 3: REAL-TIME BLUR SYSTEM")
    print("="*80)

    from optimized_realtime_blur import OptimizedRealtimeBlur

    # Create test video
    print("Creating test video...")
    test_video = 'test_performance.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 30.0, (640, 480))

    for i in range(300):  # 10 seconds at 30 FPS
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Add moving objects
        x = (i * 5) % 500
        cv2.rectangle(frame, (x, 200), (x + 100, 350), (200, 150, 100), -1)
        cv2.circle(frame, (320, 100), 40, (255, 200, 150), -1)

        out.write(frame)
    out.release()

    # Test different configurations
    configs = [
        (1, 1.0, "High Quality"),
        (3, 0.5, "Optimized"),
        (5, 0.3, "Fast")
    ]

    results = {}

    for interval, scale, name in configs:
        print(f"\nTesting {name} (detect every {interval} frames, {scale} scale):")

        blur_system = OptimizedRealtimeBlur()
        blur_system.detection_interval = interval
        blur_system.detection_scale = scale

        fps = blur_system.process_video_optimized(test_video)

        results[name] = {
            'interval': interval,
            'scale': scale,
            'fps': fps,
            'realtime': fps >= 30
        }

    # Clean up
    Path(test_video).unlink(missing_ok=True)

    return results


def test_cuda_optimizations():
    """Test impact of CUDA optimizations."""
    print("\n" + "="*80)
    print("TEST 4: CUDA OPTIMIZATION IMPACT")
    print("="*80)

    model = YOLO('yolov8n.pt')
    model.to('cuda')

    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Test without optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = model(img, verbose=False)
    torch.cuda.synchronize()

    no_opt_fps = 100 / (time.perf_counter() - start)

    # Test with optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = model(img, verbose=False)
    torch.cuda.synchronize()

    with_opt_fps = 100 / (time.perf_counter() - start)

    print(f"Without CUDA optimizations: {no_opt_fps:.1f} FPS")
    print(f"With CUDA optimizations:    {with_opt_fps:.1f} FPS")
    print(f"Improvement: {(with_opt_fps/no_opt_fps - 1)*100:.1f}%")

    return {
        'no_optimization': no_opt_fps,
        'with_optimization': with_opt_fps,
        'improvement_percent': (with_opt_fps/no_opt_fps - 1) * 100
    }


def test_memory_usage():
    """Test GPU memory usage."""
    print("\n" + "="*80)
    print("TEST 5: MEMORY USAGE ANALYSIS")
    print("="*80)

    torch.cuda.empty_cache()

    # Baseline memory
    baseline_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Baseline memory: {baseline_memory:.2f} GB")

    # Load model
    model = YOLO('yolov8n.pt')
    model.to('cuda')

    model_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"After model load: {model_memory:.2f} GB")

    # Test different batch sizes
    memory_results = {}

    for batch_size in [1, 8, 32, 64]:
        try:
            batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    for _ in range(batch_size)]

            _ = model(batch, verbose=False)

            batch_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Batch {batch_size:2d}: {batch_memory:.2f} GB")

            memory_results[f"batch_{batch_size}"] = batch_memory

        except torch.cuda.OutOfMemoryError:
            print(f"Batch {batch_size:2d}: Out of memory")
            memory_results[f"batch_{batch_size}"] = "OOM"
            torch.cuda.empty_cache()

    return memory_results


def run_all_tests():
    """Run comprehensive test suite."""
    print("="*80)
    print("REALITYGUARD COMPREHENSIVE TESTING SUITE")
    print("="*80)
    print("Testing all implementations thoroughly...")

    all_results = {}

    # System info
    all_results['system'] = test_system_info()

    # Performance tests
    all_results['baseline'] = test_baseline_performance()
    all_results['batch'] = test_batch_optimization()
    all_results['realtime_blur'] = test_realtime_blur()
    all_results['cuda'] = test_cuda_optimizations()
    all_results['memory'] = test_memory_usage()

    # Summary
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*80)

    print("\n📊 KEY METRICS:")
    print(f"  • Baseline FPS (640x640): {all_results['baseline']['640x640']['fps']:.1f}")
    print(f"  • Best batch FPS: {max(all_results['batch'].values()):.1f}")
    print(f"  • Real-time blur: {all_results['realtime_blur']['Optimized']['fps']:.1f} FPS")
    print(f"  • CUDA optimization gain: {all_results['cuda']['improvement_percent']:.1f}%")

    print("\n✅ CAPABILITIES:")
    for config, data in all_results['realtime_blur'].items():
        status = "✅" if data['realtime'] else "❌"
        print(f"  {status} {config}: {data['fps']:.1f} FPS")

    # Save results
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n📁 Results saved to: comprehensive_test_results.json")

    return all_results


if __name__ == "__main__":
    results = run_all_tests()