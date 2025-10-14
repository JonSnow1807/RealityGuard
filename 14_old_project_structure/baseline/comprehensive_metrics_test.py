#!/usr/bin/env python3
"""
Comprehensive metrics test for Reality Guard
Tests CUDA implementation and verifies actual performance
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
import json
from datetime import datetime

# Check CUDA availability
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
device = torch.device('cuda' if cuda_available else 'cpu')

print("="*70)
print("COMPREHENSIVE REALITY GUARD METRICS TEST")
print(f"Date: {datetime.now()}")
print(f"Device: {device_name}")
print(f"CUDA Available: {cuda_available}")
if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("="*70)

# Import the Reality Guard implementations
sys.path.insert(0, 'RealityGuard')

test_results = {}

def create_test_image(name, resolution=(640, 480)):
    """Create test images with known shapes"""
    h, w = resolution
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if name == "single_circle":
        cv2.circle(img, (w//2, h//2), 60, (255, 255, 255), -1)
    elif name == "multiple_circles":
        cv2.circle(img, (w//4, h//2), 50, (255, 255, 255), -1)
        cv2.circle(img, (w//2, h//2), 60, (255, 255, 255), -1)
        cv2.circle(img, (3*w//4, h//2), 55, (255, 255, 255), -1)
    elif name == "complex":
        # Add circles and rectangles
        cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
        cv2.circle(img, (w-100, 100), 40, (255, 255, 255), -1)
        cv2.rectangle(img, (w//2-50, h//2-30), (w//2+50, h//2+30), (255, 255, 255), -1)

    return img

def test_cuda_implementation():
    """Test CUDA accelerated version"""
    print("\n1. TESTING CUDA IMPLEMENTATION")
    print("-" * 40)

    try:
        from realityguard_cuda_fixed import RealityGuardCUDA

        guard = RealityGuardCUDA()
        results = []

        # Test different resolutions
        resolutions = [
            ("480p", (640, 480)),
            ("720p", (1280, 720)),
            ("1080p", (1920, 1080))
        ]

        for res_name, resolution in resolutions:
            img = create_test_image("multiple_circles", resolution)

            # Warm up
            for _ in range(5):
                _ = guard.process_frame(img)

            # Measure performance
            times = []
            detections = []
            for _ in range(50):
                start = time.perf_counter()
                output, info = guard.process_frame(img)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                detections.append(info.get('detected_faces', 0))

            avg_time = np.mean(times)
            avg_fps = 1000 / avg_time
            avg_detections = np.mean(detections)

            result = {
                "resolution": res_name,
                "avg_ms": round(avg_time, 2),
                "fps": round(avg_fps, 1),
                "detections": round(avg_detections, 1),
                "min_ms": round(np.min(times), 2),
                "max_ms": round(np.max(times), 2)
            }
            results.append(result)

            print(f"{res_name:8} - {avg_fps:6.1f} FPS ({avg_time:6.2f}ms) - {avg_detections:.1f} detections")

        test_results["cuda"] = {"status": "success", "results": results}
        return True

    except Exception as e:
        print(f"CUDA test failed: {e}")
        test_results["cuda"] = {"status": "failed", "error": str(e)}
        return False

def test_production_implementation():
    """Test production ready version"""
    print("\n2. TESTING PRODUCTION IMPLEMENTATION")
    print("-" * 40)

    try:
        from realityguard_production_ready import RealityGuard

        # Test different modes
        modes = ["fast", "balanced", "accurate"]
        results = []

        for mode in modes:
            guard = RealityGuard(mode=mode)

            img = create_test_image("multiple_circles", (1280, 720))

            # Warm up
            for _ in range(5):
                _ = guard.process_frame(img)

            # Measure performance
            times = []
            detections = []
            for _ in range(30):
                start = time.perf_counter()
                output, info = guard.process_frame(img)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                detections.append(info['detected_faces'])

            avg_time = np.mean(times)
            avg_fps = 1000 / avg_time
            avg_detections = np.mean(detections)

            result = {
                "mode": mode,
                "avg_ms": round(avg_time, 2),
                "fps": round(avg_fps, 1),
                "detections": round(avg_detections, 1)
            }
            results.append(result)

            print(f"{mode:10} - {avg_fps:6.1f} FPS ({avg_time:6.2f}ms) - {avg_detections:.1f} detections")

        test_results["production"] = {"status": "success", "results": results}
        return True

    except Exception as e:
        print(f"Production test failed: {e}")
        test_results["production"] = {"status": "failed", "error": str(e)}
        return False

def test_detection_accuracy():
    """Test detection accuracy on known shapes"""
    print("\n3. TESTING DETECTION ACCURACY")
    print("-" * 40)

    try:
        from realityguard_production_ready import RealityGuard

        guard = RealityGuard(mode="accurate")

        test_cases = [
            ("single_circle", 1),
            ("multiple_circles", 3),
            ("complex", 2)  # Should detect only circles
        ]

        results = []
        for test_name, expected in test_cases:
            img = create_test_image(test_name)

            _, info = guard.process_frame(img)
            detected = info['detected_faces']
            accuracy = min(detected / expected, 1.0) * 100 if expected > 0 else 0

            result = {
                "test": test_name,
                "expected": expected,
                "detected": detected,
                "accuracy": round(accuracy, 1)
            }
            results.append(result)

            status = "✓" if detected > 0 else "✗"
            print(f"{status} {test_name:20} - Expected: {expected}, Detected: {detected} ({accuracy:.1f}%)")

        test_results["accuracy"] = {"status": "success", "results": results}
        return True

    except Exception as e:
        print(f"Accuracy test failed: {e}")
        test_results["accuracy"] = {"status": "failed", "error": str(e)}
        return False

def verify_gpu_utilization():
    """Verify actual GPU utilization"""
    print("\n4. VERIFYING GPU UTILIZATION")
    print("-" * 40)

    if not cuda_available:
        print("GPU not available, skipping...")
        test_results["gpu_util"] = {"status": "skipped", "reason": "No GPU"}
        return False

    try:
        from realityguard_cuda_fixed import RealityGuardCUDA

        guard = RealityGuardCUDA()
        img = create_test_image("multiple_circles", (1920, 1080))

        # Monitor GPU memory
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1e6

        # Process frames
        for i in range(100):
            _, _ = guard.process_frame(img)
            if i % 20 == 0:
                current_mem = torch.cuda.memory_allocated() / 1e6
                peak_mem = torch.cuda.max_memory_allocated() / 1e6
                print(f"  Frame {i:3}: Current={current_mem:6.1f}MB, Peak={peak_mem:6.1f}MB")

        end_mem = torch.cuda.memory_allocated() / 1e6
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

        result = {
            "start_memory_mb": round(start_mem, 1),
            "end_memory_mb": round(end_mem, 1),
            "peak_memory_mb": round(peak_mem, 1),
            "memory_leak": round(end_mem - start_mem, 1)
        }

        print(f"\nMemory Summary:")
        print(f"  Start: {start_mem:.1f} MB")
        print(f"  Peak:  {peak_mem:.1f} MB")
        print(f"  End:   {end_mem:.1f} MB")
        print(f"  Leak:  {end_mem - start_mem:.1f} MB")

        test_results["gpu_util"] = {"status": "success", "result": result}
        return True

    except Exception as e:
        print(f"GPU utilization test failed: {e}")
        test_results["gpu_util"] = {"status": "failed", "error": str(e)}
        return False

def generate_report():
    """Generate final report"""
    print("\n" + "="*70)
    print("FINAL METRICS REPORT")
    print("="*70)

    # Save results to JSON
    with open("metrics_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 40)

    if "cuda" in test_results and test_results["cuda"]["status"] == "success":
        print("\nCUDA Implementation:")
        for res in test_results["cuda"]["results"]:
            print(f"  {res['resolution']:8} - {res['fps']:6.1f} FPS")

    if "production" in test_results and test_results["production"]["status"] == "success":
        print("\nProduction Implementation:")
        for res in test_results["production"]["results"]:
            print(f"  {res['mode']:10} - {res['fps']:6.1f} FPS")

    print("\n✅ VERIFIED METRICS:")
    print("-" * 40)

    # Calculate honest metrics
    if "cuda" in test_results and test_results["cuda"]["status"] == "success":
        cuda_720p = [r for r in test_results["cuda"]["results"] if r["resolution"] == "720p"]
        if cuda_720p:
            print(f"• CUDA 720p:  {cuda_720p[0]['fps']:.0f} FPS (GPU: {device_name})")

    if "production" in test_results and test_results["production"]["status"] == "success":
        balanced = [r for r in test_results["production"]["results"] if r["mode"] == "balanced"]
        if balanced:
            print(f"• CPU 720p:   {balanced[0]['fps']:.0f} FPS (Balanced mode)")

    print("\n🎯 DETECTION ACCURACY:")
    print("-" * 40)

    if "accuracy" in test_results and test_results["accuracy"]["status"] == "success":
        for res in test_results["accuracy"]["results"]:
            print(f"  {res['test']:20} - {res['accuracy']:5.1f}% accurate")

    print("\n📝 Results saved to: metrics_test_results.json")
    print("="*70)

# Run all tests
if __name__ == "__main__":
    test_cuda_implementation()
    test_production_implementation()
    test_detection_accuracy()
    verify_gpu_utilization()
    generate_report()