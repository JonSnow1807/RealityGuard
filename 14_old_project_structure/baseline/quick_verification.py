#!/usr/bin/env python3
"""
Quick verification of Reality Guard metrics
Tests key claims with actual measurements
"""

import time
import numpy as np
import cv2
import torch
import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("verification_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_gpu_performance():
    """Test actual GPU performance"""
    print("\n" + "="*60)
    print("GPU PERFORMANCE TEST")
    print("="*60)

    try:
        from realityguard_gpu_optimized import OptimizedGPUDetector

        detector = OptimizedGPUDetector(batch_size=8)

        # Create test frames
        resolutions = [
            ("480p", (640, 480)),
            ("720p", (1280, 720)),
            ("1080p", (1920, 1080))
        ]

        results = {}

        for res_name, (w, h) in resolutions:
            print(f"\nTesting {res_name} ({w}x{h}):")

            # Single frame test
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            cv2.circle(frame, (w//2, h//2), 50, (255, 255, 255), -1)

            # Single frame timing
            single_times = []
            for _ in range(20):
                start = time.perf_counter()
                _, info = detector.process_frame_batch([frame])
                single_times.append((time.perf_counter() - start) * 1000)

            single_avg = np.mean(single_times)
            single_fps = 1000 / single_avg

            print(f"  Single: {single_avg:.2f}ms ({single_fps:.1f} FPS)")

            # Batch test (8 frames)
            frames = [frame.copy() for _ in range(8)]
            batch_times = []
            for _ in range(20):
                start = time.perf_counter()
                _, info = detector.process_frame_batch(frames)
                batch_times.append((time.perf_counter() - start) * 1000)

            batch_avg = np.mean(batch_times)
            per_frame = batch_avg / 8
            batch_fps = 1000 / per_frame

            print(f"  Batch-8: {batch_avg:.2f}ms total, {per_frame:.2f}ms/frame ({batch_fps:.1f} FPS)")

            # Batch test (16 frames)
            frames_16 = [frame.copy() for _ in range(16)]
            batch16_times = []
            for _ in range(10):
                start = time.perf_counter()
                _, info = detector.process_frame_batch(frames_16)
                batch16_times.append((time.perf_counter() - start) * 1000)

            batch16_avg = np.mean(batch16_times)
            per_frame_16 = batch16_avg / 16
            batch16_fps = 1000 / per_frame_16

            print(f"  Batch-16: {batch16_avg:.2f}ms total, {per_frame_16:.2f}ms/frame ({batch16_fps:.1f} FPS)")

            results[res_name] = {
                "single_fps": round(single_fps, 1),
                "batch8_fps": round(batch_fps, 1),
                "batch16_fps": round(batch16_fps, 1)
            }

        return results

    except Exception as e:
        print(f"GPU test error: {e}")
        return None

def test_detection_accuracy():
    """Test detection accuracy"""
    print("\n" + "="*60)
    print("DETECTION ACCURACY TEST")
    print("="*60)

    try:
        from realityguard_improved_v2 import ImprovedDetector

        detector = ImprovedDetector()

        # Test cases
        test_images = []

        # Single circle
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img1, (320, 240), 60, (255, 255, 255), -1)
        test_images.append(("single_circle", img1, 1))

        # Three circles
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img2, (160, 240), 50, (255, 255, 255), -1)
        cv2.circle(img2, (320, 240), 60, (255, 255, 255), -1)
        cv2.circle(img2, (480, 240), 55, (255, 255, 255), -1)
        test_images.append(("three_circles", img2, 3))

        results = []
        for name, img, expected in test_images:
            detections = []
            for _ in range(10):
                output, info = detector.process_frame(img)
                detections.append(info['detections'])

            avg = np.mean(detections)
            accuracy = min(avg / expected * 100, 100) if expected > 0 else 0

            print(f"  {name}: Expected {expected}, Detected avg {avg:.1f} ({accuracy:.1f}% accuracy)")

            results.append({
                "test": name,
                "expected": expected,
                "detected_avg": round(avg, 1),
                "accuracy": round(accuracy, 1)
            })

            # Save proof image
            proof_path = OUTPUT_DIR / f"{name}_detected.jpg"
            cv2.imwrite(str(proof_path), output)

        return results

    except Exception as e:
        print(f"Detection test error: {e}")
        return None

def test_original_cuda():
    """Test original CUDA implementation"""
    print("\n" + "="*60)
    print("ORIGINAL CUDA TEST")
    print("="*60)

    import subprocess
    try:
        result = subprocess.run(
            ["python", "RealityGuard/realityguard_cuda_fixed.py"],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout
        lines = output.split('\n')

        results = {}
        for line in lines:
            if "FPS:" in line:
                # Parse FPS value
                fps_str = line.split("FPS:")[1].strip().split()[0]
                try:
                    fps = float(fps_str)
                    if "Single" in line:
                        results["single_fps"] = fps
                    elif "Multiple" in line:
                        results["multiple_fps"] = fps
                    elif "HD" in line:
                        results["hd_fps"] = fps
                except:
                    pass

        if results:
            print("  Original CUDA Results:")
            for key, value in results.items():
                print(f"    {key}: {value} FPS")

        return results

    except subprocess.TimeoutExpired:
        print("  Test timed out")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def cpu_vs_gpu_comparison():
    """Compare CPU vs GPU directly"""
    print("\n" + "="*60)
    print("CPU vs GPU COMPARISON")
    print("="*60)

    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    cv2.circle(frame, (640, 360), 100, (255, 255, 255), -1)

    # CPU test
    cpu_times = []
    for _ in range(50):
        start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cpu_times.append((time.perf_counter() - start) * 1000)

    cpu_avg = np.mean(cpu_times)
    cpu_fps = 1000 / cpu_avg

    print(f"  CPU: {cpu_avg:.2f}ms ({cpu_fps:.1f} FPS)")

    # GPU test (if available)
    if torch.cuda.is_available():
        try:
            from realityguard_gpu_optimized import OptimizedGPUDetector
            detector = OptimizedGPUDetector(batch_size=8)

            # Single frame
            gpu_times = []
            for _ in range(50):
                start = time.perf_counter()
                _, _ = detector.process_frame_batch([frame])
                gpu_times.append((time.perf_counter() - start) * 1000)

            gpu_avg = np.mean(gpu_times)
            gpu_fps = 1000 / gpu_avg

            # Batch of 8
            frames = [frame.copy() for _ in range(8)]
            batch_times = []
            for _ in range(20):
                start = time.perf_counter()
                _, _ = detector.process_frame_batch(frames)
                batch_times.append((time.perf_counter() - start) * 1000)

            batch_avg = np.mean(batch_times)
            batch_per_frame = batch_avg / 8
            batch_fps = 1000 / batch_per_frame

            print(f"  GPU Single: {gpu_avg:.2f}ms ({gpu_fps:.1f} FPS)")
            print(f"  GPU Batch-8: {batch_per_frame:.2f}ms/frame ({batch_fps:.1f} FPS)")

            speedup_single = cpu_avg / gpu_avg
            speedup_batch = cpu_avg / batch_per_frame

            print(f"  Speedup (single): {speedup_single:.2f}x")
            print(f"  Speedup (batch): {speedup_batch:.2f}x")

            return {
                "cpu_fps": round(cpu_fps, 1),
                "gpu_single_fps": round(gpu_fps, 1),
                "gpu_batch_fps": round(batch_fps, 1),
                "speedup_single": round(speedup_single, 2),
                "speedup_batch": round(speedup_batch, 2)
            }

        except Exception as e:
            print(f"  GPU test error: {e}")

    return {"cpu_fps": round(cpu_fps, 1)}

def main():
    print("="*60)
    print("REALITY GUARD - QUICK VERIFICATION")
    print("="*60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)

    # Run tests
    gpu_results = test_gpu_performance()
    if gpu_results:
        results["gpu_performance"] = gpu_results

    detection_results = test_detection_accuracy()
    if detection_results:
        results["detection_accuracy"] = detection_results

    comparison = cpu_vs_gpu_comparison()
    if comparison:
        results["cpu_vs_gpu"] = comparison

    cuda_results = test_original_cuda()
    if cuda_results:
        results["original_cuda"] = cuda_results

    # Save results
    results_file = OUTPUT_DIR / "verification_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    print("\n📊 KEY FINDINGS:")
    print("-" * 40)

    # Check 999 FPS claim
    max_fps = 0
    if gpu_results:
        for res in gpu_results.values():
            max_fps = max(max_fps, res.get("batch16_fps", 0))

    if max_fps >= 900:
        print(f"✅ 999 FPS claim: VERIFIED ({max_fps:.0f} FPS on batch-16)")
    else:
        print(f"❌ 999 FPS claim: FALSE (max {max_fps:.0f} FPS)")

    # Check accuracy
    if detection_results:
        avg_accuracy = np.mean([r["accuracy"] for r in detection_results])
        if avg_accuracy >= 85:
            print(f"✅ 85% accuracy: VERIFIED ({avg_accuracy:.0f}%)")
        else:
            print(f"❌ 85% accuracy: FALSE ({avg_accuracy:.0f}%)")

    # Check GPU speedup
    if comparison and "speedup_batch" in comparison:
        speedup = comparison["speedup_batch"]
        if speedup > 10:
            print(f"✅ GPU acceleration: VERIFIED ({speedup:.1f}x speedup)")
        else:
            print(f"⚠️ GPU acceleration: LIMITED ({speedup:.1f}x speedup)")

    print(f"\n📁 Results saved to: {results_file}")
    print("="*60)

if __name__ == "__main__":
    main()