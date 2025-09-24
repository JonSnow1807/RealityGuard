#!/usr/bin/env python3
"""
Final CUDA-Optimized Test with Real Performance Metrics
Testing all optimization techniques thoroughly
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import cv2
from ultralytics import YOLO
import json
from pathlib import Path
import gc

class FinalCUDATest:
    """Final comprehensive CUDA testing with all optimizations."""

    def __init__(self):
        self.device = torch.device('cuda')

        # Enable ALL CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("="*80)
        print("FINAL CUDA-OPTIMIZED TESTING")
        print("="*80)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("Optimizations Enabled:")
        print("  ✓ cuDNN benchmark mode")
        print("  ✓ TF32 for Tensor Cores")
        print("  ✓ Mixed precision (AMP)")
        print("="*80)

        self.results = {}

    def test_baseline_performance(self):
        """Test baseline YOLOv8 performance without optimizations."""
        print("\n1. BASELINE PERFORMANCE (No Optimizations)")
        print("-"*50)

        # Disable optimizations temporarily
        torch.backends.cudnn.benchmark = False

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Test single image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            _ = model(img, verbose=False, device='cuda')

        torch.cuda.synchronize()
        start = time.perf_counter()

        iterations = 100
        for _ in range(iterations):
            _ = model(img, verbose=False, device='cuda')

        torch.cuda.synchronize()
        baseline_time = time.perf_counter() - start
        baseline_fps = iterations / baseline_time

        self.results['baseline_single'] = baseline_fps
        print(f"Baseline single image: {baseline_fps:.1f} FPS")

        # Re-enable optimizations
        torch.backends.cudnn.benchmark = True

        return baseline_fps

    def test_optimized_performance(self):
        """Test with all CUDA optimizations enabled."""
        print("\n2. OPTIMIZED PERFORMANCE (All Optimizations)")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Single image with mixed precision
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup with optimizations
        for _ in range(10):
            with autocast():
                _ = model(img, verbose=False, device='cuda')

        torch.cuda.synchronize()
        start = time.perf_counter()

        iterations = 100
        for _ in range(iterations):
            with autocast():
                _ = model(img, verbose=False, device='cuda')

        torch.cuda.synchronize()
        optimized_time = time.perf_counter() - start
        optimized_fps = iterations / optimized_time

        self.results['optimized_single'] = optimized_fps
        print(f"Optimized single image: {optimized_fps:.1f} FPS")
        print(f"Speedup: {optimized_fps/self.results['baseline_single']:.2f}x")

        return optimized_fps

    def test_batch_processing(self):
        """Test batch processing performance."""
        print("\n3. BATCH PROCESSING PERFORMANCE")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        batch_sizes = [1, 2, 4, 8, 16, 32]
        batch_results = {}

        for batch_size in batch_sizes:
            # Create batch
            batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    for _ in range(batch_size)]

            # Warmup
            for _ in range(5):
                with autocast():
                    _ = model(batch, verbose=False, device='cuda')

            torch.cuda.synchronize()
            start = time.perf_counter()

            iterations = 20
            for _ in range(iterations):
                with autocast():
                    _ = model(batch, verbose=False, device='cuda')

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            total_images = iterations * batch_size
            fps = total_images / elapsed

            batch_results[batch_size] = fps
            self.results[f'batch_{batch_size}'] = fps

            print(f"Batch {batch_size:2d}: {fps:7.1f} FPS "
                  f"({elapsed/iterations*1000:.1f} ms/batch, "
                  f"{elapsed/total_images*1000:.2f} ms/img)")

        # Find optimal batch size
        optimal_batch = max(batch_results, key=batch_results.get)
        print(f"\nOptimal batch size: {optimal_batch} ({batch_results[optimal_batch]:.1f} FPS)")

        return batch_results

    def test_multi_stream(self):
        """Test multi-stream parallel processing."""
        print("\n4. MULTI-STREAM PARALLEL PROCESSING")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Test configuration: 4 streams, 4 images each
        num_streams = 4
        images_per_stream = 4

        # Create batches for each stream
        batches = []
        for _ in range(num_streams):
            batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    for _ in range(images_per_stream)]
            batches.append(batch)

        # Single stream baseline
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            for batch in batches:
                with autocast():
                    _ = model(batch, verbose=False, device='cuda')

        torch.cuda.synchronize()
        single_stream_time = time.perf_counter() - start

        # Multi-stream execution
        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            for i, (stream, batch) in enumerate(zip(streams, batches)):
                with torch.cuda.stream(stream):
                    with autocast():
                        _ = model(batch, verbose=False, device='cuda')

        # Wait for all streams
        for stream in streams:
            stream.synchronize()

        multi_stream_time = time.perf_counter() - start

        total_images = 10 * num_streams * images_per_stream
        single_fps = total_images / single_stream_time
        multi_fps = total_images / multi_stream_time

        self.results['single_stream'] = single_fps
        self.results['multi_stream'] = multi_fps

        print(f"Single stream: {single_fps:.1f} FPS")
        print(f"Multi-stream:  {multi_fps:.1f} FPS")
        print(f"Speedup: {multi_fps/single_fps:.2f}x")

        return multi_fps

    def test_memory_efficiency(self):
        """Test memory usage and efficiency."""
        print("\n5. MEMORY EFFICIENCY TEST")
        print("-"*50)

        # Clear cache first
        torch.cuda.empty_cache()
        gc.collect()

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Initial memory
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial memory: {initial_mem:.2f} GB")

        # Process increasingly large batches
        max_batch = 1
        try:
            for batch_size in [1, 2, 4, 8, 16, 32, 64]:
                batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        for _ in range(batch_size)]

                with autocast():
                    _ = model(batch, verbose=False, device='cuda')

                current_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"Batch {batch_size:2d}: {current_mem:.2f} GB used")
                max_batch = batch_size

        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory at batch size {batch_size}")

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

        self.results['memory'] = {
            'peak_gb': peak_mem,
            'total_gb': total_mem,
            'utilization': peak_mem / total_mem * 100,
            'max_batch': max_batch
        }

        print(f"\nPeak memory: {peak_mem:.2f} GB / {total_mem:.1f} GB ({peak_mem/total_mem*100:.1f}%)")
        print(f"Maximum batch size: {max_batch}")

        torch.cuda.empty_cache()

    def test_real_world_scenario(self):
        """Test realistic production scenario."""
        print("\n6. REAL-WORLD PRODUCTION SCENARIO")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Simulate real video processing (30 FPS input)
        print("Simulating 30 FPS video input for 10 seconds...")

        frames_to_process = 300  # 30 FPS × 10 seconds
        batch_size = 8  # Optimal batch size

        # Create frames
        frames = []
        for i in range(frames_to_process):
            # Simulate varying complexity
            if i % 10 == 0:
                # Complex scene
                frame = np.random.randint(50, 255, (720, 1280, 3), dtype=np.uint8)
            else:
                # Simple scene
                frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
            frames.append(frame)

        # Process in batches
        torch.cuda.synchronize()
        start = time.perf_counter()

        results_count = 0
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            with autocast():
                results = model(batch, verbose=False, device='cuda')
                results_count += len(results)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        avg_fps = frames_to_process / total_time
        real_time_factor = avg_fps / 30  # How many times faster than real-time

        self.results['real_world'] = {
            'fps': avg_fps,
            'total_time': total_time,
            'frames': frames_to_process,
            'real_time_factor': real_time_factor
        }

        print(f"Processed {frames_to_process} frames in {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Real-time factor: {real_time_factor:.1f}x")

        if real_time_factor >= 1.0:
            print("✓ Can process 30 FPS video in real-time")
        else:
            print(f"✗ Too slow for real-time (need {30/avg_fps:.1f}x speedup)")

    def test_mobile_projection(self):
        """Project performance to mobile devices."""
        print("\n7. MOBILE PROJECTION (10-15% of GPU)")
        print("-"*50)

        # Use conservative 10% projection
        conservative = 0.10
        optimistic = 0.15

        single_fps = self.results.get('optimized_single', 0)
        batch_8_fps = self.results.get('batch_8', 0)

        mobile_single_conservative = single_fps * conservative
        mobile_single_optimistic = single_fps * optimistic
        mobile_batch_conservative = batch_8_fps * conservative
        mobile_batch_optimistic = batch_8_fps * optimistic

        self.results['mobile_projection'] = {
            'single_conservative': mobile_single_conservative,
            'single_optimistic': mobile_single_optimistic,
            'batch_conservative': mobile_batch_conservative,
            'batch_optimistic': mobile_batch_optimistic
        }

        print("Single image processing:")
        print(f"  Conservative (10%): {mobile_single_conservative:.1f} FPS")
        print(f"  Optimistic (15%):   {mobile_single_optimistic:.1f} FPS")

        print("\nBatch processing (8 images):")
        print(f"  Conservative (10%): {mobile_batch_conservative:.1f} FPS")
        print(f"  Optimistic (15%):   {mobile_batch_optimistic:.1f} FPS")

        if mobile_batch_optimistic >= 30:
            print("\n✓ Mobile deployment VIABLE with batch processing")
        elif mobile_single_optimistic >= 30:
            print("\n✓ Mobile deployment VIABLE with single image processing")
        else:
            print(f"\n✗ Mobile performance below 30 FPS threshold")
            print(f"  Need {30/mobile_batch_optimistic:.1f}x more optimization")

    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*80)
        print("FINAL PERFORMANCE REPORT")
        print("="*80)

        # Key metrics
        baseline = self.results.get('baseline_single', 0)
        optimized = self.results.get('optimized_single', 0)
        batch_optimal = max([self.results.get(f'batch_{i}', 0)
                            for i in [1, 2, 4, 8, 16, 32]])

        print("\nKEY METRICS:")
        print(f"  Baseline FPS:        {baseline:.1f}")
        print(f"  Optimized FPS:       {optimized:.1f}")
        print(f"  Best Batch FPS:      {batch_optimal:.1f}")
        print(f"  Optimization Gain:   {optimized/baseline:.2f}x")
        print(f"  Batch Speedup:       {batch_optimal/optimized:.2f}x")

        # Memory efficiency
        mem = self.results.get('memory', {})
        if mem:
            print(f"\nMEMORY USAGE:")
            print(f"  Peak Usage:          {mem['peak_gb']:.2f} GB")
            print(f"  GPU Utilization:     {mem['utilization']:.1f}%")
            print(f"  Max Batch Size:      {mem['max_batch']}")

        # Real-world performance
        rw = self.results.get('real_world', {})
        if rw:
            print(f"\nREAL-WORLD PERFORMANCE:")
            print(f"  Processing Speed:    {rw['fps']:.1f} FPS")
            print(f"  Real-time Factor:    {rw['real_time_factor']:.1f}x")

        # Mobile viability
        mobile = self.results.get('mobile_projection', {})
        if mobile:
            print(f"\nMOBILE VIABILITY:")
            print(f"  Conservative:        {mobile['batch_conservative']:.1f} FPS")
            print(f"  Optimistic:          {mobile['batch_optimistic']:.1f} FPS")

            if mobile['batch_optimistic'] >= 30:
                print(f"  Status:              ✓ VIABLE")
            else:
                print(f"  Status:              ✗ NEEDS OPTIMIZATION")

        # Save results
        with open('final_cuda_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print("\nResults saved to final_cuda_results.json")

        return self.results


def main():
    """Run final comprehensive CUDA tests."""
    tester = FinalCUDATest()

    try:
        # Run all tests
        tester.test_baseline_performance()
        tester.test_optimized_performance()
        tester.test_batch_processing()
        tester.test_multi_stream()
        tester.test_memory_efficiency()
        tester.test_real_world_scenario()
        tester.test_mobile_projection()

        # Generate report
        results = tester.generate_report()

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        batch_fps = max([results.get(f'batch_{i}', 0) for i in [8, 16, 32]])
        mobile_fps = results.get('mobile_projection', {}).get('batch_optimistic', 0)

        if batch_fps > 200 and mobile_fps > 30:
            print("✅ EXCELLENT - Production Ready")
            print(f"   • GPU Performance: {batch_fps:.0f} FPS")
            print(f"   • Mobile Projection: {mobile_fps:.0f} FPS")
            print("   • All targets achieved")
        elif batch_fps > 100:
            print("✓ GOOD - Viable with optimization")
            print(f"   • GPU Performance: {batch_fps:.0f} FPS")
            print(f"   • Mobile Projection: {mobile_fps:.0f} FPS")
            print("   • Additional optimization recommended")
        else:
            print("⚠️ NEEDS WORK")
            print(f"   • GPU Performance: {batch_fps:.0f} FPS")
            print(f"   • Mobile Projection: {mobile_fps:.0f} FPS")
            print("   • Significant optimization required")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print("\n✓ Test completed and cleaned up")


if __name__ == "__main__":
    main()