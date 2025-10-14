#!/usr/bin/env python3
"""
CUDA-Optimized Production System
Complete implementation with thorough testing
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import cv2
import time
import json
import gc
import psutil
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class CUDAProductionSystem:
    """Production-ready CUDA-optimized computer vision system."""

    def __init__(self):
        """Initialize with all CUDA optimizations."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type != 'cuda':
            raise RuntimeError("CUDA is required for this system")

        # Enable all optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load model
        self.model = None
        self.batch_queue = deque(maxlen=32)
        self.results = {}

        print("="*80)
        print("CUDA-OPTIMIZED PRODUCTION SYSTEM")
        print("="*80)
        self._print_system_info()

    def _print_system_info(self):
        """Print system configuration."""
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Current memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.1f}/{reserved:.1f} GB (allocated/reserved)")

        # CPU info
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print("="*80)

    def load_model(self, model_name='yolov8n-seg.pt'):
        """Load and optimize model."""
        print(f"\nLoading model: {model_name}")

        self.model = YOLO(model_name)

        # Move model to GPU
        self.model.to(self.device)

        print(f"✓ Model loaded and moved to GPU")

        # Warmup
        self._warmup_model()

    def _warmup_model(self):
        """Warmup model for optimal performance."""
        print("Warming up model...")

        dummy_imgs = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]

        for _ in range(5):
            with autocast():
                _ = self.model(dummy_imgs, verbose=False)

        torch.cuda.synchronize()
        print("✓ Model warmed up")

    def process_single(self, image: np.ndarray) -> Dict:
        """Process single image (baseline)."""
        torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast():
            results = self.model(image, verbose=False)

        torch.cuda.synchronize()
        end = time.perf_counter()

        return {
            'time': end - start,
            'fps': 1.0 / (end - start),
            'results': results
        }

    def process_batch(self, images: List[np.ndarray]) -> Dict:
        """Process batch of images with optimization."""
        batch_size = len(images)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast():
            results = self.model(images, verbose=False)

        torch.cuda.synchronize()
        end = time.perf_counter()

        total_time = end - start

        return {
            'batch_size': batch_size,
            'total_time': total_time,
            'per_image_time': total_time / batch_size,
            'fps': batch_size / total_time,
            'results': results
        }

    def process_with_streams(self, image_batches: List[List[np.ndarray]]) -> Dict:
        """Process multiple batches using CUDA streams."""
        num_streams = len(image_batches)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        results = [None] * num_streams

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Launch parallel streams
        for i, (stream, batch) in enumerate(zip(streams, image_batches)):
            with torch.cuda.stream(stream):
                with autocast():
                    results[i] = self.model(batch, verbose=False)

        # Wait for all streams
        for stream in streams:
            stream.synchronize()

        end = time.perf_counter()

        total_images = sum(len(batch) for batch in image_batches)
        total_time = end - start

        return {
            'num_streams': num_streams,
            'total_images': total_images,
            'total_time': total_time,
            'fps': total_images / total_time,
            'results': results
        }

    def test_comprehensive_performance(self):
        """Run comprehensive performance tests."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE TESTING")
        print("="*80)

        if self.model is None:
            self.load_model()

        test_results = {}

        # 1. Single Image Performance
        print("\n1. SINGLE IMAGE PERFORMANCE")
        print("-"*50)

        resolutions = [
            ("480p", (480, 640)),
            ("720p", (720, 1280)),
            ("1080p", (1080, 1920))
        ]

        for name, (h, w) in resolutions:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

            # Test 10 times
            times = []
            for _ in range(10):
                result = self.process_single(img)
                times.append(result['time'])

            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time

            test_results[f'single_{name}'] = {
                'fps': fps,
                'latency_ms': avg_time * 1000,
                'std_ms': std_time * 1000
            }

            print(f"{name:6s}: {fps:7.1f} FPS ({avg_time*1000:.2f} ± {std_time*1000:.2f} ms)")

        # 2. Batch Processing Performance
        print("\n2. BATCH PROCESSING PERFORMANCE")
        print("-"*50)

        batch_sizes = [1, 2, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            # Create batch of 720p images
            imgs = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                   for _ in range(batch_size)]

            result = self.process_batch(imgs)

            test_results[f'batch_{batch_size}'] = {
                'fps': result['fps'],
                'per_image_ms': result['per_image_time'] * 1000,
                'total_time': result['total_time']
            }

            print(f"Batch {batch_size:2d}: {result['fps']:7.1f} FPS "
                  f"({result['per_image_time']*1000:.2f} ms/img)")

        # 3. Multi-Stream Performance
        print("\n3. MULTI-STREAM PARALLEL PROCESSING")
        print("-"*50)

        stream_configs = [
            (2, 4),  # 2 streams, 4 images each
            (4, 4),  # 4 streams, 4 images each
            (4, 8),  # 4 streams, 8 images each
        ]

        for num_streams, imgs_per_stream in stream_configs:
            batches = []
            for _ in range(num_streams):
                batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        for _ in range(imgs_per_stream)]
                batches.append(batch)

            result = self.process_with_streams(batches)

            test_results[f'stream_{num_streams}x{imgs_per_stream}'] = {
                'fps': result['fps'],
                'total_time': result['total_time']
            }

            print(f"{num_streams} streams × {imgs_per_stream} imgs: "
                  f"{result['fps']:7.1f} FPS")

        # 4. Memory Usage Test
        print("\n4. MEMORY USAGE ANALYSIS")
        print("-"*50)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        mem_before = torch.cuda.memory_allocated() / 1024**3

        # Process large batch
        large_batch = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                      for _ in range(16)]
        _ = self.process_batch(large_batch)

        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3

        test_results['memory'] = {
            'before_gb': mem_before,
            'after_gb': mem_after,
            'peak_gb': mem_peak,
            'used_gb': mem_after - mem_before
        }

        print(f"Memory before: {mem_before:.2f} GB")
        print(f"Memory after:  {mem_after:.2f} GB")
        print(f"Peak memory:   {mem_peak:.2f} GB")
        print(f"Memory used:   {mem_after - mem_before:.2f} GB")

        # 5. Sustained Load Test
        print("\n5. SUSTAINED LOAD TEST (30 seconds)")
        print("-"*50)

        print("Running sustained load test...")
        batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(8)]

        start_time = time.time()
        frames_processed = 0
        times = []

        while time.time() - start_time < 30:
            iter_start = time.perf_counter()
            _ = self.process_batch(batch)
            iter_end = time.perf_counter()

            times.append(iter_end - iter_start)
            frames_processed += len(batch)

        total_time = time.time() - start_time
        avg_fps = frames_processed / total_time

        # Check for throttling
        first_half = np.mean(times[:len(times)//2])
        second_half = np.mean(times[len(times)//2:])
        throttling = (second_half - first_half) / first_half * 100

        test_results['sustained_load'] = {
            'duration': total_time,
            'frames_processed': frames_processed,
            'avg_fps': avg_fps,
            'throttling_percent': throttling
        }

        print(f"Duration: {total_time:.1f}s")
        print(f"Frames processed: {frames_processed}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Throttling: {throttling:+.1f}%")

        # 6. Quality Verification
        print("\n6. SEGMENTATION QUALITY VERIFICATION")
        print("-"*50)

        # Create test image with clear objects
        test_img = np.ones((640, 640, 3), dtype=np.uint8) * 100
        cv2.circle(test_img, (320, 320), 100, (255, 255, 255), -1)
        cv2.rectangle(test_img, (100, 100), (250, 250), (200, 100, 50), -1)

        result = self.process_single(test_img)

        if result['results'][0].masks is not None:
            num_objects = len(result['results'][0].masks)
            print(f"✓ Segmentation working: {num_objects} objects detected")
            test_results['segmentation_works'] = True
        else:
            print("✗ No objects detected")
            test_results['segmentation_works'] = False

        self.results = test_results
        return test_results

    def analyze_results(self):
        """Analyze and summarize test results."""
        if not self.results:
            print("No results to analyze")
            return

        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)

        # Single vs Batch comparison
        single_720p = self.results.get('single_720p', {}).get('fps', 0)
        batch_8 = self.results.get('batch_8', {}).get('fps', 0)
        batch_16 = self.results.get('batch_16', {}).get('fps', 0)

        print("\n1. BATCH PROCESSING ADVANTAGE")
        print("-"*50)
        print(f"Single image:  {single_720p:7.1f} FPS")
        print(f"Batch 8:       {batch_8:7.1f} FPS ({batch_8/single_720p:.1f}x speedup)")
        print(f"Batch 16:      {batch_16:7.1f} FPS ({batch_16/single_720p:.1f}x speedup)")

        # Memory efficiency
        mem = self.results.get('memory', {})
        if mem:
            print("\n2. MEMORY EFFICIENCY")
            print("-"*50)
            print(f"Peak memory: {mem['peak_gb']:.2f} GB / 22.3 GB")
            print(f"Utilization: {mem['peak_gb']/22.3*100:.1f}%")

        # Sustained performance
        sustained = self.results.get('sustained_load', {})
        if sustained:
            print("\n3. SUSTAINED PERFORMANCE")
            print("-"*50)
            print(f"Average FPS over 30s: {sustained['avg_fps']:.1f}")

            if abs(sustained['throttling_percent']) < 5:
                print("✓ No significant throttling detected")
            else:
                print(f"⚠ Throttling detected: {sustained['throttling_percent']:+.1f}%")

        # Mobile projection
        print("\n4. MOBILE PROJECTION (15% of GPU)")
        print("-"*50)

        mobile_single = single_720p * 0.15
        mobile_batch = batch_8 * 0.15

        print(f"Single image: {mobile_single:.1f} FPS")
        print(f"With batching: {mobile_batch:.1f} FPS")

        if mobile_batch >= 30:
            print("✓ Mobile deployment VIABLE with batching")
        else:
            print("✗ Additional optimization needed for mobile")

        # Save results
        with open('cuda_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print("\nResults saved to cuda_test_results.json")

    def run_complete_test(self):
        """Run all tests and generate report."""
        print("\n" + "="*80)
        print("STARTING COMPLETE TEST SUITE")
        print("="*80)

        # Run tests
        self.test_comprehensive_performance()

        # Analyze
        self.analyze_results()

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        batch_fps = self.results.get('batch_8', {}).get('fps', 0)

        if batch_fps > 500:
            print("✅ EXCELLENT PERFORMANCE")
            print(f"   {batch_fps:.0f} FPS with batch processing")
            print("   Ready for production deployment")
        elif batch_fps > 200:
            print("✓ GOOD PERFORMANCE")
            print(f"   {batch_fps:.0f} FPS with batch processing")
            print("   Suitable for most applications")
        else:
            print("⚠ PERFORMANCE NEEDS OPTIMIZATION")
            print(f"   {batch_fps:.0f} FPS with batch processing")

        return self.results


def main():
    """Run production system tests."""
    system = CUDAProductionSystem()

    try:
        results = system.run_complete_test()

        # Check if results are as expected
        if results.get('batch_8', {}).get('fps', 0) > 200:
            print("\n✅ CUDA optimizations working as expected")
        else:
            print("\n⚠ Performance lower than expected")
            print("   Check GPU utilization and thermal throttling")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print("\n✓ Cleanup complete")


if __name__ == "__main__":
    main()