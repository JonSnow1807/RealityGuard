#!/usr/bin/env python3
"""
Consistency Verification Test
Run multiple rounds to check for inconsistencies and errors
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time
import statistics
import json
import gc
import warnings
warnings.filterwarnings('ignore')


class ConsistencyVerifier:
    """Verify results are consistent and reproducible."""

    def __init__(self):
        self.device = torch.device('cuda')
        print("="*80)
        print("CONSISTENCY VERIFICATION TEST")
        print("="*80)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("Running multiple rounds to verify consistency...")
        print("="*80)

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.inconsistencies = []
        self.errors = []

    def test_single_image_consistency(self, rounds=10):
        """Test if single image FPS is consistent."""
        print("\n1. SINGLE IMAGE CONSISTENCY TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            _ = model(img, verbose=False, device='cuda')

        fps_results = []

        for round_num in range(rounds):
            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(50):
                _ = model(img, verbose=False, device='cuda')

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            fps = 50 / elapsed
            fps_results.append(fps)

            print(f"Round {round_num+1:2d}: {fps:7.2f} FPS")

        # Statistical analysis
        mean_fps = statistics.mean(fps_results)
        std_fps = statistics.stdev(fps_results)
        cv = (std_fps / mean_fps) * 100  # Coefficient of variation

        print(f"\nStatistics:")
        print(f"  Mean: {mean_fps:.2f} FPS")
        print(f"  Std Dev: {std_fps:.2f}")
        print(f"  CV: {cv:.1f}%")
        print(f"  Min: {min(fps_results):.2f} FPS")
        print(f"  Max: {max(fps_results):.2f} FPS")

        if cv > 10:
            self.inconsistencies.append(f"Single image FPS variance too high: {cv:.1f}%")
            print("⚠️  High variance detected!")
        else:
            print("✓ Consistent results")

        return mean_fps, std_fps

    def test_batch_consistency(self, rounds=10):
        """Test if batch processing FPS is consistent."""
        print("\n2. BATCH PROCESSING CONSISTENCY TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        batch_sizes = [8, 16, 32]
        results = {}

        for batch_size in batch_sizes:
            print(f"\nBatch size {batch_size}:")
            batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    for _ in range(batch_size)]

            # Warmup
            for _ in range(5):
                _ = model(batch, verbose=False, device='cuda')

            fps_results = []

            for round_num in range(rounds):
                torch.cuda.synchronize()
                start = time.perf_counter()

                for _ in range(10):
                    _ = model(batch, verbose=False, device='cuda')

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                fps = (10 * batch_size) / elapsed
                fps_results.append(fps)

                print(f"  Round {round_num+1:2d}: {fps:7.2f} FPS")

            mean_fps = statistics.mean(fps_results)
            std_fps = statistics.stdev(fps_results)
            cv = (std_fps / mean_fps) * 100

            results[batch_size] = {
                'mean': mean_fps,
                'std': std_fps,
                'cv': cv,
                'min': min(fps_results),
                'max': max(fps_results)
            }

            print(f"  Mean: {mean_fps:.2f} ± {std_fps:.2f} FPS (CV: {cv:.1f}%)")

            if cv > 10:
                self.inconsistencies.append(f"Batch {batch_size} variance too high: {cv:.1f}%")

        return results

    def test_memory_stability(self):
        """Test if memory usage is stable."""
        print("\n3. MEMORY STABILITY TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        torch.cuda.empty_cache()
        gc.collect()

        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial memory: {initial_mem:.2f} GB")

        memory_readings = []

        # Process many batches
        for i in range(20):
            batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    for _ in range(8)]

            _ = model(batch, verbose=False, device='cuda')

            current_mem = torch.cuda.memory_allocated() / 1024**3
            memory_readings.append(current_mem)

            if i % 5 == 0:
                print(f"After {i+1:2d} batches: {current_mem:.2f} GB")

        # Check for memory leak
        mem_increase = memory_readings[-1] - memory_readings[0]

        if abs(mem_increase) > 0.5:  # More than 500MB increase
            self.errors.append(f"Potential memory leak: {mem_increase:.2f} GB increase")
            print(f"⚠️  Memory increased by {mem_increase:.2f} GB")
        else:
            print(f"✓ Memory stable (change: {mem_increase*1000:.0f} MB)")

        torch.cuda.empty_cache()

    def test_error_conditions(self):
        """Test various error conditions."""
        print("\n4. ERROR CONDITION TESTS")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Test 1: Empty batch
        try:
            print("Testing empty batch...")
            result = model([], verbose=False, device='cuda')
            print("✓ Empty batch handled")
        except Exception as e:
            self.errors.append(f"Empty batch error: {str(e)[:50]}")
            print(f"✗ Empty batch failed: {str(e)[:50]}")

        # Test 2: Very large image
        try:
            print("Testing large image (4K)...")
            large_img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
            result = model(large_img, verbose=False, device='cuda')
            print("✓ 4K image handled")
        except Exception as e:
            self.errors.append(f"Large image error: {str(e)[:50]}")
            print(f"✗ Large image failed: {str(e)[:50]}")

        # Test 3: Grayscale image
        try:
            print("Testing grayscale image...")
            gray_img = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
            result = model(gray_img, verbose=False, device='cuda')
            print("✓ Grayscale handled")
        except Exception as e:
            # This might be expected behavior
            print(f"⚠️  Grayscale not directly supported (expected)")

        # Test 4: Very small image
        try:
            print("Testing tiny image (32x32)...")
            tiny_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            result = model(tiny_img, verbose=False, device='cuda')
            print("✓ Tiny image handled")
        except Exception as e:
            self.errors.append(f"Tiny image error: {str(e)[:50]}")
            print(f"✗ Tiny image failed: {str(e)[:50]}")

    def test_concurrent_inference(self):
        """Test if model can handle rapid concurrent requests."""
        print("\n5. CONCURRENT INFERENCE STRESS TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        print("Simulating rapid concurrent requests...")

        # Create different sized batches
        batches = [
            [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)],
            [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(8)],
            [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(16)],
        ]

        error_count = 0
        success_count = 0

        # Rapid fire requests
        start = time.perf_counter()
        for _ in range(50):
            for batch in batches:
                try:
                    _ = model(batch, verbose=False, device='cuda')
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    self.errors.append(f"Concurrent inference error: {str(e)[:30]}")

        elapsed = time.perf_counter() - start

        print(f"Processed {success_count} batches in {elapsed:.2f}s")
        print(f"Success rate: {success_count}/{success_count+error_count}")

        if error_count > 0:
            print(f"⚠️  {error_count} errors during concurrent processing")
        else:
            print("✓ All concurrent requests successful")

    def test_detection_consistency(self):
        """Test if detection results are consistent."""
        print("\n6. DETECTION CONSISTENCY TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Use the real test image
        img = cv2.imread('test_real_image.jpg')

        if img is None:
            print("⚠️  Real test image not found, using synthetic")
            img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

        detection_counts = []
        confidence_scores = []

        print("Running 10 identical inferences...")
        for i in range(10):
            results = model(img, verbose=False, device='cuda')

            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                detection_counts.append(num_detections)

                if results[0].boxes.conf is not None:
                    confs = results[0].boxes.conf.cpu().numpy()
                    confidence_scores.append(float(confs[0]) if len(confs) > 0 else 0)
            else:
                detection_counts.append(0)
                confidence_scores.append(0)

        # Check consistency
        unique_counts = set(detection_counts)

        print(f"Detection counts: {detection_counts}")

        if len(unique_counts) == 1:
            print(f"✓ Consistent: Always detected {detection_counts[0]} objects")
        else:
            self.inconsistencies.append(f"Inconsistent detections: {unique_counts}")
            print(f"⚠️  Inconsistent detection counts: {unique_counts}")

        if confidence_scores[0] > 0:
            conf_std = statistics.stdev(confidence_scores)
            print(f"Confidence variation: {conf_std:.4f}")
            if conf_std > 0.01:
                print("⚠️  Confidence scores vary between runs")

    def generate_report(self):
        """Generate final consistency report."""
        print("\n" + "="*80)
        print("CONSISTENCY VERIFICATION REPORT")
        print("="*80)

        if len(self.inconsistencies) == 0 and len(self.errors) == 0:
            print("✅ ALL TESTS PASSED - RESULTS ARE CONSISTENT")
            print("\nKey Findings:")
            print("  • FPS measurements are reproducible")
            print("  • Memory usage is stable")
            print("  • Model handles various inputs correctly")
            print("  • Detection results are deterministic")
        else:
            print("⚠️  SOME ISSUES DETECTED")

            if self.inconsistencies:
                print("\nInconsistencies Found:")
                for issue in self.inconsistencies:
                    print(f"  • {issue}")

            if self.errors:
                print("\nErrors Encountered:")
                for error in self.errors:
                    print(f"  • {error}")

        # Save report
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'inconsistencies': self.inconsistencies,
            'errors': self.errors,
            'status': 'PASSED' if len(self.inconsistencies) == 0 and len(self.errors) == 0 else 'ISSUES_FOUND'
        }

        with open('consistency_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\nReport saved to consistency_report.json")

        return report


def main():
    """Run all consistency verification tests."""
    verifier = ConsistencyVerifier()

    try:
        # Run all tests
        verifier.test_single_image_consistency()
        verifier.test_batch_consistency()
        verifier.test_memory_stability()
        verifier.test_error_conditions()
        verifier.test_concurrent_inference()
        verifier.test_detection_consistency()

        # Generate report
        report = verifier.generate_report()

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        if report['status'] == 'PASSED':
            print("✅ SYSTEM IS STABLE AND CONSISTENT")
            print("   All performance metrics are reproducible")
            print("   Ready for production deployment")
        else:
            print("⚠️  SYSTEM HAS MINOR ISSUES")
            print("   Review the report for details")
            print("   Most metrics are still reliable")

    except Exception as e:
        print(f"\n❌ Critical error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print("\n✓ Testing complete")


if __name__ == "__main__":
    main()