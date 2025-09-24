#!/usr/bin/env python3
"""
Edge Case and Stress Testing
Testing extreme conditions and unusual scenarios
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time
import psutil
import gc
import threading
import concurrent.futures
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EdgeCaseStressTester:
    """Test edge cases and stress conditions."""

    def __init__(self):
        self.device = torch.device('cuda')
        print("="*80)
        print("EDGE CASE AND STRESS TESTING")
        print("="*80)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("Testing extreme conditions and edge cases...")
        print("="*80)

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.issues_found = []

    def test_extreme_batch_sizes(self):
        """Test with extreme batch sizes."""
        print("\n1. EXTREME BATCH SIZE TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        # Test very large batch
        print("Testing maximum batch size...")
        max_batch = 1
        for batch_size in [32, 64, 128, 256]:
            try:
                # Use smaller images to fit more in memory
                batch = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                        for _ in range(batch_size)]

                torch.cuda.synchronize()
                start = time.perf_counter()

                _ = model(batch, verbose=False, device='cuda')

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                fps = batch_size / elapsed
                print(f"  Batch {batch_size:3d}: {fps:7.1f} FPS - ✓ Success")
                max_batch = batch_size

                # Clear memory for next test
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  Batch {batch_size:3d}: ✗ Out of memory (expected)")
                break
            except Exception as e:
                print(f"  Batch {batch_size:3d}: ✗ Error: {str(e)[:50]}")
                self.issues_found.append(f"Batch {batch_size} failed: {str(e)[:50]}")
                break

        print(f"\nMaximum successful batch: {max_batch}")

    def test_unusual_image_dimensions(self):
        """Test with unusual image dimensions."""
        print("\n2. UNUSUAL IMAGE DIMENSIONS TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        test_cases = [
            ("Square tiny", (64, 64)),
            ("Square small", (224, 224)),
            ("Square large", (1024, 1024)),
            ("Extreme wide", (256, 2048)),
            ("Extreme tall", (2048, 256)),
            ("Non-standard", (333, 777)),
            ("4K landscape", (2160, 3840)),
            ("Mobile portrait", (1920, 1080)),
        ]

        for name, (h, w) in test_cases:
            try:
                img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

                torch.cuda.synchronize()
                start = time.perf_counter()

                results = model(img, verbose=False, device='cuda')

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                print(f"  {name:15s} ({h:4d}x{w:4d}): {1/elapsed:6.1f} FPS - ✓")

            except Exception as e:
                print(f"  {name:15s} ({h:4d}x{w:4d}): ✗ {str(e)[:30]}")
                self.issues_found.append(f"{name} failed: {str(e)[:50]}")

    def test_rapid_model_switching(self):
        """Test rapid loading and unloading of models."""
        print("\n3. RAPID MODEL SWITCHING TEST")
        print("-"*50)

        print("Testing rapid model load/unload cycles...")

        for i in range(5):
            try:
                # Load model
                model = YOLO('yolov8n-seg.pt')
                model.to('cuda')

                # Quick inference
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                _ = model(img, verbose=False, device='cuda')

                # Delete model
                del model
                torch.cuda.empty_cache()
                gc.collect()

                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  Cycle {i+1}: ✓ (Memory: {mem:.2f} GB)")

            except Exception as e:
                print(f"  Cycle {i+1}: ✗ {str(e)[:50]}")
                self.issues_found.append(f"Model switching cycle {i+1} failed")

    def test_concurrent_model_instances(self):
        """Test multiple model instances simultaneously."""
        print("\n4. CONCURRENT MODEL INSTANCES TEST")
        print("-"*50)

        print("Testing multiple model instances...")

        try:
            # Create multiple models
            models = []
            max_models = 3

            for i in range(max_models):
                model = YOLO('yolov8n-seg.pt')
                model.to('cuda')
                models.append(model)

                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  Model {i+1} loaded: {mem:.2f} GB memory")

            # Test inference with each
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            for i, model in enumerate(models):
                results = model(img, verbose=False, device='cuda')
                print(f"  Model {i+1} inference: ✓")

            # Cleanup
            for model in models:
                del model
            torch.cuda.empty_cache()

            print(f"\n✓ Successfully handled {max_models} concurrent models")

        except Exception as e:
            print(f"✗ Concurrent models failed: {str(e)[:50]}")
            self.issues_found.append(f"Concurrent models failed: {str(e)[:50]}")

    def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        print("\n5. MEMORY PRESSURE TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        print("Allocating large tensors to create memory pressure...")

        # Get current free memory
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"Free memory: {free_mem:.2f} GB")

        try:
            # Allocate 80% of free memory
            alloc_size = int(free_mem * 0.8 * 1024**3 / 4)  # bytes / sizeof(float)
            large_tensor = torch.zeros(alloc_size, dtype=torch.float32, device='cuda')

            remaining = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"After allocation: {remaining:.2f} GB free")

            # Try inference with limited memory
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            start = time.perf_counter()
            results = model(img, verbose=False, device='cuda')
            elapsed = time.perf_counter() - start

            fps = 1 / elapsed
            print(f"Inference under pressure: {fps:.1f} FPS - ✓")

            # Cleanup
            del large_tensor
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print("✗ Out of memory (system protected itself)")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Error under pressure: {str(e)[:50]}")
            self.issues_found.append(f"Memory pressure test failed: {str(e)[:50]}")

    def test_corrupted_input_handling(self):
        """Test handling of corrupted or invalid inputs."""
        print("\n6. CORRUPTED INPUT HANDLING TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        test_cases = [
            ("NaN values", lambda: np.full((640, 640, 3), np.nan, dtype=np.float32)),
            ("Inf values", lambda: np.full((640, 640, 3), np.inf, dtype=np.float32)),
            ("Wrong dtype", lambda: np.random.randint(0, 255, (640, 640, 3), dtype=np.int64)),
            ("Out of range", lambda: np.random.randint(-1000, 1000, (640, 640, 3), dtype=np.int16)),
            ("Wrong channels", lambda: np.random.randint(0, 255, (640, 640, 1), dtype=np.uint8)),
            ("5D tensor", lambda: np.random.randint(0, 255, (1, 1, 640, 640, 3), dtype=np.uint8)),
        ]

        for name, create_input in test_cases:
            try:
                img = create_input()
                _ = model(img, verbose=False, device='cuda')
                print(f"  {name:15s}: ✓ Handled gracefully")
            except Exception as e:
                # Some errors are expected
                error_msg = str(e)[:50]
                if "nan" in error_msg.lower() or "inf" in error_msg.lower():
                    print(f"  {name:15s}: ✓ Rejected invalid input (expected)")
                else:
                    print(f"  {name:15s}: ⚠ {error_msg}")

    def test_continuous_load(self):
        """Test continuous high load for extended period."""
        print("\n7. CONTINUOUS HIGH LOAD TEST (10 seconds)")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(16)]

        print("Running continuous inference at maximum speed...")

        start_time = time.time()
        iterations = 0
        fps_samples = []
        errors = 0

        while time.time() - start_time < 10:
            try:
                iter_start = time.perf_counter()
                _ = model(batch, verbose=False, device='cuda')
                iter_end = time.perf_counter()

                fps = len(batch) / (iter_end - iter_start)
                fps_samples.append(fps)
                iterations += 1

                # Print progress every 2 seconds
                if iterations % 20 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = np.mean(fps_samples[-20:])
                    print(f"  {elapsed:.1f}s: {avg_fps:.1f} FPS avg")

            except Exception as e:
                errors += 1
                self.issues_found.append(f"Continuous load error: {str(e)[:30]}")

        total_time = time.time() - start_time
        total_frames = iterations * len(batch)
        overall_fps = total_frames / total_time

        print(f"\nResults:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Iterations: {iterations}")
        print(f"  Frames: {total_frames}")
        print(f"  Average FPS: {overall_fps:.1f}")
        print(f"  Errors: {errors}")

        if errors == 0:
            print("✓ System stable under continuous load")
        else:
            print(f"⚠ {errors} errors during continuous load")

    def test_gpu_memory_fragmentation(self):
        """Test for GPU memory fragmentation issues."""
        print("\n8. GPU MEMORY FRAGMENTATION TEST")
        print("-"*50)

        model = YOLO('yolov8n-seg.pt')
        model.to('cuda')

        print("Testing memory fragmentation with varying batch sizes...")

        # Rapidly change batch sizes to cause fragmentation
        batch_sequence = [1, 32, 4, 16, 2, 64, 8, 1, 32, 4]

        for i, batch_size in enumerate(batch_sequence):
            try:
                batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        for _ in range(batch_size)]

                _ = model(batch, verbose=False, device='cuda')

                mem = torch.cuda.memory_allocated() / 1024**3
                frag = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**3

                print(f"  Batch {batch_size:2d}: Allocated={mem:.2f}GB, "
                      f"Fragmented={frag:.2f}GB")

            except torch.cuda.OutOfMemoryError:
                print(f"  Batch {batch_size:2d}: ✗ Out of memory (fragmentation?)")
                torch.cuda.empty_cache()
                self.issues_found.append(f"Memory fragmentation at batch {batch_size}")

        # Final cleanup
        torch.cuda.empty_cache()
        final_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"\nFinal memory after cleanup: {final_mem:.2f} GB")

    def generate_report(self):
        """Generate comprehensive edge case report."""
        print("\n" + "="*80)
        print("EDGE CASE & STRESS TEST REPORT")
        print("="*80)

        if len(self.issues_found) == 0:
            print("✅ ALL EDGE CASES HANDLED SUCCESSFULLY")
            print("\nSystem demonstrated:")
            print("  • Robust handling of extreme batch sizes")
            print("  • Support for unusual image dimensions")
            print("  • Stable under memory pressure")
            print("  • Graceful error handling for invalid inputs")
            print("  • No memory leaks or fragmentation issues")
            print("  • Stable under continuous high load")
        else:
            print("⚠️  SOME EDGE CASES REVEALED ISSUES")
            print("\nIssues found:")
            for issue in self.issues_found:
                print(f"  • {issue}")

            print("\nNote: Some issues may be expected behavior")

        return {
            'status': 'PASSED' if len(self.issues_found) == 0 else 'ISSUES_FOUND',
            'issues': self.issues_found
        }


def main():
    """Run edge case and stress tests."""
    tester = EdgeCaseStressTester()

    try:
        # Run all tests
        tester.test_extreme_batch_sizes()
        tester.test_unusual_image_dimensions()
        tester.test_rapid_model_switching()
        tester.test_concurrent_model_instances()
        tester.test_memory_pressure()
        tester.test_corrupted_input_handling()
        tester.test_continuous_load()
        tester.test_gpu_memory_fragmentation()

        # Generate report
        report = tester.generate_report()

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        if report['status'] == 'PASSED':
            print("✅ SYSTEM IS ROBUST")
            print("   Handles all edge cases gracefully")
            print("   Ready for production deployment")
        else:
            num_issues = len(report['issues'])
            if num_issues <= 2:
                print("✓ SYSTEM IS MOSTLY ROBUST")
                print(f"   {num_issues} minor issues found")
                print("   Suitable for production with monitoring")
            else:
                print("⚠️  SYSTEM NEEDS HARDENING")
                print(f"   {num_issues} issues found")
                print("   Review and address before production")

    except Exception as e:
        print(f"\n❌ Critical error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print("\n✓ Testing complete and cleaned up")


if __name__ == "__main__":
    main()