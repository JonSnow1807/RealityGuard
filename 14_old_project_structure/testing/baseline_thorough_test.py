#!/usr/bin/env python3
"""
Thorough baseline testing to verify actual performance.
Tests for any misleading metrics, caching effects, or false claims.
"""

import cv2
import numpy as np
import time
import hashlib
import psutil
import torch
from typing import Dict, List, Tuple
import json
import gc
import sys


class BaselineVerification:
    """Complete verification of baseline CV system performance."""

    def __init__(self):
        self.results = {}
        self.frame_sizes = [
            (640, 480),   # VGA
            (1280, 720),  # HD
            (1920, 1080), # Full HD
        ]

    def create_test_frames(self, size: Tuple[int, int], count: int = 5) -> List[np.ndarray]:
        """Create diverse test frames to prevent caching."""
        frames = []
        h, w = size

        # Random noise frame
        frames.append(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))

        # Gradient frame
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.linspace(0, 255, w)
        frames.append(gradient)

        # Checkerboard pattern
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        block_size = 32
        for i in range(0, h, block_size * 2):
            for j in range(0, w, block_size * 2):
                checker[i:i+block_size, j:j+block_size] = 255
                checker[i+block_size:i+block_size*2, j+block_size:j+block_size*2] = 255
        frames.append(checker)

        # Natural-like image with edges
        natural = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(natural, (w//2, h//2), min(w, h)//4, (100, 150, 200), -1)
        cv2.rectangle(natural, (w//4, h//4), (3*w//4, 3*h//4), (50, 100, 150), 5)
        frames.append(natural)

        # High frequency pattern
        high_freq = np.random.normal(128, 50, (h, w, 3)).astype(np.uint8)
        frames.append(high_freq)

        return frames[:count]

    def test_simple_blur(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Test simple Gaussian blur baseline."""
        # Clear any GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Hash before
        before_hash = hashlib.md5(frame.tobytes()).hexdigest()

        # Time the blur operation
        start = time.perf_counter()
        blurred = cv2.GaussianBlur(frame, (31, 31), 10)
        end = time.perf_counter()

        # Verify blur was applied
        after_hash = hashlib.md5(blurred.tobytes()).hexdigest()
        blur_applied = (before_hash != after_hash)

        # Check actual pixel changes
        pixel_diff = np.mean(np.abs(frame.astype(float) - blurred.astype(float)))

        return (end - start), blur_applied and pixel_diff > 1.0

    def test_mediapipe_baseline(self, frame: np.ndarray) -> Tuple[float, Dict]:
        """Test MediaPipe face detection baseline."""
        import mediapipe as mp

        mp_face = mp.solutions.face_detection
        face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        results = face_detection.process(rgb_frame)
        end = time.perf_counter()

        face_detection.close()

        return (end - start), {
            'faces_detected': results.detections is not None,
            'num_faces': len(results.detections) if results.detections else 0
        }

    def test_vision_transformer_claim(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Test the Vision Transformer claims from meta_ready_cv_system.py."""
        # Simulate the ViT processing (without actual model weights)
        h, w = frame.shape[:2]

        # Resize to 224x224 as ViT expects
        start = time.perf_counter()
        resized = cv2.resize(frame, (224, 224))

        # Simulate patch extraction (32x32 patches)
        patch_size = 32
        patches = []
        for i in range(0, 224, patch_size):
            for j in range(0, 224, patch_size):
                patch = resized[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())

        # Simulate transformer operations (matrix multiplications)
        patches_array = np.array(patches)  # Shape: (49, 3072)
        feature_dim = 192  # Reduced dimension as in OptimizedViT

        # Project to lower dimension
        projection = np.random.randn(patches_array.shape[1], feature_dim)
        patches_array = patches_array @ projection  # Now shape: (49, 192)

        # Simulate 3 transformer blocks with 3 heads
        for _ in range(3):  # 3 depth
            # Self-attention simulation
            attn_weights = np.random.randn(len(patches), len(patches))
            attended = patches_array @ patches_array.T  # Attention scores
            patches_array = attended @ patches_array  # Apply attention

            # MLP simulation (keeping dimensions consistent)
            mlp_weights = np.random.randn(feature_dim, feature_dim)
            patches_array = patches_array @ mlp_weights

        end = time.perf_counter()

        return (end - start), True

    def test_gpu_acceleration(self, frame: np.ndarray) -> Dict:
        """Test if GPU acceleration is actually being used."""
        results = {}

        # Check CUDA availability
        results['cuda_available'] = torch.cuda.is_available()

        if results['cuda_available']:
            # Test GPU memory before/after
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated()

            # Convert to tensor and move to GPU
            tensor = torch.from_numpy(frame).cuda()

            # Simulate GPU operations
            start = time.perf_counter()
            blurred = torch.nn.functional.conv2d(
                tensor.permute(2, 0, 1).unsqueeze(0).float(),
                torch.ones(3, 1, 5, 5).cuda() / 25,
                padding=2,
                groups=3
            )
            torch.cuda.synchronize()
            end = time.perf_counter()

            after_mem = torch.cuda.memory_allocated()

            results['gpu_time'] = end - start
            results['memory_used'] = after_mem - before_mem
            results['gpu_name'] = torch.cuda.get_device_name()
        else:
            results['gpu_time'] = None
            results['memory_used'] = 0
            results['gpu_name'] = 'No GPU'

        return results

    def test_caching_effects(self, frame: np.ndarray) -> Dict:
        """Test for caching that might inflate performance."""
        times = []

        # First run (cold cache)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = time.perf_counter()
        _ = cv2.GaussianBlur(frame, (31, 31), 10)
        end = time.perf_counter()
        cold_time = end - start

        # Multiple runs (warm cache)
        for _ in range(10):
            start = time.perf_counter()
            _ = cv2.GaussianBlur(frame, (31, 31), 10)
            end = time.perf_counter()
            times.append(end - start)

        warm_time = np.mean(times)

        return {
            'cold_cache_time': cold_time,
            'warm_cache_time': warm_time,
            'speedup_from_cache': cold_time / warm_time if warm_time > 0 else 1.0,
            'variance': np.std(times)
        }

    def run_comprehensive_test(self):
        """Run all tests comprehensively."""
        print("=" * 60)
        print("BASELINE THOROUGH VERIFICATION TEST")
        print("=" * 60)

        all_results = {}

        for size in self.frame_sizes:
            print(f"\nTesting {size[0]}x{size[1]}...")
            size_key = f"{size[0]}x{size[1]}"
            all_results[size_key] = {}

            frames = self.create_test_frames(size)

            # Test simple blur
            blur_times = []
            blur_success = []
            for i, frame in enumerate(frames):
                time_taken, success = self.test_simple_blur(frame)
                blur_times.append(time_taken)
                blur_success.append(success)
                print(f"  Frame {i+1}: {1/time_taken:.1f} FPS, Blur applied: {success}")

            all_results[size_key]['simple_blur'] = {
                'avg_fps': 1/np.mean(blur_times),
                'min_fps': 1/max(blur_times),
                'max_fps': 1/min(blur_times),
                'success_rate': sum(blur_success) / len(blur_success),
                'times': blur_times
            }

            # Test MediaPipe (on first frame only due to speed)
            if size == (1280, 720):  # Test on HD only
                mp_time, mp_results = self.test_mediapipe_baseline(frames[0])
                all_results[size_key]['mediapipe'] = {
                    'fps': 1/mp_time if mp_time > 0 else 0,
                    'details': mp_results
                }
                print(f"  MediaPipe: {1/mp_time:.1f} FPS")

            # Test ViT claims
            vit_time, _ = self.test_vision_transformer_claim(frames[0])
            all_results[size_key]['vit_simulation'] = {
                'fps': 1/vit_time,
                'note': 'Simulated ViT operations without model weights'
            }
            print(f"  ViT simulation: {1/vit_time:.1f} FPS")

            # Test GPU
            gpu_results = self.test_gpu_acceleration(frames[0])
            all_results[size_key]['gpu'] = gpu_results
            if gpu_results['cuda_available']:
                print(f"  GPU ({gpu_results['gpu_name']}): {1/gpu_results['gpu_time']:.1f} FPS")

            # Test caching
            cache_results = self.test_caching_effects(frames[0])
            all_results[size_key]['caching'] = cache_results
            print(f"  Cache speedup: {cache_results['speedup_from_cache']:.2f}x")

        # System info
        all_results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A',
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__,
            'torch_version': torch.__version__ if 'torch' in sys.modules else 'N/A'
        }

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for size_key, results in all_results.items():
            if size_key != 'system_info':
                print(f"\n{size_key}:")
                if 'simple_blur' in results:
                    blur = results['simple_blur']
                    print(f"  Simple Blur: {blur['avg_fps']:.1f} FPS (success: {blur['success_rate']*100:.0f}%)")
                if 'mediapipe' in results:
                    print(f"  MediaPipe: {results['mediapipe']['fps']:.1f} FPS")
                if 'vit_simulation' in results:
                    print(f"  ViT Simulation: {results['vit_simulation']['fps']:.1f} FPS")

        # Save results
        with open('baseline_verification_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nResults saved to baseline_verification_results.json")

        # Key findings
        print("\n" + "=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)

        hd_results = all_results.get('1280x720', {})
        if 'simple_blur' in hd_results:
            blur_fps = hd_results['simple_blur']['avg_fps']
            print(f"✓ HD Simple Blur: {blur_fps:.1f} FPS")

            if blur_fps < 200:
                print("  → This is NORMAL for CPU-based Gaussian blur")
                print("  → Claims of 200+ FPS for full processing are suspicious")

            if hd_results['simple_blur']['success_rate'] < 1.0:
                print("  ⚠ Some frames not being blurred properly!")

        if 'caching' in hd_results:
            cache = hd_results['caching']
            if cache['speedup_from_cache'] > 1.5:
                print(f"⚠ Significant caching effect: {cache['speedup_from_cache']:.2f}x speedup")
                print("  → This could inflate benchmark results")

        if 'gpu' in hd_results and hd_results['gpu']['cuda_available']:
            print(f"✓ GPU available: {hd_results['gpu']['gpu_name']}")
        else:
            print("✗ No GPU acceleration detected")

        return all_results


if __name__ == "__main__":
    verifier = BaselineVerification()
    results = verifier.run_comprehensive_test()