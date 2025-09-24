"""
Honest Performance Verification Script
Tests REAL performance with actual image processing
No fake metrics, no shortcuts
"""

import torch
import numpy as np
import cv2
import time
import psutil
import os
import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class HonestTester:
    """Thorough, honest performance testing"""

    def __init__(self):
        self.results = {}
        self.process = psutil.Process(os.getpid())

    def create_realistic_test_frames(self, num_frames=100):
        """Create realistic test frames (not random noise)"""
        frames = []

        # Create frames with actual patterns that would trigger processing
        for i in range(num_frames):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Add realistic elements
            # Face-like rectangle
            cv2.rectangle(frame, (100, 100), (200, 250), (200, 150, 100), -1)

            # Screen-like rectangle
            cv2.rectangle(frame, (400, 200), (800, 500), (100, 100, 200), -1)

            # Add some noise/texture
            noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)

            # Add motion simulation (shift elements)
            if i > 0:
                M = np.float32([[1, 0, i*2], [0, 1, i]])
                frame = cv2.warpAffine(frame, M, (1280, 720))

            frames.append(frame)

        return frames

    def measure_real_fps(self, process_func, frames, name="Unknown"):
        """Measure ACTUAL FPS with proper timing"""
        print(f"\n📊 Testing {name}...")

        # Memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            gpu_mem_before = 0

        # Warm-up (don't count these)
        print("  Warming up...")
        for _ in range(10):
            try:
                _ = process_func(frames[0])
            except:
                pass

        # Wait for GPU to finish warm-up
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ACTUAL TIMING
        print("  Measuring...")
        successful_frames = 0
        failed_frames = 0
        processing_times = []

        # Use wall clock time (the only truth)
        wall_start = time.perf_counter()

        for frame in frames:
            frame_start = time.perf_counter()
            try:
                result = process_func(frame)
                successful_frames += 1
            except Exception as e:
                failed_frames += 1
                print(f"    Error: {e}")
            frame_time = time.perf_counter() - frame_start
            processing_times.append(frame_time * 1000)  # Convert to ms

        # Wait for GPU to finish ALL work
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_end = time.perf_counter()
        total_wall_time = wall_end - wall_start

        # Memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            gpu_mem_after = 0

        # Calculate REAL metrics
        actual_fps = successful_frames / total_wall_time if total_wall_time > 0 else 0
        avg_latency = np.mean(processing_times) if processing_times else 0
        max_latency = np.max(processing_times) if processing_times else 0
        min_latency = np.min(processing_times) if processing_times else 0

        result = {
            'name': name,
            'actual_fps': actual_fps,
            'successful_frames': successful_frames,
            'failed_frames': failed_frames,
            'total_time': total_wall_time,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'memory_increase_mb': mem_after - mem_before,
            'gpu_memory_mb': gpu_mem_after - gpu_mem_before,
            'success_rate': successful_frames / len(frames) * 100 if frames else 0
        }

        return result

    def test_all_implementations(self):
        """Test all implementations honestly"""
        print("="*60)
        print("🔍 HONEST PERFORMANCE VERIFICATION")
        print("="*60)

        # Create realistic test data
        print("\n🎬 Creating realistic test frames...")
        frames = self.create_realistic_test_frames(100)
        print(f"  Created {len(frames)} frames (720x1280x3)")

        # Test implementations
        implementations = [
            ('realityguard_production', 'RealityGuardEngine'),
            ('realityguard_optimized', 'RealityGuardOptimized'),
            ('realityguard_tensorrt', 'RealityGuardTensorRT'),
            ('realityguard_1000fps', 'RealityGuard1000FPS'),
            ('patent_algorithm', 'PredictivePrivacyGradient'),
        ]

        results = []

        for module_name, class_name in implementations:
            try:
                # Import module
                module = importlib.import_module(module_name)

                # Get class
                cls = getattr(module, class_name, None)
                if cls is None:
                    print(f"❌ Could not find {class_name} in {module_name}")
                    continue

                # Create instance
                if module_name == 'realityguard_optimized':
                    # This one needs config
                    from realityguard_optimized import Config
                    system = cls(Config())
                elif module_name == 'patent_algorithm':
                    # Patent algorithm is different
                    system = cls()
                    # Create wrapper function
                    def process_func(frame):
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                        imu_data = {'gyro_x': 0, 'gyro_y': 0.1, 'gyro_z': 0}
                        threats = [{'bbox': (100, 100, 200, 150), 'confidence': 0.9}]
                        motion = system.predict_motion_vector(imu_data)
                        gradient = system.compute_privacy_gradient(frame_tensor, motion, threats)
                        return gradient.cpu().numpy()
                else:
                    system = cls()

                # Create process function
                if module_name != 'patent_algorithm':
                    if hasattr(system, 'process_frame'):
                        process_func = lambda f: system.process_frame(f)
                    elif hasattr(system, 'process_frame_optimized'):
                        process_func = lambda f: system.process_frame_optimized(f)
                    elif hasattr(system, 'process_frame_ultra_fast'):
                        process_func = lambda f: system.process_frame_ultra_fast(f)
                    else:
                        print(f"❌ No process function in {class_name}")
                        continue

                # Test it
                result = self.measure_real_fps(process_func, frames, f"{module_name}.{class_name}")
                results.append(result)

            except Exception as e:
                print(f"❌ Failed to test {module_name}: {e}")
                traceback.print_exc()

        return results

    def test_edge_cases(self):
        """Test error handling and edge cases"""
        print("\n🔥 Testing Edge Cases...")

        edge_cases = [
            ("Empty frame", np.array([])),
            ("Wrong shape", np.zeros((100, 100))),  # 2D instead of 3D
            ("Huge frame", np.zeros((4000, 4000, 3), dtype=np.uint8)),
            ("Tiny frame", np.zeros((1, 1, 3), dtype=np.uint8)),
            ("None input", None),
            ("Wrong type", "not a frame"),
        ]

        # Test with optimized version (has error handling)
        try:
            from realityguard_optimized import RealityGuardOptimized, Config
            system = RealityGuardOptimized(Config())

            for name, bad_input in edge_cases:
                try:
                    result = system.process_frame(bad_input)
                    print(f"  ✅ {name}: Handled gracefully")
                except Exception as e:
                    print(f"  ❌ {name}: Crashed with {type(e).__name__}")

        except Exception as e:
            print(f"  Could not test edge cases: {e}")

    def print_results(self, results: List[Dict]):
        """Print results in a clear table"""
        print("\n" + "="*80)
        print("📊 FINAL HONEST RESULTS")
        print("="*80)

        # Sort by FPS
        results.sort(key=lambda x: x['actual_fps'], reverse=True)

        print(f"\n{'Implementation':<30} {'FPS':<10} {'Latency':<15} {'Success':<10} {'Memory':<10}")
        print("-"*80)

        for r in results:
            name = r['name'][:30]
            fps = f"{r['actual_fps']:.1f}"
            latency = f"{r['avg_latency_ms']:.2f}ms"
            success = f"{r['success_rate']:.0f}%"
            memory = f"{r['memory_increase_mb']:.1f}MB"

            print(f"{name:<30} {fps:<10} {latency:<15} {success:<10} {memory:<10}")

        print("\n📈 Summary:")
        best = results[0] if results else None
        if best:
            print(f"  Best FPS: {best['actual_fps']:.1f} ({best['name']})")
            print(f"  Best meets 1000 FPS target: {'✅ YES' if best['actual_fps'] >= 1000 else '❌ NO'}")

            if best['actual_fps'] < 1000:
                gap = 1000 - best['actual_fps']
                print(f"  Gap to target: {gap:.1f} FPS")

        # GPU info
        if torch.cuda.is_available():
            print(f"\n💻 System Info:")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  CPU cores: {psutil.cpu_count()}")
            print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")


def main():
    """Run honest verification"""
    tester = HonestTester()

    # Test all implementations
    results = tester.test_all_implementations()

    # Test edge cases
    tester.test_edge_cases()

    # Print results
    tester.print_results(results)

    # Save to file
    with open("honest_benchmark_results.txt", "w") as f:
        f.write("HONEST PERFORMANCE VERIFICATION\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test frames: 100 realistic 720x1280x3 frames\n\n")

        for r in results:
            f.write(f"{r['name']}:\n")
            f.write(f"  FPS: {r['actual_fps']:.1f}\n")
            f.write(f"  Latency: {r['avg_latency_ms']:.2f}ms\n")
            f.write(f"  Success rate: {r['success_rate']:.0f}%\n")
            f.write(f"  Memory: {r['memory_increase_mb']:.1f}MB\n\n")

    print("\n📁 Results saved to honest_benchmark_results.txt")

    # Final verdict
    print("\n" + "="*60)
    print("🎯 FINAL VERDICT")
    print("="*60)

    if results:
        best_fps = max(r['actual_fps'] for r in results)
        if best_fps >= 1000:
            print("✅ We ACTUALLY achieved 1000+ FPS!")
        else:
            print(f"❌ Honest result: {best_fps:.1f} FPS (need {1000-best_fps:.1f} more)")
    else:
        print("❌ No successful tests")


if __name__ == "__main__":
    main()