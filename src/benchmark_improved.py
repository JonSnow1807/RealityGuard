#!/usr/bin/env python3
"""
Benchmark tool for RealityGuard Improved System
Tests the new modular architecture with YOLO/MediaPipe face detection
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys
import json
from collections import deque
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realityguard_improved import RealityGuardImproved, PrivacyMode
from src.config import Config
from src.face_detector import ModernFaceDetector


class ImprovedBenchmark:
    """Benchmark suite for improved RealityGuard system"""

    def __init__(self):
        self.config = Config()
        self.guard = None
        self.results = {}

    def setup(self):
        """Initialize the improved system"""
        print("\nInitializing RealityGuard Improved System...")
        self.guard = RealityGuardImproved()

        # Check which face detector is being used
        detector_method = self.guard.face_detector.method
        print(f"Face Detector: {detector_method.upper()}")

        # Create synthetic calibration frame
        calibration_frame = self._create_test_frame(640, 480)
        if self.guard.calibrate_user(calibration_frame):
            print("User calibration: SUCCESS")
        else:
            print("User calibration: SKIPPED (no face in synthetic frame)")

        print(f"Configuration loaded: {self.config.performance.target_fps} FPS target")
        print(f"GPU Enabled: {self.config.performance.enable_gpu}")
        print(f"Caching Enabled: {self.config.performance.enable_caching}")

    def _create_test_frame(self, width: int = 1280, height: int = 720) -> np.ndarray:
        """Create a synthetic test frame with various elements"""
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50

        # Add face-like regions (bright rectangles)
        cv2.rectangle(frame, (100, 100), (250, 280), (200, 180, 160), -1)
        cv2.rectangle(frame, (400, 150), (520, 300), (190, 170, 150), -1)

        # Add screen-like region (very bright)
        cv2.rectangle(frame, (600, 200), (1100, 600), (240, 240, 240), -1)

        # Add some noise for realism
        noise = np.random.randint(0, 25, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame

    def benchmark_privacy_mode(self, mode: PrivacyMode, num_frames: int = 100) -> Dict:
        """Benchmark a specific privacy mode"""
        print(f"\nBenchmarking {mode.name} mode...")

        self.guard.set_privacy_mode(mode)

        frame_times = []
        fps_values = []

        # Create test frames of different sizes
        test_sizes = [(640, 480), (1280, 720), (1920, 1080)]

        for width, height in test_sizes:
            frame = self._create_test_frame(width, height)
            size_times = []

            # Warm-up
            for _ in range(10):
                _ = self.guard.process_frame(frame)

            # Actual benchmark
            for i in range(num_frames // len(test_sizes)):
                start = time.perf_counter()
                processed = self.guard.process_frame(frame)
                elapsed = time.perf_counter() - start

                size_times.append(elapsed)

                if elapsed > 0:
                    fps_values.append(1.0 / elapsed)

            frame_times.extend(size_times)
            avg_time = np.mean(size_times) * 1000
            print(f"  {width}x{height}: {avg_time:.2f}ms per frame ({1000/avg_time:.1f} FPS)")

        # Calculate statistics
        frame_times_ms = [t * 1000 for t in frame_times]

        results = {
            'mode': mode.name,
            'avg_fps': np.mean(fps_values) if fps_values else 0,
            'min_fps': np.min(fps_values) if fps_values else 0,
            'max_fps': np.max(fps_values) if fps_values else 0,
            'p50_fps': np.percentile(fps_values, 50) if fps_values else 0,
            'p95_fps': np.percentile(fps_values, 95) if fps_values else 0,
            'p99_fps': np.percentile(fps_values, 99) if fps_values else 0,
            'avg_latency_ms': np.mean(frame_times_ms),
            'p99_latency_ms': np.percentile(frame_times_ms, 99) if frame_times_ms else 0,
        }

        return results

    def benchmark_face_detectors(self, num_frames: int = 100) -> Dict:
        """Compare different face detection methods"""
        print("\n" + "="*60)
        print("FACE DETECTOR COMPARISON")
        print("="*60)

        methods = ['opencv']  # Start with OpenCV as baseline

        # Check available methods
        try:
            from ultralytics import YOLO
            methods.append('yolo')
        except ImportError:
            print("YOLO not available for comparison")

        try:
            import mediapipe
            methods.append('mediapipe')
        except ImportError:
            print("MediaPipe not available for comparison")

        detector_results = {}
        frame = self._create_test_frame(1280, 720)

        for method in methods:
            print(f"\nTesting {method.upper()} detector...")
            detector = ModernFaceDetector(method=method)

            times = []
            face_counts = []

            # Warm-up
            for _ in range(10):
                _ = detector.detect_faces(frame, use_cache=False)

            # Benchmark
            for _ in range(num_frames):
                start = time.perf_counter()
                faces = detector.detect_faces(frame, use_cache=False)
                elapsed = time.perf_counter() - start

                times.append(elapsed * 1000)  # Convert to ms
                face_counts.append(len(faces))

            detector_results[method] = {
                'avg_time_ms': np.mean(times),
                'p99_time_ms': np.percentile(times, 99),
                'avg_faces': np.mean(face_counts),
                'fps': 1000 / np.mean(times) if times else 0
            }

            detector.release()

            print(f"  Avg detection time: {detector_results[method]['avg_time_ms']:.2f}ms")
            print(f"  Detection FPS: {detector_results[method]['fps']:.1f}")
            print(f"  Avg faces found: {detector_results[method]['avg_faces']:.1f}")

        return detector_results

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("REALITYGUARD IMPROVED - FULL BENCHMARK")
        print("="*60)

        self.setup()

        # Benchmark each privacy mode
        modes = [PrivacyMode.OFF, PrivacyMode.SMART, PrivacyMode.SOCIAL,
                PrivacyMode.WORKSPACE, PrivacyMode.MAXIMUM]

        mode_results = []
        for mode in modes:
            result = self.benchmark_privacy_mode(mode, num_frames=100)
            mode_results.append(result)
            self.results[mode.name] = result

        # Compare face detectors
        detector_results = self.benchmark_face_detectors(100)
        self.results['detectors'] = detector_results

        # Print summary
        self._print_summary(mode_results)

        return self.results

    def _print_summary(self, mode_results: List[Dict]):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        print("\nPrivacy Mode Performance:")
        print("-" * 40)
        print(f"{'Mode':<12} {'Avg FPS':<12} {'P99 Latency':<12} {'Status'}")
        print("-" * 40)

        for result in mode_results:
            fps = result['avg_fps']
            latency = result['p99_latency_ms']
            status = "✅ PASS" if fps >= 120 else "❌ FAIL"
            print(f"{result['mode']:<12} {fps:>8.1f} fps {latency:>8.2f}ms   {status}")

        # Overall assessment
        avg_fps = np.mean([r['avg_fps'] for r in mode_results])

        print("\n" + "="*60)
        print(f"OVERALL AVERAGE: {avg_fps:.1f} FPS")

        if avg_fps >= 120:
            print("✅ SUCCESS! System meets 120+ FPS requirement!")
            print("✅ Ready for Meta Quest 3 deployment")
        else:
            print("⚠️  WARNING: System below 120 FPS target")
            print(f"   Need {120 - avg_fps:.1f} FPS improvement")

        # Face detector comparison
        if 'detectors' in self.results:
            print("\n" + "="*60)
            print("FACE DETECTOR PERFORMANCE")
            print("-" * 40)

            detectors = self.results['detectors']
            for name, stats in detectors.items():
                print(f"{name.upper()}: {stats['fps']:.1f} FPS "
                      f"({stats['avg_time_ms']:.2f}ms avg)")

    def run_quick_test(self):
        """Run a quick performance test"""
        print("\nRunning quick test (30 frames)...")
        self.setup()

        result = self.benchmark_privacy_mode(PrivacyMode.SMART, num_frames=30)

        print(f"\nQuick Test Result: {result['avg_fps']:.1f} FPS")
        if result['avg_fps'] >= 120:
            print("✅ Performance target achieved!")
        else:
            print("⚠️  Below target performance")

        return result

    def cleanup(self):
        """Clean up resources"""
        if self.guard:
            self.guard.cleanup()


def main():
    """Main entry point"""
    print("RealityGuard Improved - Performance Benchmark")
    print("=" * 60)

    benchmark = ImprovedBenchmark()

    try:
        # Check for command line arguments
        import sys
        if len(sys.argv) > 1:
            if sys.argv[1] == '--quick':
                benchmark.run_quick_test()
            elif sys.argv[1] == '--detectors':
                benchmark.setup()
                benchmark.benchmark_face_detectors(50)
            else:
                benchmark.run_full_benchmark()
        else:
            # Interactive mode
            print("\nOptions:")
            print("1. Full benchmark (all modes)")
            print("2. Quick test (30 frames)")
            print("3. Compare face detectors")

            choice = input("\nSelect option (1-3): ").strip()

            if choice == '1':
                results = benchmark.run_full_benchmark()

                # Save results
                with open('benchmark_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to benchmark_results.json")

            elif choice == '2':
                benchmark.run_quick_test()

            elif choice == '3':
                benchmark.setup()
                benchmark.benchmark_face_detectors(100)
            else:
                print("Invalid option")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    finally:
        benchmark.cleanup()
        print("\nBenchmark completed")


if __name__ == '__main__':
    main()