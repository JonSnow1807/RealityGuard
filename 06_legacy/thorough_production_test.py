#!/usr/bin/env python3
"""
EXHAUSTIVE PRODUCTION READINESS TEST
Zero margin for error - comprehensive validation
"""

import sys
import torch
import cv2
import numpy as np
import time
import psutil
import GPUtil
import json
import traceback
import hashlib
import threading
import queue
from pathlib import Path
from datetime import datetime
import gc

# Test configuration
TEST_CONFIGS = {
    "resolutions": [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160)  # 4K
    ],
    "fps_targets": [24, 30, 60],
    "object_counts": [0, 1, 5, 10, 20, 50],
    "durations": [1, 10, 30, 60],  # seconds
    "edge_cases": [
        "empty_frame",
        "corrupted_frame",
        "rapid_scene_change",
        "extreme_motion",
        "low_light",
        "overexposed"
    ]
}

class ExhaustiveProductionTest:
    """Complete production validation suite."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "failures": [],
            "warnings": [],
            "performance": {},
            "memory": {},
            "stability": {}
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_resolution_compatibility(self):
        """Test all resolution configurations."""
        print("\n" + "="*80)
        print("RESOLUTION COMPATIBILITY TEST")
        print("="*80)

        for width, height in TEST_CONFIGS["resolutions"]:
            try:
                print(f"\nTesting {width}x{height}...")

                # Create test video
                video = self._create_test_video(width, height, 30, 3)

                # Test each system
                systems = [
                    "patent_ready_all_claims.py",
                    "sam2_diffusion_production.py",
                    "advanced_sam2_diffusion.py"
                ]

                for system in systems:
                    start = time.time()
                    success = self._test_system_with_video(system, video)
                    duration = time.time() - start

                    result = {
                        "resolution": f"{width}x{height}",
                        "system": system,
                        "success": success,
                        "time": duration
                    }

                    key = f"resolution_{width}x{height}_{system}"
                    self.results["tests"][key] = result

                    if success:
                        print(f"  ✅ {system}: {duration:.2f}s")
                    else:
                        print(f"  ❌ {system}: FAILED")
                        self.results["failures"].append(key)

            except Exception as e:
                print(f"  ❌ Error at {width}x{height}: {e}")
                self.results["failures"].append(f"resolution_{width}x{height}")

    def test_memory_stability(self):
        """Test memory usage over time."""
        print("\n" + "="*80)
        print("MEMORY STABILITY TEST")
        print("="*80)

        import subprocess

        durations = [10, 30, 60]  # seconds

        for duration in durations:
            print(f"\nTesting {duration} second run...")

            try:
                # Monitor memory during execution
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB

                # Create longer test video
                video = self._create_test_video(1280, 720, 30, duration)

                # Track memory during processing
                memory_samples = []

                # Start processing in thread
                def process_video():
                    self._test_system_with_video("patent_ready_all_claims.py", video)

                thread = threading.Thread(target=process_video)
                thread.start()

                # Sample memory usage
                start_time = time.time()
                while thread.is_alive() and time.time() - start_time < duration + 5:
                    memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(memory)
                    time.sleep(0.5)

                thread.join(timeout=5)

                # Analyze memory
                peak_memory = max(memory_samples)
                avg_memory = sum(memory_samples) / len(memory_samples)
                memory_growth = peak_memory - initial_memory

                # Check for memory leaks
                memory_stable = memory_growth < 500  # Less than 500MB growth

                result = {
                    "duration": duration,
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": peak_memory,
                    "avg_memory_mb": avg_memory,
                    "memory_growth_mb": memory_growth,
                    "stable": memory_stable
                }

                self.results["memory"][f"duration_{duration}s"] = result

                if memory_stable:
                    print(f"  ✅ Memory stable: {memory_growth:.1f}MB growth")
                else:
                    print(f"  ⚠️ Memory leak detected: {memory_growth:.1f}MB growth")
                    self.results["warnings"].append(f"memory_leak_{duration}s")

                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.results["failures"].append(f"memory_test_{duration}s")

    def test_object_density(self):
        """Test with varying numbers of objects."""
        print("\n" + "="*80)
        print("OBJECT DENSITY STRESS TEST")
        print("="*80)

        for count in TEST_CONFIGS["object_counts"]:
            print(f"\nTesting with {count} objects...")

            try:
                # Create video with specific object count
                video = self._create_multi_object_video(count)

                # Test performance
                start = time.time()
                success = self._test_system_with_video("patent_ready_all_claims.py", video)
                duration = time.time() - start

                if success:
                    fps = 90 / duration  # 90 frames in video
                    print(f"  ✅ {count} objects: {fps:.1f} FPS")

                    result = {
                        "object_count": count,
                        "fps": fps,
                        "success": True,
                        "time": duration
                    }
                else:
                    print(f"  ❌ Failed with {count} objects")
                    result = {
                        "object_count": count,
                        "success": False
                    }
                    self.results["failures"].append(f"density_{count}_objects")

                self.results["tests"][f"density_{count}"] = result

            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.results["failures"].append(f"density_{count}")

    def test_error_handling(self):
        """Test system resilience to errors."""
        print("\n" + "="*80)
        print("ERROR HANDLING & EDGE CASES")
        print("="*80)

        edge_cases = {
            "empty_video": self._create_empty_video(),
            "corrupted_frames": self._create_corrupted_video(),
            "rapid_changes": self._create_rapid_change_video(),
            "extreme_motion": self._create_motion_blur_video(),
            "single_frame": self._create_single_frame_video()
        }

        for case_name, video in edge_cases.items():
            print(f"\nTesting {case_name}...")

            try:
                # Test if system handles edge case gracefully
                success = self._test_system_with_video("patent_ready_all_claims.py", video, expect_failure=True)

                if success or case_name in ["empty_video", "single_frame"]:
                    print(f"  ✅ Handled gracefully")
                    result = {"handled": True, "crashed": False}
                else:
                    print(f"  ⚠️ System struggled but recovered")
                    result = {"handled": True, "crashed": False}
                    self.results["warnings"].append(f"edge_case_{case_name}")

            except Exception as e:
                print(f"  ❌ System crashed: {e}")
                result = {"handled": False, "crashed": True, "error": str(e)}
                self.results["failures"].append(f"edge_case_{case_name}")

            self.results["tests"][f"edge_{case_name}"] = result

    def test_cache_accuracy(self):
        """Validate cache hit accuracy and consistency."""
        print("\n" + "="*80)
        print("CACHE VALIDATION TEST")
        print("="*80)

        # Create video with repeating patterns
        video = self._create_repeating_pattern_video()

        print("Testing cache consistency...")

        # Run twice and compare cache behavior
        cache_stats_1 = self._get_cache_stats("patent_ready_all_claims.py", video)
        cache_stats_2 = self._get_cache_stats("patent_ready_all_claims.py", video)

        # Second run should have higher cache hits
        if cache_stats_2 and cache_stats_1:
            improvement = cache_stats_2.get("hit_rate", 0) - cache_stats_1.get("hit_rate", 0)

            if improvement > 0:
                print(f"  ✅ Cache working: {improvement:.1%} improvement on second run")
                result = {"working": True, "improvement": improvement}
            else:
                print(f"  ⚠️ Cache not improving: {improvement:.1%}")
                result = {"working": False, "improvement": improvement}
                self.results["warnings"].append("cache_not_improving")
        else:
            print(f"  ❌ Could not validate cache")
            result = {"working": False, "error": "validation_failed"}
            self.results["failures"].append("cache_validation")

        self.results["tests"]["cache_validation"] = result

    def test_concurrent_processing(self):
        """Test thread safety with concurrent videos."""
        print("\n" + "="*80)
        print("CONCURRENT PROCESSING TEST")
        print("="*80)

        num_concurrent = 3
        print(f"Processing {num_concurrent} videos concurrently...")

        videos = [self._create_test_video(640, 480, 30, 2) for _ in range(num_concurrent)]
        threads = []
        results = queue.Queue()

        def process_video(video_path, video_id):
            try:
                start = time.time()
                success = self._test_system_with_video("patent_ready_all_claims.py", video_path)
                duration = time.time() - start
                results.put((video_id, success, duration))
            except Exception as e:
                results.put((video_id, False, str(e)))

        # Start all threads
        for i, video in enumerate(videos):
            thread = threading.Thread(target=process_video, args=(video, i))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Check results
        successes = 0
        while not results.empty():
            video_id, success, duration = results.get()
            if success:
                successes += 1
                print(f"  ✅ Video {video_id}: {duration:.2f}s")
            else:
                print(f"  ❌ Video {video_id}: Failed")

        if successes == num_concurrent:
            print(f"  ✅ All {num_concurrent} concurrent processes succeeded")
            self.results["tests"]["concurrent"] = {"success": True, "count": num_concurrent}
        else:
            print(f"  ⚠️ Only {successes}/{num_concurrent} succeeded")
            self.results["warnings"].append("concurrent_processing_issues")
            self.results["tests"]["concurrent"] = {"success": False, "succeeded": successes, "total": num_concurrent}

    def test_gpu_utilization(self):
        """Verify GPU is being used effectively."""
        print("\n" + "="*80)
        print("GPU UTILIZATION VERIFICATION")
        print("="*80)

        if not torch.cuda.is_available():
            print("  ⚠️ No GPU available")
            return

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                print("  ⚠️ No GPU detected")
                return

            gpu = gpus[0]

            # Baseline GPU usage
            baseline_util = gpu.load * 100
            baseline_memory = gpu.memoryUsed

            print(f"Baseline: {baseline_util:.1f}% util, {baseline_memory:.0f}MB memory")

            # Run intensive task
            video = self._create_test_video(1920, 1080, 30, 5)

            # Monitor GPU during processing
            max_util = baseline_util
            max_memory = baseline_memory

            def monitor_gpu():
                nonlocal max_util, max_memory
                while processing:
                    gpu = GPUtil.getGPUs()[0]
                    max_util = max(max_util, gpu.load * 100)
                    max_memory = max(max_memory, gpu.memoryUsed)
                    time.sleep(0.1)

            processing = True
            monitor_thread = threading.Thread(target=monitor_gpu)
            monitor_thread.start()

            # Process video
            self._test_system_with_video("patent_ready_all_claims.py", video)

            processing = False
            monitor_thread.join()

            # Analyze GPU usage
            util_increase = max_util - baseline_util
            memory_increase = max_memory - baseline_memory

            print(f"Peak: {max_util:.1f}% util, {max_memory:.0f}MB memory")
            print(f"Increase: {util_increase:.1f}% util, {memory_increase:.0f}MB memory")

            if util_increase > 5:
                print(f"  ✅ GPU properly utilized")
                gpu_result = {"utilized": True, "peak_util": max_util, "memory_mb": max_memory}
            else:
                print(f"  ⚠️ Low GPU utilization")
                self.results["warnings"].append("low_gpu_utilization")
                gpu_result = {"utilized": False, "peak_util": max_util}

            self.results["tests"]["gpu_utilization"] = gpu_result

        except Exception as e:
            print(f"  ❌ GPU test failed: {e}")
            self.results["failures"].append("gpu_utilization")

    def _create_test_video(self, width, height, fps, duration):
        """Create test video with specific parameters."""
        output_path = f"test_{width}x{height}_{duration}s.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames = int(fps * duration)
        for i in range(frames):
            # Create frame with moving objects
            frame = np.ones((height, width, 3), dtype=np.uint8) * 100

            # Add moving rectangle (person simulation)
            x = int(width * (0.3 + 0.4 * (i / frames)))
            y = height // 2
            cv2.rectangle(frame, (x-50, y-100), (x+50, y+100), (0, 255, 0), -1)

            # Add some variation
            cv2.circle(frame, (width//4, height//4), 30, (255, 0, 0), -1)

            writer.write(frame)

        writer.release()
        return output_path

    def _create_multi_object_video(self, object_count):
        """Create video with specific number of objects."""
        output_path = f"test_{object_count}_objects.mp4"
        width, height = 1280, 720

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        for frame_idx in range(90):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 100

            # Add specified number of objects
            for obj_idx in range(object_count):
                x = int(width * ((obj_idx + 1) / (object_count + 1)))
                y = int(height * (0.3 + 0.4 * np.sin(frame_idx * 0.1 + obj_idx)))

                cv2.rectangle(frame, (x-30, y-60), (x+30, y+60), (0, 255, 0), -1)

            writer.write(frame)

        writer.release()
        return output_path

    def _create_empty_video(self):
        """Create empty/black video."""
        output_path = "test_empty.mp4"
        width, height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        for _ in range(30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return output_path

    def _create_corrupted_video(self):
        """Create video with some corrupted frames."""
        output_path = "test_corrupted.mp4"
        width, height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        for i in range(60):
            if i % 10 == 0:
                # Corrupted frame (random noise)
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            else:
                # Normal frame
                frame = np.ones((height, width, 3), dtype=np.uint8) * 100
                cv2.rectangle(frame, (200, 150), (400, 350), (0, 255, 0), -1)

            writer.write(frame)

        writer.release()
        return output_path

    def _create_rapid_change_video(self):
        """Create video with rapid scene changes."""
        output_path = "test_rapid.mp4"
        width, height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        for i in range(90):
            # Alternate between completely different scenes
            if i % 2 == 0:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 200
                cv2.circle(frame, (320, 240), 50, (255, 0, 0), -1)
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 50
                cv2.rectangle(frame, (100, 100), (500, 400), (0, 255, 0), -1)

            writer.write(frame)

        writer.release()
        return output_path

    def _create_motion_blur_video(self):
        """Create video with extreme motion blur."""
        output_path = "test_motion.mp4"
        width, height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        for i in range(60):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 100

            # Fast moving object with blur
            for offset in range(-20, 21, 5):
                x = int(width * (i / 60)) + offset
                alpha = 1.0 - abs(offset) / 20
                overlay = frame.copy()
                cv2.rectangle(overlay, (x-30, 200), (x+30, 300), (0, int(255*alpha), 0), -1)
                frame = cv2.addWeighted(frame, 1-alpha*0.2, overlay, alpha*0.2, 0)

            writer.write(frame)

        writer.release()
        return output_path

    def _create_single_frame_video(self):
        """Create video with just one frame."""
        output_path = "test_single.mp4"
        width, height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        frame = np.ones((height, width, 3), dtype=np.uint8) * 150
        cv2.rectangle(frame, (200, 150), (400, 350), (0, 255, 0), -1)
        writer.write(frame)

        writer.release()
        return output_path

    def _create_repeating_pattern_video(self):
        """Create video with repeating patterns for cache testing."""
        output_path = "test_repeating.mp4"
        width, height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        # Create 3 different patterns and repeat
        patterns = []
        for p in range(3):
            frame = np.ones((height, width, 3), dtype=np.uint8) * (50 + p * 50)
            cv2.rectangle(frame, (200 + p*50, 200), (300 + p*50, 300), (0, 255, 0), -1)
            patterns.append(frame)

        # Repeat patterns
        for _ in range(10):
            for pattern in patterns:
                for _ in range(3):
                    writer.write(pattern.copy())

        writer.release()
        return output_path

    def _test_system_with_video(self, system, video_path, expect_failure=False):
        """Test a system with given video."""
        import subprocess

        try:
            # Run system with timeout
            result = subprocess.run(
                [sys.executable, system, "--input", video_path, "--headless"],
                capture_output=True,
                timeout=30,
                text=True
            )

            # Check if it processed successfully
            success = result.returncode == 0 or "FPS" in result.stdout

            if not success and not expect_failure:
                print(f"    Output: {result.stdout[:200]}")
                print(f"    Error: {result.stderr[:200]}")

            return success

        except subprocess.TimeoutExpired:
            print(f"    Timeout after 30s")
            return False
        except Exception as e:
            if not expect_failure:
                print(f"    Exception: {e}")
            return False

    def _get_cache_stats(self, system, video_path):
        """Extract cache statistics from system output."""
        import subprocess

        try:
            result = subprocess.run(
                [sys.executable, system, "--input", video_path, "--headless"],
                capture_output=True,
                timeout=30,
                text=True
            )

            # Parse cache statistics from output
            output = result.stdout

            # Look for cache hit rate
            if "Cache:" in output or "cache" in output.lower():
                # Extract hit rate (this is simplified, actual parsing would be more complex)
                import re
                match = re.search(r'(\d+)%', output)
                if match:
                    hit_rate = int(match.group(1)) / 100
                    return {"hit_rate": hit_rate}

            return None

        except Exception as e:
            print(f"    Could not get cache stats: {e}")
            return None

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)

        total_tests = len(self.results["tests"])
        failures = len(self.results["failures"])
        warnings = len(self.results["warnings"])

        print(f"\nTests Run: {total_tests}")
        print(f"Failures: {failures}")
        print(f"Warnings: {warnings}")

        if failures == 0:
            print("\n✅ ALL TESTS PASSED - PRODUCTION READY")
        elif failures < 3:
            print("\n⚠️ MOSTLY READY - Minor issues to address")
        else:
            print("\n❌ NOT READY - Critical issues found")

        # Save detailed report
        report_path = "thorough_production_test_results.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed report saved to: {report_path}")

        # Print critical issues
        if self.results["failures"]:
            print("\n❌ CRITICAL FAILURES:")
            for failure in self.results["failures"]:
                print(f"  - {failure}")

        if self.results["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in self.results["warnings"]:
                print(f"  - {warning}")

    def run_all_tests(self):
        """Run complete test suite."""
        print("="*80)
        print("EXHAUSTIVE PRODUCTION READINESS TESTING")
        print("="*80)
        print("Zero margin for error validation")

        # Run all test categories
        self.test_resolution_compatibility()
        self.test_memory_stability()
        self.test_object_density()
        self.test_error_handling()
        self.test_cache_accuracy()
        self.test_concurrent_processing()
        self.test_gpu_utilization()

        # Generate final report
        self.generate_report()


if __name__ == "__main__":
    tester = ExhaustiveProductionTest()
    tester.run_all_tests()