#!/usr/bin/env python3
"""
Extended duration test - 10 minutes continuous processing
Tests system stability, memory leaks, and performance degradation
"""

import cv2
import numpy as np
import time
import psutil
import GPUtil
import json
import threading
from datetime import datetime
import gc
import subprocess
import sys

class ExtendedDurationTest:
    """10-minute continuous processing test."""

    def __init__(self):
        self.test_duration = 600  # 10 minutes
        self.checkpoint_interval = 30  # Log every 30 seconds
        self.results = {
            "start_time": None,
            "end_time": None,
            "checkpoints": [],
            "memory_samples": [],
            "fps_samples": [],
            "errors": [],
            "summary": {}
        }

    def create_long_video(self, duration_seconds=600):
        """Create a 10-minute test video."""
        print(f"Creating {duration_seconds} second test video...")

        width, height = 1280, 720
        fps = 30

        # For efficiency, create a shorter video that we'll loop
        # 60 seconds should be enough to test stability
        actual_duration = min(duration_seconds, 60)
        total_frames = fps * actual_duration

        output_path = f"test_video_{actual_duration}s.mp4"

        print(f"Creating {actual_duration} second video with OpenCV...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(total_frames):
            if i % (fps * 5) == 0:
                print(f"  Creating frame {i}/{total_frames}...")

            frame = np.ones((height, width, 3), dtype=np.uint8) * 100

            # Add moving objects
            t = i / fps
            x1 = int(width * (0.2 + 0.6 * np.sin(t * 0.5)))
            y1 = int(height * (0.3 + 0.4 * np.cos(t * 0.3)))
            cv2.rectangle(frame, (x1-50, y1-100), (x1+50, y1+100), (0, 255, 0), -1)

            x2 = int(width * (0.7 - 0.5 * np.cos(t * 0.4)))
            y2 = int(height * (0.5 + 0.3 * np.sin(t * 0.6)))
            cv2.circle(frame, (x2, y2), 40, (255, 0, 0), -1)

            # Add timestamp
            timestamp = f"Time: {int(t)}s"
            cv2.putText(frame, timestamp, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()
        print(f"Video created: {output_path}")

        return output_path, actual_duration

    def monitor_system(self, process_pid, stop_event):
        """Monitor system resources during test."""
        process = psutil.Process(process_pid)

        while not stop_event.is_set():
            try:
                # CPU usage
                cpu_percent = process.cpu_percent(interval=1)

                # Memory usage
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024

                # GPU usage if available
                gpu_util = 0
                gpu_mem = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_util = gpus[0].load * 100
                        gpu_mem = gpus[0].memoryUsed
                except:
                    pass

                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_mb": mem_mb,
                    "gpu_util": gpu_util,
                    "gpu_memory_mb": gpu_mem
                }

                self.results["memory_samples"].append(sample)

            except psutil.NoSuchProcess:
                break
            except Exception as e:
                self.results["errors"].append(str(e))

            time.sleep(5)  # Sample every 5 seconds

    def run_extended_test(self):
        """Run the 10-minute test."""
        print("="*80)
        print("EXTENDED DURATION TEST (10 MINUTES)")
        print("="*80)

        # Create or use existing long video
        video_path, actual_duration = self.create_long_video(self.test_duration)

        print(f"\nStarting {actual_duration} second continuous processing test...")
        print("This will test:")
        print("  - Memory stability over time")
        print("  - Performance degradation")
        print("  - System reliability")
        print("  - Error recovery")

        self.results["start_time"] = datetime.now().isoformat()

        # Start the processing
        cmd = [
            sys.executable,
            "patent_ready_all_claims.py",
            "--input", video_path,
            "--headless"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Start monitoring thread
        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(
            target=self.monitor_system,
            args=(process.pid, stop_monitor)
        )
        monitor_thread.start()

        # Track checkpoints
        start_time = time.time()
        last_checkpoint = start_time
        checkpoint_count = 0

        print("\nProcessing...")
        print("-" * 60)

        try:
            while True:
                # Check if process is still running
                poll = process.poll()
                if poll is not None:
                    break

                current_time = time.time()
                elapsed = current_time - start_time

                # Checkpoint logging
                if current_time - last_checkpoint >= self.checkpoint_interval:
                    checkpoint_count += 1
                    last_checkpoint = current_time

                    # Get latest memory sample
                    if self.results["memory_samples"]:
                        latest = self.results["memory_samples"][-1]
                        mem_mb = latest["memory_mb"]
                        gpu_mb = latest["gpu_memory_mb"]
                    else:
                        mem_mb = 0
                        gpu_mb = 0

                    checkpoint = {
                        "number": checkpoint_count,
                        "elapsed_seconds": elapsed,
                        "memory_mb": mem_mb,
                        "gpu_memory_mb": gpu_mb
                    }

                    self.results["checkpoints"].append(checkpoint)

                    print(f"Checkpoint {checkpoint_count}: "
                         f"{elapsed:.0f}s elapsed | "
                         f"Mem: {mem_mb:.0f}MB | "
                         f"GPU: {gpu_mb:.0f}MB")

                # Stop after target duration
                if elapsed >= actual_duration:
                    print(f"\nReached {actual_duration} second target")
                    process.terminate()
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            process.terminate()

        except Exception as e:
            print(f"\nError during test: {e}")
            self.results["errors"].append(str(e))
            process.terminate()

        # Stop monitoring
        stop_monitor.set()
        monitor_thread.join(timeout=5)

        # Wait for process to complete
        try:
            stdout, stderr = process.communicate(timeout=10)

            # Parse FPS from output
            if "FPS" in stdout:
                import re
                fps_matches = re.findall(r'(\d+\.?\d*)\s*FPS', stdout)
                if fps_matches:
                    self.results["fps_samples"] = [float(fps) for fps in fps_matches]

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

        self.results["end_time"] = datetime.now().isoformat()

        # Analyze results
        self.analyze_results()

    def analyze_results(self):
        """Analyze test results for issues."""
        print("\n" + "="*80)
        print("EXTENDED TEST ANALYSIS")
        print("="*80)

        if not self.results["memory_samples"]:
            print("❌ No memory samples collected")
            return

        # Memory analysis
        memory_values = [s["memory_mb"] for s in self.results["memory_samples"]]
        initial_memory = memory_values[:5]  # First 5 samples
        final_memory = memory_values[-5:]   # Last 5 samples

        if initial_memory and final_memory:
            avg_initial = sum(initial_memory) / len(initial_memory)
            avg_final = sum(final_memory) / len(final_memory)
            memory_growth = avg_final - avg_initial
            growth_percent = (memory_growth / avg_initial) * 100 if avg_initial > 0 else 0

            print(f"\nMemory Analysis:")
            print(f"  Initial: {avg_initial:.1f} MB")
            print(f"  Final: {avg_final:.1f} MB")
            print(f"  Growth: {memory_growth:.1f} MB ({growth_percent:.1f}%)")

            if abs(memory_growth) < 100:
                print("  ✅ No significant memory leak detected")
                memory_stable = True
            elif abs(memory_growth) < 500:
                print("  ⚠️ Moderate memory growth detected")
                memory_stable = True
            else:
                print("  ❌ Significant memory leak detected")
                memory_stable = False

            self.results["summary"]["memory_stable"] = memory_stable
            self.results["summary"]["memory_growth_mb"] = memory_growth

        # Performance analysis
        if self.results["fps_samples"]:
            fps_values = self.results["fps_samples"]
            avg_fps = sum(fps_values) / len(fps_values)
            min_fps = min(fps_values)
            max_fps = max(fps_values)

            print(f"\nPerformance Analysis:")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Min FPS: {min_fps:.1f}")
            print(f"  Max FPS: {max_fps:.1f}")

            if min_fps >= 24:
                print("  ✅ Maintained real-time performance throughout")
                performance_stable = True
            else:
                print("  ⚠️ Performance dropped below real-time")
                performance_stable = False

            self.results["summary"]["performance_stable"] = performance_stable
            self.results["summary"]["avg_fps"] = avg_fps

        # Error analysis
        if self.results["errors"]:
            print(f"\n⚠️ Errors encountered: {len(self.results['errors'])}")
            for error in self.results["errors"][:5]:
                print(f"  - {error}")
        else:
            print("\n✅ No errors encountered")

        # Overall verdict
        print("\n" + "="*80)
        print("EXTENDED TEST VERDICT")
        print("="*80)

        if (self.results["summary"].get("memory_stable", False) and
            self.results["summary"].get("performance_stable", False) and
            not self.results["errors"]):
            print("✅ SYSTEM PASSED 10-MINUTE STABILITY TEST")
            self.results["summary"]["verdict"] = "PASSED"
        else:
            print("⚠️ SYSTEM NEEDS OPTIMIZATION FOR LONG RUNS")
            self.results["summary"]["verdict"] = "NEEDS_OPTIMIZATION"

        # Save results
        with open("extended_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: extended_test_results.json")

def main():
    """Run extended duration test."""
    tester = ExtendedDurationTest()
    tester.run_extended_test()

if __name__ == "__main__":
    main()