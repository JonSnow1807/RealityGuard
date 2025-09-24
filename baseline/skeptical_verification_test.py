#!/usr/bin/env python3
"""
SKEPTICAL VERIFICATION TEST SUITE
No trust mode - verifies everything with proof
Tests all claims and generates visual evidence
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import torch
import hashlib
from datetime import datetime
from pathlib import Path
import traceback
import psutil

# Create output directory for proof
PROOF_DIR = Path("skeptical_verification_results")
PROOF_DIR.mkdir(exist_ok=True)

class SkepticalVerifier:
    """Verifies all claims with extreme skepticism"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "tests": {},
            "fraud_detected": False,
            "honest_metrics": {}
        }
        self.test_count = 0

    def get_system_info(self):
        """Get real system information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cuda_available": torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["cuda_version"] = torch.version.cuda

        return info

    def verify_file_exists(self, filepath):
        """Verify a file actually exists"""
        exists = os.path.exists(filepath)
        size = os.path.getsize(filepath) if exists else 0

        return {
            "exists": exists,
            "size_bytes": size,
            "path": str(filepath)
        }

    def measure_actual_time(self, func, *args, **kwargs):
        """Measure actual execution time multiple ways"""
        # CPU time
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        # GPU timing if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cuda_start = time.time()

        # Execute function
        result = func(*args, **kwargs)

        # Measure end times
        cpu_time = time.process_time() - cpu_start
        wall_time = time.perf_counter() - wall_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cuda_time = time.time() - cuda_start
        else:
            cuda_time = None

        return result, {
            "cpu_time_ms": cpu_time * 1000,
            "wall_time_ms": wall_time * 1000,
            "cuda_time_ms": cuda_time * 1000 if cuda_time else None
        }

    def create_test_image(self, name, size=(640, 480)):
        """Create test image with verification"""
        img = np.zeros((*size[::-1], 3), dtype=np.uint8)

        if name == "single_circle":
            cv2.circle(img, (size[0]//2, size[1]//2), 60, (255, 255, 255), -1)
            expected_detections = 1
        elif name == "three_circles":
            cv2.circle(img, (size[0]//4, size[1]//2), 50, (255, 255, 255), -1)
            cv2.circle(img, (size[0]//2, size[1]//2), 60, (255, 255, 255), -1)
            cv2.circle(img, (3*size[0]//4, size[1]//2), 55, (255, 255, 255), -1)
            expected_detections = 3
        else:
            expected_detections = 0

        # Save image as proof
        img_path = PROOF_DIR / f"test_{name}_{size[0]}x{size[1]}.jpg"
        cv2.imwrite(str(img_path), img)

        # Calculate hash for verification
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        return img, {
            "expected_detections": expected_detections,
            "image_hash": img_hash,
            "saved_to": str(img_path),
            "shape": img.shape
        }

    def test_gpu_claims(self):
        """Test GPU performance claims"""
        print("\n" + "="*60)
        print("TESTING GPU PERFORMANCE CLAIMS")
        print("="*60)

        test_name = "gpu_performance"
        self.results["tests"][test_name] = {}

        # Check if GPU implementation exists
        gpu_file_check = self.verify_file_exists("realityguard_gpu_optimized.py")
        self.results["tests"][test_name]["file_exists"] = gpu_file_check

        if not gpu_file_check["exists"]:
            print("❌ GPU implementation file not found!")
            return

        try:
            # Import and test
            from realityguard_gpu_optimized import OptimizedGPUDetector

            detector = OptimizedGPUDetector(batch_size=8)

            # Test single frame
            print("\nTesting single frame processing...")
            frame, frame_info = self.create_test_image("single_circle", (640, 480))

            times = []
            for i in range(50):
                _, timing = self.measure_actual_time(
                    detector.process_frame_batch, [frame]
                )
                times.append(timing["wall_time_ms"])

                if i % 10 == 0:
                    print(f"  Iteration {i}: {timing['wall_time_ms']:.2f}ms")

            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1000 / avg_time if avg_time > 0 else 0

            single_result = {
                "avg_ms": round(avg_time, 2),
                "std_ms": round(std_time, 2),
                "min_ms": round(np.min(times), 2),
                "max_ms": round(np.max(times), 2),
                "fps": round(fps, 1),
                "samples": len(times)
            }

            print(f"\n  Single frame: {avg_time:.2f}±{std_time:.2f}ms ({fps:.1f} FPS)")

            # Test batch processing
            print("\nTesting batch processing (8 frames)...")
            frames = [frame.copy() for _ in range(8)]

            batch_times = []
            for i in range(30):
                _, timing = self.measure_actual_time(
                    detector.process_frame_batch, frames
                )
                batch_times.append(timing["wall_time_ms"])

                if i % 10 == 0:
                    print(f"  Iteration {i}: {timing['wall_time_ms']:.2f}ms total")

            avg_batch = np.mean(batch_times)
            per_frame = avg_batch / 8
            batch_fps = 1000 / per_frame if per_frame > 0 else 0

            batch_result = {
                "total_ms": round(avg_batch, 2),
                "per_frame_ms": round(per_frame, 2),
                "fps_per_frame": round(batch_fps, 1),
                "batch_size": 8
            }

            print(f"\n  Batch (8): {avg_batch:.2f}ms total, {per_frame:.2f}ms/frame ({batch_fps:.1f} FPS)")

            self.results["tests"][test_name]["single_frame"] = single_result
            self.results["tests"][test_name]["batch_processing"] = batch_result

            # Verify claim
            claimed_999_fps = batch_fps >= 900
            self.results["tests"][test_name]["999_fps_achieved"] = claimed_999_fps

            if claimed_999_fps:
                print("\n✅ 999 FPS claim: POSSIBLE with batch processing")
            else:
                print(f"\n❌ 999 FPS claim: FALSE (actual: {batch_fps:.1f} FPS)")
                self.results["fraud_detected"] = True

        except Exception as e:
            print(f"\n❌ GPU test failed: {e}")
            self.results["tests"][test_name]["error"] = str(e)
            self.results["tests"][test_name]["traceback"] = traceback.format_exc()
            self.results["fraud_detected"] = True

    def test_detection_accuracy(self):
        """Test detection accuracy claims"""
        print("\n" + "="*60)
        print("TESTING DETECTION ACCURACY")
        print("="*60)

        test_name = "detection_accuracy"
        self.results["tests"][test_name] = {}

        # Test improved detector
        v2_file = self.verify_file_exists("realityguard_improved_v2.py")
        self.results["tests"][test_name]["v2_file"] = v2_file

        if not v2_file["exists"]:
            print("❌ Improved detector not found!")
            return

        try:
            from realityguard_improved_v2 import ImprovedDetector

            detector = ImprovedDetector()

            test_cases = [
                ("single_circle", (640, 480), 1),
                ("three_circles", (640, 480), 3),
                ("single_circle", (1280, 720), 1),
                ("three_circles", (1920, 1080), 3)
            ]

            results = []

            for test_type, resolution, expected in test_cases:
                print(f"\nTesting: {test_type} at {resolution}")

                frame, info = self.create_test_image(test_type, resolution)

                # Test detection multiple times
                detections = []
                for _ in range(10):
                    output, detect_info = detector.process_frame(frame)
                    detections.append(detect_info['detections'])

                avg_detections = np.mean(detections)
                accuracy = min(avg_detections / expected * 100, 100) if expected > 0 else 0

                result = {
                    "test": test_type,
                    "resolution": resolution,
                    "expected": expected,
                    "avg_detected": round(avg_detections, 1),
                    "accuracy_percent": round(accuracy, 1),
                    "all_detections": detections
                }

                results.append(result)

                # Save output as proof
                output_path = PROOF_DIR / f"detection_{test_type}_{resolution[0]}x{resolution[1]}.jpg"
                cv2.imwrite(str(output_path), output)
                result["output_saved"] = str(output_path)

                status = "✅" if accuracy >= 80 else "❌"
                print(f"  {status} Expected: {expected}, Detected: {avg_detections:.1f} ({accuracy:.1f}%)")

            self.results["tests"][test_name]["results"] = results

            # Calculate overall accuracy
            overall_accuracy = np.mean([r["accuracy_percent"] for r in results])
            self.results["tests"][test_name]["overall_accuracy"] = round(overall_accuracy, 1)

            if overall_accuracy < 85:
                print(f"\n❌ 85% accuracy claim: FALSE (actual: {overall_accuracy:.1f}%)")
                self.results["fraud_detected"] = True
            else:
                print(f"\n✅ 85% accuracy claim: TRUE ({overall_accuracy:.1f}%)")

        except Exception as e:
            print(f"\n❌ Detection test failed: {e}")
            self.results["tests"][test_name]["error"] = str(e)
            self.results["fraud_detected"] = True

    def test_motion_tracking(self):
        """Test motion tracking claims"""
        print("\n" + "="*60)
        print("TESTING MOTION TRACKING")
        print("="*60)

        test_name = "motion_tracking"
        self.results["tests"][test_name] = {}

        try:
            from realityguard_improved_v2 import ImprovedDetector

            detector = ImprovedDetector()

            # Simulate moving object
            print("\nSimulating moving object across 20 frames...")

            tracking_results = []
            unique_ids = set()

            for frame_num in range(20):
                # Create frame with moving circle
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                x = 100 + frame_num * 25  # Move 25 pixels per frame
                cv2.circle(frame, (x, 240), 50, (255, 255, 255), -1)

                output, info = detector.process_frame(frame)

                # Track IDs
                if 'trackers' in info:
                    tracker_count = info['trackers']
                    tracking_results.append({
                        "frame": frame_num,
                        "trackers": tracker_count,
                        "detections": info['detections']
                    })

                    # Save some frames as proof
                    if frame_num % 5 == 0:
                        proof_path = PROOF_DIR / f"tracking_frame_{frame_num}.jpg"
                        cv2.imwrite(str(proof_path), output)
                        print(f"  Frame {frame_num}: {info['detections']} detections, {tracker_count} trackers")

            # Analyze tracking consistency
            if tracking_results:
                avg_trackers = np.mean([r['trackers'] for r in tracking_results])
                tracking_consistency = len([r for r in tracking_results if r['trackers'] > 0]) / len(tracking_results) * 100

                self.results["tests"][test_name]["avg_trackers"] = round(avg_trackers, 1)
                self.results["tests"][test_name]["consistency_percent"] = round(tracking_consistency, 1)
                self.results["tests"][test_name]["frames_tested"] = len(tracking_results)

                if tracking_consistency >= 80:
                    print(f"\n✅ Motion tracking: WORKING ({tracking_consistency:.1f}% consistency)")
                else:
                    print(f"\n❌ Motion tracking: POOR ({tracking_consistency:.1f}% consistency)")
                    self.results["fraud_detected"] = True
            else:
                print("\n❌ Motion tracking: NOT IMPLEMENTED")
                self.results["tests"][test_name]["implemented"] = False
                self.results["fraud_detected"] = True

        except Exception as e:
            print(f"\n❌ Motion tracking test failed: {e}")
            self.results["tests"][test_name]["error"] = str(e)
            self.results["fraud_detected"] = True

    def test_original_implementations(self):
        """Test original Reality Guard implementations"""
        print("\n" + "="*60)
        print("TESTING ORIGINAL IMPLEMENTATIONS")
        print("="*60)

        test_name = "original_implementations"
        self.results["tests"][test_name] = {}

        # Test CUDA fixed version
        print("\nTesting realityguard_cuda_fixed.py...")
        try:
            cuda_test_cmd = "cd RealityGuard && python realityguard_cuda_fixed.py"
            result = os.popen(cuda_test_cmd).read()

            # Parse FPS from output
            if "FPS:" in result:
                lines = result.split('\n')
                for line in lines:
                    if "FPS:" in line:
                        fps = float(line.split("FPS:")[1].split()[0])
                        print(f"  CUDA Fixed FPS: {fps}")
                        self.results["tests"][test_name]["cuda_fixed_fps"] = fps

        except Exception as e:
            print(f"  CUDA test error: {e}")
            self.results["tests"][test_name]["cuda_error"] = str(e)

        # Test production version
        print("\nTesting realityguard_production_ready.py...")
        try:
            prod_test_cmd = "cd RealityGuard && python realityguard_production_ready.py"
            result = os.popen(prod_test_cmd).read()

            if "FPS:" in result:
                lines = result.split('\n')
                fps_values = []
                for line in lines:
                    if "FPS:" in line:
                        try:
                            fps = float(line.split("FPS:")[1].split()[0])
                            fps_values.append(fps)
                        except:
                            pass

                if fps_values:
                    avg_fps = np.mean(fps_values)
                    print(f"  Production Average FPS: {avg_fps:.1f}")
                    self.results["tests"][test_name]["production_avg_fps"] = round(avg_fps, 1)

        except Exception as e:
            print(f"  Production test error: {e}")
            self.results["tests"][test_name]["production_error"] = str(e)

    def compare_cpu_vs_gpu(self):
        """Direct comparison of CPU vs GPU performance"""
        print("\n" + "="*60)
        print("CPU vs GPU DIRECT COMPARISON")
        print("="*60)

        test_name = "cpu_vs_gpu"
        self.results["tests"][test_name] = {}

        # Create test frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        cv2.circle(frame, (640, 360), 100, (255, 255, 255), -1)

        # Test CPU processing
        print("\nTesting CPU (OpenCV)...")
        cpu_times = []
        for i in range(100):
            start = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Simulate blur application
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (31, 31), 0)

            cpu_times.append((time.perf_counter() - start) * 1000)

            if i % 20 == 0:
                print(f"  Iteration {i}: {cpu_times[-1]:.2f}ms")

        cpu_avg = np.mean(cpu_times)
        cpu_fps = 1000 / cpu_avg

        self.results["tests"][test_name]["cpu"] = {
            "avg_ms": round(cpu_avg, 2),
            "fps": round(cpu_fps, 1),
            "samples": len(cpu_times)
        }

        print(f"\n  CPU Result: {cpu_avg:.2f}ms ({cpu_fps:.1f} FPS)")

        # Test GPU if available
        if torch.cuda.is_available():
            print("\nTesting GPU (PyTorch)...")

            try:
                # Convert to tensor
                frame_tensor = torch.from_numpy(frame).float().cuda() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

                gpu_times = []
                for i in range(100):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    # Simple GPU processing
                    gray_tensor = torch.mean(frame_tensor, dim=1, keepdim=True)
                    edges = torch.nn.functional.conv2d(gray_tensor,
                        torch.randn(1, 1, 3, 3).cuda(), padding=1)

                    torch.cuda.synchronize()
                    gpu_times.append((time.perf_counter() - start) * 1000)

                    if i % 20 == 0:
                        print(f"  Iteration {i}: {gpu_times[-1]:.2f}ms")

                gpu_avg = np.mean(gpu_times)
                gpu_fps = 1000 / gpu_avg
                speedup = cpu_avg / gpu_avg

                self.results["tests"][test_name]["gpu"] = {
                    "avg_ms": round(gpu_avg, 2),
                    "fps": round(gpu_fps, 1),
                    "speedup": round(speedup, 2),
                    "samples": len(gpu_times)
                }

                print(f"\n  GPU Result: {gpu_avg:.2f}ms ({gpu_fps:.1f} FPS)")
                print(f"  Speedup: {speedup:.2f}x")

                if speedup < 1:
                    print("\n⚠️ GPU is SLOWER than CPU for this operation!")

            except Exception as e:
                print(f"\n❌ GPU test error: {e}")
                self.results["tests"][test_name]["gpu_error"] = str(e)

    def generate_final_report(self):
        """Generate final verification report"""
        print("\n" + "="*60)
        print("FINAL VERIFICATION REPORT")
        print("="*60)

        # Calculate honest metrics
        if "gpu_performance" in self.results["tests"]:
            if "batch_processing" in self.results["tests"]["gpu_performance"]:
                batch_fps = self.results["tests"]["gpu_performance"]["batch_processing"]["fps_per_frame"]
                self.results["honest_metrics"]["max_fps"] = batch_fps

        if "detection_accuracy" in self.results["tests"]:
            if "overall_accuracy" in self.results["tests"]["detection_accuracy"]:
                self.results["honest_metrics"]["detection_accuracy"] = \
                    self.results["tests"]["detection_accuracy"]["overall_accuracy"]

        if "motion_tracking" in self.results["tests"]:
            if "consistency_percent" in self.results["tests"]["motion_tracking"]:
                self.results["honest_metrics"]["tracking_consistency"] = \
                    self.results["tests"]["motion_tracking"]["consistency_percent"]

        # Save JSON report
        report_path = PROOF_DIR / "verification_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📊 HONEST METRICS:")
        print("-" * 40)
        for metric, value in self.results["honest_metrics"].items():
            print(f"  {metric}: {value}")

        if self.results["fraud_detected"]:
            print("\n⚠️ FRAUD DETECTED - Some claims are FALSE!")
        else:
            print("\n✅ All tested claims appear to be valid")

        print(f"\n📁 Full report saved to: {report_path}")
        print(f"📁 Proof images saved to: {PROOF_DIR}/")

        return self.results

def main():
    """Run complete skeptical verification"""
    print("="*60)
    print("SKEPTICAL VERIFICATION TEST SUITE")
    print("NO TRUST MODE ACTIVATED")
    print("="*60)

    verifier = SkepticalVerifier()

    # Run all tests
    verifier.test_gpu_claims()
    verifier.test_detection_accuracy()
    verifier.test_motion_tracking()
    verifier.compare_cpu_vs_gpu()
    verifier.test_original_implementations()

    # Generate report
    results = verifier.generate_final_report()

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

    return results

if __name__ == "__main__":
    main()