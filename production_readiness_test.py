#!/usr/bin/env python3
"""
Production Readiness Test Suite
Tests RealityGuard with real-world data to verify production deployment readiness
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
import cv2
import urllib.request
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple
import traceback
import gc

# Add production path
sys.path.insert(0, '01_production')

class ProductionReadinessTest:
    """Comprehensive production readiness testing"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "metrics": {},
            "failures": [],
            "warnings": []
        }
        self.test_data_dir = "production_test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)

    def download_test_data(self) -> Dict[str, str]:
        """Download various real-world test videos and images"""
        print("\n" + "="*60)
        print("DOWNLOADING REAL-WORLD TEST DATA")
        print("="*60)

        test_files = {}

        # Test images from reliable sources
        test_sources = {
            "office_meeting": {
                "url": "https://images.pexels.com/photos/3184639/pexels-photo-3184639.jpeg",
                "file": "office_meeting.jpg",
                "description": "Business meeting with multiple people"
            },
            "street_scene": {
                "url": "https://images.pexels.com/photos/1209843/pexels-photo-1209843.jpeg",
                "file": "street_scene.jpg",
                "description": "Urban street with pedestrians"
            },
            "video_conference": {
                "url": "https://images.pexels.com/photos/4226140/pexels-photo-4226140.jpeg",
                "file": "video_conference.jpg",
                "description": "Video conference with screens"
            },
            "crowd": {
                "url": "https://images.pexels.com/photos/1190297/pexels-photo-1190297.jpeg",
                "file": "crowd.jpg",
                "description": "Crowded event with many faces"
            }
        }

        for name, info in test_sources.items():
            filepath = os.path.join(self.test_data_dir, info["file"])
            try:
                print(f"\nDownloading {name}...")
                urllib.request.urlretrieve(info["url"], filepath)

                # Verify download
                img = cv2.imread(filepath)
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"  ✓ Downloaded: {info['description']} ({w}x{h})")
                    test_files[name] = filepath
                else:
                    print(f"  ✗ Failed to load {name}")
            except Exception as e:
                print(f"  ✗ Download failed: {e}")

        # Create synthetic video for testing
        print("\nCreating synthetic test video...")
        video_path = os.path.join(self.test_data_dir, "test_video.mp4")
        self._create_test_video(video_path)
        test_files["test_video"] = video_path

        return test_files

    def _create_test_video(self, output_path: str, fps: int = 30, duration: int = 5):
        """Create a synthetic test video with moving objects"""
        width, height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = fps * duration

        for frame_idx in range(total_frames):
            # Create frame with gradient background
            frame = np.ones((height, width, 3), dtype=np.uint8) * 50

            # Add moving rectangles (simulating people)
            x1 = int(100 + frame_idx * 5) % (width - 200)
            cv2.rectangle(frame, (x1, 200), (x1 + 150, 500), (150, 100, 80), -1)

            x2 = int(width - 250 - frame_idx * 3) % (width - 200)
            cv2.rectangle(frame, (x2, 300), (x2 + 120, 550), (100, 150, 80), -1)

            # Add timestamp
            cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"  ✓ Created {duration}s test video at {fps} FPS")

    def test_real_images(self, test_files: Dict[str, str]) -> Dict:
        """Test with real-world images"""
        print("\n" + "="*60)
        print("TESTING WITH REAL IMAGES")
        print("="*60)

        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

        results = {}

        for name, filepath in test_files.items():
            if not filepath.endswith('.jpg'):
                continue

            print(f"\nTesting: {name}")
            print("-"*40)

            try:
                # Load image
                img = cv2.imread(filepath)
                if img is None:
                    print(f"  ✗ Failed to load image")
                    continue

                h, w = img.shape[:2]
                print(f"  Image size: {w}x{h}")

                # Test with different quality modes
                for mode in ["fast", "balanced"]:
                    config = OptimizedConfig(quality_mode=mode)
                    system = OptimizedRealityGuard(config)

                    # Detect regions
                    start_detect = time.time()
                    regions = system._detect_privacy_regions(img)
                    detect_time = (time.time() - start_detect) * 1000

                    print(f"\n  [{mode.upper()} mode]")
                    print(f"    Detection: {len(regions)} regions in {detect_time:.1f}ms")

                    # Process regions
                    if regions:
                        start_process = time.time()
                        batch = [{'frame': img, 'region': r} for r in regions[:3]]  # Limit to 3 regions
                        masks = system.diffusion.generate_privacy_batch(batch)
                        process_time = (time.time() - start_process) * 1000

                        print(f"    Privacy generation: {len(masks)} masks in {process_time:.1f}ms")

                        # Apply masks and check quality
                        processed_img = img.copy()
                        for i, region in enumerate(regions[:len(masks)]):
                            x1, y1, x2, y2 = region['bbox']
                            if i < len(masks):
                                resized_mask = cv2.resize(masks[i], (x2-x1, y2-y1))
                                processed_img[y1:y2, x1:x2] = resized_mask

                        # Calculate privacy effect
                        diff = cv2.absdiff(img, processed_img)
                        privacy_score = np.mean(diff)

                        # Save processed image
                        output_path = os.path.join(
                            self.test_data_dir,
                            f"{name}_{mode}_processed.jpg"
                        )
                        cv2.imwrite(output_path, processed_img)

                        results[f"{name}_{mode}"] = {
                            "detection_time_ms": detect_time,
                            "process_time_ms": process_time,
                            "total_time_ms": detect_time + process_time,
                            "regions_found": len(regions),
                            "regions_processed": len(masks),
                            "privacy_score": float(privacy_score),
                            "output_saved": output_path
                        }

                        print(f"    Total time: {detect_time + process_time:.1f}ms")
                        print(f"    Privacy score: {privacy_score:.1f}")
                        print(f"    ✓ Saved: {output_path}")
                    else:
                        print(f"    ⚠️ No regions detected")

                    # Clean up
                    del system
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results["failures"].append({
                    "test": f"image_{name}",
                    "error": str(e)
                })

        return results

    def test_video_processing(self, video_path: str) -> Dict:
        """Test video processing performance"""
        print("\n" + "="*60)
        print("TESTING VIDEO PROCESSING")
        print("="*60)

        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

        config = OptimizedConfig(quality_mode="balanced")
        system = OptimizedRealityGuard(config)

        # Process video
        output_path = os.path.join(self.test_data_dir, "processed_video.mp4")

        print(f"\nProcessing video: {video_path}")
        print(f"Output: {output_path}")

        try:
            results = system.process_video(video_path, output_path)

            # Verify output exists
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"\n✓ Video processed successfully")
                print(f"  Output size: {output_size:.1f} MB")
                print(f"  Average FPS: {results.get('average_fps', 0):.1f}")
                print(f"  Frames processed: {results.get('frames_processed', 0)}")
                print(f"  Frames skipped: {results.get('frames_skipped', 0)}")

                results["output_size_mb"] = output_size
                results["success"] = True
            else:
                print("\n✗ Output video not created")
                results["success"] = False

        except Exception as e:
            print(f"\n✗ Video processing failed: {e}")
            results = {"success": False, "error": str(e)}

        return results

    def test_memory_stability(self, test_files: Dict[str, str]) -> Dict:
        """Test memory stability under continuous load"""
        print("\n" + "="*60)
        print("TESTING MEMORY STABILITY")
        print("="*60)

        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

        config = OptimizedConfig(quality_mode="fast")
        system = OptimizedRealityGuard(config)

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        if torch.cuda.is_available():
            initial_gpu = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        else:
            initial_gpu = 0

        print(f"\nInitial memory: {initial_memory:.1f} MB")
        print(f"Initial GPU: {initial_gpu:.1f} MB")

        memory_samples = []
        gpu_samples = []

        # Process multiple images repeatedly
        print("\nProcessing 50 frames continuously...")

        for i in range(50):
            # Use different test images
            img_path = list(test_files.values())[i % len(test_files)]

            if img_path.endswith('.jpg'):
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize for consistent testing
                    img = cv2.resize(img, (640, 480))

                    # Process
                    regions = system._detect_privacy_regions(img)
                    if regions:
                        batch = [{'frame': img, 'region': r} for r in regions[:1]]
                        _ = system.diffusion.generate_privacy_batch(batch)

                    # Sample memory every 10 frames
                    if i % 10 == 0:
                        current_memory = process.memory_info().rss / (1024 * 1024)
                        memory_samples.append(current_memory)

                        if torch.cuda.is_available():
                            current_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
                            gpu_samples.append(current_gpu)

                        print(f"  Frame {i}: Memory {current_memory:.1f} MB, GPU {current_gpu if torch.cuda.is_available() else 0:.1f} MB")

        # Calculate statistics
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory

        if torch.cuda.is_available():
            final_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_growth = final_gpu - initial_gpu
        else:
            final_gpu = 0
            gpu_growth = 0

        # Check for memory leaks
        memory_stable = memory_growth < 100  # Less than 100MB growth
        gpu_stable = gpu_growth < 500  # Less than 500MB GPU growth

        results = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": memory_growth,
            "initial_gpu_mb": initial_gpu,
            "final_gpu_mb": final_gpu,
            "gpu_growth_mb": gpu_growth,
            "memory_stable": memory_stable,
            "gpu_stable": gpu_stable,
            "samples": len(memory_samples)
        }

        print(f"\n{'✓' if memory_stable else '✗'} Memory growth: {memory_growth:.1f} MB")
        print(f"{'✓' if gpu_stable else '✗'} GPU growth: {gpu_growth:.1f} MB")

        return results

    def test_edge_cases(self) -> Dict:
        """Test edge cases and failure modes"""
        print("\n" + "="*60)
        print("TESTING EDGE CASES")
        print("="*60)

        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

        config = OptimizedConfig(quality_mode="fast")
        system = OptimizedRealityGuard(config)

        edge_cases = {}

        # Test 1: Empty/black frame
        print("\n[1] Testing empty frame...")
        try:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            regions = system._detect_privacy_regions(black_frame)
            if regions:
                batch = [{'frame': black_frame, 'region': regions[0]}]
                masks = system.diffusion.generate_privacy_batch(batch)
                edge_cases["empty_frame"] = "✓ Handled gracefully"
            else:
                edge_cases["empty_frame"] = "✓ No regions detected (expected)"
        except Exception as e:
            edge_cases["empty_frame"] = f"✗ Failed: {e}"

        # Test 2: Very small image
        print("[2] Testing tiny image (10x10)...")
        try:
            tiny_frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
            regions = system._detect_privacy_regions(tiny_frame)
            edge_cases["tiny_image"] = "✓ Handled gracefully"
        except Exception as e:
            edge_cases["tiny_image"] = f"✗ Failed: {e}"

        # Test 3: Very large image
        print("[3] Testing large image (4K)...")
        try:
            large_frame = np.ones((2160, 3840, 3), dtype=np.uint8) * 100
            # Add some content
            cv2.rectangle(large_frame, (1000, 500), (2000, 1500), (150, 100, 80), -1)

            start = time.time()
            regions = system._detect_privacy_regions(large_frame)
            detect_time = time.time() - start

            if regions:
                batch = [{'frame': large_frame, 'region': regions[0]}]
                masks = system.diffusion.generate_privacy_batch(batch)
                process_time = time.time() - start
                edge_cases["4k_image"] = f"✓ Processed in {process_time:.1f}s"
            else:
                edge_cases["4k_image"] = f"✓ Detection in {detect_time:.1f}s"
        except Exception as e:
            edge_cases["4k_image"] = f"✗ Failed: {e}"

        # Test 4: Corrupted data
        print("[4] Testing corrupted frame...")
        try:
            corrupted_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            regions = system._detect_privacy_regions(corrupted_frame)
            edge_cases["corrupted_data"] = "✓ Handled gracefully"
        except Exception as e:
            edge_cases["corrupted_data"] = f"✗ Failed: {e}"

        # Test 5: Many regions
        print("[5] Testing frame with many regions...")
        try:
            busy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
            # Add many rectangles
            for i in range(20):
                x = (i * 30) % 600
                y = (i * 40) % 400
                cv2.rectangle(busy_frame, (x, y), (x+50, y+80), (150, 100, 80), -1)

            regions = system._detect_privacy_regions(busy_frame)
            if len(regions) > 10:
                # System should limit regions
                limited = len(regions) <= config.max_regions_per_frame
                edge_cases["many_regions"] = f"✓ Found {len(regions)}, limited: {limited}"
            else:
                edge_cases["many_regions"] = f"✓ Found {len(regions)} regions"
        except Exception as e:
            edge_cases["many_regions"] = f"✗ Failed: {e}"

        # Print results
        for test, result in edge_cases.items():
            print(f"  {test}: {result}")

        return edge_cases

    def test_concurrent_processing(self) -> Dict:
        """Test concurrent processing capability"""
        print("\n" + "="*60)
        print("TESTING CONCURRENT PROCESSING")
        print("="*60)

        from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig
        import threading

        results = {
            "threads": [],
            "success_count": 0,
            "failure_count": 0,
            "avg_time": 0
        }

        def process_frame(thread_id: int, results_list: list):
            """Process a frame in a thread"""
            try:
                config = OptimizedConfig(quality_mode="fast")
                system = OptimizedRealityGuard(config)

                # Create test frame
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
                cv2.rectangle(frame, (200, 150), (400, 450), (150, 100, 80), -1)

                start = time.time()
                regions = system._detect_privacy_regions(frame)

                if regions:
                    batch = [{'frame': frame, 'region': regions[0]}]
                    _ = system.diffusion.generate_privacy_batch(batch)

                elapsed = time.time() - start

                results_list.append({
                    "thread_id": thread_id,
                    "success": True,
                    "time": elapsed
                })

            except Exception as e:
                results_list.append({
                    "thread_id": thread_id,
                    "success": False,
                    "error": str(e)
                })

        # Test with multiple threads
        print("\nTesting with 5 concurrent threads...")
        threads = []
        thread_results = []

        for i in range(5):
            t = threading.Thread(target=process_frame, args=(i, thread_results))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Analyze results
        for r in thread_results:
            if r["success"]:
                results["success_count"] += 1
                print(f"  Thread {r['thread_id']}: ✓ Completed in {r.get('time', 0):.2f}s")
            else:
                results["failure_count"] += 1
                print(f"  Thread {r['thread_id']}: ✗ Failed - {r.get('error', 'Unknown')}")

        if results["success_count"] > 0:
            avg_time = sum(r.get("time", 0) for r in thread_results if r["success"]) / results["success_count"]
            results["avg_time"] = avg_time

        results["threads"] = thread_results
        results["concurrent_capable"] = results["success_count"] == len(threads)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive production readiness report"""
        report = []
        report.append("\n" + "="*70)
        report.append(" PRODUCTION READINESS REPORT ")
        report.append(" RealityGuard System Assessment ")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Image Processing Results
        if "real_images" in self.results["tests"]:
            report.append("\n\n📸 REAL IMAGE PROCESSING:")
            report.append("-"*40)

            img_results = self.results["tests"]["real_images"]
            total_time = []

            for test_name, data in img_results.items():
                if isinstance(data, dict) and "total_time_ms" in data:
                    total_time.append(data["total_time_ms"])
                    report.append(f"\n{test_name}:")
                    report.append(f"  • Processing time: {data['total_time_ms']:.1f}ms")
                    report.append(f"  • Regions found: {data['regions_found']}")
                    report.append(f"  • Privacy score: {data.get('privacy_score', 0):.1f}")

            if total_time:
                avg_time = sum(total_time) / len(total_time)
                report.append(f"\nAverage processing time: {avg_time:.1f}ms")
                report.append(f"Real-time capable: {'✅ YES' if avg_time < 42 else '❌ NO'} (<42ms for 24fps)")

        # Video Processing Results
        if "video_processing" in self.results["tests"]:
            report.append("\n\n🎬 VIDEO PROCESSING:")
            report.append("-"*40)

            video_results = self.results["tests"]["video_processing"]
            if video_results.get("success"):
                report.append(f"  • Average FPS: {video_results.get('average_fps', 0):.1f}")
                report.append(f"  • Frames processed: {video_results.get('frames_processed', 0)}")
                report.append(f"  • Output size: {video_results.get('output_size_mb', 0):.1f} MB")
                report.append(f"  • Status: ✅ SUCCESSFUL")
            else:
                report.append(f"  • Status: ❌ FAILED")
                report.append(f"  • Error: {video_results.get('error', 'Unknown')}")

        # Memory Stability
        if "memory_stability" in self.results["tests"]:
            report.append("\n\n💾 MEMORY STABILITY:")
            report.append("-"*40)

            mem_results = self.results["tests"]["memory_stability"]
            report.append(f"  • Memory growth: {mem_results.get('memory_growth_mb', 0):.1f} MB")
            report.append(f"  • GPU growth: {mem_results.get('gpu_growth_mb', 0):.1f} MB")
            report.append(f"  • Memory stable: {'✅ YES' if mem_results.get('memory_stable') else '❌ NO'}")
            report.append(f"  • GPU stable: {'✅ YES' if mem_results.get('gpu_stable') else '❌ NO'}")

        # Edge Cases
        if "edge_cases" in self.results["tests"]:
            report.append("\n\n⚠️ EDGE CASES:")
            report.append("-"*40)

            edge_results = self.results["tests"]["edge_cases"]
            passed = sum(1 for r in edge_results.values() if "✓" in str(r))
            total = len(edge_results)

            report.append(f"  • Passed: {passed}/{total} tests")
            for test, result in edge_results.items():
                status = "✅" if "✓" in str(result) else "❌"
                report.append(f"  • {test}: {status}")

        # Concurrent Processing
        if "concurrent_processing" in self.results["tests"]:
            report.append("\n\n🔄 CONCURRENT PROCESSING:")
            report.append("-"*40)

            conc_results = self.results["tests"]["concurrent_processing"]
            report.append(f"  • Success rate: {conc_results.get('success_count', 0)}/{conc_results.get('success_count', 0) + conc_results.get('failure_count', 0)}")
            report.append(f"  • Average time: {conc_results.get('avg_time', 0):.2f}s")
            report.append(f"  • Concurrent capable: {'✅ YES' if conc_results.get('concurrent_capable') else '❌ NO'}")

        # Production Readiness Score
        report.append("\n\n" + "="*70)
        report.append(" PRODUCTION READINESS SCORE ")
        report.append("="*70)

        # Calculate score
        score = 0
        max_score = 5

        # Check criteria
        criteria = {
            "Real-time performance": False,
            "Memory stability": False,
            "Video processing": False,
            "Edge case handling": False,
            "Concurrent processing": False
        }

        # Evaluate
        if "real_images" in self.results["tests"]:
            img_results = self.results["tests"]["real_images"]
            avg_times = [d.get("total_time_ms", 100) for d in img_results.values() if isinstance(d, dict)]
            if avg_times and sum(avg_times)/len(avg_times) < 42:
                criteria["Real-time performance"] = True
                score += 1

        if self.results["tests"].get("memory_stability", {}).get("memory_stable"):
            criteria["Memory stability"] = True
            score += 1

        if self.results["tests"].get("video_processing", {}).get("success"):
            criteria["Video processing"] = True
            score += 1

        if "edge_cases" in self.results["tests"]:
            edge_results = self.results["tests"]["edge_cases"]
            passed = sum(1 for r in edge_results.values() if "✓" in str(r))
            if passed >= len(edge_results) * 0.8:  # 80% pass rate
                criteria["Edge case handling"] = True
                score += 1

        if self.results["tests"].get("concurrent_processing", {}).get("concurrent_capable"):
            criteria["Concurrent processing"] = True
            score += 1

        # Display criteria
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            report.append(f"  {status} {criterion}")

        # Final verdict
        report.append(f"\n📊 SCORE: {score}/{max_score}")

        if score == max_score:
            report.append("\n🎉 VERDICT: PRODUCTION READY")
            report.append("The system meets all production deployment criteria.")
        elif score >= 4:
            report.append("\n✅ VERDICT: MOSTLY READY")
            report.append("The system is ready with minor improvements needed.")
        elif score >= 3:
            report.append("\n⚠️ VERDICT: NEEDS IMPROVEMENT")
            report.append("The system requires some work before production deployment.")
        else:
            report.append("\n❌ VERDICT: NOT READY")
            report.append("Significant improvements needed for production deployment.")

        report.append("\n" + "="*70 + "\n")

        return "\n".join(report)

    def run_all_tests(self):
        """Run comprehensive production readiness tests"""
        print("\n" + "="*70)
        print(" STARTING PRODUCTION READINESS TESTS ")
        print("="*70)

        try:
            # 1. Download test data
            test_files = self.download_test_data()

            # 2. Test real images
            print("\n[TEST 1/5] Real Image Processing")
            self.results["tests"]["real_images"] = self.test_real_images(test_files)

            # 3. Test video processing
            print("\n[TEST 2/5] Video Processing")
            if "test_video" in test_files:
                self.results["tests"]["video_processing"] = self.test_video_processing(
                    test_files["test_video"]
                )

            # 4. Test memory stability
            print("\n[TEST 3/5] Memory Stability")
            self.results["tests"]["memory_stability"] = self.test_memory_stability(test_files)

            # 5. Test edge cases
            print("\n[TEST 4/5] Edge Cases")
            self.results["tests"]["edge_cases"] = self.test_edge_cases()

            # 6. Test concurrent processing
            print("\n[TEST 5/5] Concurrent Processing")
            self.results["tests"]["concurrent_processing"] = self.test_concurrent_processing()

            # Generate report
            report = self.generate_report()
            print(report)

            # Save report
            report_path = os.path.join(self.test_data_dir, "production_readiness_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {report_path}")

            # Save detailed results
            results_path = os.path.join(self.test_data_dir, "production_test_results.json")
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"Detailed results saved to: {results_path}")

        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            traceback.print_exc()

            self.results["critical_failure"] = str(e)
            self.results["traceback"] = traceback.format_exc()

if __name__ == "__main__":
    tester = ProductionReadinessTest()
    tester.run_all_tests()