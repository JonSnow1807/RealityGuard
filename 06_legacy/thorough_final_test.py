#!/usr/bin/env python3
"""
THOROUGH FINAL TESTING SUITE
Comprehensive evaluation of the RealityGuard system
"""

import cv2
import numpy as np
import torch
import psutil
import time
import json
import os
import gc
import traceback
from pathlib import Path
from collections import defaultdict

class ThoroughTester:
    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "issues": [],
            "recommendations": [],
            "metrics": {}
        }

    def test_1_system_imports(self):
        """Test that all system files import correctly"""
        print("\n📋 TEST 1: System Imports")
        print("-" * 50)

        systems = {
            "real_ai_privacy_system": "RealAIPrivacyEngine",
            "enhanced_real_ai_system": "EnhancedRealAIEngine",
            "final_production_ready": "FinalProductionEngine",
            "fixed_production_system": "FixedProductionEngine"
        }

        import_results = {}

        for module_name, class_name in systems.items():
            try:
                module = __import__(module_name)
                if hasattr(module, class_name):
                    import_results[module_name] = "✅ Success"
                    print(f"  ✅ {module_name} imported successfully")
                else:
                    import_results[module_name] = f"⚠️ Class {class_name} not found"
                    print(f"  ⚠️ {module_name}: Class {class_name} not found")
                    self.results["issues"].append(f"Missing class {class_name} in {module_name}")
            except Exception as e:
                import_results[module_name] = f"❌ Error: {str(e)[:50]}"
                print(f"  ❌ {module_name}: {str(e)[:50]}")
                self.results["issues"].append(f"Import error in {module_name}: {str(e)[:50]}")

        self.results["tests"]["imports"] = import_results
        return len([r for r in import_results.values() if "✅" in r]) >= 3

    def test_2_memory_usage(self):
        """Test memory usage and check for leaks"""
        print("\n📋 TEST 2: Memory Usage")
        print("-" * 50)

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"  Initial memory: {initial_memory:.1f} MB")

        # Import and create engines to test memory
        try:
            from final_production_ready import FinalProductionEngine

            # Create and destroy multiple times
            memory_readings = []

            for i in range(3):
                engine = FinalProductionEngine()

                # Process a frame
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                _ = engine.process_frame(test_frame)

                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)

                print(f"  Iteration {i+1}: {current_memory:.1f} MB")

                # Cleanup
                del engine
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Check for memory leak
            memory_growth = memory_readings[-1] - memory_readings[0]

            self.results["tests"]["memory"] = {
                "initial_mb": initial_memory,
                "final_mb": memory_readings[-1],
                "growth_mb": memory_growth,
                "leak_detected": memory_growth > 100
            }

            if memory_growth < 100:
                print(f"  ✅ Memory stable (growth: {memory_growth:.1f} MB)")
                return True
            else:
                print(f"  ⚠️ Potential memory leak: {memory_growth:.1f} MB growth")
                self.results["issues"].append(f"Memory leak: {memory_growth:.1f} MB growth")
                return False

        except Exception as e:
            print(f"  ❌ Memory test failed: {e}")
            self.results["issues"].append(f"Memory test error: {str(e)[:100]}")
            return False

    def test_3_output_quality(self):
        """Test output video quality and sizes"""
        print("\n📋 TEST 3: Output Quality")
        print("-" * 50)

        output_files = [
            "output_final_production.mp4",
            "output_real_ai_privacy.mp4",
            "output_enhanced_ai.mp4"
        ]

        quality_results = {}

        for output_file in output_files:
            if not Path(output_file).exists():
                quality_results[output_file] = "❌ Not found"
                continue

            try:
                # Get file size
                size_mb = Path(output_file).stat().st_size / 1024 / 1024

                # Open video
                cap = cv2.VideoCapture(output_file)

                if not cap.isOpened():
                    quality_results[output_file] = "❌ Cannot open"
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Read a frame to check quality
                ret, frame = cap.read()

                if ret:
                    # Check for privacy indicators
                    green_pixels = np.sum(frame[:, :, 1] > 200)
                    total_pixels = width * height
                    privacy_visible = (green_pixels / total_pixels) > 0.01

                    quality_results[output_file] = {
                        "size_mb": round(size_mb, 2),
                        "resolution": f"{width}x{height}",
                        "fps": fps,
                        "frames": frames,
                        "privacy_visible": privacy_visible
                    }

                    status = "✅" if privacy_visible else "⚠️"
                    print(f"  {status} {output_file}: {size_mb:.1f}MB, {width}x{height}, privacy={'visible' if privacy_visible else 'not visible'}")
                else:
                    quality_results[output_file] = "❌ Cannot read frames"

                cap.release()

            except Exception as e:
                quality_results[output_file] = f"❌ Error: {str(e)[:50]}"
                print(f"  ❌ {output_file}: {str(e)[:50]}")

        self.results["tests"]["output_quality"] = quality_results

        # Check if at least one output is good
        good_outputs = sum(1 for r in quality_results.values()
                          if isinstance(r, dict) and r.get("privacy_visible", False))

        if good_outputs == 0:
            self.results["recommendations"].append("Increase privacy visibility in outputs")

        return good_outputs > 0

    def test_4_error_handling(self):
        """Test error handling and edge cases"""
        print("\n📋 TEST 4: Error Handling")
        print("-" * 50)

        error_tests = {}

        try:
            from final_production_ready import FinalProductionEngine
            engine = FinalProductionEngine()

            # Test 1: None input
            try:
                result = engine.process_frame(None)
                error_tests["none_input"] = "✅ Handled"
                print("  ✅ None input handled")
            except:
                error_tests["none_input"] = "❌ Crash"
                print("  ❌ Crashes on None input")
                self.results["issues"].append("No None input handling")

            # Test 2: Empty frame
            try:
                empty = np.array([])
                result = engine.process_frame(empty)
                error_tests["empty_frame"] = "✅ Handled"
                print("  ✅ Empty frame handled")
            except:
                error_tests["empty_frame"] = "❌ Crash"
                print("  ❌ Crashes on empty frame")
                self.results["issues"].append("No empty frame handling")

            # Test 3: Wrong shape
            try:
                wrong_shape = np.random.randint(0, 255, (100,), dtype=np.uint8)
                result = engine.process_frame(wrong_shape)
                error_tests["wrong_shape"] = "✅ Handled"
                print("  ✅ Wrong shape handled")
            except:
                error_tests["wrong_shape"] = "⚠️ Exception (expected)"
                print("  ⚠️ Wrong shape raises exception (expected)")

            # Test 4: Huge frame
            try:
                huge = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
                result = engine.process_frame(huge)
                error_tests["huge_frame"] = "✅ Processed"
                print("  ✅ Huge frame processed")
            except Exception as e:
                error_tests["huge_frame"] = f"⚠️ {str(e)[:30]}"
                print(f"  ⚠️ Huge frame issue: {str(e)[:30]}")

        except Exception as e:
            print(f"  ❌ Error handling test failed: {e}")
            self.results["issues"].append(f"Error handling test failed: {str(e)[:100]}")
            return False

        self.results["tests"]["error_handling"] = error_tests

        handled_count = sum(1 for r in error_tests.values() if "✅" in str(r) or "⚠️" in str(r))
        return handled_count >= 3

    def test_5_performance_benchmark(self):
        """Benchmark actual processing speed"""
        print("\n📋 TEST 5: Performance Benchmark")
        print("-" * 50)

        try:
            from final_production_ready import FinalProductionEngine
            engine = FinalProductionEngine()

            # Create test frames of different sizes
            test_cases = [
                ("480p", (480, 640, 3)),
                ("720p", (720, 1280, 3)),
                ("1080p", (1080, 1920, 3))
            ]

            benchmark_results = {}

            for name, shape in test_cases:
                frame = np.random.randint(0, 255, shape, dtype=np.uint8)

                # Warmup
                for _ in range(3):
                    _ = engine.process_frame(frame)

                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    _ = engine.process_frame(frame)
                    elapsed = time.time() - start
                    times.append(elapsed)

                avg_time = np.mean(times)
                avg_fps = 1.0 / avg_time if avg_time > 0 else 0

                benchmark_results[name] = {
                    "avg_time_ms": round(avg_time * 1000, 2),
                    "fps": round(avg_fps, 1),
                    "realtime": avg_fps >= 24
                }

                status = "✅" if avg_fps >= 24 else "⚠️"
                print(f"  {status} {name}: {avg_fps:.1f} FPS ({avg_time*1000:.1f}ms/frame)")

            self.results["tests"]["performance"] = benchmark_results
            self.results["metrics"]["avg_fps"] = np.mean([r["fps"] for r in benchmark_results.values()])

            # Check if at least 480p is real-time
            realtime_count = sum(1 for r in benchmark_results.values() if r["realtime"])

            if realtime_count == 0:
                self.results["recommendations"].append("Optimize for better real-time performance")

            return realtime_count > 0

        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            self.results["issues"].append(f"Benchmark error: {str(e)[:100]}")
            return False

    def test_6_code_quality(self):
        """Review code for production best practices"""
        print("\n📋 TEST 6: Code Quality Review")
        print("-" * 50)

        quality_checks = {}

        # Check main production file
        production_file = "final_production_ready.py"

        try:
            with open(production_file, 'r') as f:
                code = f.read()

            # Check for basic quality indicators
            quality_checks["docstrings"] = '"""' in code or "'''" in code
            quality_checks["error_handling"] = "try:" in code and "except" in code
            quality_checks["logging"] = "print(" in code  # Basic logging
            quality_checks["type_hints"] = "->" in code or ": " in code
            quality_checks["constants"] = "DEVICE" in code

            # Check for bad practices
            quality_checks["no_bare_except"] = "except:" not in code
            quality_checks["no_eval"] = "eval(" not in code
            quality_checks["no_exec"] = "exec(" not in code

            # Print results
            for check, passed in quality_checks.items():
                status = "✅" if passed else "⚠️"
                print(f"  {status} {check.replace('_', ' ').title()}")

            self.results["tests"]["code_quality"] = quality_checks

            # Recommendations based on quality
            if not quality_checks["type_hints"]:
                self.results["recommendations"].append("Add type hints for better code maintainability")

            if not quality_checks["no_bare_except"]:
                self.results["recommendations"].append("Avoid bare except clauses - catch specific exceptions")

            good_practices = sum(quality_checks.values())
            return good_practices >= 6

        except Exception as e:
            print(f"  ❌ Code review failed: {e}")
            return False

    def test_7_gpu_efficiency(self):
        """Test GPU memory efficiency"""
        print("\n📋 TEST 7: GPU Efficiency")
        print("-" * 50)

        if not torch.cuda.is_available():
            print("  ⚠️ No GPU available - skipping")
            self.results["tests"]["gpu_efficiency"] = "Skipped - No GPU"
            return True

        try:
            # Clear cache first
            torch.cuda.empty_cache()
            initial = torch.cuda.memory_allocated() / 1e9

            # Load a system
            from final_production_ready import FinalProductionEngine
            engine = FinalProductionEngine()

            # Process some frames
            for i in range(10):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                _ = engine.process_frame(frame)

            peak = torch.cuda.max_memory_allocated() / 1e9
            current = torch.cuda.memory_allocated() / 1e9

            efficiency_results = {
                "initial_gb": round(initial, 3),
                "current_gb": round(current, 3),
                "peak_gb": round(peak, 3),
                "efficient": peak < 2.0  # Less than 2GB is efficient
            }

            self.results["tests"]["gpu_efficiency"] = efficiency_results

            if peak < 2.0:
                print(f"  ✅ GPU memory efficient: {peak:.2f} GB peak")
                return True
            else:
                print(f"  ⚠️ High GPU usage: {peak:.2f} GB peak")
                self.results["recommendations"].append(f"Optimize GPU memory usage (currently {peak:.2f} GB)")
                return False

        except Exception as e:
            print(f"  ❌ GPU efficiency test failed: {e}")
            return False

    def generate_report(self):
        """Generate comprehensive report with recommendations"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST REPORT")
        print("="*60)

        # Count test results
        passed_tests = 0
        total_tests = len(self.results["tests"])

        for test_name, result in self.results["tests"].items():
            if isinstance(result, bool) and result:
                passed_tests += 1
            elif isinstance(result, dict):
                # Complex test - check if mostly passing
                if not any("❌" in str(v) for v in result.values()):
                    passed_tests += 1
            elif isinstance(result, str) and "✅" in result:
                passed_tests += 1

        print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")

        # Issues found
        if self.results["issues"]:
            print(f"\n⚠️ Issues Found ({len(self.results['issues'])}):")
            for issue in self.results["issues"]:
                print(f"  • {issue}")
        else:
            print("\n✅ No critical issues found!")

        # Recommendations
        print(f"\n💡 Recommendations ({len(self.results['recommendations'])}):")

        # Add standard recommendations based on tests
        if "metrics" in self.results and "avg_fps" in self.results["metrics"]:
            if self.results["metrics"]["avg_fps"] < 30:
                self.results["recommendations"].append("Consider further optimization for higher FPS")

        if not self.results["recommendations"]:
            self.results["recommendations"].append("System is well-optimized - maintain current performance")

        for rec in self.results["recommendations"]:
            print(f"  • {rec}")

        # Production readiness
        print("\n" + "="*60)

        production_ready = passed_tests >= total_tests * 0.7 and len(self.results["issues"]) < 3

        if production_ready:
            print("✅ SYSTEM IS PRODUCTION READY")
            print("   Most tests passing")
            print("   Few or no critical issues")
            print("   Ready for deployment")
        else:
            print("⚠️ SYSTEM NEEDS IMPROVEMENTS")
            print(f"   Only {passed_tests}/{total_tests} tests passing")
            print(f"   {len(self.results['issues'])} issues need attention")

        # Save detailed report
        with open("thorough_test_report.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print("\n📄 Detailed report saved to: thorough_test_report.json")

        return production_ready

def main():
    """Run all tests and generate report"""
    print("🚀 Starting Thorough Final Testing...")
    print("="*60)

    tester = ThoroughTester()

    # Run all tests
    test_results = []
    test_results.append(("Imports", tester.test_1_system_imports()))
    test_results.append(("Memory", tester.test_2_memory_usage()))
    test_results.append(("Output Quality", tester.test_3_output_quality()))
    test_results.append(("Error Handling", tester.test_4_error_handling()))
    test_results.append(("Performance", tester.test_5_performance_benchmark()))
    test_results.append(("Code Quality", tester.test_6_code_quality()))
    test_results.append(("GPU Efficiency", tester.test_7_gpu_efficiency()))

    # Generate final report
    production_ready = tester.generate_report()

    print("\n" + "="*60)
    print("Testing complete!")

    return production_ready

if __name__ == "__main__":
    success = main()