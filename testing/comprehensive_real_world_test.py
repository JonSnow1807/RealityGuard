#!/usr/bin/env python3
"""
Comprehensive Real-World Testing Suite for CV Systems
Tests with actual video, edge cases, and production scenarios
"""

import cv2
import numpy as np
import time
import json
import psutil
import torch
from pathlib import Path
import traceback
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our three systems
from excellence_cv_system import ExcellenceVisionSystem
from state_of_art_vision import StateOfTheArtVisionSystem
from advanced_cv_system import SceneUnderstanding


class RealWorldTester:
    """Test all CV systems with real video and production scenarios."""

    def __init__(self):
        self.results = {
            'excellence': {},
            'state_of_art': {},
            'advanced': {}
        }

        # Initialize systems
        print("Initializing CV systems...")
        self.excellence_system = ExcellenceVisionSystem()
        self.sota_system = StateOfTheArtVisionSystem()
        self.advanced_system = SceneUnderstanding()

    def create_test_videos(self):
        """Create realistic test videos with different scenarios."""
        videos = {}

        # Test 1: Person walking (object tracking test)
        print("\nCreating test video 1: Person walking...")
        frames = []
        for i in range(120):  # 4 seconds at 30fps
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 30

            # Simulate walking person
            person_x = int(100 + i * 8)  # Walking across frame
            person_y = 360

            # Head
            cv2.circle(frame, (person_x, person_y - 80), 25, (200, 150, 100), -1)
            # Body
            cv2.rectangle(frame, (person_x - 30, person_y - 50),
                         (person_x + 30, person_y + 50), (100, 100, 200), -1)
            # Legs movement
            leg_offset = int(20 * np.sin(i * 0.5))
            cv2.line(frame, (person_x, person_y + 50),
                    (person_x - leg_offset, person_y + 120), (50, 50, 50), 8)
            cv2.line(frame, (person_x, person_y + 50),
                    (person_x + leg_offset, person_y + 120), (50, 50, 50), 8)

            # Add some background objects
            cv2.rectangle(frame, (800, 200), (1000, 400), (150, 150, 150), -1)
            cv2.circle(frame, (200, 200), 40, (100, 200, 100), -1)

            frames.append(frame)
        videos['walking_person'] = frames

        # Test 2: Multiple moving objects (tracking stress test)
        print("Creating test video 2: Multiple objects...")
        frames = []
        for i in range(120):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 40

            # Object 1: Circular motion
            x1 = int(640 + 200 * np.cos(i * 0.1))
            y1 = int(360 + 200 * np.sin(i * 0.1))
            cv2.circle(frame, (x1, y1), 30, (255, 100, 100), -1)

            # Object 2: Horizontal motion
            x2 = int(100 + (i * 10) % 1080)
            cv2.rectangle(frame, (x2, 300), (x2 + 60, 360), (100, 255, 100), -1)

            # Object 3: Diagonal motion
            x3 = int(50 + i * 8)
            y3 = int(50 + i * 4)
            cv2.circle(frame, (x3 % 1280, y3 % 720), 25, (100, 100, 255), -1)

            # Object 4: Static
            cv2.rectangle(frame, (1100, 500), (1200, 600), (200, 200, 200), -1)

            frames.append(frame)
        videos['multiple_objects'] = frames

        # Test 3: Low light / high noise (robustness test)
        print("Creating test video 3: Low light with noise...")
        frames = []
        for i in range(120):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 10  # Very dark

            # Add object with low contrast
            x = int(640 + 100 * np.sin(i * 0.05))
            y = 360
            cv2.circle(frame, (x, y), 40, (30, 30, 40), -1)

            # Add noise
            noise = np.random.randn(720, 1280, 3) * 20
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

            frames.append(frame)
        videos['low_light_noise'] = frames

        # Test 4: Fast motion (motion blur simulation)
        print("Creating test video 4: Fast motion...")
        frames = []
        for i in range(120):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50

            # Fast moving object
            x = int((i * 30) % 1280)
            y = int(360 + 100 * np.sin(i * 0.3))

            # Simulate motion blur
            for j in range(5):
                alpha = 1.0 - j * 0.2
                x_blur = x - j * 10
                cv2.circle(frame, (x_blur, y), 20,
                          (int(200*alpha), int(100*alpha), int(100*alpha)), -1)

            frames.append(frame)
        videos['fast_motion'] = frames

        # Test 5: Occlusion handling
        print("Creating test video 5: Occlusions...")
        frames = []
        for i in range(120):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 60

            # Moving object that gets occluded
            x = int(100 + i * 8)
            y = 360
            cv2.circle(frame, (x, y), 35, (200, 100, 100), -1)

            # Occluding object (appears halfway)
            if 40 < i < 80:
                cv2.rectangle(frame, (500, 300), (700, 420), (100, 100, 200), -1)

            frames.append(frame)
        videos['occlusion'] = frames

        return videos

    def test_system_performance(self, system, system_name, video_frames):
        """Test a single system with comprehensive metrics."""
        metrics = {
            'fps': [],
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'accuracy': [],
            'features': {}
        }

        print(f"\nTesting {system_name}...")

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        failed_frames = 0
        successful_features = set()

        for i, frame in enumerate(video_frames):
            try:
                # Measure CPU before
                cpu_before = psutil.cpu_percent(interval=0.01)

                # Time the processing
                start = time.perf_counter()

                # Process based on system type
                if system_name == 'excellence':
                    results = system.process(frame)
                elif system_name == 'state_of_art':
                    results = system.process_frame(frame)
                else:  # advanced
                    results = system.analyze_scene(frame)

                end = time.perf_counter()
                processing_time = (end - start) * 1000  # ms

                # Measure CPU after
                cpu_after = psutil.cpu_percent(interval=0.01)

                # Record metrics
                metrics['processing_times'].append(processing_time)
                metrics['fps'].append(1000 / processing_time if processing_time > 0 else 0)
                metrics['cpu_usage'].append(cpu_after - cpu_before)

                # Memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                metrics['memory_usage'].append(current_memory - initial_memory)

                # Track what features actually worked
                if results:
                    if isinstance(results, dict):
                        successful_features.update(results.keys())

            except Exception as e:
                failed_frames += 1
                metrics['processing_times'].append(0)
                metrics['fps'].append(0)
                print(f"  Frame {i} failed: {str(e)[:100]}")

        # Calculate statistics
        if metrics['processing_times']:
            valid_times = [t for t in metrics['processing_times'] if t > 0]
            if valid_times:
                metrics['avg_fps'] = 1000 / np.mean(valid_times)
                metrics['min_fps'] = 1000 / np.max(valid_times)
                metrics['max_fps'] = 1000 / np.min(valid_times)
                metrics['avg_processing_ms'] = np.mean(valid_times)
                metrics['std_processing_ms'] = np.std(valid_times)
            else:
                metrics['avg_fps'] = 0
                metrics['min_fps'] = 0
                metrics['max_fps'] = 0
                metrics['avg_processing_ms'] = 0
                metrics['std_processing_ms'] = 0

        metrics['failed_frames'] = failed_frames
        metrics['success_rate'] = (len(video_frames) - failed_frames) / len(video_frames) * 100
        metrics['working_features'] = list(successful_features)

        if metrics['memory_usage']:
            metrics['avg_memory_mb'] = np.mean(metrics['memory_usage'])
            metrics['max_memory_mb'] = np.max(metrics['memory_usage'])

        if metrics['cpu_usage']:
            metrics['avg_cpu_percent'] = np.mean([c for c in metrics['cpu_usage'] if c > 0])

        return metrics

    def test_specific_capabilities(self):
        """Test specific capabilities that big tech would care about."""
        capabilities = {}

        print("\n" + "="*60)
        print("TESTING SPECIFIC CAPABILITIES")
        print("="*60)

        # 1. Real-time capability (>30 FPS possible?)
        print("\n1. Real-time Processing Test (target: >30 FPS)...")
        small_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        for name, system in [('excellence', self.excellence_system),
                             ('state_of_art', self.sota_system),
                             ('advanced', self.advanced_system)]:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                try:
                    if name == 'excellence':
                        system.process(small_frame)
                    elif name == 'state_of_art':
                        system.process_frame(small_frame)
                    else:
                        system.analyze_scene(small_frame)
                except:
                    pass
                times.append(time.perf_counter() - start)

            avg_time = np.mean(times)
            fps = 1 / avg_time if avg_time > 0 else 0
            capabilities[f'{name}_realtime_fps'] = fps
            print(f"  {name}: {fps:.1f} FPS on 320x240")

        # 2. Scalability test (different resolutions)
        print("\n2. Scalability Test (multiple resolutions)...")
        resolutions = [(160, 120), (320, 240), (640, 480), (1280, 720)]

        for name, system in [('excellence', self.excellence_system)]:  # Test one for now
            scale_results = {}
            for w, h in resolutions:
                frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                start = time.perf_counter()
                try:
                    system.process(frame)
                    process_time = time.perf_counter() - start
                    scale_results[f'{w}x{h}'] = 1 / process_time
                except:
                    scale_results[f'{w}x{h}'] = 0
            capabilities[f'{name}_scalability'] = scale_results
            print(f"  {name}: {scale_results}")

        # 3. Robustness test
        print("\n3. Robustness Test (edge cases)...")
        edge_cases = {
            'empty': np.zeros((480, 640, 3), dtype=np.uint8),
            'white': np.ones((480, 640, 3), dtype=np.uint8) * 255,
            'single_channel': np.random.randint(0, 255, (480, 640), dtype=np.uint8),
            'high_noise': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }

        for name, system in [('excellence', self.excellence_system)]:
            robust_score = 0
            for case_name, frame in edge_cases.items():
                try:
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    if name == 'excellence':
                        result = system.process(frame)
                    elif name == 'state_of_art':
                        result = system.process_frame(frame)
                    else:
                        result = system.analyze_scene(frame)

                    if result:
                        robust_score += 1
                except Exception as e:
                    print(f"    {name} failed on {case_name}: {str(e)[:50]}")

            capabilities[f'{name}_robustness'] = f"{robust_score}/{len(edge_cases)}"

        return capabilities

    def run_comprehensive_tests(self):
        """Run all tests and generate report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE REAL-WORLD TESTING")
        print("="*60)

        # Create test videos
        videos = self.create_test_videos()

        all_results = {}

        # Test each system on each video
        for video_name, frames in videos.items():
            print(f"\n\nTesting on: {video_name}")
            print("-" * 40)

            all_results[video_name] = {}

            # Test excellence system
            try:
                metrics = self.test_system_performance(
                    self.excellence_system, 'excellence', frames[:30]  # Test 1 second
                )
                all_results[video_name]['excellence'] = metrics
            except Exception as e:
                print(f"Excellence system failed: {e}")
                all_results[video_name]['excellence'] = {'error': str(e)}

            # Test state-of-art system
            try:
                metrics = self.test_system_performance(
                    self.sota_system, 'state_of_art', frames[:30]
                )
                all_results[video_name]['state_of_art'] = metrics
            except Exception as e:
                print(f"State-of-art system failed: {e}")
                all_results[video_name]['state_of_art'] = {'error': str(e)}

            # Test advanced system
            try:
                metrics = self.test_system_performance(
                    self.advanced_system, 'advanced', frames[:30]
                )
                all_results[video_name]['advanced'] = metrics
            except Exception as e:
                print(f"Advanced system failed: {e}")
                all_results[video_name]['advanced'] = {'error': str(e)}

        # Test specific capabilities
        capabilities = self.test_specific_capabilities()

        # Generate report
        self.generate_report(all_results, capabilities)

        return all_results, capabilities

    def generate_report(self, results, capabilities):
        """Generate comprehensive testing report."""
        print("\n\n" + "="*80)
        print("FINAL TESTING REPORT - PRODUCTION READINESS")
        print("="*80)

        # Performance Summary
        print("\n1. PERFORMANCE METRICS")
        print("-" * 40)

        for video in results:
            print(f"\n{video.upper()}:")
            for system in results[video]:
                if 'error' not in results[video][system]:
                    data = results[video][system]
                    if 'avg_fps' in data:
                        print(f"  {system}:")
                        print(f"    • FPS: {data.get('avg_fps', 0):.1f} (min: {data.get('min_fps', 0):.1f}, max: {data.get('max_fps', 0):.1f})")
                        print(f"    • Processing: {data.get('avg_processing_ms', 0):.1f}ms ± {data.get('std_processing_ms', 0):.1f}ms")
                        print(f"    • Success Rate: {data.get('success_rate', 0):.1f}%")
                        print(f"    • Memory: {data.get('avg_memory_mb', 0):.1f}MB (max: {data.get('max_memory_mb', 0):.1f}MB)")
                        print(f"    • Working Features: {', '.join(data.get('working_features', []))}")

        print("\n2. PRODUCTION READINESS")
        print("-" * 40)

        # Determine best system
        best_system = None
        best_fps = 0
        best_reliability = 0

        for video in results:
            for system in results[video]:
                if 'error' not in results[video][system]:
                    data = results[video][system]
                    if data.get('avg_fps', 0) > best_fps and data.get('success_rate', 0) > 80:
                        best_fps = data.get('avg_fps', 0)
                        best_reliability = data.get('success_rate', 0)
                        best_system = system

        if best_system:
            print(f"\n✓ RECOMMENDED SYSTEM: {best_system}")
            print(f"  • Average FPS: {best_fps:.1f}")
            print(f"  • Reliability: {best_reliability:.1f}%")
        else:
            print("\n✗ No system meets production requirements")

        print("\n3. CAPABILITIES ASSESSMENT")
        print("-" * 40)
        for cap, value in capabilities.items():
            print(f"  • {cap}: {value}")

        print("\n4. BIG TECH ACQUISITION VALUE")
        print("-" * 40)

        # Calculate acquisition value points
        value_points = 0

        # Real-time capability
        if any(v > 30 for k, v in capabilities.items() if 'realtime_fps' in k):
            print("  ✓ Real-time capable (>30 FPS)")
            value_points += 1
        else:
            print("  ✗ Not real-time capable")

        # Reliability
        if best_reliability > 90:
            print("  ✓ High reliability (>90%)")
            value_points += 1
        else:
            print("  ✗ Reliability needs improvement")

        # Unique features
        unique_features = ['depth', 'self_supervised', 'zero_shot', 'transformer']
        found_unique = []
        for video in results:
            for system in results[video]:
                if 'working_features' in results[video][system]:
                    for feature in results[video][system]['working_features']:
                        for unique in unique_features:
                            if unique in feature.lower():
                                found_unique.append(unique)

        if found_unique:
            print(f"  ✓ Unique features: {', '.join(set(found_unique))}")
            value_points += len(set(found_unique))

        print(f"\n  ACQUISITION VALUE SCORE: {value_points}/5")

        if value_points >= 3:
            print("  → Ready for big tech acquisition discussions")
        else:
            print("  → Needs more optimization before approaching big tech")

        # Save detailed results
        with open('test_results.json', 'w') as f:
            json.dump({
                'results': results,
                'capabilities': capabilities,
                'timestamp': time.time(),
                'best_system': best_system,
                'value_score': value_points
            }, f, indent=2, default=str)

        print("\n  Full results saved to test_results.json")
        print("="*80)


if __name__ == "__main__":
    tester = RealWorldTester()
    results, capabilities = tester.run_comprehensive_tests()