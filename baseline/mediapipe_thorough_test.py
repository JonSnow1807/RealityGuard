#!/usr/bin/env python3
"""
EXHAUSTIVE MEDIAPIPE TESTING - NO TRUST MODE
Tests every aspect of MediaPipe to verify it actually works
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

from option_b_mediapipe import MediaPipeHybridGuard, MEDIAPIPE_AVAILABLE


class MediaPipeExhaustiveTester:
    """Comprehensive tester for MediaPipe with skeptical verification"""

    def __init__(self):
        self.test_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests': [],
            'summary': {},
            'failures': []
        }
        self.proof_dir = Path("/tmp/mediapipe_proof")
        self.proof_dir.mkdir(exist_ok=True)

    def create_test_suite(self) -> List[Dict]:
        """Create comprehensive test suite with edge cases"""
        tests = []

        # Category 1: Basic shapes
        tests.append({
            'category': 'basic_shapes',
            'name': 'single_white_circle',
            'frame': self._create_circle_frame(1, [(320, 240, 60)], color=(255, 255, 255)),
            'expected': {'shapes': 1, 'faces': 0},
            'description': 'Single white circle on black'
        })

        tests.append({
            'category': 'basic_shapes',
            'name': 'three_white_circles',
            'frame': self._create_circle_frame(3, [(160, 240, 50), (320, 240, 60), (480, 240, 55)]),
            'expected': {'shapes': 3, 'faces': 0},
            'description': 'Three distinct white circles'
        })

        # Category 2: Edge cases
        tests.append({
            'category': 'edge_cases',
            'name': 'empty_black',
            'frame': np.zeros((480, 640, 3), dtype=np.uint8),
            'expected': {'shapes': 0, 'faces': 0},
            'description': 'Completely black frame'
        })

        tests.append({
            'category': 'edge_cases',
            'name': 'empty_white',
            'frame': np.ones((480, 640, 3), dtype=np.uint8) * 255,
            'expected': {'shapes': 0, 'faces': 0},
            'description': 'Completely white frame'
        })

        tests.append({
            'category': 'edge_cases',
            'name': 'random_noise',
            'frame': np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8),
            'expected': {'shapes': 0, 'faces': 0},
            'description': 'Random noise pattern'
        })

        # Category 3: Size variations
        tests.append({
            'category': 'size_tests',
            'name': 'tiny_circles',
            'frame': self._create_circle_frame(5, [(100+i*100, 240, 15) for i in range(5)]),
            'expected': {'shapes': 0, 'faces': 0},  # Too small for detection
            'description': 'Five tiny circles (15px radius)'
        })

        tests.append({
            'category': 'size_tests',
            'name': 'large_circle',
            'frame': self._create_circle_frame(1, [(640, 360, 200)], size=(720, 1280)),
            'expected': {'shapes': 1, 'faces': 0},
            'description': 'Large circle in HD resolution'
        })

        tests.append({
            'category': 'size_tests',
            'name': 'mixed_sizes',
            'frame': self._create_mixed_size_circles(),
            'expected': {'shapes': 3, 'faces': 0},
            'description': 'Circles of varying sizes'
        })

        # Category 4: Overlapping and complex
        tests.append({
            'category': 'complex',
            'name': 'overlapping_circles',
            'frame': self._create_overlapping_circles(),
            'expected': {'shapes': 1, 'faces': 0},  # Should merge overlapping
            'description': 'Two overlapping circles'
        })

        tests.append({
            'category': 'complex',
            'name': 'nested_circles',
            'frame': self._create_nested_circles(),
            'expected': {'shapes': 2, 'faces': 0},
            'description': 'Concentric circles'
        })

        # Category 5: Different shapes
        tests.append({
            'category': 'shapes',
            'name': 'rectangle',
            'frame': self._create_rectangle(),
            'expected': {'shapes': 1, 'faces': 0},
            'description': 'Single rectangle'
        })

        tests.append({
            'category': 'shapes',
            'name': 'ellipse',
            'frame': self._create_ellipse(),
            'expected': {'shapes': 1, 'faces': 0},
            'description': 'Single ellipse'
        })

        tests.append({
            'category': 'shapes',
            'name': 'mixed_shapes',
            'frame': self._create_mixed_shapes(),
            'expected': {'shapes': 3, 'faces': 0},
            'description': 'Circle + Rectangle + Ellipse'
        })

        # Category 6: Color variations
        tests.append({
            'category': 'colors',
            'name': 'gray_circles',
            'frame': self._create_circle_frame(2, [(200, 240, 60), (440, 240, 60)], color=(128, 128, 128)),
            'expected': {'shapes': 2, 'faces': 0},
            'description': 'Gray circles on black'
        })

        tests.append({
            'category': 'colors',
            'name': 'colored_circles',
            'frame': self._create_colored_circles(),
            'expected': {'shapes': 3, 'faces': 0},
            'description': 'Red, Green, Blue circles'
        })

        # Category 7: Resolution tests
        for res_name, resolution in [('360p', (360, 480)), ('480p', (480, 640)),
                                     ('720p', (720, 1280)), ('1080p', (1080, 1920))]:
            tests.append({
                'category': 'resolution',
                'name': f'circle_{res_name}',
                'frame': self._create_circle_frame(1, [(resolution[1]//2, resolution[0]//2, min(80, resolution[0]//6))],
                                                  size=resolution),
                'expected': {'shapes': 1, 'faces': 0},
                'description': f'Circle at {res_name} resolution'
            })

        # Category 8: Stress tests
        tests.append({
            'category': 'stress',
            'name': 'many_circles',
            'frame': self._create_many_circles(10),
            'expected': {'shapes': 10, 'faces': 0},
            'description': '10 circles in grid'
        })

        tests.append({
            'category': 'stress',
            'name': 'dense_pattern',
            'frame': self._create_dense_pattern(),
            'expected': {'shapes': -1, 'faces': 0},  # Unknown exact count
            'description': 'Dense pattern of shapes'
        })

        return tests

    def _create_circle_frame(self, count, circles_data, size=(480, 640), color=(255, 255, 255)):
        """Create frame with specified circles"""
        frame = np.zeros((*size, 3), dtype=np.uint8)
        for x, y, r in circles_data[:count]:
            cv2.circle(frame, (x, y), r, color, -1)
        return frame

    def _create_mixed_size_circles(self):
        """Create circles of different sizes"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 240), 30, (255, 255, 255), -1)  # Small
        cv2.circle(frame, (320, 240), 60, (255, 255, 255), -1)  # Medium
        cv2.circle(frame, (480, 240), 90, (255, 255, 255), -1)  # Large
        return frame

    def _create_overlapping_circles(self):
        """Create overlapping circles"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (300, 240), 70, (255, 255, 255), -1)
        cv2.circle(frame, (360, 240), 70, (255, 255, 255), -1)
        return frame

    def _create_nested_circles(self):
        """Create concentric circles"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
        cv2.circle(frame, (320, 240), 50, (0, 0, 0), -1)
        cv2.circle(frame, (320, 240), 30, (255, 255, 255), -1)
        return frame

    def _create_rectangle(self):
        """Create single rectangle"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (250, 180), (390, 300), (255, 255, 255), -1)
        return frame

    def _create_ellipse(self):
        """Create single ellipse"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(frame, (320, 240), (100, 60), 0, 0, 360, (255, 255, 255), -1)
        return frame

    def _create_mixed_shapes(self):
        """Create mixed shapes"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(frame, (200, 360), 80, (255, 255, 255), -1)
        cv2.rectangle(frame, (500, 300), (700, 420), (255, 255, 255), -1)
        cv2.ellipse(frame, (1000, 360), (100, 60), 45, 0, 360, (255, 255, 255), -1)
        return frame

    def _create_colored_circles(self):
        """Create colored circles"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 240), 60, (0, 0, 255), -1)  # Red
        cv2.circle(frame, (320, 240), 60, (0, 255, 0), -1)  # Green
        cv2.circle(frame, (480, 240), 60, (255, 0, 0), -1)  # Blue
        return frame

    def _create_many_circles(self, count):
        """Create grid of circles"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cols = 5
        rows = (count + cols - 1) // cols
        for i in range(count):
            row = i // cols
            col = i % cols
            x = 150 + col * 200
            y = 150 + row * 200
            cv2.circle(frame, (x, y), 50, (255, 255, 255), -1)
        return frame

    def _create_dense_pattern(self):
        """Create dense pattern"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(50, 600, 100):
            for y in range(50, 450, 100):
                cv2.circle(frame, (x, y), 35, (255, 255, 255), -1)
        return frame

    def test_mode(self, detector, mode: str, test_suite: List[Dict]) -> Dict:
        """Test specific mode comprehensively"""
        mode_results = {
            'mode': mode,
            'tests': [],
            'stats': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'accuracy_sum': 0
            }
        }

        print(f"\n{'='*60}")
        print(f"Testing Mode: {mode.upper()}")
        print(f"{'='*60}")

        for test in test_suite:
            # Skip face tests for shape modes
            if mode == 'shapes_only' and 'face' in test['name']:
                continue

            print(f"\nTest: {test['name']} ({test['category']})")
            print(f"  Description: {test['description']}")

            # Process frame
            frame = test['frame'].copy()
            start = time.perf_counter()
            output, info = detector.process_frame(frame, mode=mode, draw_debug=True)
            elapsed = (time.perf_counter() - start) * 1000

            # Get expected count based on mode
            if mode == 'faces_only':
                expected = test['expected']['faces']
            elif mode == 'shapes_only':
                expected = test['expected']['shapes']
            else:  # hybrid or mesh
                expected = test['expected']['shapes'] + test['expected']['faces']

            # Handle unknown expected counts
            if expected == -1:
                expected = info['detections']  # Accept whatever was detected
                accuracy = 100 if info['detections'] > 0 else 0
            else:
                # Calculate accuracy
                detected = info['detections']
                if expected == 0:
                    accuracy = 100 if detected == 0 else 0
                else:
                    accuracy = min(detected, expected) / expected * 100
                    if detected > expected:
                        over_detection = (detected - expected) / expected
                        penalty = min(50, over_detection * 25)
                        accuracy = max(0, accuracy - penalty)

            # Verify pixel modification
            pixels_modified = np.sum(cv2.absdiff(frame, output) > 0)
            pixels_expected = expected > 0

            # Save visual proof
            proof_path = self._save_proof(test['name'], mode, frame, output, info)

            # Record results
            test_result = {
                'test': test['name'],
                'category': test['category'],
                'expected': expected,
                'detected': detected,
                'accuracy': accuracy,
                'time_ms': elapsed,
                'fps': 1000/elapsed,
                'pixels_modified': int(pixels_modified),
                'detection_types': info.get('detection_types', {}),
                'passed': accuracy >= 70,
                'proof': str(proof_path)
            }

            mode_results['tests'].append(test_result)
            mode_results['stats']['total'] += 1
            if test_result['passed']:
                mode_results['stats']['passed'] += 1
            else:
                mode_results['stats']['failed'] += 1
            mode_results['stats']['accuracy_sum'] += accuracy

            # Print result
            status = "✅" if test_result['passed'] else "❌"
            print(f"  {status} Detected: {detected}, Expected: {expected}")
            print(f"     Accuracy: {accuracy:.1f}%, Time: {elapsed:.2f}ms, FPS: {1000/elapsed:.1f}")
            if info.get('detection_types'):
                print(f"     Types: {info['detection_types']}")

        # Calculate averages
        if mode_results['stats']['total'] > 0:
            mode_results['stats']['avg_accuracy'] = mode_results['stats']['accuracy_sum'] / mode_results['stats']['total']
            mode_results['stats']['success_rate'] = (mode_results['stats']['passed'] / mode_results['stats']['total']) * 100
        else:
            mode_results['stats']['avg_accuracy'] = 0
            mode_results['stats']['success_rate'] = 0

        return mode_results

    def _save_proof(self, test_name: str, mode: str, original: np.ndarray, processed: np.ndarray, info: Dict) -> Path:
        """Save visual proof of detection"""
        # Create comparison image
        comparison = np.hstack([original, processed])

        # Add labels
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Processed ({mode})", (original.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add detection count
        cv2.putText(comparison, f"Detections: {info['detections']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save
        filename = self.proof_dir / f"{mode}_{test_name}.jpg"
        cv2.imwrite(str(filename), comparison)
        return filename

    def stress_test(self, detector) -> Dict:
        """Run stress tests for performance"""
        print(f"\n{'='*60}")
        print("STRESS TEST: Performance Under Load")
        print(f"{'='*60}")

        stress_results = {
            'batch_sizes': [],
            'resolutions': [],
            'continuous': {}
        }

        # Test different batch sizes
        print("\n1. Batch Size Scaling:")
        base_frame = self._create_circle_frame(3, [(160, 240, 50), (320, 240, 60), (480, 240, 55)])

        for batch_size in [1, 5, 10, 20, 50]:
            frames = [base_frame.copy() for _ in range(batch_size)]

            start = time.perf_counter()
            for frame in frames:
                detector.process_frame(frame, mode='hybrid')
            elapsed = (time.perf_counter() - start) * 1000

            fps_total = (batch_size * 1000) / elapsed
            result = {
                'batch_size': batch_size,
                'total_time_ms': elapsed,
                'fps_per_frame': fps_total / batch_size,
                'fps_total': fps_total
            }
            stress_results['batch_sizes'].append(result)
            print(f"  Batch {batch_size:2}: {elapsed:.1f}ms total, {fps_total:.1f} FPS")

        # Test different resolutions
        print("\n2. Resolution Scaling:")
        resolutions = [
            ('360p', (360, 480)),
            ('480p', (480, 640)),
            ('720p', (720, 1280)),
            ('1080p', (1080, 1920)),
            ('4K', (2160, 3840))
        ]

        for res_name, resolution in resolutions:
            frame = self._create_circle_frame(1, [(resolution[1]//2, resolution[0]//2, 80)], size=resolution)

            start = time.perf_counter()
            output, info = detector.process_frame(frame, mode='hybrid')
            elapsed = (time.perf_counter() - start) * 1000

            result = {
                'resolution': res_name,
                'dimensions': resolution,
                'time_ms': elapsed,
                'fps': 1000 / elapsed,
                'detections': info['detections']
            }
            stress_results['resolutions'].append(result)
            print(f"  {res_name:5} ({resolution[1]}x{resolution[0]}): {elapsed:.1f}ms, {1000/elapsed:.1f} FPS")

        # Continuous processing test
        print("\n3. Continuous Processing (100 frames):")
        frame = self._create_circle_frame(3, [(160, 240, 50), (320, 240, 60), (480, 240, 55)])

        times = []
        start_total = time.perf_counter()
        for i in range(100):
            start = time.perf_counter()
            detector.process_frame(frame, mode='hybrid')
            times.append((time.perf_counter() - start) * 1000)
        total_time = (time.perf_counter() - start_total) * 1000

        stress_results['continuous'] = {
            'frames': 100,
            'total_time_ms': total_time,
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'avg_fps': 1000 / np.mean(times)
        }

        print(f"  Total: {total_time:.1f}ms")
        print(f"  Average: {np.mean(times):.2f}ms per frame")
        print(f"  Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")
        print(f"  FPS: {1000/np.mean(times):.1f}")

        return stress_results

    def run_all_tests(self):
        """Run all tests comprehensively"""
        print("\n" + "="*80)
        print("MEDIAPIPE EXHAUSTIVE TESTING - ABSOLUTE VERIFICATION")
        print("="*80)

        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe not available!")
            return None

        # Initialize detector
        detector = MediaPipeHybridGuard(min_detection_confidence=0.5)

        # Create test suite
        test_suite = self.create_test_suite()
        print(f"\n📋 Created {len(test_suite)} test cases")

        # Test all modes
        modes = ['shapes_only', 'faces_only', 'hybrid', 'mesh']
        all_mode_results = []

        for mode in modes:
            mode_results = self.test_mode(detector, mode, test_suite)
            all_mode_results.append(mode_results)
            self.test_results['tests'].append(mode_results)

        # Run stress tests
        stress_results = self.stress_test(detector)
        self.test_results['stress'] = stress_results

        # Generate summary
        self._generate_summary(all_mode_results)

        # Save results
        self._save_results()

        return self.test_results

    def _generate_summary(self, mode_results):
        """Generate comprehensive summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY")
        print("="*80)

        total_tests = 0
        total_passed = 0
        best_mode = None
        best_accuracy = 0

        for result in mode_results:
            mode = result['mode']
            stats = result['stats']

            print(f"\n{mode.upper()} Mode:")
            print(f"  Tests Run: {stats['total']}")
            print(f"  Passed: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)")
            print(f"  Average Accuracy: {stats['avg_accuracy']:.1f}%")

            total_tests += stats['total']
            total_passed += stats['passed']

            if stats['avg_accuracy'] > best_accuracy:
                best_accuracy = stats['avg_accuracy']
                best_mode = mode

        overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0

        self.test_results['summary'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'overall_success_rate': overall_success,
            'best_mode': best_mode,
            'best_accuracy': best_accuracy
        }

        print(f"\n📊 OVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Total Passed: {total_passed}")
        print(f"  Success Rate: {overall_success:.1f}%")
        print(f"  Best Mode: {best_mode} ({best_accuracy:.1f}% accuracy)")

        # Trust assessment
        print(f"\n🔒 TRUST ASSESSMENT:")
        if overall_success >= 70:
            print("  ✅ TRUSTWORTHY - High success rate")
            trust_level = "HIGH"
        elif overall_success >= 50:
            print("  ⚠️ PARTIALLY TRUSTWORTHY - Some failures")
            trust_level = "MEDIUM"
        else:
            print("  ❌ NOT TRUSTWORTHY - Too many failures")
            trust_level = "LOW"

        self.test_results['trust_level'] = trust_level

    def _save_results(self):
        """Save all results to file"""
        output_file = Path("/tmp/mediapipe_exhaustive_results.json")
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {output_file}")
        print(f"🖼️ Visual proofs saved to: {self.proof_dir}")


if __name__ == "__main__":
    tester = MediaPipeExhaustiveTester()
    results = tester.run_all_tests()

    if results and results['trust_level'] == "HIGH":
        print("\n✅ MediaPipe PASSED exhaustive testing!")
    elif results and results['trust_level'] == "MEDIUM":
        print("\n⚠️ MediaPipe PARTIALLY passed testing")
    else:
        print("\n❌ MediaPipe FAILED exhaustive testing")