#!/usr/bin/env python3
"""
COMPREHENSIVE REAL-WORLD TESTING
No hallucinations, no optimistic reporting - just raw facts
Tests multiple implementations to find what ACTUALLY works
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import traceback

# Test all available implementations
implementations = []

try:
    from realityguard_ultimate import RealityGuardUltimate, UltimateConfig
    implementations.append(('ultimate', RealityGuardUltimate, UltimateConfig))
except Exception as e:
    print(f"Could not load ultimate: {e}")

try:
    from realityguard_gpu_optimized import RealityGuardGPU, GPUConfig
    implementations.append(('gpu', RealityGuardGPU, GPUConfig))
except Exception as e:
    print(f"Could not load GPU: {e}")

try:
    from patent_enhanced_anti_ai import PatentEnhancedAntiAISystem, PatentAntiAIConfig
    implementations.append(('patent', PatentEnhancedAntiAISystem, PatentAntiAIConfig))
except Exception as e:
    print(f"Could not load patent: {e}")


class ComprehensiveTest:
    """Honest, thorough testing with no sugar-coating"""

    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def test_single_image(self, system, image_path, system_name):
        """Test single image - measure ACTUAL performance"""

        print(f"\n  Testing {system_name} with {image_path}")

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        print(f"    Image size: {w}x{h}")

        # NO WARMUP - measure cold start too
        times = []
        pixel_diffs = []

        # Run 10 times - include first cold run
        for i in range(10):
            start = time.perf_counter()  # High precision timer

            try:
                if system_name == 'patent':
                    # Patent system has different API
                    protected, stats = system.process_frame(img)
                else:
                    protected, stats = system.process_frame(img)

                elapsed = time.perf_counter() - start
                times.append(elapsed)

                # Calculate REAL pixel difference
                diff = np.mean(np.abs(img.astype(float) - protected.astype(float)))
                pixel_diffs.append(diff)

                # Save first result for visual inspection
                if i == 0:
                    output_name = f"test_result_{system_name}_{os.path.basename(image_path)}"
                    cv2.imwrite(output_name, protected)

            except Exception as e:
                print(f"    ERROR on frame {i}: {e}")
                return None

        # Calculate HONEST metrics
        fps_values = [1.0/t for t in times]

        return {
            'image': image_path,
            'resolution': f"{w}x{h}",
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values),
            'std_fps': np.std(fps_values),
            'cold_start_fps': fps_values[0],  # First run
            'warm_fps': np.mean(fps_values[1:]),  # After warmup
            'avg_time_ms': np.mean(times) * 1000,
            'pixel_diff': np.mean(pixel_diffs),
            'pixel_diff_std': np.std(pixel_diffs)
        }

    def test_video(self, system, video_path, system_name, max_frames=100):
        """Test video - measure ACTUAL frame processing"""

        print(f"\n  Testing {system_name} with {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"    Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        # Prepare output
        output_path = f"test_output_{system_name}_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

        frame_times = []
        pixel_diffs = []
        frame_count = 0
        errors = 0

        while cap.isOpened() and frame_count < min(max_frames, total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            try:
                start = time.perf_counter()

                if system_name == 'patent':
                    protected, stats = system.process_frame(frame)
                else:
                    protected, stats = system.process_frame(frame)

                elapsed = time.perf_counter() - start
                frame_times.append(elapsed)

                # Real pixel difference
                diff = np.mean(np.abs(frame.astype(float) - protected.astype(float)))
                pixel_diffs.append(diff)

                out.write(protected)

            except Exception as e:
                errors += 1
                print(f"    Frame {frame_count} error: {e}")
                continue

            frame_count += 1

            # Progress update every 20 frames
            if frame_count % 20 == 0:
                current_fps = 1.0 / np.mean(frame_times[-20:])
                print(f"    Frame {frame_count}: {current_fps:.1f} FPS")

        cap.release()
        out.release()

        if len(frame_times) == 0:
            return None

        fps_values = [1.0/t for t in frame_times]

        return {
            'video': video_path,
            'resolution': f"{width}x{height}",
            'frames_processed': frame_count,
            'errors': errors,
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values),
            'std_fps': np.std(fps_values),
            'percentile_25_fps': np.percentile(fps_values, 25),
            'median_fps': np.median(fps_values),
            'percentile_75_fps': np.percentile(fps_values, 75),
            'avg_time_ms': np.mean(frame_times) * 1000,
            'pixel_diff': np.mean(pixel_diffs) if pixel_diffs else 0,
            'output': output_path
        }

    def run_comprehensive_test(self):
        """Run ALL tests on ALL implementations"""

        print("="*80)
        print("COMPREHENSIVE REAL-WORLD TESTING")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Available implementations: {[name for name, _, _ in implementations]}")

        # Find all test files
        test_images = []
        test_videos = []

        # Look for real test files
        for f in os.listdir('.'):
            if f.endswith('.jpg') or f.endswith('.png'):
                if 'test' in f.lower() or 'real' in f.lower():
                    test_images.append(f)
            elif f.endswith('.mp4'):
                if 'test' in f.lower() or 'real' in f.lower() or 'people' in f.lower():
                    test_videos.append(f)

        print(f"\nFound {len(test_images)} test images")
        print(f"Found {len(test_videos)} test videos")

        # Test each implementation
        for impl_name, impl_class, config_class in implementations:
            print(f"\n{'='*60}")
            print(f"Testing: {impl_name.upper()}")
            print(f"{'='*60}")

            try:
                # Create system with reasonable config
                if impl_name == 'gpu':
                    config = config_class(
                        min_strength=0.15,
                        max_strength=0.35,
                        default_strength=0.25
                    )
                elif impl_name == 'ultimate':
                    config = config_class(
                        min_strength=0.20,
                        max_strength=0.60,
                        default_strength=0.35
                    )
                elif impl_name == 'patent':
                    config = config_class()
                else:
                    config = config_class()

                system = impl_class(config)

                self.results[impl_name] = {
                    'images': [],
                    'videos': []
                }

                # Test images
                print("\nImage Tests:")
                for img_path in test_images[:3]:  # Test first 3 images
                    result = self.test_single_image(system, img_path, impl_name)
                    if result:
                        self.results[impl_name]['images'].append(result)

                # Test videos
                print("\nVideo Tests:")
                for video_path in test_videos[:2]:  # Test first 2 videos
                    result = self.test_video(system, video_path, impl_name, max_frames=50)
                    if result:
                        self.results[impl_name]['videos'].append(result)

            except Exception as e:
                print(f"  Failed to test {impl_name}: {e}")
                traceback.print_exc()
                self.results[impl_name] = {'error': str(e)}

    def generate_report(self):
        """Generate honest report with all metrics"""

        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)

        report = {
            'timestamp': self.timestamp,
            'implementations': {}
        }

        for impl_name, data in self.results.items():
            if 'error' in data:
                print(f"\n{impl_name.upper()}: FAILED - {data['error']}")
                continue

            print(f"\n{impl_name.upper()} RESULTS:")
            print("-"*40)

            # Image results
            if data.get('images'):
                img_fps = [r['avg_fps'] for r in data['images']]
                img_diff = [r['pixel_diff'] for r in data['images']]

                print(f"Images:")
                print(f"  Avg FPS: {np.mean(img_fps):.1f} (min: {np.min(img_fps):.1f}, max: {np.max(img_fps):.1f})")
                print(f"  Cold Start FPS: {np.mean([r['cold_start_fps'] for r in data['images']]):.1f}")
                print(f"  Warm FPS: {np.mean([r['warm_fps'] for r in data['images']]):.1f}")
                print(f"  Pixel Diff: {np.mean(img_diff):.2f}px")

            # Video results
            if data.get('videos'):
                vid_fps = [r['avg_fps'] for r in data['videos']]
                vid_diff = [r['pixel_diff'] for r in data['videos']]

                print(f"Videos:")
                for v in data['videos']:
                    print(f"  {v['video']}:")
                    print(f"    Resolution: {v['resolution']}")
                    print(f"    FPS: {v['avg_fps']:.1f} (min: {v['min_fps']:.1f}, median: {v['median_fps']:.1f})")
                    print(f"    Pixel Diff: {v['pixel_diff']:.2f}px")
                    print(f"    Errors: {v['errors']}")

            # Overall assessment
            all_fps = []
            if data.get('images'):
                all_fps.extend([r['avg_fps'] for r in data['images']])
            if data.get('videos'):
                all_fps.extend([r['avg_fps'] for r in data['videos']])

            if all_fps:
                overall_fps = np.mean(all_fps)
                min_fps = np.min(all_fps)

                print(f"\nOverall:")
                print(f"  Average FPS: {overall_fps:.1f}")
                print(f"  Minimum FPS: {min_fps:.1f}")

                if min_fps >= 24:
                    print(f"  ✓ REAL-TIME CAPABLE (min {min_fps:.1f} >= 24)")
                else:
                    print(f"  ✗ NOT REAL-TIME (min {min_fps:.1f} < 24)")

                report['implementations'][impl_name] = {
                    'avg_fps': overall_fps,
                    'min_fps': min_fps,
                    'real_time': min_fps >= 24
                }

        # Save report
        report_file = f"comprehensive_test_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed report saved to: {report_file}")

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        best_impl = None
        best_fps = 0

        for impl_name, metrics in report['implementations'].items():
            if metrics['avg_fps'] > best_fps:
                best_fps = metrics['avg_fps']
                best_impl = impl_name

        if best_impl and best_fps >= 24:
            print(f"✓ PRODUCTION READY: {best_impl} achieves {best_fps:.1f} FPS")
        else:
            print(f"✗ NOT PRODUCTION READY: Best is {best_impl} at {best_fps:.1f} FPS")

        return report


def main():
    """Run comprehensive testing"""
    tester = ComprehensiveTest()
    tester.run_comprehensive_test()
    report = tester.generate_report()

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("Check output files for visual verification")
    print("="*80)


if __name__ == "__main__":
    main()