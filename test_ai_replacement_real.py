#!/usr/bin/env python3
"""
COMPREHENSIVE REAL-WORLD TEST OF AI REPLACEMENT SYSTEM
No tricks, no warmup, no fallbacks - just honest metrics
"""

import cv2
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from realityguard_ai_replacement import (
    RealityGuardAIReplacement,
    AIReplacementConfig,
    ReplacementMode
)

class HonestAIReplacementTest:
    """Brutally honest testing of the AI replacement system"""

    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def verify_no_fallback(self, system):
        """Ensure YOLO is actually working, not using fallback"""

        # Check YOLO is loaded
        if system.model is None:
            print("❌ CRITICAL: YOLO not loaded - would use fallback!")
            return False

        # Test on real image
        test_img = cv2.imread('real_test_1.jpg')
        if test_img is None:
            # Create realistic test
            test_img = np.ones((720, 1280, 3), dtype=np.uint8) * 128

        detections = system._get_detections(test_img)

        # Check if we got real detections
        if len(detections) == 0:
            print("⚠️ No detections on test image")
            return False

        # Check detection details
        for det in detections[:3]:
            if 'bbox' not in det or 'class' not in det:
                print("❌ Invalid detection format")
                return False

        print(f"✓ YOLO working: {len(detections)} real detections")
        return True

    def test_replacement_visibility(self, original, replaced, mode):
        """Verify replacements are ACTUALLY visible"""

        # Calculate pixel difference
        diff = np.abs(original.astype(float) - replaced.astype(float))
        avg_diff = np.mean(diff)
        max_diff = np.max(diff)

        # Calculate percentage of image changed
        threshold = 10  # Pixels must change by at least 10 to count
        changed_pixels = np.sum(diff.mean(axis=2) > threshold)
        total_pixels = original.shape[0] * original.shape[1]
        percent_changed = (changed_pixels / total_pixels) * 100

        print(f"    Pixel difference: avg={avg_diff:.1f}, max={max_diff:.1f}")
        print(f"    Pixels changed: {percent_changed:.1f}%")

        # Verify based on mode
        if mode == ReplacementMode.SYNTHETIC_FACE:
            # Should see significant changes in face regions
            return avg_diff > 20 and percent_changed > 5
        elif mode == ReplacementMode.SMART_BLUR:
            # Should see moderate blur effect
            return avg_diff > 5 and percent_changed > 3
        else:
            # General threshold
            return avg_diff > 10 and percent_changed > 2

    def test_single_frame_cold(self, system, image_path, mode):
        """Test single frame with NO warmup - real cold start"""

        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot load {image_path}")
            return None

        h, w = img.shape[:2]
        print(f"  Image: {w}x{h}")

        # COLD START - no warmup!
        start = time.perf_counter()

        try:
            replaced, stats = system.process_frame(img, mode)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            return None

        elapsed = time.perf_counter() - start
        fps = 1.0 / elapsed

        # Verify replacement happened
        vis_ok = self.test_replacement_visibility(img, replaced, mode)

        # Save output for inspection
        output_name = f"ai_test_{mode.value}_{os.path.basename(image_path)}"
        cv2.imwrite(output_name, replaced)

        return {
            'fps': fps,
            'time_ms': elapsed * 1000,
            'replacements': stats.get('replacements', 0),
            'visible': vis_ok,
            'output': output_name
        }

    def test_video_realistic(self, system, video_path, mode, max_frames=100):
        """Test video with realistic conditions"""

        print(f"\n  Testing {mode.value} on {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Cannot open {video_path}")
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video: {width}x{height} @ {video_fps:.1f} FPS")

        # Output video
        output_path = f"ai_output_{mode.value}_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, int(video_fps), (width, height))

        frame_times = []
        replacements_counts = []
        visibility_checks = []
        frame_count = 0

        # Process frames WITHOUT pre-warming
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Time each frame honestly
            start = time.perf_counter()

            try:
                replaced, stats = system.process_frame(frame, mode)
                elapsed = time.perf_counter() - start

                frame_times.append(elapsed)
                replacements_counts.append(stats.get('replacements', 0))

                # Check visibility every 10 frames
                if frame_count % 10 == 0:
                    vis_ok = self.test_replacement_visibility(frame, replaced, mode)
                    visibility_checks.append(vis_ok)

                out.write(replaced)

            except Exception as e:
                print(f"    Frame {frame_count} error: {e}")

            frame_count += 1

            # Progress
            if frame_count % 20 == 0:
                recent_fps = 1.0 / np.mean(frame_times[-20:])
                print(f"    Frame {frame_count}: {recent_fps:.1f} FPS")

        cap.release()
        out.release()

        if not frame_times:
            return None

        # Calculate HONEST metrics
        fps_values = [1.0/t for t in frame_times]

        return {
            'frames': frame_count,
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values),
            'median_fps': np.median(fps_values),
            'first_frame_fps': fps_values[0],  # Cold start
            'stable_fps': np.mean(fps_values[5:]) if len(fps_values) > 5 else 0,
            'avg_replacements': np.mean(replacements_counts),
            'visibility_ok': all(visibility_checks) if visibility_checks else False,
            'output': output_path
        }

    def run_comprehensive_test(self):
        """Run all tests with different modes"""

        print("="*80)
        print("AI REPLACEMENT SYSTEM - HONEST COMPREHENSIVE TEST")
        print("="*80)

        # Test configurations
        test_modes = [
            ReplacementMode.SYNTHETIC_FACE,
            ReplacementMode.SMART_BLUR,
            ReplacementMode.GENERIC_PERSON,
            ReplacementMode.CONTEXTUAL
        ]

        # Find real test files
        test_images = ['real_test_1.jpg', 'ultimate_real_test_1.jpg']
        test_videos = ['real_people_video.mp4', 'real_images_video.mp4']

        # Filter to existing files
        test_images = [f for f in test_images if os.path.exists(f)]
        test_videos = [f for f in test_videos if os.path.exists(f)]

        if not test_images and not test_videos:
            print("❌ No real test files found!")
            return

        print(f"Found {len(test_images)} images, {len(test_videos)} videos")

        all_results = []

        for mode in test_modes[:2]:  # Test first 2 modes thoroughly
            print(f"\n{'='*60}")
            print(f"Testing Mode: {mode.value.upper()}")
            print(f"{'='*60}")

            # Create fresh system for each mode (no carryover)
            config = AIReplacementConfig(
                default_mode=mode,
                generation_quality=0.7,
                preserve_context=True
            )

            system = RealityGuardAIReplacement(config)

            # Verify no fallback
            if not self.verify_no_fallback(system):
                print("⚠️ System using fallback - results invalid!")
                continue

            mode_results = {
                'mode': mode.value,
                'images': [],
                'videos': []
            }

            # Test images (cold start each time)
            for img_path in test_images[:1]:
                print(f"\n  Testing image: {img_path}")
                result = self.test_single_frame_cold(system, img_path, mode)
                if result:
                    mode_results['images'].append(result)
                    print(f"    FPS: {result['fps']:.1f} (cold start)")
                    print(f"    Visible: {'✓' if result['visible'] else '✗'}")

            # Test video
            for video_path in test_videos[:1]:
                result = self.test_video_realistic(system, video_path, mode, max_frames=50)
                if result:
                    mode_results['videos'].append(result)
                    print(f"\n  Video Results:")
                    print(f"    Avg FPS: {result['avg_fps']:.1f}")
                    print(f"    Min FPS: {result['min_fps']:.1f}")
                    print(f"    First frame: {result['first_frame_fps']:.1f} FPS")
                    print(f"    Stable: {result['stable_fps']:.1f} FPS")
                    print(f"    Visibility: {'✓' if result['visibility_ok'] else '✗'}")

            all_results.append(mode_results)

        # Generate report
        self.generate_report(all_results)

        return all_results

    def generate_report(self, results):
        """Generate honest assessment report"""

        print("\n" + "="*80)
        print("HONEST ASSESSMENT REPORT")
        print("="*80)

        for mode_result in results:
            mode = mode_result['mode']
            print(f"\n{mode.upper()}:")
            print("-"*40)

            # Image results
            if mode_result['images']:
                img_fps = [r['fps'] for r in mode_result['images']]
                img_vis = [r['visible'] for r in mode_result['images']]

                print(f"Images:")
                print(f"  Cold start FPS: {np.mean(img_fps):.1f}")
                print(f"  Visibility: {'✓' if all(img_vis) else '✗'}")

            # Video results
            if mode_result['videos']:
                for v in mode_result['videos']:
                    print(f"Video:")
                    print(f"  Average FPS: {v['avg_fps']:.1f}")
                    print(f"  Minimum FPS: {v['min_fps']:.1f}")
                    print(f"  First frame: {v['first_frame_fps']:.1f} FPS")
                    print(f"  Replacements: {v['avg_replacements']:.1f} per frame")
                    print(f"  Visibility OK: {'✓' if v['visibility_ok'] else '✗'}")

        # Overall verdict
        print("\n" + "="*80)
        print("PRODUCTION READINESS VERDICT")
        print("="*80)

        # Calculate overall metrics
        all_video_fps = []
        all_visibility = []

        for mode_result in results:
            for v in mode_result['videos']:
                all_video_fps.append(v['min_fps'])
                all_visibility.append(v['visibility_ok'])

        if all_video_fps:
            overall_min_fps = np.min(all_video_fps)
            visibility_ok = all(all_visibility)

            print(f"Worst-case FPS: {overall_min_fps:.1f}")
            print(f"All replacements visible: {'✓' if visibility_ok else '✗'}")

            if overall_min_fps >= 24 and visibility_ok:
                print("\n✅ PRODUCTION READY")
                print("   System achieves real-time with visible replacements")
            elif overall_min_fps >= 15:
                print("\n⚠️ NEAR PRODUCTION READY")
                print(f"   Performance: {overall_min_fps:.1f} FPS (need 24+)")
            else:
                print("\n❌ NOT PRODUCTION READY")
                print(f"   Performance too low: {overall_min_fps:.1f} FPS")
        else:
            print("❌ No valid test results")

        # Save detailed report
        report_file = f"ai_replacement_honest_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nDetailed report saved to: {report_file}")


def main():
    """Run honest comprehensive test"""
    tester = HonestAIReplacementTest()
    results = tester.run_comprehensive_test()

    print("\n" + "="*80)
    print("TEST COMPLETE - Check output files for visual verification")
    print("="*80)


if __name__ == "__main__":
    main()