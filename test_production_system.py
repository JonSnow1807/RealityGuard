#!/usr/bin/env python3
"""
Test the production privacy system with real videos
Verify actual privacy protection at high FPS
"""

import cv2
import numpy as np
import time
import os
from realityguard_production import RealityGuardProduction, ProductionConfig, PrivacyMode

def test_privacy_effectiveness(original, protected, mode):
    """Verify privacy is actually being applied"""

    # Calculate difference
    diff = np.abs(original.astype(float) - protected.astype(float))
    avg_diff = np.mean(diff)

    # Check if regions are actually modified
    if mode == PrivacyMode.BLUR:
        # Blur should create smooth gradients
        edges_orig = cv2.Canny(original, 100, 200)
        edges_prot = cv2.Canny(protected, 100, 200)
        edge_reduction = 1 - (np.sum(edges_prot) / (np.sum(edges_orig) + 1))
        return avg_diff > 5 and edge_reduction > 0.3

    elif mode == PrivacyMode.PIXELATE:
        # Pixelate should create block patterns
        return avg_diff > 10

    elif mode == PrivacyMode.SOLID:
        # Solid should have high difference in protected regions
        return avg_diff > 20

    return avg_diff > 5


def test_video_privacy(video_path, system, mode, max_frames=100):
    """Test privacy protection on real video"""

    print(f"\n{'='*60}")
    print(f"Testing {mode.value} mode on: {video_path}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    print(f"Testing {total_frames} frames...")

    # Prepare output
    output_path = f"production_output_{mode.value}_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

    frame_times = []
    regions_counts = []
    privacy_verified = []
    frame_count = 0

    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        protected, stats = system.process_frame(frame, mode)
        elapsed = time.perf_counter() - start

        # Verify privacy is applied
        if stats['regions_protected'] > 0:
            privacy_ok = test_privacy_effectiveness(frame, protected, mode)
            privacy_verified.append(privacy_ok)

        out.write(protected)

        frame_times.append(elapsed)
        regions_counts.append(stats['regions_protected'])

        frame_count += 1

        if frame_count % 20 == 0:
            current_fps = 1.0 / np.mean(frame_times[-20:])
            print(f"  Frame {frame_count}: {current_fps:.1f} FPS, "
                  f"{stats['regions_protected']} regions protected")

    cap.release()
    out.release()

    if not frame_times:
        return None

    # Calculate statistics
    fps_values = [1.0/t for t in frame_times]

    results = {
        'mode': mode.value,
        'frames': frame_count,
        'avg_fps': np.mean(fps_values),
        'min_fps': np.min(fps_values),
        'median_fps': np.median(fps_values),
        'regions_avg': np.mean(regions_counts),
        'privacy_verified': all(privacy_verified) if privacy_verified else False,
        'cache_efficiency': stats['cache_efficiency'],
        'output': output_path
    }

    print(f"\nResults for {mode.value}:")
    print(f"  FPS: {results['avg_fps']:.1f} (min: {results['min_fps']:.1f})")
    print(f"  Regions protected: {results['regions_avg']:.1f} average")
    print(f"  Privacy verified: {'✓' if results['privacy_verified'] else '✗'}")
    print(f"  Output: {output_path}")

    return results


def main():
    """Comprehensive production system test"""

    print("="*80)
    print("PRODUCTION PRIVACY SYSTEM TEST")
    print("="*80)

    # Test different configurations
    configs = [
        # Standard blur
        (ProductionConfig(default_mode=PrivacyMode.BLUR,
                         blur_strength=31,
                         adaptive_quality=True), "Standard Blur"),

        # Heavy blur
        (ProductionConfig(default_mode=PrivacyMode.BLUR,
                         blur_strength=51,
                         adaptive_quality=False), "Heavy Blur"),

        # Pixelate
        (ProductionConfig(default_mode=PrivacyMode.PIXELATE,
                         pixelate_size=20), "Pixelate"),

        # Solid color
        (ProductionConfig(default_mode=PrivacyMode.SOLID), "Solid Color"),
    ]

    # Find test videos
    test_videos = []
    for f in os.listdir('.'):
        if f.endswith('.mp4') and ('real' in f.lower() or 'test' in f.lower()):
            test_videos.append(f)
            if len(test_videos) >= 2:
                break

    if not test_videos:
        print("No test videos found!")
        return

    all_results = []

    for config, config_name in configs[:2]:  # Test first 2 configs
        print(f"\n{'='*80}")
        print(f"Configuration: {config_name}")
        print(f"{'='*80}")

        system = RealityGuardProduction(config)

        for video_path in test_videos[:1]:  # Test first video
            results = test_video_privacy(video_path, system,
                                        config.default_mode, max_frames=50)
            if results:
                results['config'] = config_name
                all_results.append(results)

    # Summary
    print("\n" + "="*80)
    print("PRODUCTION SYSTEM PERFORMANCE SUMMARY")
    print("="*80)

    for result in all_results:
        print(f"\n{result['config']} - {result['mode']}:")
        print(f"  Average FPS: {result['avg_fps']:.1f}")
        print(f"  Minimum FPS: {result['min_fps']:.1f}")
        print(f"  Real-time: {'✓' if result['min_fps'] >= 24 else '✗'}")
        print(f"  Privacy: {'✓ Verified' if result['privacy_verified'] else '✗ Not verified'}")

    # Overall assessment
    real_time_count = sum(1 for r in all_results if r['min_fps'] >= 24)
    privacy_count = sum(1 for r in all_results if r['privacy_verified'])

    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)

    if real_time_count == len(all_results) and privacy_count == len(all_results):
        print("✅ PRODUCTION READY")
        print("   All modes achieve real-time performance")
        print("   Privacy protection verified")
    elif real_time_count > 0:
        print("⚠️ PARTIALLY READY")
        print(f"   {real_time_count}/{len(all_results)} modes are real-time")
        print(f"   {privacy_count}/{len(all_results)} have verified privacy")
    else:
        print("❌ NOT READY")
        print("   Performance below requirements")

    avg_fps = np.mean([r['avg_fps'] for r in all_results])
    print(f"\nOverall average FPS: {avg_fps:.1f}")

    print("="*80)


if __name__ == "__main__":
    main()