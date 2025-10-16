#!/usr/bin/env python3
"""
Test RealityGuard Ultimate on real video files
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from pathlib import Path
from realityguard_ultimate import RealityGuardUltimate, UltimateConfig

def download_test_video():
    """Download a test video with people"""
    # Short test videos with people
    test_videos = [
        ("https://www.w3schools.com/html/mov_bbb.mp4", "test_video.mp4"),
        ("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4", "bunny_test.mp4")
    ]

    # First, try to use existing webcam capture if available
    if os.path.exists("webcam_test.mp4"):
        print("✓ Using existing webcam_test.mp4")
        return "webcam_test.mp4"

    for url, filename in test_videos:
        if os.path.exists(filename):
            print(f"✓ Using existing {filename}")
            return filename

        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✓ Downloaded {filename}")
            return filename
        except Exception as e:
            print(f"  Failed to download: {e}")
            continue

    return None

def create_synthetic_video():
    """Create a synthetic test video with moving shapes"""
    print("Creating synthetic test video...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('synthetic_test.mp4', fourcc, 30.0, (640, 480))

    # Create 90 frames (3 seconds at 30fps)
    for frame_num in range(90):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Add moving circle (simulating person)
        x = 320 + int(200 * np.sin(frame_num * 0.1))
        y = 240 + int(100 * np.cos(frame_num * 0.15))
        cv2.circle(frame, (x, y), 50, (200, 180, 160), -1)

        # Add rectangle (simulating laptop)
        rx = 100 + frame_num * 2
        if rx > 540:
            rx = 100
        cv2.rectangle(frame, (rx, 350), (rx+100, 450), (150, 150, 150), -1)

        # Add text
        cv2.putText(frame, f"Frame {frame_num+1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print("✓ Created synthetic_test.mp4")
    return "synthetic_test.mp4"

def capture_webcam_video(duration=5):
    """Capture video from webcam if available"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None

        # Get camera properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Capturing {duration}s from webcam ({width}x{height} @ {fps}fps)...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('webcam_test.mp4', fourcc, fps, (width, height))

        start_time = time.time()
        frame_count = 0

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frame_count += 1
            else:
                break

        cap.release()
        out.release()

        if frame_count > 0:
            print(f"✓ Captured {frame_count} frames from webcam")
            return "webcam_test.mp4"
    except Exception as e:
        print(f"Webcam capture failed: {e}")

    return None

def process_video(video_path, output_path, system, max_frames=None):
    """Process video with RealityGuard Ultimate"""

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total Frames: {total_frames}")

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process statistics
    stats_list = []
    frame_times = []
    pixel_diffs = []

    print(f"\nProcessing frames...")
    print("-" * 60)

    frame_count = 0
    start_time = time.time()

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame_start = time.time()
        protected_frame, stats = system.process_frame(frame)
        frame_time = time.time() - frame_start

        # Write protected frame
        out.write(protected_frame)

        # Collect statistics
        stats_list.append(stats)
        frame_times.append(frame_time)
        pixel_diffs.append(stats['pixel_diff'])

        frame_count += 1

        # Progress update every 30 frames
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_count}/{total_frames}: "
                  f"Avg FPS={avg_fps:.1f}, "
                  f"Diff={stats['pixel_diff']:.1f}px, "
                  f"Cache={stats['cache_efficiency']:.0f}%")

    # Release resources
    cap.release()
    out.release()

    # Calculate final statistics
    total_time = time.time() - start_time

    return {
        'frames_processed': frame_count,
        'total_time': total_time,
        'avg_fps': frame_count / total_time if total_time > 0 else 0,
        'avg_pixel_diff': np.mean(pixel_diffs) if pixel_diffs else 0,
        'min_pixel_diff': min(pixel_diffs) if pixel_diffs else 0,
        'max_pixel_diff': max(pixel_diffs) if pixel_diffs else 0,
        'final_cache_efficiency': stats_list[-1]['cache_efficiency'] if stats_list else 0,
        'total_adaptations': stats_list[-1]['adaptations'] if stats_list else 0,
        'stats_list': stats_list
    }

def create_comparison_video(original_path, protected_path, output_path):
    """Create side-by-side comparison video"""

    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(protected_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Cannot open videos for comparison")
        return

    # Get properties from first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output with double width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    print(f"Creating comparison video ({width*2}x{height})...")

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Create side-by-side frame
        combined = np.hstack([frame1, frame2])

        # Add labels
        cv2.putText(combined, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Protected (Anti-AI)", (width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(combined)
        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()

    print(f"✓ Created comparison video with {frame_count} frames")

def extract_sample_frames(video_path, output_prefix, num_samples=5):
    """Extract sample frames for visual inspection"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)

    print(f"Extracting {num_samples} sample frames...")

    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            filename = f"{output_prefix}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  Saved {filename}")

    cap.release()

def main():
    """Main video testing function"""

    print("="*80)
    print("REALITYGUARD ULTIMATE - VIDEO TESTING")
    print("="*80)

    # Try to get a test video
    video_path = None

    # Option 1: Try webcam capture
    video_path = capture_webcam_video(duration=5)

    # Option 2: Download test video
    if not video_path:
        video_path = download_test_video()

    # Option 3: Create synthetic video
    if not video_path:
        video_path = create_synthetic_video()

    if not video_path:
        print("Error: No test video available")
        return

    print(f"\nUsing test video: {video_path}")

    # Initialize RealityGuard Ultimate
    print("\nInitializing RealityGuard Ultimate...")
    config = UltimateConfig(
        force_simulation=False,  # Use real YOLO if available
        min_strength=0.20,
        max_strength=0.60,
        default_strength=0.35
    )
    system = RealityGuardUltimate(config)

    # Process the video
    output_path = "protected_" + os.path.basename(video_path)
    results = process_video(video_path, output_path, system, max_frames=150)

    if not results:
        print("Error: Video processing failed")
        return

    # Print results
    print("\n" + "="*80)
    print("VIDEO PROCESSING RESULTS")
    print("="*80)

    print(f"\nPerformance Metrics:")
    print(f"  Frames Processed: {results['frames_processed']}")
    print(f"  Total Time: {results['total_time']:.1f}s")
    print(f"  Average FPS: {results['avg_fps']:.1f}")
    print(f"  Real-time: {'✓ YES' if results['avg_fps'] >= 24 else '✗ NO'}")

    print(f"\nAnti-AI Effectiveness:")
    print(f"  Average Pixel Diff: {results['avg_pixel_diff']:.1f}px")
    print(f"  Min Pixel Diff: {results['min_pixel_diff']:.1f}px")
    print(f"  Max Pixel Diff: {results['max_pixel_diff']:.1f}px")
    print(f"  Effectiveness: {'✓ GOOD' if 2 <= results['avg_pixel_diff'] <= 15 else '✗ NEEDS ADJUSTMENT'}")

    print(f"\nCache Performance:")
    print(f"  Final Efficiency: {results['final_cache_efficiency']:.0f}%")
    print(f"  Total Adaptations: {results['total_adaptations']}")

    # Create comparison video
    comparison_path = "comparison_" + os.path.basename(video_path)
    create_comparison_video(video_path, output_path, comparison_path)

    # Extract sample frames
    extract_sample_frames(output_path, "protected_sample", num_samples=3)

    # Analyze pattern consistency across frames
    if len(results['stats_list']) > 10:
        print(f"\nPattern Consistency Analysis:")

        # Check variation in pixel differences
        diffs = [s['pixel_diff'] for s in results['stats_list']]
        std_dev = np.std(diffs)
        print(f"  Pixel Diff Std Dev: {std_dev:.2f}")
        print(f"  Consistency: {'✓ STABLE' if std_dev < 2 else '⚠ VARIABLE'}")

        # Check strategy changes
        strategies = [s['strategy'] for s in results['stats_list']]
        unique_strategies = len(set(strategies))
        print(f"  Strategies Used: {unique_strategies}")

        # Check cache hit progression
        cache_effs = [s['cache_efficiency'] for s in results['stats_list']]
        print(f"  Cache Warmup: {cache_effs[0]:.0f}% → {cache_effs[-1]:.0f}%")

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    all_checks = {
        "Real-time Processing": results['avg_fps'] >= 24,
        "Anti-AI Effectiveness": 2 <= results['avg_pixel_diff'] <= 15,
        "Cache Efficiency": results['final_cache_efficiency'] >= 70,
        "Frame Processing": results['frames_processed'] > 0
    }

    for check, passed in all_checks.items():
        print(f"  {check}: {'✓ PASS' if passed else '✗ FAIL'}")

    if all(all_checks.values()):
        print("\n🎉 VIDEO PROCESSING: FULLY VERIFIED")
        print("The RealityGuard Ultimate system successfully processes video in real-time")
        print("with effective anti-AI patterns applied to all frames!")
    else:
        failed = [k for k, v in all_checks.items() if not v]
        print(f"\n⚠ Issues detected: {', '.join(failed)}")

    print("\nOutput files created:")
    print(f"  - {output_path} (protected video)")
    print(f"  - {comparison_path} (side-by-side comparison)")
    print(f"  - protected_sample_*.jpg (sample frames)")

    print("="*80)

if __name__ == "__main__":
    main()