#!/usr/bin/env python3
"""
Test GPU-optimized RealityGuard with real images and videos
Measure actual FPS without fallback mechanisms
"""

import cv2
import numpy as np
import time
import os
from realityguard_gpu_optimized import RealityGuardGPU, GPUConfig

def test_real_image(system, image_path):
    """Test with real image"""
    print(f"\n{'='*60}")
    print(f"Testing with: {image_path}")
    print(f"{'='*60}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load {image_path}")
        return None

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Warm-up (important for GPU)
    print("Warming up GPU...")
    for _ in range(5):
        _, _ = system.process_frame(img)

    # Actual measurement
    print("\nMeasuring performance:")
    frame_times = []
    pixel_diffs = []
    regions_counts = []

    for i in range(20):
        start = time.time()
        protected, stats = system.process_frame(img)
        elapsed = time.time() - start

        fps = 1.0 / elapsed
        frame_times.append(fps)
        pixel_diffs.append(stats['pixel_diff'])
        regions_counts.append(stats['regions_detected'])

        if i % 5 == 0:
            print(f"  Frame {i+1:2d}: {fps:6.1f} FPS | "
                  f"Diff: {stats['pixel_diff']:5.2f}px | "
                  f"Regions: {stats['regions_detected']:2d} detected, "
                  f"{stats['regions_predicted']:2d} predicted")

    # Save result
    output_path = f"gpu_result_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, protected)

    # Calculate statistics
    avg_fps = np.mean(frame_times)
    std_fps = np.std(frame_times)
    min_fps = np.min(frame_times)
    max_fps = np.max(frame_times)

    avg_diff = np.mean(pixel_diffs)
    avg_regions = np.mean(regions_counts)

    print(f"\nResults:")
    print(f"  FPS: {avg_fps:.1f} ± {std_fps:.1f} (min: {min_fps:.1f}, max: {max_fps:.1f})")
    print(f"  Pixel Difference: {avg_diff:.2f}px")
    print(f"  Average Regions: {avg_regions:.1f}")
    print(f"  Cache Efficiency: {stats['cache_efficiency']:.1f}%")
    print(f"  GPU Memory: {stats.get('gpu_memory', 0):.2f} GB")
    print(f"  Output saved to: {output_path}")

    # Check if real-time
    if avg_fps >= 24:
        print(f"  ✓ REAL-TIME ACHIEVED ({avg_fps:.1f} >= 24 FPS)")
    else:
        print(f"  ✗ Below real-time ({avg_fps:.1f} < 24 FPS)")

    # Check effectiveness
    if 2 <= avg_diff <= 15:
        print(f"  ✓ Effective anti-AI ({avg_diff:.2f}px in 2-15px range)")
    else:
        print(f"  ✗ Ineffective ({avg_diff:.2f}px outside 2-15px range)")

    return {
        'avg_fps': avg_fps,
        'pixel_diff': avg_diff,
        'regions': avg_regions,
        'cache_eff': stats['cache_efficiency']
    }


def test_real_video(system, video_path, max_frames=100):
    """Test with real video"""
    print(f"\n{'='*60}")
    print(f"Testing video: {video_path}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return None

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print(f"Processing first {min(max_frames, total_frames)} frames...")

    # Prepare output
    output_path = f"gpu_result_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_times = []
    pixel_diffs = []
    regions_counts = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        start = time.time()
        protected, stats = system.process_frame(frame)
        elapsed = time.time() - start

        out.write(protected)

        fps_current = 1.0 / elapsed if elapsed > 0 else 0
        frame_times.append(fps_current)
        pixel_diffs.append(stats['pixel_diff'])
        regions_counts.append(stats['regions_detected'])

        frame_count += 1

        # Progress update
        if frame_count % 20 == 0:
            avg_fps_current = np.mean(frame_times[-20:])
            print(f"  Frame {frame_count:3d}: {avg_fps_current:6.1f} FPS | "
                  f"Diff: {stats['pixel_diff']:5.2f}px | "
                  f"Regions: {stats['regions_detected']:2d}")

    cap.release()
    out.release()

    # Calculate statistics
    avg_fps = np.mean(frame_times)
    std_fps = np.std(frame_times)
    min_fps = np.min(frame_times)
    max_fps = np.max(frame_times)

    avg_diff = np.mean(pixel_diffs)
    avg_regions = np.mean(regions_counts)

    print(f"\nVideo Results:")
    print(f"  Frames Processed: {frame_count}")
    print(f"  FPS: {avg_fps:.1f} ± {std_fps:.1f} (min: {min_fps:.1f}, max: {max_fps:.1f})")
    print(f"  Pixel Difference: {avg_diff:.2f}px")
    print(f"  Average Regions: {avg_regions:.1f}")
    print(f"  Cache Efficiency: {stats['cache_efficiency']:.1f}%")
    print(f"  Output saved to: {output_path}")

    # Check if real-time
    if avg_fps >= 24:
        print(f"  ✓ REAL-TIME VIDEO PROCESSING ({avg_fps:.1f} >= 24 FPS)")
    else:
        print(f"  ✗ Below real-time ({avg_fps:.1f} < 24 FPS)")

    return {
        'avg_fps': avg_fps,
        'pixel_diff': avg_diff,
        'regions': avg_regions,
        'frames': frame_count
    }


def main():
    """Main test function"""
    print("="*80)
    print("REALITYGUARD GPU PERFORMANCE TEST")
    print("="*80)

    # Create system with stronger settings for visibility
    config = GPUConfig(
        min_strength=0.15,  # Increased for visibility
        max_strength=0.40,  # Increased
        default_strength=0.25,  # Increased
        detection_interval=1,  # Process every frame for accuracy
        batch_size=16  # GPU batch processing
    )

    system = RealityGuardGPU(config)

    # Test images
    test_images = [
        'real_test_1.jpg',
        'ultimate_real_test_1.jpg',
        'realityguard_ultimate_result.jpg'
    ]

    image_results = []
    for img_path in test_images:
        if os.path.exists(img_path):
            result = test_real_image(system, img_path)
            if result:
                image_results.append(result)
            break

    # Test videos
    test_videos = [
        'real_people_video.mp4',
        'real_images_video.mp4',
        'test_video.mp4'
    ]

    video_results = []
    for video_path in test_videos:
        if os.path.exists(video_path):
            result = test_real_video(system, video_path, max_frames=50)
            if result:
                video_results.append(result)
            # Test only first available video
            break

    # Final summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    if image_results:
        img_fps = np.mean([r['avg_fps'] for r in image_results])
        img_diff = np.mean([r['pixel_diff'] for r in image_results])
        print(f"\nImage Processing:")
        print(f"  Average FPS: {img_fps:.1f}")
        print(f"  Pixel Difference: {img_diff:.2f}px")

    if video_results:
        vid_fps = np.mean([r['avg_fps'] for r in video_results])
        vid_diff = np.mean([r['pixel_diff'] for r in video_results])
        print(f"\nVideo Processing:")
        print(f"  Average FPS: {vid_fps:.1f}")
        print(f"  Pixel Difference: {vid_diff:.2f}px")

    # Overall assessment
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)

    all_fps = []
    if image_results:
        all_fps.extend([r['avg_fps'] for r in image_results])
    if video_results:
        all_fps.extend([r['avg_fps'] for r in video_results])

    if all_fps:
        overall_fps = np.mean(all_fps)
        if overall_fps >= 24:
            print(f"✓ SYSTEM IS PRODUCTION READY")
            print(f"  Overall FPS: {overall_fps:.1f} (>= 24 FPS target)")
        else:
            print(f"⚠ NEEDS OPTIMIZATION")
            print(f"  Overall FPS: {overall_fps:.1f} (< 24 FPS target)")

    print("="*80)


if __name__ == "__main__":
    main()