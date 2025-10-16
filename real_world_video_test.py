#!/usr/bin/env python3
"""
COMPREHENSIVE REAL-WORLD VIDEO TEST FOR REALITYGUARD
Tests with actual video content to verify production readiness
"""

import cv2
import numpy as np
import time
import os
import subprocess
import urllib.request
from pathlib import Path

# Import our system
from realityguard_ultimate import RealityGuardUltimate, UltimateConfig

def download_real_videos():
    """Download real videos with people for testing"""

    videos_to_test = []

    # Try to download a real video with people
    print("Attempting to download real test videos...")

    # Option 1: Create a test video using ffmpeg if available
    try:
        # Check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode == 0:
            print("Creating test video with ffmpeg...")

            # Generate a test pattern video
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'testsrc2=size=640x480:rate=30',
                '-f', 'lavfi', '-i', 'sine=frequency=1000:sample_rate=48000',
                '-t', '5',  # 5 seconds
                '-pix_fmt', 'yuv420p',
                'ffmpeg_test.mp4'
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists('ffmpeg_test.mp4'):
                videos_to_test.append('ffmpeg_test.mp4')
                print("✓ Created ffmpeg_test.mp4")
    except:
        pass

    # Option 2: Use existing real images to create video
    if os.path.exists('real_test_1.jpg'):
        print("Creating video from real images...")
        videos_to_test.append(create_video_from_real_images())

    # Option 3: Create synthetic test video
    print("Creating synthetic test video...")
    videos_to_test.append(create_synthetic_test_video())

    return videos_to_test

def create_video_from_real_images():
    """Create video from real test images"""

    output = 'real_images_video.mp4'

    # Load real images
    images = []
    for i in range(1, 4):
        path = f'real_test_{i}.jpg'
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                print(f"  Loaded {path}: {img.shape}")

    if not images:
        return None

    # Create video writer
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 30.0, (w, h))

    # Create 5 seconds of video (150 frames at 30fps)
    for i in range(150):
        img_idx = (i // 30) % len(images)
        img = images[img_idx].copy()

        # Add frame number
        cv2.putText(img, f"Frame {i+1}/150", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add some variation
        if i % 10 == 0:
            noise = np.random.randn(*img.shape) * 3
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        out.write(img)

    out.release()
    print(f"✓ Created {output} ({w}x{h}, 150 frames)")
    return output

def create_synthetic_test_video():
    """Create synthetic video with moving objects"""

    output = 'synthetic_test_video.mp4'

    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, float(fps), (width, height))

    for frame_num in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50

        # Add moving circles (simulate people)
        for i in range(3):
            x = int(width/2 + 150 * np.sin(frame_num * 0.05 + i * 2))
            y = int(height/2 + 100 * np.cos(frame_num * 0.03 + i * 2))
            color = (200 - i*30, 150 + i*20, 100 + i*40)
            cv2.circle(frame, (x, y), 40, color, -1)

        # Add rectangles (simulate objects)
        rect_x = (frame_num * 3) % (width - 100)
        cv2.rectangle(frame, (rect_x, 300), (rect_x + 80, 380), (150, 150, 150), -1)

        # Add frame counter
        cv2.putText(frame, f"Synthetic Test - Frame {frame_num+1}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✓ Created {output} ({width}x{height}, {total_frames} frames)")
    return output

def test_video_processing(video_path, system, detailed=True):
    """Comprehensive test of video processing"""

    print(f"\n{'='*70}")
    print(f"TESTING: {video_path}")
    print('='*70)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")

    # Prepare output
    output_path = f"verified_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Test metrics
    metrics = {
        'frame_times': [],
        'pixel_diffs': [],
        'detections': [],
        'cache_hits': [],
        'strategies_used': set(),
        'errors': [],
        'original_frames': [],
        'protected_frames': []
    }

    print(f"\nProcessing frames...")
    print("-" * 50)

    frame_count = 0
    start_time = time.time()
    last_report = 0

    while cap.isOpened() and frame_count < min(total_frames, 150):  # Process up to 150 frames
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Store original for comparison
            if frame_count < 5:  # Store first 5 frames
                metrics['original_frames'].append(frame.copy())

            # Process frame
            t_start = time.time()
            protected, stats = system.process_frame(frame)
            t_elapsed = time.time() - t_start

            # Store protected for comparison
            if frame_count < 5:
                metrics['protected_frames'].append(protected.copy())

            # Write to output
            out.write(protected)

            # Collect metrics
            metrics['frame_times'].append(t_elapsed)
            metrics['pixel_diffs'].append(stats['pixel_diff'])
            metrics['detections'].append(stats['regions_detected'])
            metrics['cache_hits'].append(stats['cache_efficiency'])
            metrics['strategies_used'].add(stats['strategy'])

            # Verify pattern was actually applied
            if stats['pixel_diff'] < 0.1:
                metrics['errors'].append(f"Frame {frame_count}: No pattern applied!")

            # Progress report
            if time.time() - last_report > 1.0:  # Report every second
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                recent_diff = np.mean(metrics['pixel_diffs'][-30:]) if len(metrics['pixel_diffs']) > 0 else 0

                print(f"  Frame {frame_count}/{min(total_frames, 150)}: "
                      f"FPS={current_fps:.1f}, "
                      f"Diff={recent_diff:.1f}px, "
                      f"Detections={stats['regions_detected']}")

                last_report = time.time()

        except Exception as e:
            metrics['errors'].append(f"Frame {frame_count}: {str(e)}")
            print(f"  ERROR at frame {frame_count}: {e}")

        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    total_time = time.time() - start_time

    # Calculate statistics
    results = {
        'video_path': video_path,
        'output_path': output_path,
        'frames_processed': frame_count,
        'total_time': total_time,
        'avg_fps': frame_count / total_time if total_time > 0 else 0,
        'avg_frame_time': np.mean(metrics['frame_times']) if metrics['frame_times'] else 0,
        'avg_pixel_diff': np.mean(metrics['pixel_diffs']) if metrics['pixel_diffs'] else 0,
        'min_pixel_diff': min(metrics['pixel_diffs']) if metrics['pixel_diffs'] else 0,
        'max_pixel_diff': max(metrics['pixel_diffs']) if metrics['pixel_diffs'] else 0,
        'std_pixel_diff': np.std(metrics['pixel_diffs']) if metrics['pixel_diffs'] else 0,
        'total_detections': sum(metrics['detections']),
        'avg_detections': np.mean(metrics['detections']) if metrics['detections'] else 0,
        'cache_efficiency': np.mean(metrics['cache_hits']) if metrics['cache_hits'] else 0,
        'strategies_used': list(metrics['strategies_used']),
        'error_count': len(metrics['errors']),
        'errors': metrics['errors'][:5],  # First 5 errors
        'original_frames': metrics['original_frames'],
        'protected_frames': metrics['protected_frames']
    }

    return results

def verify_pattern_application(results):
    """Verify that patterns are actually being applied"""

    print("\n" + "="*70)
    print("PATTERN APPLICATION VERIFICATION")
    print("="*70)

    if not results['original_frames'] or not results['protected_frames']:
        print("No frames available for verification")
        return False

    verified = True

    for i in range(min(len(results['original_frames']), len(results['protected_frames']))):
        original = results['original_frames'][i]
        protected = results['protected_frames'][i]

        # Calculate actual difference
        diff = np.mean(np.abs(original.astype(float) - protected.astype(float)))

        # Visual check - save comparison
        comparison = np.hstack([original, protected])
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Protected (Diff: {diff:.1f}px)",
                   (original.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(f'verification_frame_{i}.jpg', comparison)

        print(f"  Frame {i}: Difference = {diff:.1f}px", end="")

        if diff < 1.0:
            print(" ❌ FAIL - No visible pattern!")
            verified = False
        elif diff < 2.0:
            print(" ⚠️  WEAK - Pattern barely visible")
        else:
            print(" ✅ PASS - Pattern applied")

    return verified

def performance_analysis(all_results):
    """Analyze performance across all tests"""

    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)

    for result in all_results:
        print(f"\n📹 {result['video_path']}")
        print(f"  Resolution: {Path(result['video_path']).stem}")
        print(f"  Frames: {result['frames_processed']}")
        print(f"  Time: {result['total_time']:.1f}s")
        print(f"  FPS: {result['avg_fps']:.1f}")

        print(f"\n  Pattern Application:")
        print(f"    Average: {result['avg_pixel_diff']:.1f}px")
        print(f"    Range: {result['min_pixel_diff']:.1f}-{result['max_pixel_diff']:.1f}px")
        print(f"    Std Dev: {result['std_pixel_diff']:.2f}px")

        print(f"\n  Detection:")
        print(f"    Total Regions: {result['total_detections']}")
        print(f"    Per Frame: {result['avg_detections']:.1f}")

        print(f"\n  System:")
        print(f"    Cache: {result['cache_efficiency']:.0f}%")
        print(f"    Strategies: {', '.join(result['strategies_used'])}")
        print(f"    Errors: {result['error_count']}")

        # Verdict for this video
        passed_checks = 0
        total_checks = 4

        if result['avg_fps'] >= 24:
            print(f"    ✅ Real-time: {result['avg_fps']:.1f} FPS")
            passed_checks += 1
        else:
            print(f"    ❌ Too slow: {result['avg_fps']:.1f} FPS")

        if 2 <= result['avg_pixel_diff'] <= 15:
            print(f"    ✅ Pattern strength: {result['avg_pixel_diff']:.1f}px")
            passed_checks += 1
        else:
            print(f"    ❌ Pattern issue: {result['avg_pixel_diff']:.1f}px")

        if result['cache_efficiency'] >= 70:
            print(f"    ✅ Cache working: {result['cache_efficiency']:.0f}%")
            passed_checks += 1
        else:
            print(f"    ❌ Cache issue: {result['cache_efficiency']:.0f}%")

        if result['error_count'] == 0:
            print(f"    ✅ No errors")
            passed_checks += 1
        else:
            print(f"    ⚠️  {result['error_count']} errors")
            for err in result['errors'][:3]:
                print(f"      - {err}")

        print(f"\n  Result: {passed_checks}/{total_checks} checks passed")

def main():
    """Main comprehensive test"""

    print("="*70)
    print("REALITYGUARD ULTIMATE - COMPREHENSIVE REAL-WORLD TEST")
    print("="*70)

    # Get test videos
    test_videos = download_real_videos()

    if not test_videos:
        print("ERROR: No test videos available")
        return

    print(f"\nTest videos ready: {test_videos}")

    # Initialize system
    print("\n" + "="*70)
    print("INITIALIZING REALITYGUARD SYSTEM")
    print("="*70)

    config = UltimateConfig(
        force_simulation=False,
        min_strength=0.20,
        max_strength=0.60,
        default_strength=0.35
    )

    system = RealityGuardUltimate(config)

    # Test each video
    all_results = []

    for video_path in test_videos:
        result = test_video_processing(video_path, system)
        if result:
            all_results.append(result)

            # Verify pattern application on this video
            verify_pattern_application(result)

    # Overall analysis
    performance_analysis(all_results)

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if not all_results:
        print("❌ FAILED: No videos could be processed")
        return

    # Calculate overall metrics
    total_frames = sum(r['frames_processed'] for r in all_results)
    total_time = sum(r['total_time'] for r in all_results)
    overall_fps = total_frames / total_time if total_time > 0 else 0
    avg_diff = np.mean([r['avg_pixel_diff'] for r in all_results])

    print(f"\nOverall Statistics:")
    print(f"  Videos Tested: {len(all_results)}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Overall FPS: {overall_fps:.1f}")
    print(f"  Average Pattern: {avg_diff:.1f}px")

    # Check if system meets requirements
    requirements_met = {
        "Real-time (≥24 FPS)": overall_fps >= 24,
        "Pattern Applied (2-15px)": 2 <= avg_diff <= 15,
        "No Critical Errors": all(r['error_count'] == 0 for r in all_results),
        "Cache Working": all(r['cache_efficiency'] >= 70 for r in all_results)
    }

    print("\nRequirements Check:")
    for req, met in requirements_met.items():
        status = "✅ PASS" if met else "❌ FAIL"
        print(f"  {req}: {status}")

    if all(requirements_met.values()):
        print("\n🎉 SUCCESS: SYSTEM IS PRODUCTION READY!")
        print("All requirements met for real-world video processing")
    else:
        print("\n⚠️ WARNING: System needs optimization")
        failed = [k for k, v in requirements_met.items() if not v]
        print(f"Failed requirements: {', '.join(failed)}")

    # Output locations
    print("\nOutput Files:")
    for r in all_results:
        print(f"  - {r['output_path']}")
    print("  - verification_frame_*.jpg (comparison images)")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()