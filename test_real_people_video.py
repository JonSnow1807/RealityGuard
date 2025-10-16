#!/usr/bin/env python3
"""
Test RealityGuard Ultimate on videos with real people
Downloads a video with actual humans for proper YOLO detection
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from realityguard_ultimate import RealityGuardUltimate, UltimateConfig

def download_people_video():
    """Download videos containing real people"""
    videos = [
        # Public domain videos with people
        ("https://www.pexels.com/download/video/3209828/", "people_meeting.mp4"),
        ("https://sample-videos.com/video123/mp4/480/big_buck_bunny_480p_1mb.mp4", "sample_480p.mp4"),
    ]

    # First check if we already have real people images to convert to video
    if os.path.exists('real_test_1.jpg'):
        print("Creating video from real people images...")
        return create_video_from_images()

    for url, filename in videos:
        if os.path.exists(filename):
            print(f"✓ Using existing {filename}")
            return filename

    # If downloads fail, create from images
    return create_video_from_images()

def create_video_from_images():
    """Create a video from existing real people images"""

    # Check for real test images
    real_images = []
    for i in range(1, 4):
        img_path = f'real_test_{i}.jpg'
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                real_images.append(img)
                print(f"  Loaded {img_path}")

    if not real_images:
        print("No real images found")
        return None

    # Create video from images
    height, width = real_images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('real_people_video.mp4', fourcc, 10.0, (width, height))

    print(f"Creating video from {len(real_images)} real images...")

    # Create 10 seconds of video by repeating images with transitions
    for cycle in range(3):  # 3 cycles through images
        for img_idx, img in enumerate(real_images):
            # Write each image for 10 frames (1 second at 10fps)
            for frame in range(10):
                # Add slight variation to simulate video
                varied_img = img.copy()

                # Add temporal noise
                if frame % 3 == 0:
                    noise = np.random.randn(*varied_img.shape) * 2
                    varied_img = np.clip(varied_img + noise, 0, 255).astype(np.uint8)

                # Add frame counter
                cv2.putText(varied_img, f"Frame {cycle*30 + img_idx*10 + frame + 1}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out.write(varied_img)

    out.release()
    print("✓ Created real_people_video.mp4")
    return 'real_people_video.mp4'

def analyze_video_with_people(video_path, system):
    """Detailed analysis of video processing with people detection"""

    print(f"\nAnalyzing: {video_path}")
    print("-" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Process video and collect detailed stats
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = 'protected_people_' + os.path.basename(video_path)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    people_detected = 0
    total_regions = 0
    fps_samples = []
    diff_samples = []
    detection_stats = []

    print("\nProcessing with people detection...")

    while frame_count < min(total_frames, 90):  # Process up to 90 frames
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        protected, stats = system.process_frame(frame)
        elapsed = time.time() - start

        out.write(protected)

        # Collect stats
        fps_samples.append(1.0/elapsed if elapsed > 0 else 0)
        diff_samples.append(stats['pixel_diff'])

        if stats['regions_detected'] > 0:
            people_detected += 1
            total_regions += stats['regions_detected']

        detection_stats.append({
            'frame': frame_count,
            'detected': stats['regions_detected'],
            'predicted': stats['regions_predicted'],
            'processed': stats['regions_processed']
        })

        frame_count += 1

        # Progress report
        if frame_count % 20 == 0:
            avg_fps = np.mean(fps_samples[-20:])
            avg_diff = np.mean(diff_samples[-20:])
            recent_detections = sum(1 for d in detection_stats[-20:] if d['detected'] > 0)

            print(f"  Frames {frame_count-19}-{frame_count}: "
                  f"FPS={avg_fps:.1f}, Diff={avg_diff:.1f}px, "
                  f"People in {recent_detections}/20 frames")

    cap.release()
    out.release()

    # Analysis results
    print("\n" + "="*60)
    print("DETECTION ANALYSIS")
    print("="*60)

    print(f"\nPeople Detection:")
    print(f"  Frames with people: {people_detected}/{frame_count} "
          f"({100*people_detected/frame_count:.1f}%)")
    print(f"  Total regions detected: {total_regions}")
    print(f"  Avg regions per frame: {total_regions/frame_count:.1f}")

    print(f"\nPerformance:")
    print(f"  Average FPS: {np.mean(fps_samples):.1f}")
    print(f"  Min FPS: {min(fps_samples):.1f}")
    print(f"  Max FPS: {max(fps_samples):.1f}")

    print(f"\nAnti-AI Effect:")
    print(f"  Average pixel diff: {np.mean(diff_samples):.1f}px")
    print(f"  Std deviation: {np.std(diff_samples):.2f}px")

    # Extract key frames showing people detection
    if people_detected > 0:
        print("\nExtracting frames with detected people...")
        cap2 = cv2.VideoCapture(out_path)

        for stat in detection_stats[:3]:  # First 3 frames with detections
            if stat['detected'] > 0:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, stat['frame'])
                ret, frame = cap2.read()
                if ret:
                    fname = f"people_frame_{stat['frame']:03d}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"  Saved {fname} ({stat['detected']} people detected)")

        cap2.release()

    return {
        'frames_processed': frame_count,
        'people_detected_frames': people_detected,
        'total_regions': total_regions,
        'avg_fps': np.mean(fps_samples),
        'avg_diff': np.mean(diff_samples),
        'output_path': out_path
    }

def main():
    """Main test function for real people videos"""

    print("="*80)
    print("REALITYGUARD ULTIMATE - REAL PEOPLE VIDEO TEST")
    print("="*80)

    # Get video with real people
    video_path = download_people_video()
    if not video_path:
        print("Error: Could not get video with people")
        return

    # Initialize system
    print("\nInitializing RealityGuard Ultimate...")
    config = UltimateConfig(
        force_simulation=False,  # Use real YOLO
        min_strength=0.20,
        max_strength=0.60,
        default_strength=0.35
    )
    system = RealityGuardUltimate(config)

    # Analyze the video
    results = analyze_video_with_people(video_path, system)

    if not results:
        print("Error: Analysis failed")
        return

    # Final verdict
    print("\n" + "="*80)
    print("FINAL RESULTS - REAL PEOPLE VIDEO")
    print("="*80)

    detection_rate = results['people_detected_frames'] / results['frames_processed']

    checks = {
        "YOLO People Detection": detection_rate > 0.5,
        "Real-time Processing": results['avg_fps'] >= 24,
        "Anti-AI Effectiveness": 2 <= results['avg_diff'] <= 15,
        "Regions Processed": results['total_regions'] > 0
    }

    print("\nVerification:")
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")

    if detection_rate > 0.5:
        print(f"\n✓ SUCCESS: YOLO detected people in {detection_rate*100:.0f}% of frames!")
        print(f"  - Total regions protected: {results['total_regions']}")
        print(f"  - Average {results['total_regions']/results['frames_processed']:.1f} people per frame")
    else:
        print(f"\n⚠ WARNING: Low detection rate ({detection_rate*100:.0f}%)")
        print("  The video may not contain clear people or YOLO needs adjustment")

    print(f"\nProtected video saved as: {results['output_path']}")
    print("="*80)

if __name__ == "__main__":
    main()