#!/usr/bin/env python3
"""
Realistic performance testing with actual image processing
Tests with real computational load, not just synthetic frames
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.realityguard_improved import RealityGuardImproved, PrivacyMode
from src.face_detector import ModernFaceDetector


def create_realistic_image(width=1280, height=720):
    """Create a realistic test image with actual complexity"""
    # Create base image with noise
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Add realistic elements
    # Add face-like regions (skin-colored rectangles)
    for _ in range(3):
        x = np.random.randint(0, width - 150)
        y = np.random.randint(0, height - 200)
        face_color = (np.random.randint(150, 200),  # B
                     np.random.randint(130, 180),  # G
                     np.random.randint(160, 210))  # R (skin tone)
        cv2.rectangle(image, (x, y), (x + 150, y + 200), face_color, -1)
        # Add eyes
        cv2.circle(image, (x + 50, y + 70), 15, (50, 50, 50), -1)
        cv2.circle(image, (x + 100, y + 70), 15, (50, 50, 50), -1)
        # Add mouth
        cv2.ellipse(image, (x + 75, y + 140), (40, 20), 0, 0, 180, (100, 50, 50), 2)

    # Add screen-like regions (bright rectangles)
    for _ in range(2):
        x = np.random.randint(0, width - 300)
        y = np.random.randint(0, height - 200)
        cv2.rectangle(image, (x, y), (x + 300, y + 200), (240, 240, 240), -1)
        # Add some content
        cv2.putText(image, "SCREEN", (x + 50, y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add text regions
    for _ in range(5):
        x = np.random.randint(0, width - 200)
        y = np.random.randint(20, height)
        cv2.putText(image, "Sample Text 123", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def test_face_detection_performance():
    """Test actual face detection performance"""
    print("\n" + "="*60)
    print("FACE DETECTION PERFORMANCE TEST")
    print("="*60)

    # Test each detector
    methods = ['opencv', 'yolo', 'mediapipe']
    results = {}

    for method in methods:
        try:
            print(f"\nTesting {method.upper()}...")
            detector = ModernFaceDetector(method=method)

            # Create test image
            image = create_realistic_image()

            # Warm up
            for _ in range(5):
                _ = detector.detect_faces(image, use_cache=False)

            # Actual test
            times = []
            face_counts = []

            for _ in range(50):
                start = time.perf_counter()
                faces = detector.detect_faces(image, use_cache=False)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                face_counts.append(len(faces))

            avg_time = np.mean(times) * 1000  # Convert to ms
            fps = 1000 / avg_time if avg_time > 0 else 0

            results[method] = {
                'avg_time_ms': avg_time,
                'fps': fps,
                'faces_detected': np.mean(face_counts)
            }

            print(f"  Average time: {avg_time:.2f}ms")
            print(f"  FPS: {fps:.1f}")
            print(f"  Faces detected: {np.mean(face_counts):.1f}")

            detector.release()

        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = {'error': str(e)}

    return results


def test_full_system_performance():
    """Test full RealityGuard system with realistic load"""
    print("\n" + "="*60)
    print("FULL SYSTEM REALISTIC PERFORMANCE TEST")
    print("="*60)

    guard = RealityGuardImproved()

    # Test different resolutions
    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD")
    ]

    all_results = []

    for width, height, name in resolutions:
        print(f"\nTesting {name} ({width}x{height})...")

        # Create realistic test images
        test_images = [create_realistic_image(width, height) for _ in range(10)]

        # Test each privacy mode
        for mode in [PrivacyMode.OFF, PrivacyMode.SMART, PrivacyMode.MAXIMUM]:
            guard.set_privacy_mode(mode)

            # Warm up
            for img in test_images[:3]:
                _ = guard.process_frame(img)

            # Actual measurement
            times = []

            for img in test_images:
                start = time.perf_counter()
                processed = guard.process_frame(img)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

                # Verify output is valid
                assert processed is not None
                assert processed.shape == img.shape

            avg_time = np.mean(times) * 1000  # ms
            fps = 1000 / avg_time if avg_time > 0 else 0

            print(f"  {mode.name}: {avg_time:.2f}ms ({fps:.1f} FPS)")

            all_results.append({
                'resolution': name,
                'mode': mode.name,
                'avg_ms': avg_time,
                'fps': fps
            })

    guard.cleanup()

    # Calculate overall average
    avg_fps = np.mean([r['fps'] for r in all_results])
    print(f"\n  OVERALL AVERAGE: {avg_fps:.1f} FPS")

    return all_results, avg_fps


def test_with_real_video():
    """Test with actual video processing"""
    print("\n" + "="*60)
    print("VIDEO PROCESSING TEST")
    print("="*60)

    # Create a synthetic video sequence
    print("\nCreating synthetic video sequence...")
    frames = []
    for i in range(30):  # 1 second at 30fps
        frame = create_realistic_image()
        # Add motion
        shift = int(10 * np.sin(i * 0.2))
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        frames.append(frame)

    guard = RealityGuardImproved()
    guard.set_privacy_mode(PrivacyMode.SMART)

    print("Processing video frames...")
    start_time = time.perf_counter()

    for frame in frames:
        processed = guard.process_frame(frame)

    total_time = time.perf_counter() - start_time
    fps = len(frames) / total_time

    print(f"  Processed {len(frames)} frames in {total_time:.2f}s")
    print(f"  Average FPS: {fps:.1f}")

    guard.cleanup()

    return fps


def verify_actual_filtering():
    """Verify that filtering actually works"""
    print("\n" + "="*60)
    print("FILTERING VERIFICATION TEST")
    print("="*60)

    # Create image with known bright region (screen)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.rectangle(image, (200, 150), (440, 330), (250, 250, 250), -1)

    guard = RealityGuardImproved()

    # Test OFF mode - should not change
    guard.set_privacy_mode(PrivacyMode.OFF)
    off_result = guard.process_frame(image)
    off_diff = np.mean(np.abs(off_result.astype(float) - image.astype(float)))
    print(f"  OFF mode difference: {off_diff:.2f}")

    # Test MAXIMUM mode - should change
    guard.set_privacy_mode(PrivacyMode.MAXIMUM)
    max_result = guard.process_frame(image)
    max_diff = np.mean(np.abs(max_result.astype(float) - image.astype(float)))
    print(f"  MAXIMUM mode difference: {max_diff:.2f}")

    guard.cleanup()

    filtering_works = max_diff > off_diff
    print(f"\n  Filtering {'WORKS' if filtering_works else 'DOES NOT WORK'}")

    return filtering_works


def main():
    """Run all realistic tests"""
    print("="*60)
    print("REALISTIC PERFORMANCE VALIDATION")
    print("Running actual tests, not synthetic benchmarks")
    print("="*60)

    # 1. Test face detectors
    face_results = test_face_detection_performance()

    # 2. Test full system
    system_results, avg_fps = test_full_system_performance()

    # 3. Test video processing
    video_fps = test_with_real_video()

    # 4. Verify filtering works
    filtering_works = verify_actual_filtering()

    # Summary
    print("\n" + "="*60)
    print("REALISTIC TEST SUMMARY")
    print("="*60)

    print("\n1. Face Detection Performance:")
    for method, stats in face_results.items():
        if 'fps' in stats:
            print(f"   {method.upper()}: {stats['fps']:.1f} FPS")

    print(f"\n2. Full System Performance:")
    print(f"   Average across all modes/resolutions: {avg_fps:.1f} FPS")

    print(f"\n3. Video Processing:")
    print(f"   Real-time video: {video_fps:.1f} FPS")

    print(f"\n4. Filtering Verification:")
    print(f"   {'✅ PASSED' if filtering_works else '❌ FAILED'}")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT:")

    if avg_fps >= 120:
        print(f"✅ System achieves {avg_fps:.1f} FPS - MEETS 120 FPS requirement")
    else:
        print(f"❌ System achieves {avg_fps:.1f} FPS - BELOW 120 FPS requirement")

    print("\nNOTE: These are REALISTIC measurements with actual image")
    print("processing, not synthetic benchmarks.")
    print("="*60)


if __name__ == "__main__":
    main()