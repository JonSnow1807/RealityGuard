#!/usr/bin/env python3
"""
Truth verification script - what actually works vs claims
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.realityguard_improved import RealityGuardImproved, PrivacyMode
from src.face_detector import ModernFaceDetector


def test_truth():
    """Test what actually works"""

    print("="*60)
    print("TRUTH VERIFICATION")
    print("="*60)

    # 1. Does face detection actually work?
    print("\n1. FACE DETECTION TEST")
    print("-"*40)

    detector = ModernFaceDetector(method="mediapipe")

    # Create a simple white image
    white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    faces = detector.detect_faces(white_img)
    print(f"Faces in white image: {len(faces)}")

    # Create image with a dark circle (face-like)
    face_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.circle(face_img, (320, 240), 100, (100, 100, 100), -1)
    # Add eyes
    cv2.circle(face_img, (290, 220), 20, (50, 50, 50), -1)
    cv2.circle(face_img, (350, 220), 20, (50, 50, 50), -1)
    faces = detector.detect_faces(face_img)
    print(f"Faces in synthetic face image: {len(faces)}")

    detector.release()

    # 2. Does the system actually process frames fast?
    print("\n2. ACTUAL PROCESSING SPEED")
    print("-"*40)

    guard = RealityGuardImproved()

    # Small image
    small_img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = guard.process_frame(small_img)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"Small image (240x320): {avg_time*1000:.2f}ms = {fps:.1f} FPS")

    # Large image
    large_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = guard.process_frame(large_img)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"Large image (1080x1920): {avg_time*1000:.2f}ms = {fps:.1f} FPS")

    guard.cleanup()

    # 3. Does filtering actually change the image?
    print("\n3. FILTERING EFFECTIVENESS")
    print("-"*40)

    guard = RealityGuardImproved()

    # Create test image
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 100

    # OFF mode
    guard.set_privacy_mode(PrivacyMode.OFF)
    off_result = guard.process_frame(test_img)

    # MAXIMUM mode
    guard.set_privacy_mode(PrivacyMode.MAXIMUM)
    max_result = guard.process_frame(test_img)

    # Check if they're different
    diff = np.sum(np.abs(off_result.astype(float) - max_result.astype(float)))

    if diff > 0:
        print(f"✅ Filtering changes image (diff: {diff:.0f})")
    else:
        print(f"❌ Filtering does NOT change image (diff: {diff:.0f})")

    guard.cleanup()

    # 4. What's the REAL performance with all features?
    print("\n4. REAL-WORLD PERFORMANCE")
    print("-"*40)

    guard = RealityGuardImproved()
    guard.set_privacy_mode(PrivacyMode.SMART)

    # Realistic image size (720p)
    real_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Add some bright regions (screens)
    cv2.rectangle(real_img, (100, 100), (400, 300), (240, 240, 240), -1)
    cv2.rectangle(real_img, (600, 400), (1000, 600), (250, 250, 250), -1)

    times = []
    for i in range(50):
        start = time.perf_counter()
        result = guard.process_frame(real_img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(f"Real-world 720p processing: {avg_time*1000:.2f}ms = {fps:.1f} FPS")

    if fps >= 120:
        print(f"✅ Meets 120 FPS requirement ({fps:.1f} FPS)")
    else:
        print(f"❌ Below 120 FPS requirement ({fps:.1f} FPS)")

    guard.cleanup()

    # 5. Check actual screen detection
    print("\n5. SCREEN DETECTION TEST")
    print("-"*40)

    from src.realityguard_improved import ScreenDetector
    from src.config import Config

    config = Config()
    screen_det = ScreenDetector(config)

    # Create image with bright rectangle (screen)
    screen_img = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.rectangle(screen_img, (200, 150), (440, 330), (250, 250, 250), -1)

    screens = screen_det.detect_screens(screen_img, use_cache=False)
    print(f"Screens detected: {len(screens)}")

    if len(screens) > 0:
        print("✅ Screen detection works")
    else:
        print("❌ Screen detection NOT working")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)


if __name__ == "__main__":
    test_truth()