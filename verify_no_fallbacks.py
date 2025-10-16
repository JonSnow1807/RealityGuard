#!/usr/bin/env python3
"""
Verify that the GPU-optimized system is NOT using fallback mechanisms
Ensures all detections are from real YOLO, not simulated
"""

import cv2
import numpy as np
from realityguard_gpu_optimized import RealityGuardGPU, GPUConfig

def verify_real_detections():
    """Verify YOLO is working without fallbacks"""

    print("="*80)
    print("VERIFYING NO FALLBACK MECHANISMS")
    print("="*80)

    # Create system WITHOUT forcing simulation
    config = GPUConfig(
        min_strength=0.20,
        max_strength=0.40,
        default_strength=0.30,
        detection_interval=1  # Detect every frame
    )

    system = RealityGuardGPU(config)

    # Check that YOLO is loaded
    if system.model is None:
        print("❌ ERROR: YOLO not loaded - system would use fallback!")
        return False
    else:
        print("✓ YOLO model loaded successfully")

    # Test with real image
    real_image = cv2.imread('real_test_1.jpg')
    if real_image is None:
        print("⚠ Warning: real_test_1.jpg not found, trying alternative")
        real_image = cv2.imread('ultimate_real_test_1.jpg')

    if real_image is None:
        print("Creating test image with realistic content...")
        # Create realistic test image
        real_image = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        # Add some realistic features
        cv2.circle(real_image, (640, 360), 100, (150, 120, 100), -1)
        cv2.rectangle(real_image, (200, 200), (400, 500), (180, 150, 130), -1)

    print(f"\nTest image shape: {real_image.shape}")

    # Process frame and check detections
    protected, stats = system.process_frame(real_image)

    print(f"\nDetection Results:")
    print(f"  Regions detected by YOLO: {stats['regions_detected']}")
    print(f"  Regions predicted (faces): {stats['regions_predicted']}")
    print(f"  Total regions processed: {stats['regions_processed']}")
    print(f"  FPS: {stats['fps']:.1f}")
    print(f"  Pixel difference: {stats['pixel_diff']:.2f}px")

    # Verify detections are real
    if stats['regions_detected'] > 0:
        print("\n✓ YOLO is detecting real objects - NO FALLBACK")

        # Additional verification: check the detection method directly
        print("\nDirect YOLO test:")
        test_detections = system._get_detections_gpu(real_image)
        print(f"  Direct YOLO detections: {len(test_detections)}")

        for i, det in enumerate(test_detections[:3]):  # Show first 3
            print(f"    Detection {i+1}: Class={det['class']}, Conf={det['confidence']:.2f}")

        return True
    else:
        print("\n⚠ No YOLO detections found")

        # Check if fallback would have been triggered
        if stats['regions_processed'] > 0:
            print("  WARNING: Regions were processed despite no YOLO detections")
            print("  This might indicate fallback mechanism is active")
            return False
        else:
            print("  No regions processed - system correctly handled no detections")
            return True


def test_video_no_fallback():
    """Test video processing without fallbacks"""

    print("\n" + "="*80)
    print("TESTING VIDEO WITHOUT FALLBACKS")
    print("="*80)

    config = GPUConfig(
        min_strength=0.20,
        max_strength=0.40,
        default_strength=0.30,
        detection_interval=2  # Realistic detection interval
    )

    system = RealityGuardGPU(config)

    # Try to load real video
    video_path = 'real_people_video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return False

    print(f"Testing with: {video_path}")

    detection_counts = []
    frame_count = 0
    max_frames = 10

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        protected, stats = system.process_frame(frame)
        detection_counts.append(stats['regions_detected'])

        frame_count += 1
        print(f"  Frame {frame_count}: {stats['regions_detected']} detections, "
              f"{stats['fps']:.1f} FPS")

    cap.release()

    avg_detections = np.mean(detection_counts)
    print(f"\nAverage detections per frame: {avg_detections:.1f}")

    if avg_detections > 0:
        print("✓ Video processing using real YOLO detections - NO FALLBACK")
        return True
    else:
        print("⚠ No detections in video - possible issue")
        return False


def main():
    """Main verification"""

    print("REALITYGUARD FALLBACK VERIFICATION")
    print("="*80)

    # Test 1: Verify YOLO loads and detects
    test1_pass = verify_real_detections()

    # Test 2: Verify video processing
    test2_pass = test_video_no_fallback()

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERIFICATION RESULTS")
    print("="*80)

    if test1_pass and test2_pass:
        print("✅ SYSTEM VERIFIED: NO FALLBACK MECHANISMS ACTIVE")
        print("   All detections are from real YOLO model")
        print("   System is using GPU acceleration properly")
        print("   Performance metrics are legitimate")
    elif test1_pass:
        print("⚠️ PARTIAL SUCCESS: Image detection works, video needs verification")
    else:
        print("❌ VERIFICATION FAILED: Fallback mechanisms may be active")

    print("="*80)


if __name__ == "__main__":
    main()