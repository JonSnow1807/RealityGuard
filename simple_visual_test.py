#!/usr/bin/env python3
"""
Simple visual test to see what's happening with privacy application
"""

import cv2
import numpy as np
from realityguard_optimized import OptimizedRealityGuard, OptimizedConfig, PrivacyStrength

def simple_test():
    """Simple test with visual output."""
    print("=" * 80)
    print("SIMPLE VISUAL TEST")
    print("=" * 80)

    # Create a simple frame with clear objects
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Gray background

    # Add a blue square (simulating person)
    cv2.rectangle(frame, (100, 100), (300, 400), (255, 0, 0), -1)
    cv2.putText(frame, "PERSON", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add a green square (simulating object)
    cv2.rectangle(frame, (400, 200), (600, 400), (0, 255, 0), -1)
    cv2.putText(frame, "OBJECT", (450, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save original
    cv2.imwrite("test_original.jpg", frame)
    print(f"Original saved to test_original.jpg")

    # Process with MAXIMUM privacy
    config = OptimizedConfig(
        privacy_strength=PrivacyStrength.MAXIMUM,
        min_pixel_difference=10.0,  # Lower threshold for testing
        debug_mode=False
    )

    system = OptimizedRealityGuard(config)

    # Process frame
    result, proc_time = system.process_frame(frame)

    # Save result
    cv2.imwrite("test_blurred.jpg", result)
    print(f"Processed saved to test_blurred.jpg")

    # Calculate difference
    diff = np.mean(np.abs(frame.astype(np.float32) - result.astype(np.float32)))
    print(f"\nPixel difference: {diff:.2f}")

    # Create comparison
    comparison = np.hstack([frame, result])
    cv2.putText(comparison, "ORIGINAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(comparison, "PROCESSED", (640 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imwrite("test_comparison.jpg", comparison)
    print(f"Comparison saved to test_comparison.jpg")

    # Show what regions were detected
    regions = system._detect_objects(frame)
    print(f"\nDetected {len(regions)} regions:")
    for i, region in enumerate(regions):
        print(f"  Region {i}: type={region.get('type')}, bbox={region.get('bbox')[:4] if 'bbox' in region else 'N/A'}")

    return diff > 10

if __name__ == "__main__":
    success = simple_test()
    print("\n" + "=" * 80)
    if success:
        print("✅ PRIVACY SUCCESSFULLY APPLIED")
    else:
        print("❌ PRIVACY NOT SUFFICIENTLY APPLIED")
    print("=" * 80)