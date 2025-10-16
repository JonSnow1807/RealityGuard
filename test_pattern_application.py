#!/usr/bin/env python3
"""
Simple test to verify pattern application is working
This will help us understand why we're getting 0.0px difference
"""

import cv2
import numpy as np

def test_pattern_application():
    """Test if patterns are actually being applied"""

    print("="*60)
    print("PATTERN APPLICATION TEST")
    print("="*60)

    # Create test image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    original = img.copy()

    print(f"Original image mean: {np.mean(original):.2f}")

    # Test 1: Simple addition
    print("\n1. Testing simple pattern addition:")
    pattern1 = np.ones_like(img, dtype=np.float32) * 10
    test1 = img.copy().astype(np.float32)
    test1 += pattern1
    test1 = np.clip(test1, 0, 255).astype(np.uint8)
    diff1 = np.mean(np.abs(original.astype(float) - test1.astype(float)))
    print(f"   Pattern strength: 10")
    print(f"   Result mean: {np.mean(test1):.2f}")
    print(f"   Pixel difference: {diff1:.2f}")

    # Test 2: Stronger pattern
    print("\n2. Testing stronger pattern:")
    pattern2 = np.random.randn(*img.shape) * 30
    test2 = img.copy().astype(np.float32)
    test2 += pattern2
    test2 = np.clip(test2, 0, 255).astype(np.uint8)
    diff2 = np.mean(np.abs(original.astype(float) - test2.astype(float)))
    print(f"   Pattern strength: 30 (random)")
    print(f"   Result mean: {np.mean(test2):.2f}")
    print(f"   Pixel difference: {diff2:.2f}")

    # Test 3: Alpha blending
    print("\n3. Testing alpha blending:")
    pattern3 = np.ones_like(img) * 200
    alpha = 0.3
    test3 = cv2.addWeighted(img, 1-alpha, pattern3, alpha, 0)
    diff3 = np.mean(np.abs(original.astype(float) - test3.astype(float)))
    print(f"   Pattern value: 200")
    print(f"   Alpha: {alpha}")
    print(f"   Result mean: {np.mean(test3):.2f}")
    print(f"   Pixel difference: {diff3:.2f}")

    # Test 4: Check the patent system's method
    print("\n4. Testing patent system's _apply_adversarial logic:")

    # Simulate the patent system's approach
    strength = 0.08
    pattern4 = np.random.randn(*img.shape) * strength  # This might be too small!

    # Method 1: Direct addition (what might be happening)
    test4a = img.copy().astype(np.float32)
    test4a += pattern4 * 255  # Scale up
    test4a = np.clip(test4a, 0, 255).astype(np.uint8)
    diff4a = np.mean(np.abs(original.astype(float) - test4a.astype(float)))

    print(f"   Method A (scaled): strength={strength}, diff={diff4a:.2f}")

    # Method 2: Without proper scaling (possible bug)
    test4b = img.copy().astype(np.float32)
    test4b += pattern4  # No scaling - values between -0.08 and 0.08
    test4b = np.clip(test4b, 0, 255).astype(np.uint8)
    diff4b = np.mean(np.abs(original.astype(float) - test4b.astype(float)))

    print(f"   Method B (unscaled): strength={strength}, diff={diff4b:.2f}")

    # Save visual comparison
    comparison = np.hstack([original, test1, test2, test3, test4a])
    cv2.imwrite('pattern_test_comparison.jpg', comparison)

    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)

    if diff4b < 0.1:
        print("❌ PROBLEM FOUND: Patterns are too weak or not scaled properly!")
        print("   The strength value needs to be multiplied by 255 or increased")

    if diff1 > 5 and diff2 > 5 and diff3 > 5:
        print("✓ Pattern application logic works when values are correct")

    print("\nSaved visual comparison to pattern_test_comparison.jpg")
    print("(Shows: original | +10 | random | blend | scaled)")

    return diff4a

if __name__ == "__main__":
    diff = test_pattern_application()

    print("\n" + "="*60)
    print("CONCLUSION:")
    print(f"With proper scaling, we get {diff:.1f}px difference")
    print(f"Target range is 2-15px, so we need strength * 255 * factor")
    print("="*60)