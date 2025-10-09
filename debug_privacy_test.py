#!/usr/bin/env python3
"""
Debug test to understand why privacy isn't being applied
"""

import cv2
import numpy as np
from realityguard_optimized import OptimizedRealityGuard, OptimizedConfig, PrivacyStrength

def debug_privacy_application():
    """Debug privacy mask application step by step."""
    print("=" * 80)
    print("DEBUGGING PRIVACY APPLICATION")
    print("=" * 80)

    # Create system with maximum privacy for best visibility
    config = OptimizedConfig(
        privacy_strength=PrivacyStrength.MAXIMUM,
        debug_mode=True
    )

    system = OptimizedRealityGuard(config)

    # Create a simple test frame with clear objects
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100  # Gray background

    # Add a clear blue rectangle (person-like)
    cv2.rectangle(frame, (300, 200), (500, 600), (255, 0, 0), -1)

    # Add a green circle (object)
    cv2.circle(frame, (800, 400), 100, (0, 255, 0), -1)

    # Add text to make it clear
    cv2.putText(frame, "PERSON", (350, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "OBJECT", (750, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("\n1. Original frame statistics:")
    print(f"   Shape: {frame.shape}")
    print(f"   Mean pixel value: {np.mean(frame):.2f}")
    print(f"   Blue rectangle mean: {np.mean(frame[200:600, 300:500]):.2f}")
    print(f"   Green circle area mean: {np.mean(frame[300:500, 700:900]):.2f}")

    # Process frame
    print("\n2. Processing frame...")

    result, privacy_score = system.process_frame(frame)

    print("\n3. Processed frame statistics:")
    print(f"   Shape: {result.shape}")
    print(f"   Mean pixel value: {np.mean(result):.2f}")
    print(f"   Blue rectangle area mean: {np.mean(result[200:600, 300:500]):.2f}")
    print(f"   Green circle area mean: {np.mean(result[300:500, 700:900]):.2f}")

    # Calculate difference
    diff = np.mean(np.abs(frame.astype(np.float32) - result.astype(np.float32)))
    print(f"\n4. Pixel difference: {diff:.2f}")

    # Check if regions were detected
    print("\n5. Detection analysis:")

    # Try detection directly
    regions = system.segmentation.detect(frame, confidence_threshold=0.1)
    print(f"   Regions detected by segmentation: {len(regions)}")

    if regions:
        for i, region in enumerate(regions):
            print(f"   Region {i}: bbox={region.get('bbox', 'N/A')}, class={region.get('class', 'N/A')}")

    # Check simulation
    simulated_regions = system.segmentation._simulate_detection(frame)
    print(f"   Simulated regions: {len(simulated_regions)}")
    if simulated_regions:
        for i, region in enumerate(simulated_regions):
            print(f"   Sim Region {i}: bbox={region.get('bbox', 'N/A')}")

    # Visual comparison
    print("\n6. Visual comparison test:")

    # Save frames for visual inspection
    cv2.imwrite("debug_original.jpg", frame)
    cv2.imwrite("debug_processed.jpg", result)

    # Create side-by-side comparison
    comparison = np.hstack([frame, result])
    cv2.putText(comparison, "ORIGINAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(comparison, "PROCESSED", (1280 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite("debug_comparison.jpg", comparison)

    print("   Saved: debug_original.jpg, debug_processed.jpg, debug_comparison.jpg")

    # Test privacy generator directly
    print("\n7. Testing privacy generator directly:")
    test_roi = frame[200:600, 300:500].copy()  # Extract blue rectangle

    # Test each strategy
    for strategy in ['geometric', 'neural', 'cached', 'diffusion']:
        mask = system.generator.generate(
            test_roi,
            {'bbox': [0, 0, 200, 400], 'class': 0, 'confidence': 0.9},
            strategy
        )
        mask_diff = np.mean(np.abs(test_roi.astype(np.float32) - mask.astype(np.float32)))
        print(f"   {strategy}: diff = {mask_diff:.2f}")

    # Final verdict
    print("\n" + "=" * 80)
    if diff > 10:
        print("✅ PRIVACY IS BEING APPLIED")
    else:
        print("❌ PRIVACY IS NOT BEING APPLIED")
        print("\nPossible issues:")
        print("- Regions not being detected")
        print("- Masks not being applied to result")
        print("- Generator not modifying content enough")
    print("=" * 80)

    return diff > 10

if __name__ == "__main__":
    success = debug_privacy_application()
    exit(0 if success else 1)