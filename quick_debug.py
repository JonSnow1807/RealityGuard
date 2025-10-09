#!/usr/bin/env python3
"""Quick debug to check what's happening"""

import cv2
import numpy as np
from realityguard_final import FinalRealityGuard, FinalConfig, PrivacyStrength

# Create simple test frame
frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
cv2.rectangle(frame, (100, 100), (300, 400), (255, 0, 0), -1)
cv2.rectangle(frame, (400, 200), (600, 400), (0, 255, 0), -1)

config = FinalConfig(
    privacy_strength=PrivacyStrength.MAXIMUM,
    min_pixel_difference=10.0,
    debug_mode=False
)

system = FinalRealityGuard(config)

# Get regions
regions = system._detect_objects(frame)
print(f"\nDetected {len(regions)} regions:")
for i, r in enumerate(regions):
    print(f"  Region {i}: {r['bbox'][:4]} - type: {r.get('type')}")

# Process frame
result, _ = system.process_frame(frame)

# Check difference
diff = np.mean(np.abs(frame.astype(np.float32) - result.astype(np.float32)))
print(f"\nPixel difference: {diff:.2f}")

# Check a specific region
if len(regions) > 0:
    bbox = regions[0]['bbox']
    x1, y1, x2, y2 = [int(b) for b in bbox]
    roi = frame[y1:y2, x1:x2]
    roi_result = result[y1:y2, x1:x2]
    roi_diff = np.mean(np.abs(roi.astype(np.float32) - roi_result.astype(np.float32)))
    print(f"Region 0 difference: {roi_diff:.2f}")

# Test generator directly
if len(regions) > 0:
    print("\nTesting generator directly:")
    roi = frame[100:300, 100:300]
    mask = system.generator._maximum_privacy(roi)
    mask_diff = np.mean(np.abs(roi.astype(np.float32) - mask.astype(np.float32)))
    print(f"Maximum privacy diff: {mask_diff:.2f}")

    # Check if mask is actually different
    print(f"ROI mean: {np.mean(roi):.2f}")
    print(f"Mask mean: {np.mean(mask):.2f}")
    print(f"Are they identical? {np.array_equal(roi, mask)}")