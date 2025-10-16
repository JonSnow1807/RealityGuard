#!/usr/bin/env python3
"""
Debug detection to see why patterns aren't being applied
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load test image
test_img = cv2.imread('test_person_laptop.jpg')
if test_img is None:
    test_img = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    cv2.circle(test_img, (640, 360), 100, (200, 180, 160), -1)

print("Test image shape:", test_img.shape)

# Test YOLO detection
model = YOLO('yolov8n.pt')
results = model(test_img, verbose=False)

detections = []
for r in results:
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            detections.append({
                'bbox': box.tolist(),
                'class': int(cls),
                'confidence': float(conf)
            })

print(f"\nDetections found: {len(detections)}")
for i, det in enumerate(detections):
    print(f"  {i+1}. Class: {det['class']}, Conf: {det['confidence']:.2f}, BBox: {[int(b) for b in det['bbox']]}")

# If no detections, use simulation
if len(detections) == 0:
    print("\n⚠ No YOLO detections! Using simulated regions...")
    h, w = test_img.shape[:2]
    sim_detections = [
        {'bbox': [w//3, h//4, 2*w//3, 3*h//4], 'class': 0, 'confidence': 0.9},
        {'bbox': [w//10, h//10, w//4, h//3], 'class': 0, 'confidence': 0.7}
    ]
    for i, det in enumerate(sim_detections):
        print(f"  Sim {i+1}. BBox: {[int(b) for b in det['bbox']]}")

# Test pattern application directly
print("\n" + "="*60)
print("TESTING DIRECT PATTERN APPLICATION")
print("="*60)

original = test_img.copy()

# Apply simple pattern to verify it works
if len(detections) > 0 or True:  # Force application
    # Use first detection or simulation
    bbox = detections[0]['bbox'] if detections else [100, 100, 400, 400]
    x1, y1, x2, y2 = [int(b) for b in bbox]

    print(f"Applying pattern to region: [{x1}, {y1}, {x2}, {y2}]")

    # Extract ROI
    roi = test_img[y1:y2, x1:x2]
    print(f"ROI shape: {roi.shape}")

    # Generate visible pattern
    h, w = roi.shape[:2]
    pattern = np.random.randn(h, w, 3) * 30  # Strong pattern

    # Apply pattern
    roi_with_pattern = roi.astype(np.float32) + pattern
    roi_with_pattern = np.clip(roi_with_pattern, 0, 255).astype(np.uint8)

    # Put back
    test_img[y1:y2, x1:x2] = roi_with_pattern

    # Calculate difference
    diff = np.mean(np.abs(original.astype(float) - test_img.astype(float)))
    print(f"Pixel difference after pattern: {diff:.1f}px")

    # Save result
    cv2.imwrite('debug_with_pattern.jpg', test_img)
    print("Saved result to debug_with_pattern.jpg")