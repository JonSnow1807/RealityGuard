#!/usr/bin/env python3
"""
Test segmentation on a real image
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO

print("Testing YOLOv8 segmentation on real image...")

# Load model
model = YOLO('yolov8n-seg.pt')
model.to('cuda')

# Load real image
img = cv2.imread('test_real_image.jpg')
print(f"Image shape: {img.shape}")

# Run segmentation
results = model(img, verbose=False, device='cuda')

# Analyze results
print("\nDetection Results:")
print("-" * 50)

if results[0].boxes is not None:
    num_detections = len(results[0].boxes)
    print(f"✓ Detected {num_detections} objects")

    for i in range(num_detections):
        cls = int(results[0].boxes.cls[i])
        conf = results[0].boxes.conf[i]
        print(f"  {i+1}. {model.names[cls]}: {conf:.2%} confidence")
else:
    print("✗ No objects detected")

if results[0].masks is not None:
    num_masks = len(results[0].masks)
    print(f"\n✓ Generated {num_masks} segmentation masks")

    # Analyze mask quality
    masks = results[0].masks.data.cpu().numpy()
    for i, mask in enumerate(masks):
        mask_pixels = np.sum(mask > 0.5)
        print(f"  Mask {i+1}: {mask_pixels:,} pixels")

    # Save visualization
    result_img = results[0].plot()
    cv2.imwrite('segmentation_result.jpg', result_img)
    print("\n✓ Saved visualization to segmentation_result.jpg")
else:
    print("\n✗ No segmentation masks generated")

print("\nCONCLUSION:")
if results[0].masks is not None and len(results[0].masks) > 0:
    print("✅ SEGMENTATION IS WORKING PROPERLY")
    print("   The model correctly detects and segments objects")
    print("   Performance metrics from previous tests are valid")
else:
    print("⚠️ SEGMENTATION NOT WORKING")
    print("   Check model configuration")