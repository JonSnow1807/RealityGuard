#!/usr/bin/env python3
"""
Verify that segmentation is actually working and producing quality results
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

def test_segmentation_quality():
    """Test if segmentation produces meaningful results."""

    print("="*70)
    print("SEGMENTATION QUALITY VERIFICATION")
    print("="*70)

    # Load model
    model = YOLO('yolov8n-seg.pt')
    model.to('cuda')

    print("\n1. Testing with synthetic objects")
    print("-"*50)

    # Create test image with clear objects
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Add white rectangle (should be detected as something)
    cv2.rectangle(test_img, (100, 100), (300, 300), (255, 255, 255), -1)

    # Add gray circle
    cv2.circle(test_img, (450, 450), 80, (128, 128, 128), -1)

    # Add colored triangle
    pts = np.array([[320, 50], [270, 150], [370, 150]], np.int32)
    cv2.fillPoly(test_img, [pts], (0, 255, 0))

    # Save test image
    cv2.imwrite('test_image.jpg', test_img)

    # Run segmentation
    results = model(test_img, verbose=False, device='cuda')

    # Check results
    if results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"✓ Detected {num_detections} objects")

        # Print classes detected
        if results[0].boxes.cls is not None:
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                class_name = model.names[int(cls)]
                print(f"  Object {i+1}: {class_name} (confidence: {conf:.2%})")
    else:
        print("✗ No objects detected in synthetic image")

    if results[0].masks is not None:
        num_masks = len(results[0].masks)
        print(f"✓ Generated {num_masks} segmentation masks")

        # Check mask quality
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            mask_pixels = np.sum(mask > 0.5)
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage = mask_pixels / total_pixels * 100
            print(f"  Mask {i+1}: {mask_pixels:,} pixels ({coverage:.1f}% of image)")
    else:
        print("✗ No segmentation masks generated")

    print("\n2. Testing with real-world-like image")
    print("-"*50)

    # Create more realistic test image
    real_img = np.ones((640, 640, 3), dtype=np.uint8) * 50  # Dark background

    # Add "person-like" shape
    cv2.ellipse(real_img, (320, 200), (40, 80), 0, 0, 360, (200, 150, 100), -1)  # Head
    cv2.rectangle(real_img, (280, 250), (360, 450), (100, 100, 200), -1)  # Body

    # Add "car-like" shape
    cv2.rectangle(real_img, (100, 400), (250, 480), (150, 150, 150), -1)  # Car body
    cv2.circle(real_img, (130, 490), 15, (50, 50, 50), -1)  # Wheel
    cv2.circle(real_img, (220, 490), 15, (50, 50, 50), -1)  # Wheel

    # Save and process
    cv2.imwrite('real_world_test.jpg', real_img)
    results = model(real_img, verbose=False, device='cuda')

    if results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"Detected {num_detections} objects in real-world-like image")

        for i in range(min(3, num_detections)):  # Show first 3 detections
            cls = int(results[0].boxes.cls[i])
            conf = results[0].boxes.conf[i]
            print(f"  • {model.names[cls]}: {conf:.2%} confidence")
    else:
        print("No objects detected in real-world-like image")

    print("\n3. Testing with noise (should detect nothing)")
    print("-"*50)

    # Pure noise image
    noise_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(noise_img, verbose=False, device='cuda')

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"⚠️  Detected {len(results[0].boxes)} objects in pure noise (false positives)")
    else:
        print("✓ Correctly detected no objects in noise")

    print("\n4. Performance with segmentation enabled")
    print("-"*50)

    # Test performance impact of segmentation
    import time

    # Batch of real images
    batch = [np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8) for _ in range(8)]

    # Add some shapes to each image
    for img in batch:
        # Random rectangle
        x1, y1 = np.random.randint(100, 400, 2)
        x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(50, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Warmup
    for _ in range(5):
        _ = model(batch, verbose=False, device='cuda')

    # Time with segmentation
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(10):
        results = model(batch, verbose=False, device='cuda')

    torch.cuda.synchronize()
    seg_time = time.perf_counter() - start

    seg_fps = (10 * 8) / seg_time
    print(f"Segmentation FPS: {seg_fps:.1f}")

    # Check if we're getting masks
    total_masks = 0
    for r in results:
        if r.masks is not None:
            total_masks += len(r.masks)

    print(f"Total masks generated: {total_masks}")

    print("\n" + "="*70)
    print("QUALITY VERIFICATION SUMMARY")
    print("="*70)

    if total_masks > 0:
        print("✅ SEGMENTATION WORKING PROPERLY")
        print("   • Model detects objects")
        print("   • Generates segmentation masks")
        print(f"   • Performance: {seg_fps:.1f} FPS")
    else:
        print("⚠️  SEGMENTATION NEEDS VERIFICATION")
        print("   • Check if model is properly loaded")
        print("   • Verify CUDA is being used")
        print("   • Test with real images")

if __name__ == "__main__":
    test_segmentation_quality()