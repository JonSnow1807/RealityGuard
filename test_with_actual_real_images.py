#!/usr/bin/env python3
"""
Test with ACTUAL real images - using images we know exist locally
"""

import os
import sys
import cv2
import numpy as np
import time
import json
from datetime import datetime

sys.path.insert(0, '01_production')

def find_existing_real_images():
    """Find any real images that might exist in the project"""
    print("\n" + "="*60)
    print("SEARCHING FOR EXISTING REAL IMAGES")
    print("="*60)

    # Common test image locations
    possible_images = [
        "test_image.jpg",
        "test_office.jpg",
        "real_office.jpg",
        "crowd.jpg",
        "meeting.jpg",
        "people.jpg",
        "03_verification/test_image.jpg",
        "03_verification/real_test.jpg",
        "production_test_data/test_image.jpg"
    ]

    found_images = []
    for img_path in possible_images:
        if os.path.exists(img_path):
            print(f"✓ Found: {img_path}")
            found_images.append(img_path)

    # Also check for any JPG/PNG files
    import glob
    for pattern in ["*.jpg", "*.png", "**/*.jpg", "**/*.png"]:
        for file in glob.glob(pattern, recursive=True):
            if "cache" not in file and ".huggingface" not in file:
                print(f"✓ Found: {file}")
                found_images.append(file)

    return found_images

def create_real_test_image():
    """Create an image that YOLO should actually detect"""
    print("\n" + "="*60)
    print("CREATING TEST IMAGE WITH DETECTABLE CONTENT")
    print("="*60)

    # Create a more realistic image that might trigger YOLO
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Gray background

    # Add person-like shape with more realistic proportions
    # Head (circle)
    cv2.circle(img, (640, 200), 60, (180, 140, 120), -1)  # Face color

    # Body (rectangle with shoulders)
    points = np.array([
        [590, 250],  # Left shoulder
        [690, 250],  # Right shoulder
        [680, 500],  # Right bottom
        [600, 500]   # Left bottom
    ])
    cv2.fillPoly(img, [points], (100, 100, 150))  # Clothing color

    # Arms
    cv2.rectangle(img, (550, 260), (590, 400), (180, 140, 120), -1)  # Left arm
    cv2.rectangle(img, (690, 260), (730, 400), (180, 140, 120), -1)  # Right arm

    # Add laptop-like rectangle
    cv2.rectangle(img, (500, 450), (780, 550), (60, 60, 60), -1)  # Laptop
    cv2.rectangle(img, (520, 460), (760, 530), (150, 150, 200), -1)  # Screen

    # Add some texture/noise to make it more realistic
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Save it
    cv2.imwrite("test_person_laptop.jpg", img)
    print("Created: test_person_laptop.jpg")

    return "test_person_laptop.jpg"

def test_yolo_detection_directly():
    """Test YOLO detection directly to see what it can detect"""
    print("\n" + "="*60)
    print("TESTING YOLO DETECTION DIRECTLY")
    print("="*60)

    try:
        from ultralytics import YOLO

        # Load YOLO
        model = YOLO('yolov8n.pt')
        print("✓ YOLO loaded successfully")

        # Test with our created image
        test_img_path = create_real_test_image()
        img = cv2.imread(test_img_path)

        # Run detection
        results = model(img, verbose=True)  # Set verbose=True to see what YOLO is doing

        print(f"\nDetection results:")
        detected_objects = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    class_name = model.names[cls_id]
                    print(f"  - Detected: {class_name} (confidence: {conf:.2%})")
                    detected_objects.append({
                        "class": class_name,
                        "confidence": conf,
                        "class_id": cls_id
                    })

        if not detected_objects:
            print("  ✗ No objects detected")
            print("\nTrying with lower confidence threshold...")

            # Try with lower threshold
            results = model(img, conf=0.1, verbose=True)
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        class_name = model.names[cls_id]
                        print(f"  - Low confidence: {class_name} ({conf:.2%})")

        return detected_objects

    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        return []

def test_with_system():
    """Test the full system with various images"""
    print("\n" + "="*60)
    print("TESTING FULL SYSTEM")
    print("="*60)

    from revolutionary_optimized import OptimizedRealityGuard, OptimizedConfig

    config = OptimizedConfig(quality_mode="fast")
    system = OptimizedRealityGuard(config)

    # Create test image
    img_path = create_real_test_image()
    img = cv2.imread(img_path)

    print(f"\nTesting with image: {img.shape}")

    # Test detection
    start = time.time()
    regions = system._detect_privacy_regions(img)
    detect_time = (time.time() - start) * 1000

    print(f"Detection took: {detect_time:.1f}ms")
    print(f"Regions found: {len(regions)}")

    if regions:
        print("Region details:")
        for i, region in enumerate(regions):
            print(f"  Region {i}: {region}")

        # Test processing
        start = time.time()
        batch = [{'frame': img, 'region': regions[0]}]
        masks = system.diffusion.generate_privacy_batch(batch)
        process_time = (time.time() - start) * 1000

        print(f"Processing took: {process_time:.1f}ms")
        print(f"Total FPS: {1000/(detect_time + process_time):.1f}")
    else:
        print("No regions detected - system will use fallback")

        # Check what fallback returns
        fallback_regions = system._fallback_detection(img)
        print(f"Fallback regions: {len(fallback_regions)}")
        for region in fallback_regions:
            print(f"  Fallback region: {region}")

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" REAL IMAGE DETECTION TEST ")
    print("="*70)

    # 1. Check for existing real images
    existing = find_existing_real_images()

    # 2. Test YOLO directly
    yolo_results = test_yolo_detection_directly()

    # 3. Test full system
    test_with_system()

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY ")
    print("="*70)

    print(f"\nExisting images found: {len(existing)}")
    print(f"YOLO detections: {len(yolo_results)}")

    if len(yolo_results) == 0:
        print("\n⚠️ IMPORTANT: YOLO is not detecting synthetic images")
        print("This explains why all tests show 0 regions detected")
        print("The system is falling back to center region detection")
        print("\nTo get accurate performance metrics, you need:")
        print("1. Real photographs with actual people")
        print("2. Or download a YOLO model trained on synthetic data")
    else:
        print("\n✓ YOLO is detecting objects")
        print("Performance metrics should be accurate")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "existing_images": existing,
        "yolo_detections": yolo_results,
        "conclusion": "YOLO needs real photos" if len(yolo_results) == 0 else "Detection working"
    }

    with open("real_detection_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: real_detection_test_results.json")

if __name__ == "__main__":
    main()