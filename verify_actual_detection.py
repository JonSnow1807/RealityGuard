#!/usr/bin/env python3
"""
Verify that the system is actually detecting and protecting real objects,
not just using fallback simulation.
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
from realityguard_final import FinalRealityGuard, FinalConfig, PrivacyStrength

def create_realistic_test_image():
    """Create a test image with actual person-like and object-like shapes."""
    # Create a more realistic scene
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Light gray background

    # Add floor
    cv2.rectangle(img, (0, 500), (1280, 720), (100, 80, 70), -1)

    # Add wall
    cv2.rectangle(img, (0, 0), (1280, 300), (230, 220, 210), -1)

    # Add a person-like silhouette (more realistic proportions)
    # Head
    cv2.ellipse(img, (400, 200), (40, 50), 0, 0, 360, (180, 140, 100), -1)
    # Body
    cv2.ellipse(img, (400, 320), (60, 100), 0, 0, 360, (100, 100, 150), -1)
    # Arms
    cv2.ellipse(img, (340, 320), (20, 80), -30, 0, 360, (100, 100, 150), -1)
    cv2.ellipse(img, (460, 320), (20, 80), 30, 0, 360, (100, 100, 150), -1)
    # Legs
    cv2.rectangle(img, (370, 400), (395, 500), (80, 80, 120), -1)
    cv2.rectangle(img, (405, 400), (430, 500), (80, 80, 120), -1)

    # Add a laptop-like object
    # Screen
    cv2.rectangle(img, (700, 300), (900, 450), (40, 40, 40), -1)
    cv2.rectangle(img, (710, 310), (890, 440), (100, 150, 200), -1)
    # Keyboard base
    pts = np.array([[680, 450], [920, 450], [940, 500], [660, 500]], np.int32)
    cv2.fillPoly(img, [pts], (60, 60, 60))

    # Add a phone-like object
    cv2.rectangle(img, (1000, 350), (1080, 500), (30, 30, 30), -1)
    cv2.rectangle(img, (1010, 360), (1070, 480), (150, 150, 180), -1)

    # Add some text/labels
    cv2.putText(img, "PERSON", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "LAPTOP", (780, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "PHONE", (1000, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def test_yolo_directly():
    """Test YOLO model directly to see what it detects."""
    print("=" * 80)
    print("TESTING YOLO DETECTION DIRECTLY")
    print("=" * 80)

    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO: {e}")
        return

    # Create test image
    test_img = create_realistic_test_image()
    cv2.imwrite("test_scene.jpg", test_img)
    print("\nTest image saved as 'test_scene.jpg'")

    # Run YOLO detection with different confidence thresholds
    print("\nTesting different confidence thresholds:")
    print("-" * 40)

    for conf_threshold in [0.5, 0.25, 0.1, 0.05]:
        print(f"\nConfidence threshold: {conf_threshold}")
        results = model(test_img, conf=conf_threshold, verbose=False)

        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = model.names[int(cls)]
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': box
                    })

        if detections:
            print(f"  Found {len(detections)} objects:")
            for det in detections:
                print(f"    - {det['class']}: conf={det['confidence']:.3f}")
        else:
            print("  No objects detected")

    # Test with a real image if available
    print("\n" + "-" * 40)
    print("Testing with downloaded test image...")

    # Download a test image with people
    os.system("wget -q -O test_people.jpg 'https://images.pexels.com/photos/1181671/pexels-photo-1181671.jpeg?w=640'")

    if os.path.exists("test_people.jpg"):
        real_img = cv2.imread("test_people.jpg")
        if real_img is not None:
            print("Testing real image with people...")
            results = model(real_img, conf=0.25, verbose=False)

            detections = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()

                    for box, cls, conf in zip(boxes, classes, confs):
                        class_name = model.names[int(cls)]
                        detections.append({
                            'class': class_name,
                            'confidence': conf
                        })

            if detections:
                print(f"  Found {len(detections)} objects in real image:")
                for det in detections:
                    print(f"    - {det['class']}: conf={det['confidence']:.3f}")
            else:
                print("  No objects detected in real image")

    return len(detections) > 0

def test_system_detection():
    """Test what the full system actually detects and processes."""
    print("\n" + "=" * 80)
    print("TESTING FULL SYSTEM DETECTION")
    print("=" * 80)

    config = FinalConfig(
        privacy_strength=PrivacyStrength.MAXIMUM,
        min_pixel_difference=10.0,
        debug_mode=False
    )

    system = FinalRealityGuard(config)

    # Test 1: Synthetic image
    print("\n1. Testing with synthetic scene:")
    test_img = create_realistic_test_image()

    # Get detections
    regions = system._detect_objects(test_img)

    print(f"   Detected {len(regions)} regions:")
    yolo_detections = 0
    fallback_detections = 0

    for i, region in enumerate(regions):
        region_type = region.get('type', 'unknown')
        confidence = region.get('confidence', 0)

        # Check if this is a YOLO detection or fallback
        if region_type in ['person', 'screen', 'vehicle', 'object']:
            if confidence > 0.25:  # Likely YOLO
                yolo_detections += 1
                detection_source = "YOLO"
            else:
                fallback_detections += 1
                detection_source = "Fallback"
        else:
            fallback_detections += 1
            detection_source = "Fallback"

        bbox = region['bbox']
        print(f"   Region {i}: type={region_type}, conf={confidence:.2f}, source={detection_source}")
        print(f"            bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    print(f"\n   Summary: {yolo_detections} YOLO detections, {fallback_detections} fallback detections")

    # Process the image
    result, proc_time = system.process_frame(test_img)

    # Check privacy application
    diff = np.mean(np.abs(test_img.astype(np.float32) - result.astype(np.float32)))
    print(f"   Privacy applied: {diff:.1f} pixel difference")

    # Save comparison
    comparison = np.hstack([test_img, result])
    cv2.putText(comparison, "ORIGINAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(comparison, "PROCESSED", (1280 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite("detection_comparison.jpg", comparison)
    print("   Comparison saved as 'detection_comparison.jpg'")

    # Test 2: Real image if available
    if os.path.exists("test_people.jpg"):
        print("\n2. Testing with real image:")
        real_img = cv2.imread("test_people.jpg")

        regions = system._detect_objects(real_img)

        print(f"   Detected {len(regions)} regions in real image:")
        yolo_detections = 0
        fallback_detections = 0

        for i, region in enumerate(regions[:5]):  # Show first 5
            region_type = region.get('type', 'unknown')
            confidence = region.get('confidence', 0)

            if region_type in ['person', 'screen', 'vehicle', 'object'] and confidence > 0.25:
                yolo_detections += 1
                detection_source = "YOLO"
            else:
                fallback_detections += 1
                detection_source = "Fallback"

            print(f"   Region {i}: type={region_type}, conf={confidence:.2f}, source={detection_source}")

        print(f"\n   Summary: {yolo_detections} YOLO detections, {fallback_detections} fallback detections")

        # Process the real image
        result, _ = system.process_frame(real_img)
        diff = np.mean(np.abs(real_img.astype(np.float32) - result.astype(np.float32)))
        print(f"   Privacy applied: {diff:.1f} pixel difference")

        # Save real comparison
        real_comparison = np.hstack([real_img, result])
        cv2.imwrite("real_detection_comparison.jpg", real_comparison)
        print("   Real image comparison saved as 'real_detection_comparison.jpg'")

    return yolo_detections > 0

def analyze_detection_quality():
    """Analyze the quality of detections and privacy application."""
    print("\n" + "=" * 80)
    print("DETECTION QUALITY ANALYSIS")
    print("=" * 80)

    # Create a video with moving person
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    test_video = "detection_test.mp4"
    out = cv2.VideoWriter(test_video, fourcc, 10, (1280, 720))

    for frame_num in range(30):
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 200

        # Moving person
        x = 200 + frame_num * 20
        # Head
        cv2.circle(img, (x, 250), 35, (180, 140, 100), -1)
        # Body
        cv2.rectangle(img, (x-40, 285), (x+40, 450), (100, 100, 150), -1)
        # Arms
        cv2.rectangle(img, (x-60, 300), (x-45, 400), (100, 100, 150), -1)
        cv2.rectangle(img, (x+45, 300), (x+60, 400), (100, 100, 150), -1)
        # Legs
        cv2.rectangle(img, (x-30, 450), (x-10, 550), (80, 80, 120), -1)
        cv2.rectangle(img, (x+10, 450), (x+30, 550), (80, 80, 120), -1)

        # Static laptop
        cv2.rectangle(img, (800, 400), (1000, 550), (40, 40, 40), -1)
        cv2.rectangle(img, (810, 410), (990, 540), (100, 150, 200), -1)

        out.write(img)

    out.release()
    print("Test video created: detection_test.mp4")

    # Process the video
    config = FinalConfig(
        privacy_strength=PrivacyStrength.HIGH,
        max_frames=30,
        debug_mode=False
    )

    system = FinalRealityGuard(config)

    print("\nProcessing video to analyze detections...")
    cap = cv2.VideoCapture(test_video)

    frame_count = 0
    yolo_frames = 0
    fallback_frames = 0
    privacy_applied_frames = 0

    while cap.isOpened() and frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        regions = system._detect_objects(frame)
        result, _ = system.process_frame(frame)

        # Check detection source
        has_yolo = any(r.get('confidence', 0) > 0.25 and r.get('type') in ['person', 'object']
                      for r in regions)

        if has_yolo:
            yolo_frames += 1
        else:
            fallback_frames += 1

        # Check privacy application
        diff = np.mean(np.abs(frame.astype(np.float32) - result.astype(np.float32)))
        if diff > 10:
            privacy_applied_frames += 1

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"   Frame {frame_count}: {'YOLO' if has_yolo else 'Fallback'}, "
                  f"Privacy: {diff:.1f} pixels")

    cap.release()

    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    print(f"Total frames analyzed: {frame_count}")
    print(f"YOLO detection frames: {yolo_frames} ({yolo_frames/frame_count*100:.1f}%)")
    print(f"Fallback detection frames: {fallback_frames} ({fallback_frames/frame_count*100:.1f}%)")
    print(f"Privacy applied frames: {privacy_applied_frames} ({privacy_applied_frames/frame_count*100:.1f}%)")

    # Verdict
    print("\n" + "=" * 80)
    if yolo_frames > frame_count * 0.5:
        print("✅ YOLO IS WORKING: Detecting objects in majority of frames")
        print("   The system is performing its intended task.")
    elif yolo_frames > 0:
        print("⚠️ YOLO PARTIALLY WORKING: Some detections, but mostly using fallback")
        print("   The system is partially performing its intended task.")
    else:
        print("❌ YOLO NOT WORKING: Completely relying on fallback")
        print("   The system is NOT performing its intended task properly.")

    if privacy_applied_frames > frame_count * 0.8:
        print("✅ PRIVACY APPLICATION: Working well")
    elif privacy_applied_frames > frame_count * 0.5:
        print("⚠️ PRIVACY APPLICATION: Partially working")
    else:
        print("❌ PRIVACY APPLICATION: Not working properly")

    print("=" * 80)

def main():
    """Run all verification tests."""
    print("=" * 80)
    print("VERIFYING ACTUAL DETECTION CAPABILITY")
    print("=" * 80)
    print("\nThis test verifies whether the system is actually detecting")
    print("real objects (people, laptops, etc.) or just using fallback.")
    print("=" * 80)

    # Test 1: Direct YOLO testing
    yolo_works = test_yolo_directly()

    # Test 2: System detection testing
    system_detects = test_system_detection()

    # Test 3: Quality analysis
    analyze_detection_quality()

    print("\n" + "=" * 80)
    print("FINAL VERIFICATION VERDICT")
    print("=" * 80)

    if yolo_works:
        print("✅ YOLO model is capable of detecting objects")
    else:
        print("❌ YOLO model is not detecting objects properly")

    if system_detects:
        print("✅ System is using YOLO detections")
    else:
        print("❌ System is relying entirely on fallback")

    print("\nCONCLUSION:")
    if yolo_works and system_detects:
        print("The system IS performing its intended task - detecting and")
        print("protecting real objects using AI-based segmentation.")
    else:
        print("The system is NOT performing its intended task properly.")
        print("It's mostly using fallback detection instead of real AI detection.")

    print("=" * 80)

if __name__ == "__main__":
    main()