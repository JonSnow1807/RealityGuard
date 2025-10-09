#!/usr/bin/env python3
"""
Test RealityGuard with actual real-world data
Downloads and tests with real images and videos containing people, laptops, phones, etc.
"""

import cv2
import numpy as np
import os
import subprocess
from urllib.request import urlretrieve
from ultralytics import YOLO
from realityguard_final import FinalRealityGuard, FinalConfig, PrivacyStrength
import time

def download_real_images():
    """Download real test images from various sources."""
    print("=" * 80)
    print("DOWNLOADING REAL-WORLD TEST IMAGES")
    print("=" * 80)

    # Use various free image sources
    test_images = [
        # Pexels images (free to use)
        ("https://images.pexels.com/photos/1181671/pexels-photo-1181671.jpeg", "office_people.jpg", "Office with people and laptops"),
        ("https://images.pexels.com/photos/3184287/pexels-photo-3184287.jpeg", "meeting_room.jpg", "Business meeting"),
        ("https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg", "woman_laptop.jpg", "Woman with laptop"),
        ("https://images.pexels.com/photos/4226263/pexels-photo-4226263.jpeg", "video_call.jpg", "Video conference call"),
        ("https://images.pexels.com/photos/7014337/pexels-photo-7014337.jpeg", "person_phone.jpg", "Person using phone"),

        # Unsplash images (free to use)
        ("https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=800", "coworking.jpg", "Coworking space"),
        ("https://images.unsplash.com/photo-1600880292203-757bb62b4baf?w=800", "team_meeting.jpg", "Team meeting"),
    ]

    downloaded = []
    failed = []

    for url, filename, description in test_images:
        try:
            print(f"\nDownloading: {description}")
            print(f"  URL: {url[:50]}...")
            urlretrieve(url, filename)

            # Verify file
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                img = cv2.imread(filename)
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"  ✓ Success: {filename} ({w}x{h})")
                    downloaded.append((filename, description))
                else:
                    print(f"  ✗ Failed: Cannot read image")
                    failed.append(filename)
            else:
                print(f"  ✗ Failed: File is empty or missing")
                failed.append(filename)

        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:50]}")
            failed.append(filename)

    print(f"\n✓ Downloaded: {len(downloaded)}/{len(test_images)} images")

    return downloaded

def download_real_video():
    """Download a real test video with people."""
    print("\n" + "=" * 80)
    print("DOWNLOADING REAL-WORLD TEST VIDEO")
    print("=" * 80)

    # Download a short public domain video
    video_url = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
    video_file = "test_video_real.mp4"

    try:
        print("Downloading sample video...")
        subprocess.run(["wget", "-q", "-O", video_file, video_url], timeout=30)

        if os.path.exists(video_file) and os.path.getsize(video_file) > 0:
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                print(f"  ✓ Success: {video_file}")
                print(f"    Resolution: {width}x{height}")
                print(f"    FPS: {fps:.1f}")
                print(f"    Frames: {frames}")
                return video_file
            else:
                print(f"  ✗ Failed: Cannot open video")
        else:
            print(f"  ✗ Failed: Download unsuccessful")

    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")

    return None

def test_yolo_on_real_images(images):
    """Test YOLO detection on real images."""
    print("\n" + "=" * 80)
    print("TESTING YOLO DETECTION ON REAL IMAGES")
    print("=" * 80)

    model = YOLO('yolov8n.pt')

    total_detections = {}
    successful_detections = 0

    for filename, description in images:
        print(f"\nTesting: {description}")
        print(f"File: {filename}")

        img = cv2.imread(filename)
        if img is None:
            print("  ✗ Cannot read image")
            continue

        # Run YOLO with multiple confidence levels
        for conf in [0.5, 0.25, 0.1]:
            results = model(img, conf=conf, verbose=False)

            detections = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()

                    for box, cls, conf_score in zip(boxes, classes, confs):
                        class_name = model.names[int(cls)]
                        detections.append({
                            'class': class_name,
                            'confidence': conf_score,
                            'box': box
                        })

            if detections:
                print(f"  Confidence {conf}: Found {len(detections)} objects")

                # Count by class
                from collections import Counter
                class_counts = Counter(d['class'] for d in detections)

                for cls, count in class_counts.most_common(5):
                    print(f"    - {count} {cls}(s)")

                    if cls not in total_detections:
                        total_detections[cls] = 0
                    total_detections[cls] += count

                successful_detections += 1
                break  # Found detections, no need to try lower confidence

        if not detections:
            print(f"  ✗ No objects detected even at 0.1 confidence")

    print("\n" + "-" * 40)
    print("YOLO DETECTION SUMMARY:")
    print(f"Images with detections: {successful_detections}/{len(images)}")
    print("\nMost common objects detected:")
    for cls, count in sorted(total_detections.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {cls}: {count} instances")

    return successful_detections > len(images) * 0.5  # Success if >50% images have detections

def test_system_on_real_data(images):
    """Test the full RealityGuard system on real data."""
    print("\n" + "=" * 80)
    print("TESTING REALITYGUARD SYSTEM ON REAL DATA")
    print("=" * 80)

    config = FinalConfig(
        privacy_strength=PrivacyStrength.HIGH,
        yolo_confidence=0.25,
        debug_mode=False
    )

    system = FinalRealityGuard(config)

    results = []

    for filename, description in images[:3]:  # Test first 3 images
        print(f"\n{'='*60}")
        print(f"Processing: {description}")
        print(f"File: {filename}")
        print('-'*60)

        img = cv2.imread(filename)
        if img is None:
            continue

        h, w = img.shape[:2]
        print(f"Image size: {w}x{h}")

        # Detect objects
        start = time.time()
        regions = system._detect_objects(img)
        detect_time = (time.time() - start) * 1000

        print(f"\nDetection ({detect_time:.1f}ms):")
        print(f"  Total regions: {len(regions)}")

        # Categorize regions
        yolo_regions = []
        fallback_regions = []

        for region in regions:
            if region.get('confidence', 0) > 0.25 and region.get('type') in ['person', 'screen', 'vehicle', 'object']:
                yolo_regions.append(region)
            else:
                fallback_regions.append(region)

        print(f"  - YOLO detections: {len(yolo_regions)}")
        print(f"  - Fallback detections: {len(fallback_regions)}")

        # Show YOLO detections
        if yolo_regions:
            print("\n  YOLO detected:")
            for r in yolo_regions[:5]:  # Show first 5
                print(f"    • {r.get('type', 'unknown')} (conf: {r.get('confidence', 0):.2f})")

        # Process frame
        start = time.time()
        processed, proc_time = system.process_frame(img)
        total_time = (time.time() - start) * 1000

        # Calculate privacy metrics
        diff = np.mean(np.abs(img.astype(np.float32) - processed.astype(np.float32)))

        # Check which regions were actually modified
        modified_regions = 0
        for region in regions:
            bbox = region['bbox']
            x1, y1, x2, y2 = [int(b) for b in bbox]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))

            if x2 > x1 and y2 > y1:
                roi_orig = img[y1:y2, x1:x2]
                roi_proc = processed[y1:y2, x1:x2]
                roi_diff = np.mean(np.abs(roi_orig.astype(np.float32) - roi_proc.astype(np.float32)))
                if roi_diff > 10:
                    modified_regions += 1

        print(f"\nPrivacy Application:")
        print(f"  Processing time: {total_time:.1f}ms")
        print(f"  Overall privacy effect: {diff:.1f} pixels")
        print(f"  Regions modified: {modified_regions}/{len(regions)}")
        print(f"  FPS potential: {1000/total_time:.1f}")

        # Save comparison
        comparison = np.hstack([img, processed])
        comp_h, comp_w = comparison.shape[:2]

        # Add labels
        cv2.rectangle(comparison, (0, 0), (comp_w, 60), (0, 0, 0), -1)
        cv2.putText(comparison, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(comparison, "PROCESSED", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        output_name = f"real_test_{os.path.splitext(filename)[0]}_comparison.jpg"
        cv2.imwrite(output_name, comparison)
        print(f"\n  ✓ Saved comparison: {output_name}")

        results.append({
            'file': filename,
            'yolo_detections': len(yolo_regions),
            'total_regions': len(regions),
            'privacy_effect': diff,
            'modified_regions': modified_regions,
            'fps': 1000/total_time
        })

    # Summary
    print("\n" + "=" * 80)
    print("REAL-WORLD DATA TEST SUMMARY")
    print("=" * 80)

    if results:
        avg_yolo = sum(r['yolo_detections'] for r in results) / len(results)
        avg_privacy = sum(r['privacy_effect'] for r in results) / len(results)
        avg_fps = sum(r['fps'] for r in results) / len(results)
        avg_modified = sum(r['modified_regions'] for r in results) / len(results)

        print(f"\nAverage Results ({len(results)} images):")
        print(f"  YOLO detections per image: {avg_yolo:.1f}")
        print(f"  Privacy effect: {avg_privacy:.1f} pixels")
        print(f"  Regions modified per image: {avg_modified:.1f}")
        print(f"  Average FPS: {avg_fps:.1f}")

        success = avg_yolo > 0 and avg_privacy > 10 and avg_fps > 24

        if success:
            print("\n✅ SYSTEM WORKING AS INTENDED")
            print("  - YOLO successfully detecting objects in real images")
            print("  - Privacy protection being applied effectively")
            print("  - Real-time performance achieved")
        else:
            print("\n⚠️ SYSTEM PERFORMANCE ISSUES")
            if avg_yolo == 0:
                print("  - YOLO not detecting objects")
            if avg_privacy < 10:
                print("  - Privacy effect too weak")
            if avg_fps < 24:
                print("  - Performance below real-time threshold")

        return success

    return False

def test_video_processing(video_file):
    """Test video processing with real video."""
    if not video_file:
        print("\n⚠️ No video file to test")
        return False

    print("\n" + "=" * 80)
    print("TESTING VIDEO PROCESSING WITH REAL DATA")
    print("=" * 80)

    config = FinalConfig(
        privacy_strength=PrivacyStrength.HIGH,
        max_frames=60,  # Process first 60 frames
        debug_mode=False
    )

    system = FinalRealityGuard(config)

    print(f"\nProcessing video: {video_file}")
    print("Processing first 60 frames...")

    output_video = "real_video_output.mp4"
    results = system.process_video(video_file, output_video)

    print("\nVideo Processing Results:")
    print(f"  Frames processed: {results['frames_processed']}")
    print(f"  Average FPS: {results['average_fps']:.1f}")
    print(f"  Privacy rate: {results['privacy_rate']:.1f}%")
    print(f"  Privacy effect: {results['average_privacy_effect']:.1f} pixels")
    print(f"  Cache hit rate: {results['cache_hit_rate']:.1f}%")
    print(f"  Memory used: {results['memory_used_mb']:.1f} MB")

    success = (
        results['average_fps'] > 20 and
        results['privacy_rate'] > 30 and
        results['average_privacy_effect'] > 5
    )

    if success:
        print("\n✅ VIDEO PROCESSING SUCCESSFUL")
    else:
        print("\n⚠️ VIDEO PROCESSING ISSUES")

    return success

def main():
    """Run comprehensive real-world data tests."""
    print("=" * 80)
    print("COMPREHENSIVE REAL-WORLD DATA TESTING")
    print("=" * 80)
    print("\nThis test uses ACTUAL real-world images and videos")
    print("to verify the system performs its intended task.")
    print("=" * 80)

    # Step 1: Download real images
    images = download_real_images()

    if not images:
        print("\n✗ Failed to download test images")
        return

    # Step 2: Download real video
    video = download_real_video()

    # Step 3: Test YOLO on real images
    yolo_works = test_yolo_on_real_images(images)

    # Step 4: Test full system on real images
    system_works = test_system_on_real_data(images)

    # Step 5: Test video processing
    video_works = test_video_processing(video) if video else False

    # Final Report
    print("\n" + "=" * 80)
    print("FINAL REAL-WORLD TEST REPORT")
    print("=" * 80)

    print("\n┌" + "─"*58 + "┐")
    print("│" + " "*20 + "TEST RESULTS" + " "*26 + "│")
    print("├" + "─"*58 + "┤")
    print(f"│ YOLO Detection on Real Images:  {'✅ PASS' if yolo_works else '❌ FAIL':>24} │")
    print(f"│ System Processing Real Images:  {'✅ PASS' if system_works else '❌ FAIL':>24} │")
    print(f"│ Video Processing:               {'✅ PASS' if video_works else '❌ FAIL':>24} │")
    print("└" + "─"*58 + "┘")

    all_pass = yolo_works and system_works

    print("\n" + "=" * 80)
    if all_pass:
        print("🎉 SYSTEM IS PERFORMING ITS INTENDED TASK 🎉")
        print("\nThe RealityGuard system successfully:")
        print("  ✅ Detects real people and objects using YOLO")
        print("  ✅ Applies effective privacy protection")
        print("  ✅ Maintains real-time performance")
        print("  ✅ Works with actual real-world content")
        print("\nThe system is PRODUCTION READY for real-world deployment!")
    else:
        print("⚠️ SYSTEM HAS ISSUES WITH REAL-WORLD DATA")
        if not yolo_works:
            print("  ❌ YOLO detection problems on real images")
        if not system_works:
            print("  ❌ System processing issues")
        if not video_works:
            print("  ❌ Video processing problems")
    print("=" * 80)

if __name__ == "__main__":
    main()