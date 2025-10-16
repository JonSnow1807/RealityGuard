#!/usr/bin/env python3
"""
Test RealityGuard Ultimate with real-world images
"""

import cv2
import numpy as np
import urllib.request
import os
from realityguard_ultimate import RealityGuardUltimate, UltimateConfig

def download_test_image():
    """Download a real test image with people"""
    # Test images with people
    test_urls = [
        "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=800",  # Office people
        "https://images.unsplash.com/photo-1517486808906-6ca8b3d66f4f?w=800",  # Meeting
        "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?w=800"   # Team
    ]

    for i, url in enumerate(test_urls):
        try:
            print(f"Downloading test image {i+1}...")
            filename = f'real_test_{i+1}.jpg'
            urllib.request.urlretrieve(url, filename)

            # Test with this image
            img = cv2.imread(filename)
            if img is not None:
                print(f"✓ Downloaded {filename}: {img.shape}")
                return img, filename
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Fallback to local image if exists
    if os.path.exists('test_person_laptop.jpg'):
        img = cv2.imread('test_person_laptop.jpg')
        if img is not None:
            print("✓ Using existing test_person_laptop.jpg")
            return img, 'test_person_laptop.jpg'

    # Create synthetic as last resort
    print("⚠ Using synthetic test image")
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    cv2.circle(img, (640, 360), 150, (200, 180, 160), -1)
    return img, 'synthetic.jpg'

def main():
    """Test with real-world data"""

    print("="*80)
    print("REALITYGUARD ULTIMATE - REAL WORLD TEST")
    print("="*80)

    # Download test image
    test_img, img_name = download_test_image()
    print(f"\nTesting with: {img_name}")
    print(f"Image shape: {test_img.shape}")

    # Create system (will use YOLO if available)
    config = UltimateConfig(
        force_simulation=False,  # Use real YOLO detection
        min_strength=0.20,
        max_strength=0.60,
        default_strength=0.35
    )
    system = RealityGuardUltimate(config)

    print("\n" + "="*80)
    print("PROCESSING REAL-WORLD IMAGE")
    print("="*80)

    # Process the image multiple times to test adaptation
    results = []
    for i in range(5):
        protected, stats = system.process_frame(test_img)
        results.append(stats)

        print(f"\nPass {i+1}:")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Pixel Difference: {stats['pixel_diff']:.1f}px")
        print(f"  Strength: {stats['strength']:.3f}")
        print(f"  Strategy: {stats['strategy']}")
        print(f"  YOLO Regions: {stats['regions_detected']}")
        print(f"  Predicted Regions: {stats['regions_predicted']}")
        print(f"  Total Processed: {stats['regions_processed']}")
        print(f"  Cache Efficiency: {stats['cache_efficiency']:.0f}%")

        # Save first result
        if i == 0:
            output_name = f'ultimate_{img_name}'
            cv2.imwrite(output_name, protected)
            print(f"  ✓ Saved: {output_name}")

    # Final summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    avg_fps = sum(r['fps'] for r in results) / len(results)
    avg_diff = sum(r['pixel_diff'] for r in results) / len(results)
    final_cache = results[-1]['cache_efficiency']

    print(f"\nAverage FPS: {avg_fps:.1f}")
    print(f"Average Pixel Difference: {avg_diff:.1f}px")
    print(f"Final Cache Efficiency: {final_cache:.0f}%")
    print(f"Total Adaptations: {results[-1]['adaptations']}")

    # Verify effectiveness
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    checks = {
        "Real-time (>24 FPS)": avg_fps >= 24,
        "Visible Effect (2-15px)": 2 <= avg_diff <= 15,
        "Cache Working (>70%)": final_cache >= 70,
        "YOLO Detection": results[0]['regions_detected'] > 0 or config.force_simulation,
        "Adaptive Control": results[-1]['adaptations'] > 0
    }

    for check, passed in checks.items():
        print(f"  {check}: {'✓ PASS' if passed else '✗ FAIL'}")

    if all(checks.values()):
        print("\n✓ SYSTEM FULLY OPERATIONAL")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n⚠ Issues with: {', '.join(failed)}")

    print("="*80)

if __name__ == "__main__":
    main()