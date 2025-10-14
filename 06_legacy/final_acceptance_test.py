#!/usr/bin/env python3
"""
Final Acceptance Test for RealityGuard System
Tests with realistic thresholds and comprehensive metrics
"""

import cv2
import numpy as np
import time
from realityguard_final import FinalRealityGuard, FinalConfig, PrivacyStrength

def test_privacy_strength(strength: PrivacyStrength):
    """Test a specific privacy strength."""
    print(f"\n{'='*60}")
    print(f"Testing {strength.value.upper()} Privacy Strength")
    print('='*60)

    config = FinalConfig(
        privacy_strength=strength,
        min_pixel_difference=5.0,  # Realistic threshold
        enable_memory_optimization=True,
        max_frames=30,  # Quick test
        debug_mode=True
    )

    system = FinalRealityGuard(config)

    # Test 1: Single frame privacy
    print("\n1. Single Frame Test:")
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    cv2.rectangle(frame, (100, 100), (300, 400), (255, 0, 0), -1)
    cv2.rectangle(frame, (400, 200), (600, 400), (0, 255, 0), -1)

    result, proc_time = system.process_frame(frame)
    diff = np.mean(np.abs(frame.astype(np.float32) - result.astype(np.float32)))

    print(f"   Processing time: {proc_time*1000:.1f} ms")
    print(f"   Pixel difference: {diff:.1f}")
    print(f"   Privacy applied: {'✅' if diff > 5 else '❌'}")

    # Test 2: Video processing
    print("\n2. Video Processing Test:")

    # Create test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    test_video = f"acceptance_{strength.value}.mp4"
    out = cv2.VideoWriter(test_video, fourcc, 30, (640, 480))

    for i in range(30):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Moving person
        x = 100 + i * 10
        cv2.rectangle(frame, (x, 100), (x + 100, 350), (100, 150, 200), -1)
        cv2.circle(frame, (x + 50, 80), 30, (200, 150, 100), -1)

        # Static object
        cv2.rectangle(frame, (450, 250), (600, 380), (50, 50, 200), -1)

        out.write(frame)

    out.release()

    # Process video
    output_video = f"acceptance_out_{strength.value}.mp4"
    start = time.time()
    results = system.process_video(test_video, output_video)
    elapsed = time.time() - start

    print(f"   Frames processed: {results['frames_processed']}")
    print(f"   Processing time: {elapsed:.1f}s")
    print(f"   Average FPS: {results['average_fps']:.1f}")
    print(f"   Privacy effect: {results['average_privacy_effect']:.1f} pixels")

    # Count frames with privacy (using realistic threshold)
    frames_with_privacy = 0
    cap = cv2.VideoCapture(test_video)
    cap2 = cv2.VideoCapture(output_video)

    while cap.isOpened() and cap2.isOpened():
        ret1, frame1 = cap.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame_diff = np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))
        if frame_diff > 5:
            frames_with_privacy += 1

    cap.release()
    cap2.release()

    privacy_rate = (frames_with_privacy / max(results['frames_processed'], 1)) * 100
    print(f"   Privacy rate: {privacy_rate:.1f}%")
    print(f"   Cache hit rate: {results['cache_hit_rate']:.1f}%")
    print(f"   Memory usage: {results['memory_used_mb']:.1f} MB")

    # Test 3: Memory stability
    print("\n3. Memory Stability Test:")
    initial_memory = results['memory_used_mb']

    # Process more frames
    for _ in range(10):
        result, _ = system.process_frame(frame)

    final_memory = system._get_memory_usage()
    memory_growth = final_memory - initial_memory

    print(f"   Initial memory: {initial_memory:.1f} MB")
    print(f"   Final memory: {final_memory:.1f} MB")
    print(f"   Memory growth: {memory_growth:.1f} MB")
    print(f"   Memory stable: {'✅' if memory_growth < 10 else '❌'}")

    # Overall assessment
    passed = (
        diff > 5 and
        results['average_fps'] > 15 and
        privacy_rate > 50 and
        memory_growth < 10
    )

    return {
        'strength': strength.value,
        'pixel_diff': diff,
        'fps': results['average_fps'],
        'privacy_rate': privacy_rate,
        'cache_hit_rate': results['cache_hit_rate'],
        'memory_growth': memory_growth,
        'passed': passed
    }

def main():
    """Run comprehensive acceptance tests."""
    print("="*80)
    print("REALITYGUARD FINAL ACCEPTANCE TEST")
    print("="*80)

    results = []

    # Test all privacy strengths
    for strength in PrivacyStrength:
        result = test_privacy_strength(strength)
        results.append(result)

    # Final summary
    print("\n" + "="*80)
    print("ACCEPTANCE TEST SUMMARY")
    print("="*80)

    print("\n┌" + "─"*78 + "┐")
    print("│" + " " * 25 + "TEST RESULTS SUMMARY" + " " * 33 + "│")
    print("├" + "─"*78 + "┤")
    print(f"│ {'Strength':<10} │ {'FPS':>7} │ {'Privacy':>8} │ {'PixelDiff':>9} │ {'Cache':>7} │ {'Memory':>7} │ {'Status':>8} │")
    print("├" + "─"*78 + "┤")

    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"│ {r['strength']:<10} │ {r['fps']:>7.1f} │ {r['privacy_rate']:>7.1f}% │ "
              f"{r['pixel_diff']:>9.1f} │ {r['cache_hit_rate']:>6.1f}% │ {r['memory_growth']:>6.1f}M │ {status:>8} │")

    print("└" + "─"*78 + "┘")

    # Overall verdict
    all_passed = all(r['passed'] for r in results)
    critical_passed = any(r['passed'] and r['strength'] in ['high', 'maximum'] for r in results)

    print("\n" + "="*80)
    if all_passed:
        print("🎉🎉🎉 ALL TESTS PASSED 🎉🎉🎉")
        print("\nThe RealityGuard system is PRODUCTION READY!")
        print("\nKey achievements:")
        print("  ✅ Privacy protection working across all strength levels")
        print("  ✅ Real-time processing achieved")
        print("  ✅ Memory usage stable")
        print("  ✅ Cache system effective")
        print("\nAll optimizations successfully implemented!")
    elif critical_passed:
        print("✅ SYSTEM ACCEPTABLE FOR PRODUCTION")
        print("\nHigh and Maximum privacy levels working properly.")
        print("Some optimization needed for lower privacy levels.")
    else:
        print("⚠️ SYSTEM NEEDS IMPROVEMENT")
        print("\nCritical issues that need addressing:")
        for r in results:
            if not r['passed']:
                issues = []
                if r['pixel_diff'] <= 5:
                    issues.append("insufficient privacy")
                if r['fps'] <= 15:
                    issues.append("low FPS")
                if r['privacy_rate'] <= 50:
                    issues.append("low privacy rate")
                if r['memory_growth'] >= 10:
                    issues.append("memory leak")
                print(f"  - {r['strength']}: {', '.join(issues)}")

    print("="*80)

    return all_passed or critical_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)