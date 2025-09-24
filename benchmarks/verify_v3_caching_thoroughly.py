#!/usr/bin/env python3
"""
Thorough verification of V3 caching claims
Tests accuracy, timing, and actual performance
"""

import cv2
import numpy as np
import time
import hashlib
from mediapipe_excellence_v3_caching import MediaPipeWithCaching
from mediapipe_excellence_v1_baseline import MediaPipeBaseline


def test_detection_accuracy():
    """Verify that caching doesn't break detection accuracy"""
    print("\n" + "="*60)
    print("DETECTION ACCURACY TEST")
    print("="*60)

    cached = MediaPipeWithCaching(cache_size=30)
    baseline = MediaPipeBaseline()

    # Test cases with known objects
    test_cases = [
        ("empty", np.zeros((480, 640, 3), dtype=np.uint8), 0),
        ("one_circle", np.zeros((480, 640, 3), dtype=np.uint8), 1),
        ("three_circles", np.zeros((480, 640, 3), dtype=np.uint8), 3)
    ]

    # Add shapes to test frames
    cv2.circle(test_cases[1][1], (320, 240), 60, (255, 255, 255), -1)

    cv2.circle(test_cases[2][1], (160, 240), 50, (255, 255, 255), -1)
    cv2.circle(test_cases[2][1], (320, 240), 50, (255, 255, 255), -1)
    cv2.circle(test_cases[2][1], (480, 240), 50, (255, 255, 255), -1)

    results = []

    for name, frame, expected in test_cases:
        # Test cached version
        cached_detections = cached.detect_with_cache(frame)

        # Test baseline version
        baseline_detections = baseline.detect_shapes(frame)

        print(f"\n{name}:")
        print(f"  Expected: {expected} objects")
        print(f"  Baseline detected: {len(baseline_detections)}")
        print(f"  Cached detected: {len(cached_detections)}")

        # Check if detection counts match
        accuracy_match = len(cached_detections) == len(baseline_detections)
        print(f"  Match: {'✅' if accuracy_match else '❌'}")

        results.append({
            'name': name,
            'expected': expected,
            'baseline': len(baseline_detections),
            'cached': len(cached_detections),
            'match': accuracy_match
        })

    return results


def test_cache_behavior():
    """Test if cache actually works as claimed"""
    print("\n" + "="*60)
    print("CACHE BEHAVIOR TEST")
    print("="*60)

    cached = MediaPipeWithCaching(cache_size=5)

    # Create identical frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame1, (320, 240), 60, (255, 255, 255), -1)

    frame2 = frame1.copy()  # Identical frame

    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame3, (321, 240), 60, (255, 255, 255), -1)  # Slightly different

    # First detection (should be cache miss)
    print("\n1. First frame detection:")
    start = time.perf_counter()
    det1 = cached.detect_with_cache(frame1)
    time1 = (time.perf_counter() - start) * 1000
    print(f"   Time: {time1:.2f}ms")
    print(f"   Cache hits: {cached.metrics['cache_hits']}")
    print(f"   Cache misses: {cached.metrics['cache_misses']}")
    print(f"   Full detections: {cached.metrics['full_detections']}")

    # Second detection of identical frame (should be cache hit)
    print("\n2. Identical frame detection:")
    start = time.perf_counter()
    det2 = cached.detect_with_cache(frame2)
    time2 = (time.perf_counter() - start) * 1000
    print(f"   Time: {time2:.2f}ms")
    print(f"   Cache hits: {cached.metrics['cache_hits']}")
    print(f"   Cache misses: {cached.metrics['cache_misses']}")
    print(f"   Full detections: {cached.metrics['full_detections']}")

    # Check if it was actually cached
    cache_worked = cached.metrics['cache_hits'] > 0
    speedup = time1 / time2 if time2 > 0 else 0

    print(f"\n   Cache worked: {'✅' if cache_worked else '❌'}")
    print(f"   Speedup: {speedup:.1f}x")

    # Third detection of different frame (should be cache miss)
    print("\n3. Different frame detection:")
    start = time.perf_counter()
    det3 = cached.detect_with_cache(frame3)
    time3 = (time.perf_counter() - start) * 1000
    print(f"   Time: {time3:.2f}ms")
    print(f"   Cache hits: {cached.metrics['cache_hits']}")
    print(f"   Cache misses: {cached.metrics['cache_misses']}")
    print(f"   Full detections: {cached.metrics['full_detections']}")

    # Repeat identical frame to verify cache
    print("\n4. Repeat identical frame:")
    start = time.perf_counter()
    det4 = cached.detect_with_cache(frame2)
    time4 = (time.perf_counter() - start) * 1000
    print(f"   Time: {time4:.2f}ms")
    print(f"   Cache hits: {cached.metrics['cache_hits']}")

    return {
        'cache_worked': cache_worked,
        'speedup': speedup,
        'hit_rate': cached.detection_cache.get_hit_rate()
    }


def test_realistic_performance():
    """Test with realistic video-like sequences"""
    print("\n" + "="*60)
    print("REALISTIC PERFORMANCE TEST")
    print("="*60)

    cached = MediaPipeWithCaching(cache_size=30)
    baseline = MediaPipeBaseline()

    # Create realistic video sequence
    frames = []

    # Static scene (30 identical frames)
    print("\nGenerating static scene...")
    static_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(static_frame, (640, 360), 80, (255, 255, 255), -1)
    for _ in range(30):
        frames.append(static_frame.copy())

    # Slow motion (30 frames with gradual movement)
    print("Generating slow motion scene...")
    for i in range(30):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x = 400 + i * 5  # Slow movement
        cv2.circle(frame, (x, 360), 80, (255, 255, 255), -1)
        frames.append(frame)

    # Test cached version
    print("\n[Cached Version]")
    cached_times = []
    start_total = time.perf_counter()

    for i, frame in enumerate(frames):
        start = time.perf_counter()
        _, info = cached.process_frame(frame)
        elapsed = (time.perf_counter() - start) * 1000
        cached_times.append(elapsed)

        if i % 10 == 0:
            print(f"  Frame {i}: {elapsed:.2f}ms")

    cached_total = (time.perf_counter() - start_total) * 1000

    print(f"\nCached Statistics:")
    print(f"  Total time: {cached_total:.2f}ms")
    print(f"  Average: {np.mean(cached_times):.2f}ms")
    print(f"  FPS: {len(frames) * 1000 / cached_total:.1f}")
    print(f"  Cache hit rate: {cached.detection_cache.get_hit_rate():.2%}")
    print(f"  Full detections: {cached.metrics['full_detections']}")

    # Test baseline version
    print("\n[Baseline Version]")
    baseline_times = []
    start_total = time.perf_counter()

    for i, frame in enumerate(frames):
        start = time.perf_counter()
        _, info = baseline.process_frame(frame)
        elapsed = (time.perf_counter() - start) * 1000
        baseline_times.append(elapsed)

        if i % 10 == 0:
            print(f"  Frame {i}: {elapsed:.2f}ms")

    baseline_total = (time.perf_counter() - start_total) * 1000

    print(f"\nBaseline Statistics:")
    print(f"  Total time: {baseline_total:.2f}ms")
    print(f"  Average: {np.mean(baseline_times):.2f}ms")
    print(f"  FPS: {len(frames) * 1000 / baseline_total:.1f}")

    # Calculate real speedup
    real_speedup = baseline_total / cached_total

    print(f"\n📊 REAL SPEEDUP: {real_speedup:.2f}x")

    return {
        'cached_fps': len(frames) * 1000 / cached_total,
        'baseline_fps': len(frames) * 1000 / baseline_total,
        'speedup': real_speedup,
        'cache_hit_rate': cached.detection_cache.get_hit_rate()
    }


def test_frame_hashing():
    """Verify frame hashing works correctly"""
    print("\n" + "="*60)
    print("FRAME HASHING TEST")
    print("="*60)

    cached = MediaPipeWithCaching()

    # Test that identical frames produce same hash
    frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

    hash1 = cached.compute_frame_hash(frame1)
    hash2 = cached.compute_frame_hash(frame2)

    print(f"Identical frames same hash: {'✅' if hash1 == hash2 else '❌'}")

    # Test that different frames produce different hashes
    frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 129
    hash3 = cached.compute_frame_hash(frame3)

    print(f"Different frames different hash: {'✅' if hash1 != hash3 else '❌'}")

    # Test hash performance
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = cached.compute_frame_hash(frame1)
        times.append((time.perf_counter() - start) * 1000)

    print(f"Average hash time: {np.mean(times):.3f}ms")

    return np.mean(times)


def test_motion_prediction():
    """Test if motion prediction actually works"""
    print("\n" + "="*60)
    print("MOTION PREDICTION TEST")
    print("="*60)

    cached = MediaPipeWithCaching()

    # Create frames with moving object
    frames = []
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 100 + i * 20  # Linear motion
        cv2.circle(frame, (x, 240), 50, (255, 255, 255), -1)
        frames.append(frame)

    # Process frames to build motion history
    for i, frame in enumerate(frames[:5]):
        detections = cached.detect_with_cache(frame)
        cached.motion_predictor.update(detections, i)

    # Test prediction
    predicted = cached.motion_predictor.predict_next(5)

    print(f"Motion history built: {len(cached.motion_predictor.history)} objects")
    print(f"Predictions made: {len(predicted)}")

    if predicted:
        print(f"Last detection: {cached.motion_predictor.history['det_0'][-1]['bbox']}")
        print(f"Predicted next: {predicted[0]}")

    return len(predicted) > 0


def verify_extreme_fps_claim():
    """Verify the 6812 FPS claim for static scenes"""
    print("\n" + "="*60)
    print("EXTREME FPS CLAIM VERIFICATION")
    print("="*60)

    cached = MediaPipeWithCaching(cache_size=100)

    # Create a static frame
    static_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(static_frame, (320, 240), 60, (255, 255, 255), -1)

    # First pass - populate cache
    print("\nFirst pass (populating cache):")
    _ = cached.detect_with_cache(static_frame)

    # Measure cached retrieval speed
    print("\nMeasuring cached retrieval speed:")
    times = []

    for i in range(1000):
        start = time.perf_counter()
        _ = cached.detect_with_cache(static_frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    fps_avg = 1000 / avg_time if avg_time > 0 else 0
    fps_max = 1000 / min_time if min_time > 0 else 0

    print(f"  Average time: {avg_time:.4f}ms ({fps_avg:.1f} FPS)")
    print(f"  Min time: {min_time:.4f}ms ({fps_max:.1f} FPS)")
    print(f"  Max time: {max_time:.4f}ms")
    print(f"  Cache hit rate: {cached.detection_cache.get_hit_rate():.2%}")

    print(f"\n  Claimed: 6812 FPS")
    print(f"  Actual: {fps_avg:.1f} FPS")
    print(f"  Peak: {fps_max:.1f} FPS")

    if fps_avg > 6000:
        print("  ✅ Claim VERIFIED!")
    elif fps_avg > 1000:
        print("  ⚠️ High FPS but not quite 6812")
    else:
        print("  ❌ Claim NOT verified")

    return {
        'avg_fps': fps_avg,
        'peak_fps': fps_max,
        'claimed_fps': 6812
    }


if __name__ == "__main__":
    print("="*80)
    print("V3 CACHING THOROUGH VERIFICATION")
    print("="*80)

    # Run all tests
    accuracy_results = test_detection_accuracy()
    cache_results = test_cache_behavior()
    realistic_results = test_realistic_performance()
    hash_time = test_frame_hashing()
    motion_works = test_motion_prediction()
    fps_verification = verify_extreme_fps_claim()

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERIFICATION RESULTS")
    print("="*80)

    print("\n✅ VERIFIED:")
    print(f"  • Cache mechanism works: {cache_results['cache_worked']}")
    print(f"  • Detection accuracy maintained: {all(r['match'] for r in accuracy_results)}")
    print(f"  • Motion prediction functional: {motion_works}")
    print(f"  • Hash computation fast: {hash_time:.3f}ms")

    print("\n📊 PERFORMANCE REALITY CHECK:")
    print(f"  • Realistic speedup: {realistic_results['speedup']:.2f}x")
    print(f"  • Realistic FPS: {realistic_results['cached_fps']:.1f}")
    print(f"  • Cache hit rate: {realistic_results['cache_hit_rate']:.2%}")

    print("\n⚠️ FPS CLAIM ANALYSIS:")
    print(f"  • Claimed: 6812 FPS")
    print(f"  • Measured avg: {fps_verification['avg_fps']:.1f} FPS")
    print(f"  • Measured peak: {fps_verification['peak_fps']:.1f} FPS")

    if fps_verification['avg_fps'] > 6000:
        print("  • Verdict: ✅ CLAIM VERIFIED")
    elif fps_verification['avg_fps'] > 1000:
        print("  • Verdict: ⚠️ VERY FAST but claim EXAGGERATED")
    else:
        print("  • Verdict: ❌ CLAIM FALSE")

    print("\n🔍 CONCLUSION:")
    if realistic_results['speedup'] > 1.5:
        print(f"  V3 Caching provides REAL {realistic_results['speedup']:.1f}x speedup")
        print("  Performance gains are LEGITIMATE but context-dependent")
    else:
        print("  V3 Caching gains are MARGINAL in realistic scenarios")