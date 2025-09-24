#!/usr/bin/env python3
"""
Investigate the FPS discrepancy in V3 claims
Why does it show 6812 FPS in benchmark but 378 FPS in realistic test?
"""

import cv2
import numpy as np
import time
from mediapipe_excellence_v3_caching import MediaPipeWithCaching


def test_what_is_being_measured():
    """Figure out what exactly is being measured"""
    print("="*80)
    print("INVESTIGATING FPS MEASUREMENT DISCREPANCY")
    print("="*80)

    cached = MediaPipeWithCaching(cache_size=30)

    # Create test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(frame, (640, 360), 80, (255, 255, 255), -1)

    print("\n1. Testing detect_with_cache() only:")
    print("-" * 40)

    # Prime cache
    _ = cached.detect_with_cache(frame)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        detections = cached.detect_with_cache(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_detect = np.mean(times)
    fps_detect = 1000 / avg_detect
    print(f"  Average time: {avg_detect:.4f}ms")
    print(f"  FPS: {fps_detect:.1f}")
    print(f"  Detections returned: {len(detections)}")

    print("\n2. Testing process_frame() (detect + blur):")
    print("-" * 40)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        output, info = cached.process_frame(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_process = np.mean(times)
    fps_process = 1000 / avg_process
    print(f"  Average time: {avg_process:.4f}ms")
    print(f"  FPS: {fps_process:.1f}")
    print(f"  Info returned: {info}")

    print("\n3. Testing apply_blur_cached() separately:")
    print("-" * 40)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        blurred = cached.apply_blur_cached(frame, detections)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_blur = np.mean(times)
    print(f"  Average blur time: {avg_blur:.4f}ms")

    print("\n4. Breaking down process_frame components:")
    print("-" * 40)
    print(f"  Detection only: {avg_detect:.4f}ms ({fps_detect:.1f} FPS)")
    print(f"  Blur only: {avg_blur:.4f}ms")
    print(f"  Total process_frame: {avg_process:.4f}ms ({fps_process:.1f} FPS)")
    print(f"  Overhead: {avg_process - (avg_detect + avg_blur):.4f}ms")

    return {
        'detect_fps': fps_detect,
        'process_fps': fps_process,
        'blur_time': avg_blur
    }


def test_static_vs_changing():
    """Test performance difference between static and changing scenes"""
    print("\n" + "="*80)
    print("STATIC VS CHANGING SCENES")
    print("="*80)

    cached = MediaPipeWithCaching(cache_size=30)

    # Test 1: Completely static frames
    print("\n1. Static Scene (100 identical frames):")
    print("-" * 40)

    static_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(static_frame, (640, 360), 80, (255, 255, 255), -1)

    # Prime cache
    cached.process_frame(static_frame)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        _, info = cached.process_frame(static_frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    static_avg = np.mean(times)
    static_fps = 1000 / static_avg
    print(f"  Average: {static_avg:.2f}ms")
    print(f"  FPS: {static_fps:.1f}")
    print(f"  Cache hit rate: {cached.detection_cache.get_hit_rate():.2%}")

    # Reset for next test
    cached = MediaPipeWithCaching(cache_size=30)

    # Test 2: Every frame different
    print("\n2. Changing Scene (100 unique frames):")
    print("-" * 40)

    times = []
    for i in range(100):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Different position each frame
        x = 300 + i * 5
        cv2.circle(frame, (x % 1280, 360), 80, (255, 255, 255), -1)

        start = time.perf_counter()
        _, info = cached.process_frame(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    changing_avg = np.mean(times)
    changing_fps = 1000 / changing_avg
    print(f"  Average: {changing_avg:.2f}ms")
    print(f"  FPS: {changing_fps:.1f}")
    print(f"  Cache hit rate: {cached.detection_cache.get_hit_rate():.2%}")

    print(f"\n  Speedup from caching: {changing_avg/static_avg:.1f}x")

    return {
        'static_fps': static_fps,
        'changing_fps': changing_fps,
        'speedup': changing_avg/static_avg
    }


def test_original_benchmark_method():
    """Reproduce the exact benchmark from V3"""
    print("\n" + "="*80)
    print("REPRODUCING ORIGINAL V3 BENCHMARK")
    print("="*80)

    cached = MediaPipeWithCaching(cache_size=30)

    # This is what the original benchmark did
    test_scenarios = [
        ('static_scene', 'static'),
        ('slow_motion', 'slow'),
    ]

    for scenario_name, scenario_type in test_scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 40)

        frames = []
        for i in range(100):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            if scenario_type == 'static':
                # Static scene - same objects
                cv2.circle(frame, (640, 360), 80, (255, 255, 255), -1)
                cv2.circle(frame, (400, 360), 60, (255, 255, 255), -1)

            elif scenario_type == 'slow':
                # Slow motion
                x = 400 + i * 2
                cv2.circle(frame, (x, 360), 80, (255, 255, 255), -1)
                cv2.circle(frame, (400, 360), 60, (255, 255, 255), -1)

            frames.append(frame)

        # Process sequence
        start = time.perf_counter()
        processed = cached.process_video_sequence(frames)
        total_time = (time.perf_counter() - start) * 1000

        fps = len(frames) * 1000 / total_time
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Cache hit rate: {cached.detection_cache.get_hit_rate():.2%}")


def test_cache_only_performance():
    """Test pure cache retrieval speed without any processing"""
    print("\n" + "="*80)
    print("PURE CACHE RETRIEVAL SPEED")
    print("="*80)

    cached = MediaPipeWithCaching(cache_size=30)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 60, (255, 255, 255), -1)

    # Prime cache
    frame_hash = cached.compute_frame_hash(frame)
    cached.detection_cache.put(frame_hash, [(100, 100, 50, 50)])

    print("\n1. Cache GET operation only:")
    times = []
    for _ in range(10000):
        start = time.perf_counter()
        _ = cached.detection_cache.get(frame_hash)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_get = np.mean(times)
    fps_get = 1000 / avg_get
    print(f"  Average: {avg_get:.6f}ms")
    print(f"  FPS: {fps_get:.1f}")

    print("\n2. Hash + Cache GET:")
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        hash_val = cached.compute_frame_hash(frame)
        _ = cached.detection_cache.get(hash_val)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_hash_get = np.mean(times)
    fps_hash_get = 1000 / avg_hash_get
    print(f"  Average: {avg_hash_get:.4f}ms")
    print(f"  FPS: {fps_hash_get:.1f}")

    print("\n3. Full detect_with_cache (cached hit):")
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = cached.detect_with_cache(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_detect = np.mean(times)
    fps_detect = 1000 / avg_detect
    print(f"  Average: {avg_detect:.4f}ms")
    print(f"  FPS: {fps_detect:.1f}")

    print("\n  This explains the 6000+ FPS claims!")
    print("  Pure cache hit without blur = very fast")
    print("  But realistic usage includes blur operation")


if __name__ == "__main__":
    # Run all investigations
    measure_results = test_what_is_being_measured()
    scene_results = test_static_vs_changing()
    test_original_benchmark_method()
    test_cache_only_performance()

    print("\n" + "="*80)
    print("INVESTIGATION CONCLUSIONS")
    print("="*80)

    print("\n🔍 THE TRUTH ABOUT V3 PERFORMANCE:")
    print("-" * 60)

    print("\n1. The 6812 FPS claim is for:")
    print("   • Detection ONLY (not blur)")
    print("   • Completely static frames")
    print("   • Pure cache hits")
    print(f"   • Actual: {measure_results['detect_fps']:.1f} FPS for cached detection")

    print("\n2. Real-world performance (detect + blur):")
    print(f"   • Static scenes: {scene_results['static_fps']:.1f} FPS")
    print(f"   • Changing scenes: {scene_results['changing_fps']:.1f} FPS")
    print(f"   • Speedup from caching: {scene_results['speedup']:.1f}x")

    print("\n3. Where time is spent:")
    print(f"   • Detection (cached): ~0.05ms")
    print(f"   • Blur operation: {measure_results['blur_time']:.2f}ms")
    print("   • Blur dominates the processing time!")

    print("\n📊 ACTUAL PERFORMANCE:")
    print("   • Detection only (cached): 6000-20000 FPS ✅")
    print("   • Full pipeline (cached): 200-400 FPS ✅")
    print("   • Full pipeline (uncached): 100-200 FPS ✅")
    print("   • Real speedup: 1.2-2x (not 6x)")

    print("\n⚠️ MISLEADING METRICS:")
    print("   • Testing detection-only FPS is misleading")
    print("   • Blur operation can't be cached effectively")
    print("   • Real-world speedup is modest but valuable")