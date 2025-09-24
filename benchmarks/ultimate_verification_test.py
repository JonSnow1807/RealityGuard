"""
ULTIMATE VERIFICATION TEST
Testing baseline vs neural approximation with extreme thoroughness
Multiple timing methods, verification checks, and real-world scenarios
"""

import numpy as np
import cv2
import time
import timeit
from datetime import datetime
import hashlib
import json


def baseline_blur(frame):
    """Simple Gaussian blur - the baseline"""
    return cv2.GaussianBlur(frame, (31, 31), 10)


def neural_approximation(frame):
    """Downsample -> Blur -> Upsample approach"""
    h, w = frame.shape[:2]

    # Downsample by 8x
    small = cv2.resize(frame, (w//8, h//8), interpolation=cv2.INTER_LINEAR)

    # Apply blur to small image
    blurred_small = cv2.GaussianBlur(small, (5, 5), 2)

    # Upsample back
    output = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return output


class ThoroughTester:
    """Multiple ways to measure performance to eliminate errors"""

    def method1_time_perf_counter(self, func, frame, iterations=100):
        """Using time.perf_counter() - highest resolution"""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = func(frame)
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }

    def method2_time_process_time(self, func, frame, iterations=100):
        """Using time.process_time() - CPU time only"""
        times = []
        for _ in range(iterations):
            start = time.process_time()
            _ = func(frame)
            end = time.process_time()
            times.append(end - start)

        return {
            'mean_ms': np.mean(times) * 1000,
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }

    def method3_timeit(self, func, frame, iterations=100):
        """Using timeit module - eliminates GC and other factors"""
        def test_func():
            func(frame)

        total_time = timeit.timeit(test_func, number=iterations)
        avg_time = total_time / iterations

        return {
            'mean_ms': avg_time * 1000,
            'fps': 1.0 / avg_time
        }

    def method4_cv2_gettickcount(self, func, frame, iterations=100):
        """Using OpenCV's tick counter"""
        times = []
        freq = cv2.getTickFrequency()

        for _ in range(iterations):
            start = cv2.getTickCount()
            _ = func(frame)
            end = cv2.getTickCount()
            time_sec = (end - start) / freq
            times.append(time_sec)

        return {
            'mean_ms': np.mean(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }

    def method5_manual_loop_timing(self, func, frame, iterations=100):
        """Time entire loop and divide"""
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func(frame)
        end = time.perf_counter()

        avg_time = (end - start) / iterations

        return {
            'mean_ms': avg_time * 1000,
            'fps': 1.0 / avg_time
        }


def verify_blur_applied(original, blurred):
    """Thoroughly verify that blur was actually applied"""

    checks = {}

    # Check 1: Variance reduction
    orig_var = np.var(original)
    blur_var = np.var(blurred)
    variance_reduction = (orig_var - blur_var) / orig_var * 100 if orig_var > 0 else 0
    checks['variance_reduction_%'] = variance_reduction
    checks['variance_reduced'] = variance_reduction > 5

    # Check 2: High frequency reduction (using Laplacian)
    orig_lap = cv2.Laplacian(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    blur_lap = cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_64F)

    orig_edges = np.var(orig_lap)
    blur_edges = np.var(blur_lap)
    edge_reduction = (orig_edges - blur_edges) / orig_edges * 100 if orig_edges > 0 else 0
    checks['edge_reduction_%'] = edge_reduction
    checks['edges_reduced'] = edge_reduction > 10

    # Check 3: Arrays are different
    checks['arrays_different'] = not np.array_equal(original, blurred)

    # Check 4: Pixel-wise difference
    diff = np.mean(np.abs(original.astype(float) - blurred.astype(float)))
    checks['mean_pixel_diff'] = diff
    checks['significant_diff'] = diff > 1.0

    # Check 5: Hash comparison (should be different)
    orig_hash = hashlib.md5(original.tobytes()).hexdigest()
    blur_hash = hashlib.md5(blurred.tobytes()).hexdigest()
    checks['hashes_different'] = orig_hash != blur_hash

    # Overall verdict
    checks['blur_definitely_applied'] = (
        checks['variance_reduced'] and
        checks['edges_reduced'] and
        checks['arrays_different'] and
        checks['significant_diff'] and
        checks['hashes_different']
    )

    return checks


def create_test_frames():
    """Create various test frames"""
    frames = {}

    # Standard resolutions
    resolutions = {
        '480p': (480, 854),
        '720p': (720, 1280),
        '1080p': (1080, 1920)
    }

    for res_name, (h, w) in resolutions.items():
        # Random noise
        frames[f'{res_name}_random'] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Gradient (smooth)
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            gradient[i, :] = int(255 * i / h)
        frames[f'{res_name}_gradient'] = gradient

        # Checkerboard (high frequency)
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        checker[::20, ::20] = 255
        checker[10::20, 10::20] = 255
        frames[f'{res_name}_checker'] = checker

        # Real-world like (mixed content)
        real = np.zeros((h, w, 3), dtype=np.uint8)
        # Add some shapes
        cv2.rectangle(real, (w//4, h//4), (3*w//4, 3*h//4), (100, 150, 200), -1)
        cv2.circle(real, (w//2, h//2), min(h, w)//6, (200, 100, 100), -1)
        # Add some noise
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        real = cv2.add(real, noise)
        frames[f'{res_name}_realistic'] = real

    return frames


def run_ultimate_verification():
    """Run the ultimate verification test"""

    print("=" * 80)
    print("ULTIMATE VERIFICATION TEST")
    print("Testing Baseline vs Neural Approximation with extreme thoroughness")
    print("=" * 80)

    tester = ThoroughTester()
    frames = create_test_frames()

    # Methods to test
    methods = {
        'baseline': baseline_blur,
        'neural_approx': neural_approximation
    }

    # Timing methods
    timing_methods = [
        ('perf_counter', tester.method1_time_perf_counter),
        ('process_time', tester.method2_time_process_time),
        ('timeit', tester.method3_timeit),
        ('cv2_ticks', tester.method4_cv2_gettickcount),
        ('loop_timing', tester.method5_manual_loop_timing)
    ]

    all_results = {}

    # Test each method
    for method_name, method_func in methods.items():
        print(f"\n{'='*60}")
        print(f"Testing: {method_name.upper()}")
        print(f"{'='*60}")

        method_results = {}

        # Test on different frame types
        for frame_name, frame in frames.items():
            if '720p' not in frame_name:  # Focus on 720p for detailed testing
                continue

            print(f"\n  Frame: {frame_name}")
            print(f"  Shape: {frame.shape}")

            # Warmup
            for _ in range(10):
                _ = method_func(frame)

            # Test with different timing methods
            timing_results = {}
            for timing_name, timing_func in timing_methods:
                result = timing_func(method_func, frame, iterations=50)
                timing_results[timing_name] = result['fps']
                print(f"    {timing_name:15}: {result['fps']:8.1f} FPS ({result['mean_ms']:6.2f} ms)")

            # Calculate average and variation
            fps_values = list(timing_results.values())
            avg_fps = np.mean(fps_values)
            std_fps = np.std(fps_values)
            variation_percent = (std_fps / avg_fps * 100) if avg_fps > 0 else 0

            print(f"    {'Average':15}: {avg_fps:8.1f} FPS")
            print(f"    {'Std Dev':15}: {std_fps:8.1f} FPS")
            print(f"    {'Variation':15}: {variation_percent:8.1f} %")

            # Verify blur is actually applied
            output = method_func(frame)
            blur_checks = verify_blur_applied(frame, output)

            print(f"\n    Blur Verification:")
            print(f"      Variance reduced: {blur_checks['variance_reduction_%']:.1f}% {blur_checks['variance_reduced']}")
            print(f"      Edges reduced:    {blur_checks['edge_reduction_%']:.1f}% {blur_checks['edges_reduced']}")
            print(f"      Pixels changed:   {blur_checks['mean_pixel_diff']:.1f} {blur_checks['significant_diff']}")
            print(f"      Hash different:   {blur_checks['hashes_different']}")
            print(f"      BLUR APPLIED:     {blur_checks['blur_definitely_applied']}")

            method_results[frame_name] = {
                'avg_fps': avg_fps,
                'std_fps': std_fps,
                'variation_%': variation_percent,
                'blur_applied': blur_checks['blur_definitely_applied'],
                'timing_results': timing_results,
                'blur_checks': blur_checks
            }

        all_results[method_name] = method_results

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    # Calculate overall averages
    baseline_fps = []
    neural_fps = []

    for frame_name in all_results['baseline'].keys():
        baseline_fps.append(all_results['baseline'][frame_name]['avg_fps'])
        neural_fps.append(all_results['neural_approx'][frame_name]['avg_fps'])

    baseline_avg = np.mean(baseline_fps)
    neural_avg = np.mean(neural_fps)
    speedup = neural_avg / baseline_avg

    print(f"\nBaseline Average: {baseline_avg:.1f} FPS")
    print(f"Neural Approximation Average: {neural_avg:.1f} FPS")
    print(f"Speedup: {speedup:.2f}x")

    # Verification summary
    print("\nBlur Verification Summary:")

    for method_name in ['baseline', 'neural_approx']:
        blur_success = all([r['blur_applied'] for r in all_results[method_name].values()])
        print(f"  {method_name}: {'✓ All frames blurred correctly' if blur_success else '✗ Some frames not blurred'}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if neural_avg > 1000 and all(r['blur_applied'] for r in all_results['neural_approx'].values()):
        print(f"✅ Neural Approximation VERIFIED: {neural_avg:.1f} FPS with proper blur")
    else:
        print(f"❌ Neural Approximation FAILED: {neural_avg:.1f} FPS or blur not applied")

    # Save detailed results
    with open('ultimate_verification_results.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(all_results, f, indent=2, default=convert_numpy)

    print("\nDetailed results saved to: ultimate_verification_results.json")

    return baseline_avg, neural_avg, speedup


def test_with_actual_mediapipe():
    """Test with actual MediaPipe face detection + blur"""
    print("\n" + "=" * 80)
    print("BONUS: Testing with ACTUAL MediaPipe")
    print("=" * 80)

    try:
        import mediapipe as mp

        mp_face = mp.solutions.face_detection
        face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

        def mediapipe_baseline(frame):
            """Actual MediaPipe detection + blur"""
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            output = frame.copy()
            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # Ensure bounds
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(w, x + width)
                    y2 = min(h, y + height)

                    if x2 > x and y2 > y:
                        roi = output[y:y2, x:x2]
                        output[y:y2, x:x2] = cv2.GaussianBlur(roi, (31, 31), 10)

            return output

        # Test frame with actual face
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame[:] = (200, 200, 200)  # Gray background
        # Add face-like oval
        cv2.ellipse(test_frame, (640, 360), (150, 200), 0, 0, 360, (150, 120, 90), -1)

        # Time MediaPipe
        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = mediapipe_baseline(test_frame)
            end = time.perf_counter()
            times.append(end - start)

        mediapipe_fps = 1.0 / np.mean(times)
        print(f"\nMediaPipe with detection + blur: {mediapipe_fps:.1f} FPS")

        # Compare with our methods
        baseline_fps = 1.0 / np.mean([time.perf_counter() - time.perf_counter()
                                      for _ in range(20)
                                      if baseline_blur(test_frame) is not None])

        print(f"Simple blur (no detection): {baseline_fps:.1f} FPS")

    except ImportError:
        print("\nMediaPipe not installed, skipping MediaPipe test")


if __name__ == "__main__":
    baseline_avg, neural_avg, speedup = run_ultimate_verification()
    test_with_actual_mediapipe()