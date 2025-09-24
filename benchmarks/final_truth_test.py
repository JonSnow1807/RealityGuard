"""
FINAL TRUTH TEST
Focusing on the most reliable measurement method
"""

import numpy as np
import cv2
import time


def baseline_blur(frame):
    """Standard Gaussian blur"""
    return cv2.GaussianBlur(frame, (31, 31), 10)


def neural_approximation(frame):
    """Downsample -> Blur -> Upsample"""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w//8, h//8), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(small, (5, 5), 2)
    output = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return output


def measure_accurately(func, frame, iterations=100):
    """Most accurate measurement using perf_counter"""
    # Warmup
    for _ in range(10):
        _ = func(frame)

    # Measure individual times
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = func(frame)
        end = time.perf_counter()
        times.append(end - start)

    # Remove outliers (top and bottom 10%)
    times.sort()
    trimmed = times[int(len(times)*0.1):int(len(times)*0.9)]

    mean_time = np.mean(trimmed)
    fps = 1.0 / mean_time

    return {
        'fps': fps,
        'mean_ms': mean_time * 1000,
        'min_ms': min(trimmed) * 1000,
        'max_ms': max(trimmed) * 1000,
        'std_ms': np.std(trimmed) * 1000
    }


def verify_blur(original, blurred):
    """Check if blur was actually applied"""
    # Variance check
    orig_var = np.var(original)
    blur_var = np.var(blurred)
    variance_reduced = (orig_var - blur_var) / orig_var * 100 if orig_var > 0 else 0

    # Edge check using Sobel
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    orig_edges = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1)
    blur_edges = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 1)

    edge_var_orig = np.var(orig_edges)
    edge_var_blur = np.var(blur_edges)
    edge_reduction = (edge_var_orig - edge_var_blur) / edge_var_orig * 100 if edge_var_orig > 0 else 0

    return {
        'variance_reduced_%': variance_reduced,
        'edge_reduced_%': edge_reduction,
        'blur_applied': variance_reduced > 5 or edge_reduction > 10
    }


def main():
    print("=" * 70)
    print("FINAL TRUTH TEST - Most Accurate Measurements")
    print("=" * 70)

    # Test at different resolutions
    resolutions = [
        ('480p', (480, 854)),
        ('720p', (720, 1280)),
        ('1080p', (1080, 1920))
    ]

    all_results = []

    for res_name, (h, w) in resolutions:
        print(f"\n{res_name} Resolution ({w}x{h})")
        print("-" * 50)

        # Create test frames
        frames = {
            'random': np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            'photo_like': np.ones((h, w, 3), dtype=np.uint8) * 128  # Uniform gray
        }

        # Add some structure to photo_like
        cv2.rectangle(frames['photo_like'], (w//4, h//4), (3*w//4, 3*h//4), (200, 150, 100), -1)
        cv2.circle(frames['photo_like'], (w//2, h//2), min(h,w)//6, (100, 150, 200), -1)

        for frame_type, frame in frames.items():
            print(f"\n  {frame_type.upper()} frame:")

            # Test baseline
            baseline_result = measure_accurately(baseline_blur, frame, iterations=100)
            baseline_verify = verify_blur(frame, baseline_blur(frame))

            print(f"    Baseline:")
            print(f"      FPS: {baseline_result['fps']:.1f}")
            print(f"      Time: {baseline_result['mean_ms']:.2f} ± {baseline_result['std_ms']:.2f} ms")
            print(f"      Blur: {'✓' if baseline_verify['blur_applied'] else '✗'} "
                  f"(var: {baseline_verify['variance_reduced_%']:.1f}%, "
                  f"edge: {baseline_verify['edge_reduced_%']:.1f}%)")

            # Test neural approximation
            neural_result = measure_accurately(neural_approximation, frame, iterations=100)
            neural_verify = verify_blur(frame, neural_approximation(frame))

            print(f"    Neural Approx:")
            print(f"      FPS: {neural_result['fps']:.1f}")
            print(f"      Time: {neural_result['mean_ms']:.2f} ± {neural_result['std_ms']:.2f} ms")
            print(f"      Blur: {'✓' if neural_verify['blur_applied'] else '✗'} "
                  f"(var: {neural_verify['variance_reduced_%']:.1f}%, "
                  f"edge: {neural_verify['edge_reduced_%']:.1f}%)")

            # Speedup
            speedup = neural_result['fps'] / baseline_result['fps']
            print(f"    Speedup: {speedup:.2f}x")

            all_results.append({
                'resolution': res_name,
                'frame_type': frame_type,
                'baseline_fps': baseline_result['fps'],
                'neural_fps': neural_result['fps'],
                'speedup': speedup,
                'baseline_blur': baseline_verify['blur_applied'],
                'neural_blur': neural_verify['blur_applied']
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Calculate averages
    baseline_fps_all = [r['baseline_fps'] for r in all_results]
    neural_fps_all = [r['neural_fps'] for r in all_results]

    print(f"\nAverage FPS across all tests:")
    print(f"  Baseline: {np.mean(baseline_fps_all):.1f} FPS")
    print(f"  Neural:   {np.mean(neural_fps_all):.1f} FPS")
    print(f"  Speedup:  {np.mean(neural_fps_all) / np.mean(baseline_fps_all):.2f}x")

    # Check blur success rate
    baseline_blur_success = sum(r['baseline_blur'] for r in all_results) / len(all_results) * 100
    neural_blur_success = sum(r['neural_blur'] for r in all_results) / len(all_results) * 100

    print(f"\nBlur Success Rate:")
    print(f"  Baseline: {baseline_blur_success:.0f}%")
    print(f"  Neural:   {neural_blur_success:.0f}%")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    avg_neural = np.mean(neural_fps_all)

    if avg_neural > 1000 and neural_blur_success >= 80:
        print(f"✅ Neural Approximation WORKS: {avg_neural:.1f} FPS average")
        print(f"   Blur success rate: {neural_blur_success:.0f}%")
    else:
        print(f"⚠️  Neural Approximation: {avg_neural:.1f} FPS")
        print(f"   Blur success rate: {neural_blur_success:.0f}%")

        if avg_neural > 1000:
            print("   FPS target met but blur quality issues on some frames")
        else:
            print("   FPS below 1000 target")

    print("\nNOTE: Gradient/uniform frames don't show variance reduction")
    print("because there's no high-frequency content to blur.")
    print("This is expected behavior, not a failure.")


if __name__ == "__main__":
    main()