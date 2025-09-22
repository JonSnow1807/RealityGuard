"""
BRUTAL HONEST VERIFICATION
No tricks, no shortcuts, actual privacy processing
Tests the REAL system with REAL operations
"""

import torch
import numpy as np
import time
import sys
import importlib
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def create_test_frame_with_faces(frame_num):
    """Create a frame with actual face-like regions that need blurring"""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Add background
    frame[:] = [50, 50, 50]

    # Add face-like regions (skin color)
    # Face 1
    cv2.ellipse(frame, (200 + frame_num % 100, 200), (80, 100), 0, 0, 360, (210, 180, 140), -1)
    cv2.circle(frame, (170 + frame_num % 100, 180), 10, (50, 50, 50), -1)  # eye
    cv2.circle(frame, (230 + frame_num % 100, 180), 10, (50, 50, 50), -1)  # eye

    # Face 2
    cv2.ellipse(frame, (600, 400 + frame_num % 50), (70, 90), 0, 0, 360, (190, 160, 120), -1)

    # Add screen-like region
    cv2.rectangle(frame, (800, 200), (1100, 500), (100, 100, 200), -1)
    cv2.putText(frame, "PRIVATE DATA", (850, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add noise for realism
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

    return frame

def verify_blur_actually_applied(original, processed):
    """Check if blur was actually applied to the frame"""
    # Calculate difference
    diff = cv2.absdiff(original, processed)

    # Check if there's significant difference (blur applied)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    blur_applied = mean_diff > 5 or max_diff > 50

    return blur_applied, mean_diff, max_diff

def test_single_implementation(module_name, class_name, frames):
    """Test a single implementation with REAL processing"""
    print(f"\n{'='*60}")
    print(f"Testing: {module_name}.{class_name}")
    print(f"{'='*60}")

    try:
        # Import and instantiate
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # Create instance
        if module_name == 'realityguard_optimized':
            from realityguard_optimized import Config
            system = cls(Config())
        else:
            system = cls()

        # Find the process method
        process_func = None
        for method_name in ['process_frame', 'process_frame_optimized',
                           'process_frame_ultra_fast', 'process_batch_final',
                           'process_batch_ultimate']:
            if hasattr(system, method_name):
                process_func = getattr(system, method_name)
                break

        if not process_func:
            print(f"❌ No process function found")
            return None

        # Warm up
        print("Warming up...")
        for _ in range(10):
            try:
                _ = process_func(frames[0])
            except:
                pass

        # ACTUAL TEST
        print("Running actual test with real frames...")
        successful = 0
        failed = 0
        blur_detected = 0
        processing_times = []

        # Synchronize before starting
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Wall clock timing (THE TRUTH)
        wall_start = time.perf_counter()

        for i, frame in enumerate(frames):
            frame_start = time.perf_counter()

            try:
                # Process frame
                if 'batch' in process_func.__name__:
                    # Handle batch processing
                    if isinstance(frame, np.ndarray):
                        frame_gpu = torch.from_numpy(frame).cuda().unsqueeze(0)
                    else:
                        frame_gpu = frame.unsqueeze(0) if frame.dim() == 3 else frame
                    result = process_func(frame_gpu)
                    if result is not None:
                        if torch.is_tensor(result):
                            result = result.squeeze(0).cpu().numpy()
                        successful += 1
                else:
                    # Single frame processing
                    result = process_func(frame)
                    if result is not None:
                        successful += 1

                # Verify blur was applied (only for first few frames)
                if i < 5 and result is not None:
                    if isinstance(result, tuple):
                        result = result[0]

                    if isinstance(result, torch.Tensor):
                        result_np = result.cpu().numpy() if result.is_cuda else result.numpy()
                    else:
                        result_np = result

                    # Check if blur was applied
                    if isinstance(frame, torch.Tensor):
                        frame_np = frame.cpu().numpy() if frame.is_cuda else frame.numpy()
                    else:
                        frame_np = frame

                    blur_applied, mean_diff, max_diff = verify_blur_actually_applied(
                        frame_np, result_np
                    )
                    if blur_applied:
                        blur_detected += 1

            except Exception as e:
                failed += 1
                if i == 0:  # Print first error only
                    print(f"  Error: {e}")

            frame_time = time.perf_counter() - frame_start
            processing_times.append(frame_time * 1000)

        # Wait for GPU to finish everything
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_end = time.perf_counter()
        total_wall_time = wall_end - wall_start

        # Calculate REAL metrics
        actual_fps = successful / total_wall_time if total_wall_time > 0 else 0
        avg_latency = np.mean(processing_times) if processing_times else 0

        print(f"\n📊 RESULTS:")
        print(f"  Frames processed: {successful}/{len(frames)}")
        print(f"  Failed: {failed}")
        print(f"  Wall time: {total_wall_time:.2f}s")
        print(f"  ACTUAL FPS: {actual_fps:.1f}")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Blur detected: {blur_detected}/5 samples")

        if actual_fps > 1000:
            print(f"  ⚠️  SUSPICIOUS! Likely not doing real work")

        return {
            'name': f"{module_name}.{class_name}",
            'fps': actual_fps,
            'latency_ms': avg_latency,
            'success_rate': successful / len(frames) * 100,
            'blur_detected': blur_detected,
            'wall_time': total_wall_time
        }

    except Exception as e:
        print(f"❌ Failed completely: {e}")
        return None

def brutal_honest_test():
    """Run the most honest test possible"""
    print("="*60)
    print("🔍 BRUTAL HONEST VERIFICATION")
    print("No tricks, real processing, actual results")
    print("="*60)

    # Create test frames with actual content
    print("\n📹 Creating test frames with faces and screens...")
    frames = [create_test_frame_with_faces(i) for i in range(100)]
    print(f"Created {len(frames)} frames with face and screen regions")

    # Test implementations
    implementations = [
        ('realityguard_1000fps_final', 'RealityGuard1000FPSFinal'),
        ('realityguard_1000fps_ultimate', 'RealityGuard1000FPSUltimate'),
        ('realityguard_1000fps', 'RealityGuard1000FPS'),
        ('realityguard_tensorrt', 'RealityGuardTensorRT'),
        ('patent_algorithm', 'PredictivePrivacyGradient'),
    ]

    results = []

    for module_name, class_name in implementations:
        result = test_single_implementation(module_name, class_name, frames)
        if result:
            results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("📊 BRUTAL HONEST SUMMARY")
    print("="*80)

    if results:
        # Sort by FPS
        results.sort(key=lambda x: x['fps'], reverse=True)

        print(f"\n{'Implementation':<40} {'FPS':<10} {'Latency':<10} {'Blur?':<10}")
        print("-"*70)

        for r in results:
            blur_status = "Yes" if r['blur_detected'] > 0 else "NO!"
            print(f"{r['name']:<40} {r['fps']:<10.1f} {r['latency_ms']:<10.2f} {blur_status:<10}")

        best = results[0]
        print(f"\n📈 Best HONEST performance: {best['fps']:.1f} FPS")

        if best['fps'] >= 1000:
            print("⚠️  WARNING: Even if >1000 FPS, verify it's doing actual work!")
            print("    Check if blur_detected > 0 to confirm processing")

        if best['fps'] < 1000:
            print(f"❌ Honest result: {best['fps']:.1f} FPS (not 6916!)")
            print(f"   Gap to 1000: {1000 - best['fps']:.1f} FPS")

    # Save honest results
    with open("BRUTAL_HONEST_RESULTS.txt", "w") as f:
        f.write("BRUTAL HONEST PERFORMANCE TEST\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test: 100 frames with faces and screens\n\n")

        for r in results:
            f.write(f"{r['name']}:\n")
            f.write(f"  FPS: {r['fps']:.1f}\n")
            f.write(f"  Latency: {r['latency_ms']:.2f}ms\n")
            f.write(f"  Success: {r['success_rate']:.0f}%\n")
            f.write(f"  Blur detected: {r['blur_detected']}/5\n\n")

    print("\n📁 Results saved to BRUTAL_HONEST_RESULTS.txt")

    # Final verdict
    print("\n" + "="*60)
    print("🎯 FINAL HONEST VERDICT")
    print("="*60)

    if results:
        best_fps = max(r['fps'] for r in results)
        best_with_blur = max((r['fps'] for r in results if r['blur_detected'] > 0), default=0)

        if best_with_blur >= 1000:
            print(f"✅ We have {best_with_blur:.0f} FPS with actual blur!")
        else:
            print(f"❌ HONEST FPS with blur: {best_with_blur:.0f}")
            print(f"❌ The 6916 FPS was likely fake/empty operations")

if __name__ == "__main__":
    brutal_honest_test()