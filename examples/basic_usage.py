#!/usr/bin/env python3
"""
RealityGuard Basic Usage Example
Demonstrates simple video and image processing with AI replacement
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from realityguard_ai_replacement import (
    RealityGuardAIReplacement,
    AIReplacementConfig,
    ReplacementMode
)


def process_single_image():
    """Process a single image with AI replacement"""
    print("\n=== Processing Single Image ===")

    # Configure system for high quality
    config = AIReplacementConfig(
        default_mode=ReplacementMode.SYNTHETIC_FACE,
        generation_quality=0.8,
        preserve_context=True
    )

    # Initialize system
    system = RealityGuardAIReplacement(config)

    # Load image
    image_path = "test_image.jpg"
    if Path(image_path).exists():
        image = cv2.imread(image_path)

        # Process image
        protected_image, stats = system.process_frame(image)

        # Save result
        output_path = "protected_image.jpg"
        cv2.imwrite(output_path, protected_image)

        print(f"✓ Image processed successfully")
        print(f"  Output: {output_path}")
        print(f"  Processing time: {1000/stats['fps']:.1f}ms")
        print(f"  Replacements: {stats['replacements']}")
    else:
        print(f"  Image not found: {image_path}")
        print("  Creating synthetic test image...")

        # Create test image
        test_image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        protected_image, stats = system.process_frame(test_image)

        print(f"✓ Test image processed")
        print(f"  FPS: {stats['fps']:.1f}")


def process_video_file():
    """Process a video file with AI replacement"""
    print("\n=== Processing Video File ===")

    # Configure for balanced performance
    config = AIReplacementConfig(
        default_mode=ReplacementMode.SYNTHETIC_FACE,
        detection_interval=2,  # Detect every 2nd frame
        batch_size=8,
        cache_size=300,
        adaptive_quality=True,
        target_fps=30
    )

    # Initialize system
    system = RealityGuardAIReplacement(config)

    # Process video
    input_video = "input.mp4"
    output_video = "output_protected.mp4"

    if Path(input_video).exists():
        print(f"Processing: {input_video}")
        stats = system.process_video(input_video, output_video)

        print(f"✓ Video processed successfully")
        print(f"  Output: {output_video}")
        print(f"  Average FPS: {stats['avg_fps']:.1f}")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Cache efficiency: {stats['cache_efficiency']:.1%}")
    else:
        print(f"  Video not found: {input_video}")
        print("  Please provide an input video file")


def process_webcam_stream():
    """Process live webcam stream"""
    print("\n=== Processing Webcam Stream ===")
    print("Press 'q' to quit, 's' to save screenshot")

    # Configure for real-time performance
    config = AIReplacementConfig(
        default_mode=ReplacementMode.SYNTHETIC_FACE,
        detection_interval=3,  # Less frequent detection
        batch_size=4,
        use_half_precision=True,
        adaptive_quality=True,
        target_fps=30
    )

    # Initialize system
    system = RealityGuardAIReplacement(config)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Error: Cannot access webcam")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    screenshot_count = 0
    frame_count = 0
    fps_history = []

    print("  Webcam started. Processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        protected_frame, stats = system.process_frame(frame)

        # Update FPS history
        fps_history.append(stats['fps'])
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Add overlay
        overlay = protected_frame.copy()
        cv2.putText(overlay, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, f"Mode: {config.default_mode.value}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"Replacements: {stats['replacements']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        cv2.imshow('RealityGuard - Protected Stream', overlay)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_path = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(screenshot_path, protected_frame)
            print(f"  Screenshot saved: {screenshot_path}")
            screenshot_count += 1

        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Session ended")
    print(f"  Frames processed: {frame_count}")
    print(f"  Average FPS: {avg_fps:.1f}")


def test_different_modes():
    """Test different replacement modes"""
    print("\n=== Testing Different Replacement Modes ===")

    # Create test image
    test_image = np.ones((720, 1280, 3), dtype=np.uint8) * 128

    modes = [
        ReplacementMode.SYNTHETIC_FACE,
        ReplacementMode.CONTEXTUAL,
        ReplacementMode.GENERIC_PERSON,
        ReplacementMode.SMART_BLUR,
        ReplacementMode.ABSTRACT
    ]

    for mode in modes:
        print(f"\nTesting mode: {mode.value}")

        config = AIReplacementConfig(
            default_mode=mode,
            generation_quality=0.7
        )

        system = RealityGuardAIReplacement(config)
        protected, stats = system.process_frame(test_image)

        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Processing time: {1000/stats['fps']:.1f}ms")

        # Save result
        output_path = f"test_{mode.value}.jpg"
        cv2.imwrite(output_path, protected)
        print(f"  Saved: {output_path}")


def benchmark_performance():
    """Benchmark system performance"""
    print("\n=== Performance Benchmark ===")

    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p")
    ]

    config = AIReplacementConfig(
        default_mode=ReplacementMode.SYNTHETIC_FACE,
        batch_size=8,
        use_half_precision=True,
        cache_size=300
    )

    system = RealityGuardAIReplacement(config)

    for width, height, name in resolutions:
        print(f"\nTesting {name} ({width}x{height})")

        # Create test frame
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Warm up
        for _ in range(5):
            system.process_frame(frame)

        # Benchmark
        fps_values = []
        for _ in range(20):
            _, stats = system.process_frame(frame)
            fps_values.append(stats['fps'])

        avg_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)

        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Min FPS: {min_fps:.1f}")
        print(f"  Max FPS: {max_fps:.1f}")
        print(f"  Processing time: {1000/avg_fps:.1f}ms")


def main():
    """Run all examples"""
    print("=" * 50)
    print("RealityGuard Usage Examples")
    print("=" * 50)

    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU: Not available (running on CPU)")

    # Run examples
    try:
        # Basic operations
        process_single_image()
        test_different_modes()

        # Performance testing
        benchmark_performance()

        # Interactive examples (comment out if running non-interactively)
        # process_video_file()
        # process_webcam_stream()

    except Exception as e:
        print(f"\nError in examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()