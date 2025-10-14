#!/usr/bin/env python3
"""
Interactive Demo for RealityGuard
Shows the system actually working with visual output
"""

import cv2
import numpy as np
import time
from src.realityguard_fixed import RealityGuardFixed
from src.config import PrivacyMode


def create_demo_frame():
    """Create a demo frame with recognizable elements"""
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 60

    # Add "screens" (bright rectangles)
    cv2.rectangle(frame, (100, 100), (500, 400), (240, 240, 240), -1)
    cv2.putText(frame, "MONITOR 1", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(frame, (700, 300), (1150, 600), (250, 250, 250), -1)
    cv2.putText(frame, "LAPTOP SCREEN", (800, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add "sensitive text"
    cv2.putText(frame, "SSN: 123-45-6789", (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Credit Card: 4532-1234-5678-9012", (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add some face-like circles (simplified faces)
    # Face 1
    cv2.circle(frame, (300, 500), 60, (200, 180, 160), -1)
    cv2.circle(frame, (280, 490), 8, (50, 50, 50), -1)
    cv2.circle(frame, (320, 490), 8, (50, 50, 50), -1)

    # Face 2
    cv2.circle(frame, (900, 150), 60, (190, 170, 150), -1)
    cv2.circle(frame, (880, 140), 8, (50, 50, 50), -1)
    cv2.circle(frame, (920, 140), 8, (50, 50, 50), -1)

    return frame


def run_interactive_demo():
    """Run interactive demo with visual output"""
    print("="*60)
    print("REALITYGUARD INTERACTIVE DEMO")
    print("="*60)
    print("\nControls:")
    print("  1-5: Switch privacy modes")
    print("  s: Save screenshot")
    print("  q: Quit")
    print("="*60)

    guard = RealityGuardFixed()
    modes = [PrivacyMode.OFF, PrivacyMode.SMART, PrivacyMode.SOCIAL,
             PrivacyMode.WORKSPACE, PrivacyMode.MAXIMUM]

    current_mode_idx = 0
    guard.set_privacy_mode(modes[current_mode_idx])

    # Performance tracking
    fps_history = []

    while True:
        # Create demo frame
        frame = create_demo_frame()

        # Process with RealityGuard
        start = time.perf_counter()
        processed = guard.process_frame(frame)
        elapsed = time.perf_counter() - start

        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)

        avg_fps = np.mean(fps_history)

        # Add overlay information
        overlay = processed.copy()

        # Add status bar
        cv2.rectangle(overlay, (0, 0), (1280, 80), (0, 0, 0), -1)
        cv2.putText(overlay, f"Mode: {modes[current_mode_idx].name}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"FPS: {avg_fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add description
        descriptions = {
            PrivacyMode.OFF: "No filtering applied",
            PrivacyMode.SMART: "Blur unknown faces only",
            PrivacyMode.SOCIAL: "Blur faces and screens",
            PrivacyMode.WORKSPACE: "Professional privacy",
            PrivacyMode.MAXIMUM: "Maximum privacy protection"
        }
        cv2.putText(overlay, descriptions[modes[current_mode_idx]], (300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Create side-by-side comparison
        comparison = np.hstack([frame, overlay])

        # Add labels
        cv2.putText(comparison, "ORIGINAL", (20, 700),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "FILTERED", (1300, 700),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display
        cv2.imshow('RealityGuard Demo', comparison)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('5'):
            current_mode_idx = key - ord('1')
            guard.set_privacy_mode(modes[current_mode_idx])
            print(f"Switched to {modes[current_mode_idx].name} mode")
        elif key == ord('s'):
            filename = f"screenshot_{modes[current_mode_idx].name}.png"
            cv2.imwrite(filename, comparison)
            print(f"Saved screenshot: {filename}")

    cv2.destroyAllWindows()
    guard.cleanup()

    print(f"\nAverage FPS: {avg_fps:.1f}")


def run_performance_test():
    """Run performance test on different image sizes"""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)

    guard = RealityGuardFixed()
    guard.set_privacy_mode(PrivacyMode.SMART)

    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD")
    ]

    for width, height, name in resolutions:
        frame = create_demo_frame()
        frame = cv2.resize(frame, (width, height))

        times = []
        for _ in range(100):
            start = time.perf_counter()
            result = guard.process_frame(frame)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time

        print(f"{name:10} ({width}x{height}): {avg_time:.2f}ms = {fps:.1f} FPS")

    guard.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_performance_test()
    else:
        print("\nStarting interactive demo...")
        print("(Use --test flag for performance test)\n")
        run_interactive_demo()