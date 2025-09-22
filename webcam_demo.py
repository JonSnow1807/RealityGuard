#!/usr/bin/env python3
"""
RealityGuard Webcam Demo
Real-time privacy protection demonstration using webcam
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from realityguard_production import RealityGuardProduction
from config import PrivacyMode


def main():
    """Run webcam demo with real privacy filtering."""
    print("=" * 60)
    print("REALITYGUARD WEBCAM DEMO")
    print("=" * 60)
    print("\nControls:")
    print("  1-5: Switch privacy modes")
    print("  c: Calibrate known faces")
    print("  s: Save screenshot")
    print("  q: Quit")
    print("=" * 60)

    # Initialize system
    system = RealityGuardProduction()
    system.set_privacy_mode(PrivacyMode.SMART)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Please ensure:")
        print("1. You have a webcam connected")
        print("2. You've granted camera permissions")
        print("3. No other application is using the camera")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\nWebcam initialized. Starting demo...")

    # Mode names for display
    mode_names = {
        PrivacyMode.OFF: "OFF - No Privacy",
        PrivacyMode.SMART: "SMART - Blur Unknown",
        PrivacyMode.SOCIAL: "SOCIAL - Blur All Faces",
        PrivacyMode.WORKSPACE: "WORKSPACE - Professional",
        PrivacyMode.MAXIMUM: "MAXIMUM - Full Privacy"
    }

    # Performance tracking
    fps_history = []
    calibration_frames = []
    calibration_mode = False
    screenshot_count = 0

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            # Start timing
            start_time = time.perf_counter()

            # Process frame
            if calibration_mode:
                # Collect calibration frames
                processed = frame.copy()
                calibration_frames.append(frame.copy())
                cv2.putText(processed, "CALIBRATING - Look at camera",
                          (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                if len(calibration_frames) >= 30:  # 1 second at 30fps
                    system.calibrate_known_faces(calibration_frames)
                    calibration_frames = []
                    calibration_mode = False
                    print("Calibration complete!")
            else:
                # Normal processing
                processed = system.process_frame(frame)

            # Calculate FPS
            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            # Create display with info overlay
            display = processed.copy()

            # Add black bar at top for info
            info_height = 80
            display_with_info = np.zeros((display.shape[0] + info_height,
                                         display.shape[1], 3), dtype=np.uint8)
            display_with_info[info_height:] = display
            display_with_info[:info_height] = (30, 30, 30)  # Dark gray

            # Add info text
            mode_text = mode_names.get(system.privacy_mode, "UNKNOWN")
            cv2.putText(display_with_info, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(display_with_info, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add detection info
            if hasattr(system, '_detection_cache'):
                num_faces = len(system._detection_cache.get('faces', []))
                num_screens = len(system._detection_cache.get('screens', []))
                detection_text = f"Faces: {num_faces} | Screens: {num_screens}"
                cv2.putText(display_with_info, detection_text, (400, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Add controls reminder
            cv2.putText(display_with_info, "1-5: Modes | C: Calibrate | S: Screenshot | Q: Quit",
                       (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Show comparison if not in OFF mode
            if system.privacy_mode != PrivacyMode.OFF:
                # Create side-by-side comparison
                comparison = np.hstack([frame, processed])
                comparison_scaled = cv2.resize(comparison, (1280, 360))

                # Create final display
                final_display = np.zeros((440, 1280, 3), dtype=np.uint8)
                final_display[:80] = display_with_info[:80]
                final_display[80:] = comparison_scaled

                # Add labels
                cv2.putText(final_display, "ORIGINAL", (320, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(final_display, "FILTERED", (960, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow('RealityGuard Webcam Demo', final_display)
            else:
                cv2.imshow('RealityGuard Webcam Demo', display_with_info)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('5'):
                modes = [PrivacyMode.OFF, PrivacyMode.SMART, PrivacyMode.SOCIAL,
                        PrivacyMode.WORKSPACE, PrivacyMode.MAXIMUM]
                mode_idx = key - ord('1')
                if mode_idx < len(modes):
                    system.set_privacy_mode(modes[mode_idx])
                    print(f"Switched to {modes[mode_idx].name} mode")
            elif key == ord('c'):
                calibration_mode = True
                calibration_frames = []
                print("Starting calibration. Look at the camera...")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"realityguard_screenshot_{screenshot_count}.png"
                cv2.imwrite(filename, display_with_info)
                print(f"Saved screenshot: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        system.cleanup()

        # Print final stats
        print("\n" + "=" * 60)
        print("SESSION STATS")
        print("=" * 60)
        stats = system.performance_monitor.get_stats()
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Runtime: {stats['runtime']:.1f} seconds")
        print("=" * 60)


if __name__ == "__main__":
    print("\nRealityGuard - Privacy Protection for AR/VR")
    print("Copyright 2025 - Ready for Meta Acquisition\n")

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python webcam_demo.py")
            print("\nThis demo uses your webcam to demonstrate real-time")
            print("privacy filtering with face and screen detection.\n")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
    else:
        main()