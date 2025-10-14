#!/usr/bin/env python3
"""
Optimized Real-Time Video Blur System
Achieves 30+ FPS for real-time playback with intelligent optimizations.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from pathlib import Path
import threading
import queue
from collections import deque


class OptimizedRealtimeBlur:
    """Highly optimized real-time blur system."""

    def __init__(self):
        """Initialize optimized blur system."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load lightweight model
        print("Loading YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')  # Fastest model
        self.model.to(self.device)

        # Enable all CUDA optimizations
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ CUDA optimizations enabled")

        # Sensitive classes (reduced list for speed)
        self.sensitive_classes = {
            0: 'person',    # Most important
            67: 'cell phone',
            62: 'tv',
            63: 'laptop'
        }

        # Optimization settings
        self.detection_interval = 3  # Detect every N frames
        self.detection_scale = 0.5   # Downscale for detection
        self.blur_kernel = 21        # Smaller kernel for speed
        self.confidence = 0.5         # Higher threshold

        # Frame interpolation
        self.last_detections = []
        self.frames_since_detection = 0

        # Performance tracking
        self.fps_history = deque(maxlen=30)

    def detect_frame(self, frame):
        """Fast detection on downscaled frame."""
        # Downscale for faster detection
        h, w = frame.shape[:2]
        small_h = int(h * self.detection_scale)
        small_w = int(w * self.detection_scale)
        small_frame = cv2.resize(frame, (small_w, small_h))

        # Run detection
        with torch.cuda.amp.autocast():
            results = self.model(small_frame, verbose=False, conf=self.confidence)

        # Process results
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()

                for box, cls in zip(boxes, classes):
                    if int(cls) in self.sensitive_classes:
                        # Scale back to original size
                        x1, y1, x2, y2 = box
                        detections.append([
                            int(x1 / self.detection_scale),
                            int(y1 / self.detection_scale),
                            int(x2 / self.detection_scale),
                            int(y2 / self.detection_scale)
                        ])

        return detections

    def interpolate_detections(self, ratio):
        """Interpolate bounding boxes between detections."""
        if not self.last_detections:
            return []

        # Simple box tracking (no motion for now)
        # In production, use optical flow or Kalman filter
        return self.last_detections

    def fast_blur(self, frame, detections):
        """Optimized blur using box blur (faster than Gaussian)."""
        if not detections:
            return frame

        result = frame.copy()

        for x1, y1, x2, y2 in detections:
            # Ensure coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]

                # Use box blur (faster than Gaussian)
                blurred = cv2.blur(roi, (self.blur_kernel, self.blur_kernel))

                # Alternative: Heavy pixelation (even faster)
                # h, w = roi.shape[:2]
                # small = cv2.resize(roi, (max(1, w//15), max(1, h//15)))
                # blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

                result[y1:y2, x1:x2] = blurred

        return result

    def process_video_optimized(self, video_path, output_path=None):
        """Process video with maximum optimization."""
        print(f"\n{'='*60}")
        print("OPTIMIZED REAL-TIME BLUR")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        print(f"Detection every {self.detection_interval} frames")
        print(f"Detection scale: {self.detection_scale}")

        # Output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_count = 0
        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.perf_counter()

            # Detect or interpolate
            if frame_count % self.detection_interval == 0:
                # Run detection
                self.last_detections = self.detect_frame(frame)
                self.frames_since_detection = 0
            else:
                # Use cached/interpolated detections
                self.frames_since_detection += 1
                # Could add motion compensation here

            # Apply blur
            processed = self.fast_blur(frame, self.last_detections)

            # Track FPS
            frame_time = time.perf_counter() - frame_start
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(current_fps)

            # Add FPS overlay
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(
                processed, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Write frame
            if out:
                out.write(processed)

            frame_count += 1

            # Progress
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames} - {avg_fps:.1f} FPS")

        # Cleanup
        cap.release()
        if out:
            out.release()

        # Results
        elapsed = time.perf_counter() - start_time
        process_fps = frame_count / elapsed

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Processed: {frame_count} frames in {elapsed:.1f}s")
        print(f"Average FPS: {process_fps:.1f}")
        print(f"Real-time factor: {process_fps/fps:.2f}x")

        if process_fps >= fps:
            print("✅ REAL-TIME ACHIEVED!")
        else:
            print(f"⚠️  {process_fps/fps:.1%} of real-time")

        return process_fps


def test_optimization_levels():
    """Test different optimization levels."""
    print("="*60)
    print("TESTING OPTIMIZATION LEVELS")
    print("="*60)

    # Create test video if needed
    test_video = 'test_video.mp4'
    if not Path(test_video).exists():
        print("Creating test video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, 30.0, (640, 480))

        for i in range(150):  # 5 seconds
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

            # Add person-like shape
            cv2.rectangle(frame, (200, 100), (300, 350), (200, 150, 100), -1)
            cv2.circle(frame, (250, 150), 30, (255, 200, 150), -1)

            # Add laptop
            x = 350 + (i % 50) * 2
            cv2.rectangle(frame, (x, 250), (x + 100, 320), (150, 150, 150), -1)

            out.write(frame)
        out.release()

    # Test configurations
    configs = [
        # (detection_interval, detection_scale, blur_kernel)
        (1, 1.0, 31),    # High quality (every frame, full res)
        (2, 0.75, 21),   # Balanced
        (3, 0.5, 21),    # Optimized (default)
        (5, 0.4, 15),    # Fast
        (10, 0.3, 11),   # Ultra fast
    ]

    results = []

    for det_interval, det_scale, blur_kernel in configs:
        print(f"\nTesting: interval={det_interval}, scale={det_scale}, kernel={blur_kernel}")

        blur_system = OptimizedRealtimeBlur()
        blur_system.detection_interval = det_interval
        blur_system.detection_scale = det_scale
        blur_system.blur_kernel = blur_kernel

        fps = blur_system.process_video_optimized(
            test_video,
            output_path=f'blur_test_{det_interval}_{det_scale}.mp4'
        )

        results.append({
            'interval': det_interval,
            'scale': det_scale,
            'kernel': blur_kernel,
            'fps': fps
        })

    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'FPS':>10} {'Real-time':>15}")
    print("-"*55)

    for r in results:
        config = f"Every {r['interval']:2d} frames, {r['scale']:.1f} scale"
        rt = "✅ YES" if r['fps'] >= 30 else f"❌ {r['fps']/30:.1%}"
        print(f"{config:<30} {r['fps']:>10.1f} {rt:>15}")

    # Find best configuration
    realtime_configs = [r for r in results if r['fps'] >= 30]
    if realtime_configs:
        best = max(realtime_configs, key=lambda x: x['interval'])  # Best quality that's realtime
        print(f"\n✅ BEST REAL-TIME CONFIG:")
        print(f"   Detection every {best['interval']} frames")
        print(f"   Scale: {best['scale']}")
        print(f"   FPS: {best['fps']:.1f}")
    else:
        fastest = max(results, key=lambda x: x['fps'])
        print(f"\n⚠️  No real-time config found")
        print(f"   Fastest: {fastest['fps']:.1f} FPS")
        print(f"   Need {30/fastest['fps']:.1f}x speedup")

    return results


if __name__ == "__main__":
    results = test_optimization_levels()