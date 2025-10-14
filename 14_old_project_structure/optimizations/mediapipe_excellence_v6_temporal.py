#!/usr/bin/env python3
"""
MediaPipe Excellence - Version 6: Temporal Optimization
Leverages temporal coherence across frames
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque
import mediapipe as mp
from scipy import interpolate


class TemporalFilter:
    """Temporal filtering and smoothing"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def smooth(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Apply temporal smoothing to detections"""
        self.history.append(detections)

        if len(self.history) < 2:
            return detections

        # Group detections across frames
        smoothed = []
        for i in range(len(detections)):
            if i >= len(detections):
                break

            x_hist, y_hist, w_hist, h_hist = [], [], [], []

            for frame_dets in self.history:
                if i < len(frame_dets):
                    x, y, w, h = frame_dets[i]
                    x_hist.append(x)
                    y_hist.append(y)
                    w_hist.append(w)
                    h_hist.append(h)

            if x_hist:
                # Weighted average (recent frames weighted more)
                weights = np.linspace(0.5, 1.0, len(x_hist))
                weights /= weights.sum()

                x_smooth = int(np.average(x_hist, weights=weights))
                y_smooth = int(np.average(y_hist, weights=weights))
                w_smooth = int(np.average(w_hist, weights=weights))
                h_smooth = int(np.average(h_hist, weights=weights))

                smoothed.append((x_smooth, y_smooth, w_smooth, h_smooth))

        return smoothed


class OpticalFlowTracker:
    """Track motion using optical flow"""

    def __init__(self):
        self.prev_gray = None
        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def track(self, gray_frame: np.ndarray, detections: List) -> List:
        """Track detections using optical flow"""
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return detections

        if not detections:
            self.prev_gray = gray_frame
            return []

        # Convert detections to points
        old_points = []
        for x, y, w, h in detections:
            # Track center point
            old_points.append([x + w/2, y + h/2])

        old_points = np.array(old_points, dtype=np.float32).reshape(-1, 1, 2)

        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, old_points, None, **self.flow_params
        )

        # Update detections based on flow
        tracked = []
        for i, (new_pt, old_det, st) in enumerate(zip(new_points, detections, status)):
            if st[0] == 1:  # Successfully tracked
                x_old, y_old, w, h = old_det
                cx_new, cy_new = new_pt[0]
                x_new = int(cx_new - w/2)
                y_new = int(cy_new - h/2)
                tracked.append((x_new, y_new, w, h))
            else:
                tracked.append(old_det)

        self.prev_gray = gray_frame
        return tracked


class FrameInterpolator:
    """Interpolate between frames for smoother playback"""

    def __init__(self):
        self.prev_frame = None
        self.interpolation_factor = 0.5

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Interpolate between two frames"""
        return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)

    def generate_intermediate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Generate intermediate frame"""
        if self.prev_frame is None:
            self.prev_frame = frame
            return None

        intermediate = self.interpolate(self.prev_frame, frame, self.interpolation_factor)
        self.prev_frame = frame
        return intermediate


class MediaPipeTemporal:
    """MediaPipe with temporal optimizations"""

    def __init__(self):
        # MediaPipe setup
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.5)

        # Temporal components
        self.temporal_filter = TemporalFilter(window_size=5)
        self.optical_flow = OpticalFlowTracker()
        self.frame_interpolator = FrameInterpolator()

        # Motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.motion_threshold = 0.01  # Percentage of frame with motion

        # Frame difference tracking
        self.prev_frame = None
        self.static_frames = 0
        self.max_static_frames = 10

        # Parameters
        self.params = {
            'canny_low': 50,
            'canny_high': 150,
            'min_area': 1000,
            'circularity': 0.65,
            'blur_kernel': 31
        }

        # Metrics
        self.metrics = {
            'static_frames': 0,
            'motion_frames': 0,
            'interpolated_frames': 0,
            'flow_tracked': 0,
            'processing_times': []
        }

    def detect_motion(self, frame: np.ndarray) -> float:
        """Detect motion level in frame"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Calculate motion percentage
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels

        return motion_ratio

    def detect_shapes_temporal(self, frame: np.ndarray, use_flow: bool = True) -> List[Tuple[int, int, int, int]]:
        """Temporal-aware shape detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check motion level
        motion_level = self.detect_motion(frame)

        if motion_level < self.motion_threshold:
            # Very little motion - reuse previous detections
            self.static_frames += 1
            self.metrics['static_frames'] += 1

            if self.static_frames < self.max_static_frames and hasattr(self, 'last_detections'):
                # Use optical flow to track slight movements
                if use_flow:
                    tracked = self.optical_flow.track(gray, self.last_detections)
                    self.metrics['flow_tracked'] += 1
                    return tracked
                return self.last_detections
        else:
            self.static_frames = 0
            self.metrics['motion_frames'] += 1

        # Full detection for frames with motion
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, self.params['canny_low'], self.params['canny_high'])

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.params['min_area']:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.params['circularity']:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append((x, y, w, h))

        # Apply temporal smoothing
        smoothed = self.temporal_filter.smooth(detections)
        self.last_detections = smoothed

        return smoothed

    def apply_blur_temporal(self, frame: np.ndarray, regions: List) -> np.ndarray:
        """Apply blur with temporal optimization"""
        if not regions:
            return frame

        # Check if we can reuse previous blur
        if self.prev_frame is not None and self.static_frames > 2:
            # Frame is static, blend with previous for temporal consistency
            diff = cv2.absdiff(frame, self.prev_frame)
            avg_diff = np.mean(diff)

            if avg_diff < 5:  # Very similar frames
                # Return interpolated result for smoother appearance
                return self.frame_interpolator.interpolate(self.prev_frame, frame, 0.3)

        output = frame.copy()
        kernel_size = self.params['blur_kernel']

        for x, y, w, h in regions:
            roi = output[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            output[y:y+h, x:x+w] = blurred

        self.prev_frame = output
        return output

    def process_frame_temporal(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame with temporal optimizations"""
        start = time.perf_counter()

        # Temporal detection
        detections = self.detect_shapes_temporal(frame, use_flow=True)

        # Temporal blur
        output = self.apply_blur_temporal(frame, detections)

        # Generate intermediate frame if applicable
        intermediate = self.frame_interpolator.generate_intermediate(output)
        if intermediate is not None:
            self.metrics['interpolated_frames'] += 1

        elapsed = (time.perf_counter() - start) * 1000
        self.metrics['processing_times'].append(elapsed)

        return output, {
            'detections': len(detections),
            'time_ms': elapsed,
            'fps': 1000 / elapsed if elapsed > 0 else 0,
            'static': self.static_frames > 0,
            'motion_level': self.detect_motion(frame),
            'interpolated': intermediate is not None
        }

    def process_video_temporal(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process video with temporal coherence"""
        results = []

        for i, frame in enumerate(frames):
            output, info = self.process_frame_temporal(frame)
            results.append(output)

            # Add interpolated frame for smoother playback
            if i > 0 and i < len(frames) - 1:
                # Generate intermediate frame
                intermediate = self.frame_interpolator.interpolate(
                    results[-2] if len(results) > 1 else frame,
                    output,
                    0.5
                )
                if self.static_frames < 2:  # Only for moving scenes
                    results.append(intermediate)
                    self.metrics['interpolated_frames'] += 1

        return results


def benchmark_temporal():
    """Benchmark temporal optimization"""
    print("="*80)
    print("TEMPORAL OPTIMIZATION BENCHMARK")
    print("="*80)

    temporal = MediaPipeTemporal()

    # Test scenarios
    test_scenarios = [
        ('static_scene', 'static'),
        ('slow_motion', 'slow'),
        ('fast_motion', 'fast'),
        ('intermittent', 'intermittent')
    ]

    results = {}

    for scenario_name, scenario_type in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario_name}")
        print('='*60)

        temporal = MediaPipeTemporal()  # Reset for each scenario

        # Generate test frames
        frames = []
        for i in range(100):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            if scenario_type == 'static':
                # Static scene
                cv2.circle(frame, (640, 360), 80, (255, 255, 255), -1)
                cv2.circle(frame, (400, 360), 60, (255, 255, 255), -1)

            elif scenario_type == 'slow':
                # Slow motion
                x = 400 + i * 3
                cv2.circle(frame, (x, 360), 80, (255, 255, 255), -1)

            elif scenario_type == 'fast':
                # Fast motion
                x = 200 + i * 15
                cv2.circle(frame, (x % 1280, 360), 80, (255, 255, 255), -1)

            elif scenario_type == 'intermittent':
                # Intermittent motion
                if i % 20 < 10:
                    x = 640
                else:
                    x = 640 + (i % 20 - 10) * 20
                cv2.circle(frame, (x, 360), 80, (255, 255, 255), -1)

            frames.append(frame)

        # Process
        start = time.perf_counter()
        processed = temporal.process_video_temporal(frames)
        total_time = (time.perf_counter() - start) * 1000

        # Metrics
        avg_time = np.mean(temporal.metrics['processing_times']) if temporal.metrics['processing_times'] else 0
        fps = len(frames) * 1000 / total_time

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Avg frame time: {avg_time:.2f}ms")
        print(f"  Static frames: {temporal.metrics['static_frames']}")
        print(f"  Motion frames: {temporal.metrics['motion_frames']}")
        print(f"  Flow tracked: {temporal.metrics['flow_tracked']}")
        print(f"  Interpolated: {temporal.metrics['interpolated_frames']}")

        results[scenario_name] = {
            'fps': fps,
            'static_ratio': temporal.metrics['static_frames'] / len(frames),
            'flow_usage': temporal.metrics['flow_tracked'] / len(frames),
            'avg_time_ms': avg_time
        }

    return results


if __name__ == "__main__":
    results = benchmark_temporal()

    print("\n" + "="*80)
    print("TEMPORAL OPTIMIZATION SUMMARY")
    print("="*80)

    # Find best improvement
    best_scenario = max(results.items(), key=lambda x: x[1]['fps'])
    worst_scenario = min(results.items(), key=lambda x: x[1]['fps'])

    print(f"\nBest performance: {best_scenario[0]}")
    print(f"  FPS: {best_scenario[1]['fps']:.1f}")
    print(f"  Static frame ratio: {best_scenario[1]['static_ratio']:.2%}")

    print(f"\nWorst performance: {worst_scenario[0]}")
    print(f"  FPS: {worst_scenario[1]['fps']:.1f}")
    print(f"  Flow usage: {worst_scenario[1]['flow_usage']:.2%}")

    print("\nKey Features:")
    print("• Optical flow tracking for static scenes")
    print("• Temporal smoothing of detections")
    print("• Frame interpolation for smoother playback")
    print("• Motion-based processing decisions")
    print("• Background subtraction for motion detection")