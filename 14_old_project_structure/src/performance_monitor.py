"""
Performance Monitoring Module
Tracks FPS, latency, and other performance metrics
"""

import time
from collections import deque
from typing import Dict, Optional


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)

        self.total_frames = 0
        self.start_time = time.perf_counter()

    def log_frame(self, total_time: float, detection_time: float = 0,
                  processing_time: float = 0):
        """Log frame processing metrics.

        Args:
            total_time: Total frame processing time in seconds
            detection_time: Time spent on detection
            processing_time: Time spent on filtering/processing
        """
        self.frame_times.append(total_time)
        self.detection_times.append(detection_time)
        self.processing_times.append(processing_time)
        self.total_frames += 1

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.frame_times:
            return {
                'fps': 0,
                'avg_total_time': 0,
                'avg_detection_time': 0,
                'avg_processing_time': 0,
                'total_frames': 0,
                'runtime': 0
            }

        avg_total = sum(self.frame_times) / len(self.frame_times)
        avg_detection = sum(self.detection_times) / len(self.detection_times)
        avg_processing = sum(self.processing_times) / len(self.processing_times)

        runtime = time.perf_counter() - self.start_time

        return {
            'fps': 1.0 / avg_total if avg_total > 0 else 0,
            'avg_total_time': avg_total,
            'avg_detection_time': avg_detection,
            'avg_processing_time': avg_processing,
            'total_frames': self.total_frames,
            'runtime': runtime,
            'avg_fps': self.total_frames / runtime if runtime > 0 else 0
        }

    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print(f"Performance Stats:")
        print(f"  FPS: {stats['fps']:.1f} (current), {stats['avg_fps']:.1f} (average)")
        print(f"  Frame Time: {stats['avg_total_time']*1000:.2f}ms")
        print(f"  Detection: {stats['avg_detection_time']*1000:.2f}ms")
        print(f"  Processing: {stats['avg_processing_time']*1000:.2f}ms")
        print(f"  Total Frames: {stats['total_frames']}")

    def reset(self):
        """Reset all metrics."""
        self.frame_times.clear()
        self.detection_times.clear()
        self.processing_times.clear()
        self.total_frames = 0
        self.start_time = time.perf_counter()