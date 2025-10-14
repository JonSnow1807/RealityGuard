#!/usr/bin/env python3
"""
MediaPipe Excellence - Version 5: Adaptive Quality Modes
Dynamically adjusts processing quality based on performance
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import mediapipe as mp
from collections import deque
import psutil


class AdaptiveQualityManager:
    """Manages dynamic quality adjustment"""

    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.target_fps = 60
        self.quality_levels = {
            'ultra': {'blur_kernel': 51, 'detection_confidence': 0.6, 'skip_frames': 0},
            'high': {'blur_kernel': 31, 'detection_confidence': 0.5, 'skip_frames': 0},
            'medium': {'blur_kernel': 21, 'detection_confidence': 0.4, 'skip_frames': 1},
            'low': {'blur_kernel': 15, 'detection_confidence': 0.3, 'skip_frames': 2},
            'performance': {'blur_kernel': 9, 'detection_confidence': 0.2, 'skip_frames': 3}
        }
        self.current_quality = 'high'
        self.cpu_threshold = 80  # CPU usage threshold

    def update_quality(self, current_fps: float) -> str:
        """Dynamically adjust quality based on performance"""
        self.fps_history.append(current_fps)

        if len(self.fps_history) < 5:
            return self.current_quality

        avg_fps = np.mean(list(self.fps_history))
        cpu_usage = psutil.cpu_percent(interval=0.01)

        # Quality decision logic
        if avg_fps < self.target_fps * 0.5:
            # Very low FPS - drop quality
            if self.current_quality == 'ultra':
                self.current_quality = 'high'
            elif self.current_quality == 'high':
                self.current_quality = 'medium'
            elif self.current_quality == 'medium':
                self.current_quality = 'low'
            elif self.current_quality == 'low':
                self.current_quality = 'performance'

        elif avg_fps > self.target_fps * 1.5 and cpu_usage < self.cpu_threshold:
            # High FPS and low CPU - increase quality
            if self.current_quality == 'performance':
                self.current_quality = 'low'
            elif self.current_quality == 'low':
                self.current_quality = 'medium'
            elif self.current_quality == 'medium':
                self.current_quality = 'high'
            elif self.current_quality == 'high':
                self.current_quality = 'ultra'

        return self.current_quality

    def get_params(self) -> Dict:
        """Get current quality parameters"""
        return self.quality_levels[self.current_quality]


class MediaPipeAdaptive:
    """MediaPipe with adaptive quality"""

    def __init__(self, target_fps=60):
        # MediaPipe setup
        self.mp_face = mp.solutions.face_detection

        # Adaptive quality manager
        self.quality_manager = AdaptiveQualityManager()
        self.quality_manager.target_fps = target_fps

        # Multiple detectors for different quality levels
        self.detectors = {
            'high': self.mp_face.FaceDetection(min_detection_confidence=0.5),
            'low': self.mp_face.FaceDetection(min_detection_confidence=0.3)
        }

        # Frame skip counter
        self.frame_counter = 0
        self.last_detections = []

        # Parameters (will be adjusted dynamically)
        self.params = {
            'canny_low': 50,
            'canny_high': 150,
            'min_area': 1000,
            'circularity': 0.65
        }

        # Performance metrics
        self.metrics = {
            'quality_changes': [],
            'fps_history': [],
            'cpu_history': [],
            'quality_distribution': {'ultra': 0, 'high': 0, 'medium': 0, 'low': 0, 'performance': 0}
        }

    def detect_shapes_adaptive(self, frame: np.ndarray, quality_params: Dict) -> List[Tuple[int, int, int, int]]:
        """Adaptive shape detection based on quality level"""
        # Skip frames based on quality setting
        if quality_params['skip_frames'] > 0:
            if self.frame_counter % (quality_params['skip_frames'] + 1) != 0:
                # Return cached detections with slight motion prediction
                predicted = []
                for x, y, w, h in self.last_detections:
                    # Simple motion prediction
                    predicted.append((x, y, w, h))
                return predicted

        # Adjust detection parameters based on quality
        if self.quality_manager.current_quality == 'performance':
            # Use simplified detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for faster processing
            small = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
            blurred = cv2.GaussianBlur(small, (3, 3), 1)
            edges = cv2.Canny(blurred, 100, 200)
        else:
            # Full quality detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            edges = cv2.Canny(blurred, self.params['canny_low'], self.params['canny_high'])

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        min_area = self.params['min_area'] if self.quality_manager.current_quality != 'performance' else 500

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.params['circularity']:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Scale up if downsampled
            if self.quality_manager.current_quality == 'performance':
                x, y, w, h = x*2, y*2, w*2, h*2

            detections.append((x, y, w, h))

        self.last_detections = detections
        return detections

    def apply_blur_adaptive(self, frame: np.ndarray, regions: List, kernel_size: int) -> np.ndarray:
        """Adaptive blur based on quality level"""
        if not regions:
            return frame

        output = frame.copy()

        # Adaptive blur strategy
        if self.quality_manager.current_quality == 'performance':
            # Box blur for performance
            for x, y, w, h in regions:
                roi = output[y:y+h, x:x+w]
                blurred = cv2.blur(roi, (kernel_size, kernel_size))
                output[y:y+h, x:x+w] = blurred

        elif self.quality_manager.current_quality in ['low', 'medium']:
            # Standard Gaussian blur
            for x, y, w, h in regions:
                roi = output[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                output[y:y+h, x:x+w] = blurred

        else:  # high or ultra
            # High quality bilateral filter
            for x, y, w, h in regions:
                roi = output[y:y+h, x:x+w]
                if self.quality_manager.current_quality == 'ultra':
                    blurred = cv2.bilateralFilter(roi, kernel_size//2, 75, 75)
                else:
                    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                output[y:y+h, x:x+w] = blurred

        return output

    def process_frame_adaptive(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame with adaptive quality"""
        start = time.perf_counter()
        self.frame_counter += 1

        # Get current quality parameters
        quality_params = self.quality_manager.get_params()

        # Detection
        detections = self.detect_shapes_adaptive(frame, quality_params)

        # Blur
        output = self.apply_blur_adaptive(frame, detections, quality_params['blur_kernel'])

        # Calculate FPS and update quality
        elapsed = (time.perf_counter() - start) * 1000
        current_fps = 1000 / elapsed if elapsed > 0 else 0

        old_quality = self.quality_manager.current_quality
        new_quality = self.quality_manager.update_quality(current_fps)

        # Track quality changes
        if old_quality != new_quality:
            self.metrics['quality_changes'].append({
                'frame': self.frame_counter,
                'from': old_quality,
                'to': new_quality,
                'fps': current_fps
            })

        # Update metrics
        self.metrics['fps_history'].append(current_fps)
        self.metrics['cpu_history'].append(psutil.cpu_percent(interval=0))
        self.metrics['quality_distribution'][new_quality] += 1

        return output, {
            'detections': len(detections),
            'time_ms': elapsed,
            'fps': current_fps,
            'quality': new_quality,
            'blur_kernel': quality_params['blur_kernel'],
            'skipped': quality_params['skip_frames'] > 0 and self.frame_counter % (quality_params['skip_frames'] + 1) != 0
        }

    def process_video_adaptive(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process video with adaptive quality"""
        results = []

        for frame in frames:
            output, info = self.process_frame_adaptive(frame)
            results.append(output)

        return results


def benchmark_adaptive():
    """Benchmark adaptive quality system"""
    print("="*80)
    print("ADAPTIVE QUALITY BENCHMARK")
    print("="*80)

    # Test different target FPS values
    target_fps_values = [30, 60, 120]
    test_resolutions = [
        ('480p', (480, 640)),
        ('720p', (720, 1280)),
        ('1080p', (1080, 1920))
    ]

    results = {}

    for target_fps in target_fps_values:
        print(f"\n{'='*60}")
        print(f"Target FPS: {target_fps}")
        print('='*60)

        adaptive = MediaPipeAdaptive(target_fps=target_fps)
        target_results = {}

        for res_name, (h, w) in test_resolutions:
            print(f"\n{res_name} Resolution:")

            # Create test frames with varying complexity
            frames = []
            for i in range(100):
                frame = np.zeros((h, w, 3), dtype=np.uint8)

                # Vary complexity
                if i < 30:
                    # Simple scene
                    cv2.circle(frame, (w//2, h//2), 60, (255, 255, 255), -1)
                elif i < 60:
                    # Medium complexity
                    cv2.circle(frame, (w//2, h//2), 80, (255, 255, 255), -1)
                    cv2.circle(frame, (w//3, h//3), 60, (255, 255, 255), -1)
                else:
                    # Complex scene
                    for j in range(5):
                        x = np.random.randint(100, w-100)
                        y = np.random.randint(100, h-100)
                        cv2.circle(frame, (x, y), 40, (255, 255, 255), -1)

                frames.append(frame)

            # Process
            start = time.perf_counter()
            processed = adaptive.process_video_adaptive(frames)
            total_time = (time.perf_counter() - start) * 1000

            # Analyze results
            avg_fps = np.mean(adaptive.metrics['fps_history']) if adaptive.metrics['fps_history'] else 0
            quality_changes = len(adaptive.metrics['quality_changes'])

            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Quality changes: {quality_changes}")

            # Quality distribution
            total_frames = sum(adaptive.metrics['quality_distribution'].values())
            if total_frames > 0:
                print(f"  Quality distribution:")
                for q, count in adaptive.metrics['quality_distribution'].items():
                    pct = count * 100 / total_frames
                    if pct > 0:
                        print(f"    {q}: {pct:.1f}%")

            target_results[res_name] = {
                'avg_fps': avg_fps,
                'quality_changes': quality_changes,
                'quality_dist': adaptive.metrics['quality_distribution'].copy()
            }

        results[f'target_{target_fps}'] = target_results

    return results


def test_adaptive_behavior():
    """Test adaptive quality behavior"""
    print("\n" + "="*60)
    print("ADAPTIVE BEHAVIOR TEST")
    print("="*60)

    adaptive = MediaPipeAdaptive(target_fps=60)

    # Simulate different load scenarios
    scenarios = [
        ('low_load', 150),   # 150 FPS - should increase quality
        ('normal_load', 65),  # 65 FPS - should maintain quality
        ('high_load', 25),    # 25 FPS - should decrease quality
        ('overload', 10)      # 10 FPS - should go to performance mode
    ]

    print("\nSimulating load scenarios:")
    for name, simulated_fps in scenarios:
        # Update quality manager with simulated FPS
        for _ in range(10):
            adaptive.quality_manager.update_quality(simulated_fps)

        print(f"\n{name} ({simulated_fps} FPS):")
        print(f"  Quality level: {adaptive.quality_manager.current_quality}")
        params = adaptive.quality_manager.get_params()
        print(f"  Blur kernel: {params['blur_kernel']}")
        print(f"  Skip frames: {params['skip_frames']}")


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_adaptive()

    # Test behavior
    test_adaptive_behavior()

    print("\n" + "="*80)
    print("ADAPTIVE QUALITY SUMMARY")
    print("="*80)

    print("\nKey Features:")
    print("• Dynamically adjusts quality based on FPS")
    print("• Multiple quality levels (ultra/high/medium/low/performance)")
    print("• Frame skipping for low-end systems")
    print("• Adaptive blur algorithms (bilateral/gaussian/box)")
    print("• CPU usage monitoring")

    # Find best configuration
    if results:
        best_config = None
        best_fps = 0

        for target, res_data in results.items():
            if '720p' in res_data:
                fps = res_data['720p']['avg_fps']
                if fps > best_fps:
                    best_fps = fps
                    best_config = target

        print(f"\nBest 720p configuration: {best_config}")
        print(f"Average FPS: {best_fps:.1f}")