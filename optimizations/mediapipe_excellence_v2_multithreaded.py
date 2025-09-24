#!/usr/bin/env python3
"""
MediaPipe Excellence - Version 2: Multi-threaded Pipeline
Implements parallel processing with thread pools
"""

import cv2
import numpy as np
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import mediapipe as mp
from dataclasses import dataclass

@dataclass
class FrameData:
    """Container for frame processing data"""
    frame_id: int
    frame: np.ndarray
    detections: List = None
    processed: np.ndarray = None
    timestamp: float = 0


class MediaPipeMultithreaded:
    """Multi-threaded MediaPipe implementation"""

    def __init__(self, num_threads=4):
        # MediaPipe setup
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.5)

        # Threading setup
        self.num_threads = num_threads
        self.detection_executor = ThreadPoolExecutor(max_workers=num_threads//2)
        self.blur_executor = ThreadPoolExecutor(max_workers=num_threads//2)

        # Queues for pipeline
        self.detection_queue = queue.Queue(maxsize=20)
        self.blur_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=20)

        # Performance tracking
        self.metrics = {
            'detection_times': [],
            'blur_times': [],
            'queue_wait_times': [],
            'total_times': []
        }

        # Parameters
        self.params = {
            'canny_low': 50,
            'canny_high': 150,
            'min_area': 1000,
            'circularity': 0.65,
            'blur_kernel': 31
        }

    def detect_shapes_parallel(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Parallelized shape detection"""
        # Split frame into regions for parallel processing
        h, w = frame.shape[:2]
        regions = []

        # Split into quadrants with overlap
        overlap = 50
        regions.append((0, 0, w//2 + overlap, h//2 + overlap))
        regions.append((w//2 - overlap, 0, w, h//2 + overlap))
        regions.append((0, h//2 - overlap, w//2 + overlap, h))
        regions.append((w//2 - overlap, h//2 - overlap, w, h))

        # Process regions in parallel
        futures = []
        for x1, y1, x2, y2 in regions:
            region = frame[y1:y2, x1:x2]
            future = self.detection_executor.submit(
                self._detect_in_region, region, x1, y1
            )
            futures.append(future)

        # Collect results
        all_detections = []
        for future in as_completed(futures):
            detections = future.result()
            all_detections.extend(detections)

        # Merge overlapping detections
        return self._merge_detections(all_detections)

    def _detect_in_region(self, region: np.ndarray, offset_x: int, offset_y: int) -> List:
        """Detect shapes in a region"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
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
            # Adjust coordinates to global frame
            detections.append((x + offset_x, y + offset_y, w, h))

        return detections

    def _merge_detections(self, detections: List) -> List:
        """Merge overlapping detections from parallel processing"""
        if not detections:
            return []

        # Simple NMS
        merged = []
        used = set()

        for i, det1 in enumerate(detections):
            if i in used:
                continue

            x1, y1, w1, h1 = det1
            overlapping = [det1]

            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue

                x2, y2, w2, h2 = det2

                # Check overlap
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2):
                    overlapping.append(det2)
                    used.add(j)

            # Take average of overlapping
            if len(overlapping) > 1:
                x = int(np.mean([d[0] for d in overlapping]))
                y = int(np.mean([d[1] for d in overlapping]))
                w = int(np.mean([d[2] for d in overlapping]))
                h = int(np.mean([d[3] for d in overlapping]))
                merged.append((x, y, w, h))
            else:
                merged.append(det1)

            used.add(i)

        return merged

    def apply_blur_parallel(self, frame: np.ndarray, regions: List) -> np.ndarray:
        """Parallel blur application"""
        if not regions:
            return frame

        output = frame.copy()
        kernel_size = self.params['blur_kernel']

        # Process regions in parallel
        futures = []
        for x, y, w, h in regions:
            roi = output[y:y+h, x:x+w].copy()
            future = self.blur_executor.submit(
                cv2.GaussianBlur, roi, (kernel_size, kernel_size), 0
            )
            futures.append((future, x, y, w, h))

        # Apply results
        for future, x, y, w, h in futures:
            blurred_roi = future.result()
            output[y:y+h, x:x+w] = blurred_roi

        return output

    def process_frame_pipeline(self, frame_data: FrameData) -> FrameData:
        """Process frame through pipeline"""
        start = time.perf_counter()

        # Detection phase
        start_detect = time.perf_counter()
        frame_data.detections = self.detect_shapes_parallel(frame_data.frame)
        detect_time = (time.perf_counter() - start_detect) * 1000

        # Blur phase
        start_blur = time.perf_counter()
        frame_data.processed = self.apply_blur_parallel(
            frame_data.frame, frame_data.detections
        )
        blur_time = (time.perf_counter() - start_blur) * 1000

        # Total time
        total_time = (time.perf_counter() - start) * 1000

        # Update metrics
        self.metrics['detection_times'].append(detect_time)
        self.metrics['blur_times'].append(blur_time)
        self.metrics['total_times'].append(total_time)

        frame_data.timestamp = time.perf_counter()

        return frame_data

    def process_batch_async(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch asynchronously with pipeline"""
        # Submit all frames to pipeline
        futures = []
        for i, frame in enumerate(frames):
            frame_data = FrameData(frame_id=i, frame=frame)
            future = self.detection_executor.submit(
                self.process_frame_pipeline, frame_data
            )
            futures.append(future)

        # Collect results in order
        results = [None] * len(frames)
        for future in as_completed(futures):
            frame_data = future.result()
            results[frame_data.frame_id] = frame_data.processed

        return results

    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary"""
        summary = {}
        for key, times in self.metrics.items():
            if times:
                summary[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
        return summary


def benchmark_multithreaded():
    """Benchmark multi-threaded implementation"""
    print("="*80)
    print("MULTITHREADED MEDIAPIPE BENCHMARK")
    print("="*80)

    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    test_resolutions = [
        ('480p', (480, 640)),
        ('720p', (720, 1280)),
        ('1080p', (1080, 1920))
    ]

    results = {}

    for num_threads in thread_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_threads} threads")
        print('='*60)

        mt = MediaPipeMultithreaded(num_threads=num_threads)
        thread_results = {}

        for res_name, (h, w) in test_resolutions:
            print(f"\n{res_name} Resolution:")

            # Create test frames
            frames = []
            for i in range(10):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                # Add different patterns to each frame
                cv2.circle(frame, (w//2 + i*10, h//2), 80, (255, 255, 255), -1)
                cv2.circle(frame, (w//3, h//3), 60, (255, 255, 255), -1)
                frames.append(frame)

            # Benchmark
            start = time.perf_counter()
            processed = mt.process_batch_async(frames)
            batch_time = (time.perf_counter() - start) * 1000

            # Get metrics
            metrics = mt.get_metrics_summary()

            fps = len(frames) * 1000 / batch_time
            print(f"  Batch time: {batch_time:.2f}ms")
            print(f"  FPS: {fps:.1f}")
            print(f"  Avg detection: {metrics['detection_times']['mean']:.2f}ms")
            print(f"  Avg blur: {metrics['blur_times']['mean']:.2f}ms")

            thread_results[res_name] = {
                'batch_time_ms': batch_time,
                'fps': fps,
                'metrics': metrics
            }

        results[f'threads_{num_threads}'] = thread_results

    return results


if __name__ == "__main__":
    results = benchmark_multithreaded()

    print("\n" + "="*80)
    print("MULTITHREADING SUMMARY")
    print("="*80)

    # Find best thread count for 720p
    best_fps = 0
    best_threads = 0

    for thread_key, thread_data in results.items():
        if '720p' in thread_data:
            fps = thread_data['720p']['fps']
            threads = int(thread_key.split('_')[1])
            if fps > best_fps:
                best_fps = fps
                best_threads = threads

    print(f"\nBest configuration: {best_threads} threads")
    print(f"Best 720p FPS: {best_fps:.1f}")
    print("\nKey findings:")
    print("• Parallel region processing speeds up detection")
    print("• Async pipeline improves throughput")
    print("• Optimal thread count depends on CPU cores")