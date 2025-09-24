#!/usr/bin/env python3
"""
Reality Guard Improved V2 - With fixed detection and tracking
Honest implementation with real performance metrics
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a detected privacy region"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    detection_type: str
    timestamp: float
    velocity: Optional[Tuple[float, float]] = None

class KalmanTracker:
    """Kalman filter for motion tracking"""

    def __init__(self, detection: Detection):
        self.id = detection.id
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)  # Increased for smoother tracking

        # Initialize with detection
        x, y, w, h = detection.bbox
        center_x = x + w/2
        center_y = y + h/2
        self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)

        self.age = 0
        self.hits = 1
        self.misses = 0
        self.last_detection = detection

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next position"""
        prediction = self.kalman.predict()
        x = int(prediction[0] - self.last_detection.bbox[2]/2)
        y = int(prediction[1] - self.last_detection.bbox[3]/2)
        w = self.last_detection.bbox[2]
        h = self.last_detection.bbox[3]
        return (x, y, w, h)

    def update(self, detection: Detection):
        """Update with new detection"""
        x, y, w, h = detection.bbox
        center_x = x + w/2
        center_y = y + h/2
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        self.kalman.correct(measurement)

        self.hits += 1
        self.misses = 0
        self.last_detection = detection
        self.age += 1

class ImprovedDetector:
    """Improved detection with better thresholds and tracking"""

    # Fixed thresholds to reduce over-detection
    IMPROVED_CONFIG = {
        'canny_low': 50,        # Increased to reduce noise
        'canny_high': 150,      # Higher threshold for cleaner edges
        'min_area': 1000,       # Increased to filter small noise
        'max_area': 50000,      # Maximum area limit
        'circularity': 0.65,    # Stricter circularity requirement
        'min_radius': 20,       # Larger minimum radius
        'max_radius': 200,      # Maximum radius limit
        'haar_scale': 1.2,      # Less aggressive Haar cascade
        'haar_neighbors': 5,    # More neighbors for validation
        'blur_kernel': 31,      # Blur strength
        'tracker_max_age': 10,  # Frames to keep tracker alive
        'confidence_threshold': 0.5,  # Minimum confidence to accept detection
        'max_trackers': 20,     # Maximum number of simultaneous trackers
        'nms_threshold': 0.3,   # Non-maximum suppression IOU threshold
        'tracker_init_threshold': 0.6,  # Minimum confidence to create tracker
    }

    def __init__(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # Tracking system
        self.trackers: Dict[int, KalmanTracker] = {}
        self.next_id = 0

        # Performance monitoring
        self.frame_times = deque(maxlen=30)

    def detect_geometric_shapes(self, frame: np.ndarray) -> List[Detection]:
        """Improved geometric shape detection"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Better edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(
            blurred,
            self.IMPROVED_CONFIG['canny_low'],
            self.IMPROVED_CONFIG['canny_high']
        )

        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Check area bounds
            if area < self.IMPROVED_CONFIG['min_area'] or area > self.IMPROVED_CONFIG['max_area']:
                continue

            # Check circularity with more lenient threshold
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.IMPROVED_CONFIG['circularity']:
                continue

            # Additional confidence check
            if circularity < self.IMPROVED_CONFIG['confidence_threshold']:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Additional validation
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            detections.append(Detection(
                id=-1,  # Will be assigned by tracker
                bbox=(x, y, w, h),
                confidence=circularity,
                detection_type='geometric',
                timestamp=time.time()
            ))

        return detections

    def detect_haar_faces(self, frame: np.ndarray) -> List[Detection]:
        """Haar cascade detection with better parameters"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)

        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.IMPROVED_CONFIG['haar_scale'],
            minNeighbors=self.IMPROVED_CONFIG['haar_neighbors'],
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            detections.append(Detection(
                id=-1,
                bbox=(x, y, w, h),
                confidence=0.8,
                detection_type='haar_frontal',
                timestamp=time.time()
            ))

        # Detect profile faces
        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=self.IMPROVED_CONFIG['haar_scale'],
            minNeighbors=self.IMPROVED_CONFIG['haar_neighbors'],
            minSize=(30, 30)
        )

        for (x, y, w, h) in profiles:
            detections.append(Detection(
                id=-1,
                bbox=(x, y, w, h),
                confidence=0.7,
                detection_type='haar_profile',
                timestamp=time.time()
            ))

        return detections

    def detect_blob_shapes(self, frame: np.ndarray) -> List[Detection]:
        """Blob detection for circular shapes"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Setup SimpleBlobDetector with better parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.IMPROVED_CONFIG['min_area']
        params.maxArea = self.IMPROVED_CONFIG['max_area']
        params.filterByCircularity = True
        params.minCircularity = self.IMPROVED_CONFIG['circularity']
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        for kp in keypoints:
            x = int(kp.pt[0] - kp.size/2)
            y = int(kp.pt[1] - kp.size/2)
            w = h = int(kp.size)

            detections.append(Detection(
                id=-1,
                bbox=(x, y, w, h),
                confidence=0.6,
                detection_type='blob',
                timestamp=time.time()
            ))

        return detections

    def merge_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping detections using NMS"""
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Convert to x1, y1, x2, y2 format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])

        # Apply NMS with configurable threshold
        indices = self.non_max_suppression(boxes_xyxy, scores, self.IMPROVED_CONFIG['nms_threshold'])

        return [detections[i] for i in indices]

    def non_max_suppression(self, boxes, scores, threshold):
        """Non-maximum suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / areas[order[1:]]

            order = order[np.where(overlap <= threshold)[0] + 1]

        return keep

    def update_trackers(self, detections: List[Detection]) -> List[Detection]:
        """Update trackers with new detections"""
        tracked_detections = []

        # Match detections to existing trackers
        unmatched_detections = list(detections)
        matched_trackers = set()

        for tracker_id, tracker in list(self.trackers.items()):
            best_match = None
            best_iou = 0

            predicted_bbox = tracker.predict()

            for detection in unmatched_detections:
                iou = self.calculate_iou(predicted_bbox, detection.bbox)
                if iou > best_iou and iou > 0.4:  # Increased IOU threshold for matching
                    best_iou = iou
                    best_match = detection

            if best_match:
                # Update tracker with matched detection
                best_match.id = tracker_id
                tracker.update(best_match)
                tracked_detections.append(best_match)
                unmatched_detections.remove(best_match)
                matched_trackers.add(tracker_id)
            else:
                # No match, use prediction
                tracker.misses += 1
                if tracker.misses < self.IMPROVED_CONFIG['tracker_max_age']:
                    predicted_detection = Detection(
                        id=tracker_id,
                        bbox=predicted_bbox,
                        confidence=0.5,
                        detection_type='predicted',
                        timestamp=time.time()
                    )
                    tracked_detections.append(predicted_detection)
                else:
                    # Remove dead tracker
                    del self.trackers[tracker_id]

        # Create new trackers for unmatched detections (with limits)
        for detection in unmatched_detections:
            # Only create tracker if confidence is high enough and not too many trackers
            if (detection.confidence >= self.IMPROVED_CONFIG['tracker_init_threshold'] and
                len(self.trackers) < self.IMPROVED_CONFIG['max_trackers']):
                detection.id = self.next_id
                self.trackers[self.next_id] = KalmanTracker(detection)
                tracked_detections.append(detection)
                self.next_id += 1
            elif detection.confidence >= self.IMPROVED_CONFIG['confidence_threshold']:
                # Still include detection but without tracking
                detection.id = -1
                tracked_detections.append(detection)

        return tracked_detections

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def apply_privacy_blur(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Apply adaptive blur to detected regions"""
        output = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox

            # Ensure bbox is within frame bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            if x2 <= x or y2 <= y:
                continue

            # Extract region
            roi = output[y:y2, x:x2]

            # Adaptive blur strength based on confidence
            kernel_size = self.IMPROVED_CONFIG['blur_kernel']
            if detection.confidence < 0.6:
                kernel_size = max(15, kernel_size - 10)

            # Ensure kernel size is odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            # Apply blur
            blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            output[y:y2, x:x2] = blurred

        return output

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame with all improvements"""
        start_time = time.perf_counter()

        # Run all detection methods
        detections = []
        detections.extend(self.detect_geometric_shapes(frame))
        detections.extend(self.detect_haar_faces(frame))
        detections.extend(self.detect_blob_shapes(frame))

        # Merge overlapping detections
        detections = self.merge_detections(detections)

        # Update trackers and get tracked detections
        tracked_detections = self.update_trackers(detections)

        # Apply privacy blur
        output = self.apply_privacy_blur(frame, tracked_detections)

        # Calculate metrics
        process_time = (time.perf_counter() - start_time) * 1000
        self.frame_times.append(process_time)

        avg_time = np.mean(self.frame_times) if self.frame_times else process_time
        fps = 1000 / avg_time if avg_time > 0 else 0

        # Prepare info
        info = {
            'detections': len(tracked_detections),
            'trackers': len(self.trackers),
            'process_time_ms': round(process_time, 2),
            'fps': round(fps, 1),
            'detection_types': {}
        }

        # Count detection types
        for det in tracked_detections:
            det_type = det.detection_type
            info['detection_types'][det_type] = info['detection_types'].get(det_type, 0) + 1

        # Draw debug info if needed
        if len(tracked_detections) > 0:
            for det in tracked_detections:
                x, y, w, h = det.bbox
                color = (0, 255, 0) if det.detection_type != 'predicted' else (255, 255, 0)
                cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                cv2.putText(output, f"ID:{det.id}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return output, info

def run_tests():
    """Test the improved detector"""
    detector = ImprovedDetector()

    print("="*60)
    print("TESTING IMPROVED REALITY GUARD V2")
    print("="*60)

    # Test different scenarios
    test_cases = [
        ("single_circle", (640, 480)),
        ("multiple_circles", (640, 480)),
        ("moving_object", (640, 480)),
        ("hd_resolution", (1280, 720))
    ]

    for test_name, resolution in test_cases:
        print(f"\nTesting: {test_name} at {resolution}")

        # Create test frame
        frame = np.zeros((*resolution[::-1], 3), dtype=np.uint8)

        if test_name == "single_circle":
            cv2.circle(frame, (320, 240), 60, (255, 255, 255), -1)
        elif test_name == "multiple_circles":
            cv2.circle(frame, (160, 240), 50, (255, 255, 255), -1)
            cv2.circle(frame, (320, 240), 60, (255, 255, 255), -1)
            cv2.circle(frame, (480, 240), 55, (255, 255, 255), -1)
        elif test_name == "moving_object":
            # Simulate movement over 10 frames
            for i in range(10):
                frame = np.zeros((*resolution[::-1], 3), dtype=np.uint8)
                x = 100 + i * 50
                cv2.circle(frame, (x, 240), 50, (255, 255, 255), -1)
                output, info = detector.process_frame(frame)
                if i % 3 == 0:
                    print(f"  Frame {i}: {info['detections']} detections, "
                          f"{info['trackers']} trackers, {info['fps']:.1f} FPS")
            continue
        else:
            # HD test with multiple shapes
            cv2.circle(frame, (200, 200), 80, (255, 255, 255), -1)
            cv2.circle(frame, (1000, 500), 100, (255, 255, 255), -1)
            cv2.ellipse(frame, (640, 360), (120, 80), 0, 0, 360, (255, 255, 255), -1)

        # Process frame
        output, info = detector.process_frame(frame)

        print(f"  Detections: {info['detections']}")
        print(f"  Types: {info['detection_types']}")
        print(f"  Process time: {info['process_time_ms']}ms")
        print(f"  FPS: {info['fps']}")

    print("\n" + "="*60)
    print("IMPROVED V2 TEST COMPLETE")
    print("Key improvements implemented:")
    print("✓ Better detection thresholds")
    print("✓ Kalman filter tracking")
    print("✓ Non-maximum suppression")
    print("✓ Multi-method detection")
    print("✓ Adaptive blur strength")
    print("="*60)

if __name__ == "__main__":
    run_tests()