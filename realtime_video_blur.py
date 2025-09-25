#!/usr/bin/env python3
"""
Real-Time Video Blur System
Automatically detects and blurs sensitive content in video streams.
Optimized for real-time playback.
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


class RealtimeVideoBlur:
    """Real-time video blur system for sensitive content."""

    def __init__(self, model_type='yolov8n', target_fps=30):
        """
        Initialize the blur system.

        Args:
            model_type: YOLO model to use (yolov8n, yolov8s, etc.)
            target_fps: Target playback FPS
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load models
        print(f"Loading models on {self.device}...")

        # Object detection model
        self.detector = YOLO(f'{model_type}.pt')
        self.detector.to(self.device)

        # Face detection (using YOLOv8-face if available, else cascade)
        try:
            self.face_detector = YOLO('yolov8n-face.pt')
            self.face_detector.to(self.device)
            self.use_yolo_face = True
        except:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_yolo_face = False
            print("Using OpenCV cascade for faces")

        # Sensitive object classes to blur
        self.sensitive_classes = {
            'person', 'face', 'laptop', 'cell phone', 'tv',
            'monitor', 'keyboard', 'book', 'credit card',
            'passport', 'license plate'
        }

        # Performance settings
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.skip_frames = False
        self.blur_strength = 31  # Must be odd

        # Frame buffer for smooth playback
        self.frame_buffer = queue.Queue(maxsize=5)
        self.detection_cache = {}
        self.cache_duration = 5  # Cache for 5 frames

        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)

        # Enable CUDA optimizations
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def detect_sensitive_content(self, frame, frame_id):
        """
        Detect sensitive content in frame.
        Uses caching for performance.
        """
        # Check cache
        if frame_id in self.detection_cache:
            cache_entry = self.detection_cache[frame_id]
            if cache_entry['valid_until'] > frame_id:
                return cache_entry['detections']

        detections = []

        # Detect objects
        start = time.perf_counter()

        # Downsample for faster detection if needed
        if self.skip_frames:
            small_frame = cv2.resize(frame, (320, 320))
            results = self.detector(small_frame, verbose=False, conf=0.4)
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 320
        else:
            results = self.detector(frame, verbose=False, conf=0.4)
            scale_x = scale_y = 1

        # Process detections
        for r in results:
            if r.boxes is not None:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy()):
                    class_name = self.detector.names[int(cls)]

                    # Check if sensitive
                    if class_name in self.sensitive_classes:
                        x1, y1, x2, y2 = box
                        detections.append({
                            'type': class_name,
                            'bbox': [
                                int(x1 * scale_x),
                                int(y1 * scale_y),
                                int(x2 * scale_x),
                                int(y2 * scale_y)
                            ]
                        })

        # Detect faces (if not using YOLO-face)
        if not self.use_yolo_face:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )
            for (x, y, w, h) in faces:
                detections.append({
                    'type': 'face',
                    'bbox': [x, y, x + w, y + h]
                })

        detection_time = time.perf_counter() - start
        self.detection_times.append(detection_time)

        # Cache results
        self.detection_cache[frame_id] = {
            'detections': detections,
            'valid_until': frame_id + self.cache_duration
        }

        # Clean old cache entries
        if len(self.detection_cache) > 30:
            min_frame = frame_id - 30
            self.detection_cache = {
                k: v for k, v in self.detection_cache.items()
                if k >= min_frame
            }

        return detections

    def blur_regions(self, frame, detections):
        """Apply blur to detected regions."""
        blurred = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']

            # Ensure coordinates are within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                # Extract region
                roi = frame[y1:y2, x1:x2]

                # Apply strong blur
                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(
                        roi, (self.blur_strength, self.blur_strength), 0
                    )

                    # Optional: Add pixelation for stronger privacy
                    if detection['type'] in ['face', 'person']:
                        h, w = roi.shape[:2]
                        temp = cv2.resize(roi, (w//10, h//10))
                        blurred_roi = cv2.resize(
                            temp, (w, h),
                            interpolation=cv2.INTER_NEAREST
                        )

                    blurred[y1:y2, x1:x2] = blurred_roi

        return blurred

    def process_video(self, video_path, output_path=None, show_preview=True):
        """
        Process a video file with real-time blur.

        Args:
            video_path: Path to input video
            output_path: Path to save processed video (optional)
            show_preview: Show live preview window
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing: {video_path.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.1f}")
        print(f"Frames: {total_frames}")

        # Setup output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )

        # Process frames
        frame_id = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.perf_counter()

            # Detect sensitive content
            detections = self.detect_sensitive_content(frame, frame_id)

            # Apply blur
            if detections:
                processed = self.blur_regions(frame, detections)
            else:
                processed = frame

            # Calculate FPS
            frame_time = time.perf_counter() - frame_start
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(current_fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # Add info overlay
            info_text = [
                f"FPS: {avg_fps:.1f}",
                f"Detections: {len(detections)}",
                f"Frame: {frame_id}/{total_frames}"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(
                    processed, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2
                )
                y_offset += 30

            # Draw bounding boxes (optional)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(
                    processed, (x1, y1), (x2, y2),
                    (0, 0, 255), 2
                )
                cv2.putText(
                    processed, det['type'], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2
                )

            # Write output
            if out:
                out.write(processed)

            # Show preview (skip if no display)
            if show_preview:
                try:
                    cv2.imshow('RealityGuard - Real-time Blur', processed)
                    # Maintain target FPS
                    wait_time = max(1, int((self.frame_time - frame_time) * 1000))
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # No display available, continue without preview
                    show_preview = False

            frame_id += 1

            # Adaptive quality
            if avg_fps < self.target_fps * 0.8:
                self.skip_frames = True
                self.blur_strength = max(21, self.blur_strength - 10)
            elif avg_fps > self.target_fps * 1.2:
                self.skip_frames = False
                self.blur_strength = min(51, self.blur_strength + 10)

        # Cleanup
        cap.release()
        if out:
            out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # No display to destroy

        # Report performance
        elapsed = time.time() - start_time
        process_fps = frame_id / elapsed

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {frame_id} frames")
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Average FPS: {process_fps:.1f}")
        print(f"Real-time factor: {process_fps/fps:.2f}x")

        if process_fps >= fps:
            print("✅ REAL-TIME ACHIEVED!")
        else:
            print(f"⚠️  Below real-time ({process_fps/fps:.1%})")

        if output_path:
            print(f"Output saved: {output_path}")

        return process_fps

    def process_stream(self, stream_url):
        """Process a live stream (webcam, RTSP, etc.)."""
        cap = cv2.VideoCapture(stream_url)

        frame_id = 0
        print(f"Processing stream: {stream_url}")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or error")
                break

            # Process frame
            detections = self.detect_sensitive_content(frame, frame_id)
            processed = self.blur_regions(frame, detections)

            # Show result
            cv2.imshow('RealityGuard Live Stream', processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass


def create_test_video():
    """Create a test video with faces and objects."""
    print("Creating test video...")

    # Create synthetic video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))

    for i in range(150):  # 5 seconds at 30 FPS
        # Create frame with moving objects
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50

        # Add moving rectangle (simulated laptop)
        x = 100 + i * 2
        cv2.rectangle(frame, (x, 200), (x + 150, 300), (200, 200, 200), -1)
        cv2.putText(frame, 'Laptop', (x + 50, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add face area
        cv2.circle(frame, (320, 100), 50, (255, 200, 150), -1)
        cv2.putText(frame, 'Face', (300, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print("Test video created: test_video.mp4")
    return 'test_video.mp4'


def main():
    """Test the real-time blur system."""
    print("="*60)
    print("REALTIME VIDEO BLUR SYSTEM")
    print("="*60)

    # Initialize system
    blur_system = RealtimeVideoBlur(model_type='yolov8n', target_fps=30)

    # Test options
    print("\nSelect test mode:")
    print("1. Process test video")
    print("2. Process webcam")
    print("3. Create and process synthetic video")

    # For automated testing, use option 3
    choice = '3'

    if choice == '1':
        # Process existing video
        video_path = input("Enter video path: ")
        blur_system.process_video(
            video_path,
            output_path='blurred_output.mp4',
            show_preview=True
        )

    elif choice == '2':
        # Process webcam
        blur_system.process_stream(0)  # 0 for default webcam

    else:
        # Create and process test video
        test_video = create_test_video()
        fps = blur_system.process_video(
            test_video,
            output_path='blurred_test.mp4',
            show_preview=False  # Headless for testing
        )

        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")

        if fps >= 30:
            print(f"✅ REAL-TIME BLUR ACHIEVED: {fps:.1f} FPS")
            print("   System can blur video in real-time during playback")
        else:
            print(f"⚠️  Performance: {fps:.1f} FPS")
            print(f"   Need {30/fps:.1f}x speedup for real-time")


if __name__ == "__main__":
    main()