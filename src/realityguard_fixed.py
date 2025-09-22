"""
RealityGuard Fixed - Actually working privacy filtering
This version properly applies filtering instead of skipping frames
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional
from pathlib import Path

try:
    from .config import Config, PrivacyMode, get_config
    from .face_detector import ModernFaceDetector, FaceBlurrer
except ImportError:
    from config import Config, PrivacyMode, get_config
    from face_detector import ModernFaceDetector, FaceBlurrer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealityGuardFixed:
    """Fixed version that actually applies privacy filtering"""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with proper filtering"""
        self.config = get_config() if not config_path else Config(config_path)
        self.face_detector = ModernFaceDetector(method="mediapipe")  # Use fastest
        self.privacy_mode = PrivacyMode.SMART
        self.frame_count = 0
        self.last_processed = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with actual filtering applied"""

        # Get privacy settings
        privacy_settings = self.config.get_privacy_settings(self.privacy_mode)

        # Only skip if OFF mode
        if self.privacy_mode == PrivacyMode.OFF:
            return frame

        output = frame.copy()

        # Process every Nth frame for detection, but ALWAYS apply filtering
        self.frame_count += 1
        should_detect = (self.frame_count % self.config.detection.frame_skip_interval) == 0

        if should_detect or self.last_processed is None:
            # Do new detection

            # Downscale for detection performance
            small_frame = cv2.resize(frame, None,
                                    fx=self.config.detection.downscale_factor,
                                    fy=self.config.detection.downscale_factor)

            # Detect faces
            if privacy_settings.get('blur_faces', False):
                faces = self.face_detector.detect_faces(small_frame, use_cache=False)

                # Scale faces back up
                scale_factor = 1.0 / self.config.detection.downscale_factor
                for face in faces:
                    x, y, w, h = face.bbox
                    face.bbox = (
                        int(x * scale_factor),
                        int(y * scale_factor),
                        int(w * scale_factor),
                        int(h * scale_factor)
                    )

                # Apply blur
                output = FaceBlurrer.blur_faces(output, faces, blur_type="gaussian", blur_strength=31)
                self.last_processed = {'faces': faces}

            # Detect and blur screens
            if privacy_settings.get('blur_screens', False):
                screens = self._detect_screens(small_frame)

                # Scale and apply pixelation
                scale_factor = 1.0 / self.config.detection.downscale_factor
                for x, y, w, h in screens:
                    x = int(x * scale_factor)
                    y = int(y * scale_factor)
                    w = int(w * scale_factor)
                    h = int(h * scale_factor)

                    # Pixelate screen area
                    roi = output[y:y+h, x:x+w]
                    if roi.size > 0:
                        temp = cv2.resize(roi, (max(1, w//20), max(1, h//20)), cv2.INTER_LINEAR)
                        pixelated = cv2.resize(temp, (w, h), cv2.INTER_NEAREST)
                        output[y:y+h, x:x+w] = pixelated

                if self.last_processed:
                    self.last_processed['screens'] = screens
                else:
                    self.last_processed = {'screens': screens}

        elif self.last_processed:
            # Reuse last detection but apply to current frame
            if 'faces' in self.last_processed and privacy_settings.get('blur_faces', False):
                output = FaceBlurrer.blur_faces(output, self.last_processed['faces'],
                                               blur_type="gaussian", blur_strength=31)

            if 'screens' in self.last_processed and privacy_settings.get('blur_screens', False):
                for x, y, w, h in self.last_processed['screens']:
                    roi = output[y:y+h, x:x+w]
                    if roi.size > 0:
                        temp = cv2.resize(roi, (max(1, w//20), max(1, h//20)), cv2.INTER_LINEAR)
                        pixelated = cv2.resize(temp, (w, h), cv2.INTER_NEAREST)
                        output[y:y+h, x:x+w] = pixelated

        return output

    def _detect_screens(self, frame: np.ndarray) -> list:
        """Simple screen detection that actually works"""
        screens = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find bright regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum screen size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Check if aspect ratio looks like a screen
                if 0.5 < aspect_ratio < 3.0:
                    screens.append((x, y, w, h))

        return screens

    def set_privacy_mode(self, mode: PrivacyMode):
        """Set privacy mode"""
        self.privacy_mode = mode
        logger.info(f"Privacy mode set to: {mode.name}")

    def cleanup(self):
        """Clean up resources"""
        if self.face_detector:
            self.face_detector.release()


def test_fixed_version():
    """Test that filtering actually works"""
    print("\n" + "="*60)
    print("TESTING FIXED VERSION")
    print("="*60)

    guard = RealityGuardFixed()

    # Create test image with bright rectangle (screen)
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.rectangle(test_img, (200, 150), (440, 330), (250, 250, 250), -1)

    # Test OFF mode
    guard.set_privacy_mode(PrivacyMode.OFF)
    off_result = guard.process_frame(test_img)

    # Test MAXIMUM mode
    guard.set_privacy_mode(PrivacyMode.MAXIMUM)
    max_result = guard.process_frame(test_img)

    # Check difference
    diff = np.mean(np.abs(max_result.astype(float) - off_result.astype(float)))

    if diff > 0:
        print(f"✅ FILTERING WORKS! (difference: {diff:.2f})")
    else:
        print(f"❌ Filtering still not working (difference: {diff:.2f})")

    # Test performance
    guard.set_privacy_mode(PrivacyMode.SMART)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = guard.process_frame(test_img)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(f"\nPerformance with filtering: {avg_time*1000:.2f}ms = {fps:.1f} FPS")

    if fps >= 120:
        print(f"✅ Still meets 120 FPS requirement!")
    else:
        print(f"⚠️  Below 120 FPS with filtering enabled")

    guard.cleanup()


if __name__ == "__main__":
    test_fixed_version()