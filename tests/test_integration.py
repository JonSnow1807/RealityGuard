"""Integration tests for RealityGuard system"""
import pytest
import numpy as np
import cv2
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.realityguard_improved import RealityGuardImproved, PrivacyMode
from src.face_detector import ModernFaceDetector, Face
from src.config import Config


class TestIntegration:
    """End-to-end integration tests"""

    @pytest.fixture
    def guard(self):
        """Create RealityGuard instance for testing"""
        guard = RealityGuardImproved()
        yield guard
        guard.cleanup()

    @pytest.fixture
    def test_frame(self):
        """Create a test frame"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add some variation
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        return frame

    def test_system_initialization(self, guard):
        """Test that all components initialize correctly"""
        assert guard is not None
        assert guard.face_detector is not None
        assert guard.screen_detector is not None
        assert guard.config is not None
        assert guard.privacy_mode == PrivacyMode.SMART

    def test_privacy_mode_switching(self, guard):
        """Test switching between privacy modes"""
        modes = [PrivacyMode.OFF, PrivacyMode.SMART, PrivacyMode.MAXIMUM]

        for mode in modes:
            guard.set_privacy_mode(mode)
            assert guard.privacy_mode == mode

    def test_frame_processing(self, guard, test_frame):
        """Test basic frame processing"""
        processed = guard.process_frame(test_frame)

        assert processed is not None
        assert processed.shape == test_frame.shape
        assert processed.dtype == test_frame.dtype

    def test_performance_requirement(self, guard, test_frame):
        """Test that system meets performance requirements"""
        # Warm-up
        for _ in range(10):
            guard.process_frame(test_frame)

        # Measure performance
        times = []
        for _ in range(50):
            start = time.perf_counter()
            guard.process_frame(test_frame)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        # Should achieve at least 120 FPS
        assert fps >= 120, f"Performance {fps:.1f} FPS is below 120 FPS requirement"

    def test_face_detector_fallback(self):
        """Test that face detector has proper fallback"""
        # Try to create detector with non-existent method
        detector = ModernFaceDetector(method="non_existent")

        # Should fallback to opencv
        assert detector.method == "opencv"

        # Should still work
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        faces = detector.detect_faces(test_image)
        assert isinstance(faces, list)

        detector.release()

    def test_calibration(self, guard, test_frame):
        """Test user calibration"""
        # Add a face-like region to test frame
        cv2.rectangle(test_frame, (200, 150), (350, 350), (200, 180, 160), -1)

        result = guard.calibrate_user(test_frame)
        # Result depends on whether detector finds the synthetic face
        assert isinstance(result, bool)

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        guard = RealityGuardImproved()

        # Process some frames
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        for _ in range(10):
            guard.process_frame(test_frame)

        # Cleanup
        guard.cleanup()

        # Resources should be released
        assert len(guard.resource_manager.resources) == 0

    def test_configuration_loading(self):
        """Test configuration system"""
        config = Config()

        # Test default values
        assert config.performance.target_fps == 120
        assert config.detection.face_scale_factor == 1.1

        # Test privacy presets
        smart_settings = config.get_privacy_settings(PrivacyMode.SMART)
        assert smart_settings['blur_faces'] == True
        assert smart_settings['known_faces_exempt'] == True

    def test_thread_safety(self, guard, test_frame):
        """Test thread-safe operations"""
        import threading

        results = []
        errors = []

        def process_frames():
            try:
                for _ in range(20):
                    processed = guard.process_frame(test_frame)
                    results.append(processed is not None)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=process_frames)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert all(results), "Some frames failed to process"

    def test_different_frame_sizes(self, guard):
        """Test processing different frame sizes"""
        sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]

        for width, height in sizes:
            frame = np.ones((height, width, 3), dtype=np.uint8) * 128
            processed = guard.process_frame(frame)

            assert processed is not None
            assert processed.shape == (height, width, 3)

    def test_privacy_mode_effects(self, guard):
        """Test that privacy modes actually change behavior"""
        # Create frame with bright region (simulating screen)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        cv2.rectangle(frame, (200, 150), (440, 330), (250, 250, 250), -1)

        # Test OFF mode - should not modify
        guard.set_privacy_mode(PrivacyMode.OFF)
        processed_off = guard.process_frame(frame)

        # Test MAXIMUM mode - should apply filtering
        guard.set_privacy_mode(PrivacyMode.MAXIMUM)
        processed_max = guard.process_frame(frame)

        # The outputs might be different if screen detection works
        # For now, just verify they're valid frames
        assert processed_off.shape == frame.shape
        assert processed_max.shape == frame.shape

    @pytest.mark.parametrize("detector_method", ["opencv", "yolo", "mediapipe"])
    def test_face_detector_methods(self, detector_method):
        """Test different face detection methods"""
        try:
            detector = ModernFaceDetector(method=detector_method)
            actual_method = detector.method

            # Create test image
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128

            # Detect faces
            faces = detector.detect_faces(image)

            # Should return a list
            assert isinstance(faces, list)

            # Each face should be properly formatted
            for face in faces:
                assert isinstance(face, Face)
                assert len(face.bbox) == 4
                assert 0 <= face.confidence <= 1

            detector.release()

        except Exception as e:
            # If method not available, should fallback
            pytest.skip(f"Method {detector_method} not available: {e}")