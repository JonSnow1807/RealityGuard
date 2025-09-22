"""Unit tests for face detection module."""
import pytest
import numpy as np
import cv2
import sys
sys.path.append('../src')

from src.face_detector import ModernFaceDetector, FaceBlurrer, Face


class TestModernFaceDetector:
    """Test modern face detection."""

    @pytest.fixture
    def detector(self):
        """Create a face detector instance."""
        return ModernFaceDetector(method="opencv")  # Use OpenCV for testing

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        # Create a simple test image (640x480 RGB)
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add some variation to simulate a real image
        image[100:300, 200:400] = 255  # Bright region
        return image

    def test_detector_initialization(self):
        """Test detector initialization with different methods."""
        # Test auto mode
        detector = ModernFaceDetector(method="auto")
        assert detector.method in ["yolo", "mediapipe", "opencv"]

        # Test OpenCV fallback
        detector = ModernFaceDetector(method="opencv")
        assert detector.method == "opencv"
        assert detector.detector is not None

    def test_detect_faces_empty_image(self, detector):
        """Test face detection on empty image."""
        # Create empty image
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        faces = detector.detect_faces(image)

        # Should return empty list for blank image
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_detect_faces_with_cache(self, detector, test_image):
        """Test face detection caching."""
        # First detection
        faces1 = detector.detect_faces(test_image, use_cache=True)

        # Second detection (should use cache)
        faces2 = detector.detect_faces(test_image, use_cache=True)

        # Should return the same cached results
        assert len(faces1) == len(faces2)

        # Detection without cache
        faces3 = detector.detect_faces(test_image, use_cache=False)

        # Should perform new detection
        assert isinstance(faces3, list)

    def test_face_object_structure(self):
        """Test Face dataclass structure."""
        face = Face(
            bbox=(10, 20, 100, 120),
            confidence=0.95,
            face_id=1,
            is_known=True
        )

        assert face.bbox == (10, 20, 100, 120)
        assert face.confidence == 0.95
        assert face.face_id == 1
        assert face.is_known == True

    def test_identify_known_faces(self, detector, test_image):
        """Test known face identification."""
        # Create mock known faces
        known_faces = {
            0: np.random.rand(512)  # Mock histogram
        }

        # Create mock detected faces
        faces = [
            Face(bbox=(100, 100, 50, 50), confidence=0.9)
        ]

        # Identify faces
        identified = detector.identify_known_faces(test_image, faces, known_faces)

        assert isinstance(identified, list)
        assert len(identified) == len(faces)

    def test_draw_faces(self, detector, test_image):
        """Test drawing face bounding boxes."""
        faces = [
            Face(bbox=(100, 100, 50, 50), confidence=0.9, is_known=False),
            Face(bbox=(200, 200, 60, 60), confidence=0.8, face_id=1, is_known=True)
        ]

        # Draw faces
        output = detector.draw_faces(test_image, faces)

        # Output should be same shape as input
        assert output.shape == test_image.shape
        assert output.dtype == test_image.dtype

        # Image should be modified (not same as input)
        assert not np.array_equal(output, test_image)

    def test_release_resources(self, detector):
        """Test resource cleanup."""
        # Add some cache
        detector.face_cache = {0: Face(bbox=(0, 0, 10, 10), confidence=0.5)}

        # Release resources
        detector.release()

        # Cache should be cleared
        assert len(detector.face_cache) == 0


class TestFaceBlurrer:
    """Test face blurring functionality."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with recognizable patterns."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add some patterns to verify blurring
        cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
        cv2.rectangle(image, (300, 300), (400, 400), (0, 255, 0), -1)

        return image

    @pytest.fixture
    def test_faces(self):
        """Create test face objects."""
        return [
            Face(bbox=(100, 100, 100, 100), confidence=0.9, is_known=False),
            Face(bbox=(300, 300, 100, 100), confidence=0.8, is_known=True)
        ]

    def test_gaussian_blur(self, test_image, test_faces):
        """Test Gaussian blur application."""
        output = FaceBlurrer.blur_faces(
            test_image, test_faces,
            blur_type="gaussian",
            blur_strength=21
        )

        # Output shape should match input
        assert output.shape == test_image.shape

        # First face region should be blurred (unknown face)
        face1_roi = output[100:200, 100:200]
        original1_roi = test_image[100:200, 100:200]

        # Blurred region should be different from original
        assert not np.array_equal(face1_roi, original1_roi)

        # Second face should NOT be blurred (known face)
        face2_roi = output[300:400, 300:400]
        original2_roi = test_image[300:400, 300:400]

        # Known face region should be unchanged
        assert np.array_equal(face2_roi, original2_roi)

    def test_pixelate_blur(self, test_image, test_faces):
        """Test pixelation effect."""
        # Make all faces unknown for this test
        for face in test_faces:
            face.is_known = False

        output = FaceBlurrer.blur_faces(
            test_image, test_faces,
            blur_type="pixelate",
            blur_strength=10
        )

        # Output shape should match input
        assert output.shape == test_image.shape

        # Both face regions should be pixelated
        for face in test_faces:
            x, y, w, h = face.bbox
            roi = output[y:y+h, x:x+w]
            original_roi = test_image[y:y+h, x:x+w]

            # Pixelated region should be different
            assert not np.array_equal(roi, original_roi)

    def test_solid_blur(self, test_image, test_faces):
        """Test solid color fill."""
        # Make all faces unknown
        for face in test_faces:
            face.is_known = False

        output = FaceBlurrer.blur_faces(
            test_image, test_faces,
            blur_type="solid",
            blur_strength=0  # Not used for solid
        )

        # Check that face regions are filled with gray
        for face in test_faces:
            x, y, w, h = face.bbox
            roi = output[y:y+h, x:x+w]

            # Should be filled with gray (128, 128, 128)
            assert np.all(roi == 128)

    def test_blur_with_out_of_bounds(self, test_image):
        """Test blurring with face partially out of bounds."""
        faces = [
            # Face partially out of left boundary
            Face(bbox=(-10, 100, 50, 50), confidence=0.9, is_known=False),
            # Face partially out of bottom boundary
            Face(bbox=(100, 450, 50, 50), confidence=0.8, is_known=False)
        ]

        # Should handle out-of-bounds gracefully
        output = FaceBlurrer.blur_faces(test_image, faces, blur_type="gaussian")

        # Should not crash and return valid image
        assert output.shape == test_image.shape

    def test_empty_faces_list(self, test_image):
        """Test blurring with no faces."""
        output = FaceBlurrer.blur_faces(test_image, [], blur_type="gaussian")

        # Output should be identical to input
        assert np.array_equal(output, test_image)