"""Modern face detection module using YOLO and MediaPipe."""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


@dataclass
class Face:
    """Represents a detected face."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[np.ndarray] = None
    face_id: Optional[int] = None
    is_known: bool = False


class ModernFaceDetector:
    """Modern face detection using YOLO or MediaPipe as fallback."""

    def __init__(self, method: str = "auto", config: Optional[dict] = None):
        """Initialize face detector.

        Args:
            method: Detection method ('yolo', 'mediapipe', 'auto')
            config: Optional configuration dictionary
        """
        self.method = method
        self.config = config or {}
        self.detector = None
        self.face_cache: Dict[int, Face] = {}
        self.cache_ttl = self.config.get('cache_ttl', 0.5)  # 500ms cache
        self.last_cache_time = 0

        # Initialize detector based on availability and preference
        if method == "auto":
            if YOLO_AVAILABLE:
                self._init_yolo()
            elif MEDIAPIPE_AVAILABLE:
                self._init_mediapipe()
            else:
                self._init_opencv()  # Fallback to OpenCV
        elif method == "yolo" and YOLO_AVAILABLE:
            self._init_yolo()
        elif method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            self._init_opencv()

    def _init_yolo(self):
        """Initialize YOLO face detector."""
        try:
            # Use YOLO face model if available
            model_path = self.config.get('yolo_model', 'yolov8n-face.pt')
            if not Path(model_path).exists():
                # Download model if not exists
                self.detector = YOLO('yolov8n.pt')  # Use general model as fallback
            else:
                self.detector = YOLO(model_path)
            self.method = "yolo"
            print("Using YOLO for face detection")
        except Exception as e:
            print(f"Failed to initialize YOLO: {e}")
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe face detector."""
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Long-range model
                min_detection_confidence=self.config.get('confidence', 0.5)
            )
            self.method = "mediapipe"
            print("Using MediaPipe for face detection")
        else:
            self._init_opencv()

    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade as fallback."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.method = "opencv"
        print("Using OpenCV Haar Cascade for face detection (fallback)")

    def detect_faces(self, image: np.ndarray, use_cache: bool = True) -> List[Face]:
        """Detect faces in image.

        Args:
            image: Input image (BGR format)
            use_cache: Whether to use cached results

        Returns:
            List of detected faces
        """
        # Check cache
        current_time = time.time()
        if use_cache and (current_time - self.last_cache_time) < self.cache_ttl:
            return list(self.face_cache.values())

        faces = []

        if self.method == "yolo":
            faces = self._detect_yolo(image)
        elif self.method == "mediapipe":
            faces = self._detect_mediapipe(image)
        else:
            faces = self._detect_opencv(image)

        # Update cache
        self.face_cache = {i: face for i, face in enumerate(faces)}
        self.last_cache_time = current_time

        return faces

    def _detect_yolo(self, image: np.ndarray) -> List[Face]:
        """Detect faces using YOLO.

        Args:
            image: Input image

        Returns:
            List of detected faces
        """
        faces = []

        try:
            # Run inference
            results = self.detector(image, conf=self.config.get('confidence', 0.5))

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if this is a person/face detection
                        cls = int(box.cls[0])
                        if cls == 0:  # Person class in COCO
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])

                            # Convert to x, y, w, h format
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                            # Estimate face region from person detection
                            # (This is a simplification - ideally use a face-specific model)
                            face_y = y
                            face_h = min(h // 3, h)  # Assume face is top portion

                            faces.append(Face(
                                bbox=(x, face_y, w, face_h),
                                confidence=conf
                            ))

        except Exception as e:
            print(f"YOLO detection error: {e}")

        return faces

    def _detect_mediapipe(self, image: np.ndarray) -> List[Face]:
        """Detect faces using MediaPipe.

        Args:
            image: Input image

        Returns:
            List of detected faces
        """
        faces = []

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            # Process image
            results = self.detector.process(rgb_image)

            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    # Get landmarks if available
                    landmarks = None
                    if detection.location_data.relative_keypoints:
                        landmarks = np.array([
                            [kp.x * width, kp.y * height]
                            for kp in detection.location_data.relative_keypoints
                        ])

                    faces.append(Face(
                        bbox=(x, y, w, h),
                        confidence=detection.score[0] if detection.score else 0.9,
                        landmarks=landmarks
                    ))

        except Exception as e:
            print(f"MediaPipe detection error: {e}")

        return faces

    def _detect_opencv(self, image: np.ndarray) -> List[Face]:
        """Detect faces using OpenCV Haar Cascade.

        Args:
            image: Input image

        Returns:
            List of detected faces
        """
        faces = []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            detected = self.detector.detectMultiScale(
                gray,
                scaleFactor=self.config.get('scale_factor', 1.1),
                minNeighbors=self.config.get('min_neighbors', 5),
                minSize=tuple(self.config.get('min_size', [30, 30]))
            )

            for (x, y, w, h) in detected:
                faces.append(Face(
                    bbox=(x, y, w, h),
                    confidence=0.8  # Haar cascade doesn't provide confidence
                ))

        except Exception as e:
            print(f"OpenCV detection error: {e}")

        return faces

    def identify_known_faces(self, image: np.ndarray, faces: List[Face],
                           known_faces: Dict[int, np.ndarray]) -> List[Face]:
        """Identify known faces using feature matching.

        Args:
            image: Input image
            faces: List of detected faces
            known_faces: Dictionary of known face features

        Returns:
            Updated list of faces with identification
        """
        for face in faces:
            x, y, w, h = face.bbox
            face_roi = image[y:y+h, x:x+w]

            # Simple histogram comparison for known face matching
            # In production, use face embeddings from deep learning models
            face_hist = cv2.calcHist([face_roi], [0, 1, 2], None,
                                    [8, 8, 8], [0, 256, 0, 256, 0, 256])
            face_hist = cv2.normalize(face_hist, face_hist).flatten()

            # Compare with known faces
            best_match = None
            best_score = 0.7  # Threshold

            for face_id, known_hist in known_faces.items():
                score = cv2.compareHist(face_hist, known_hist, cv2.HISTCMP_CORREL)
                if score > best_score:
                    best_score = score
                    best_match = face_id

            if best_match is not None:
                face.face_id = best_match
                face.is_known = True

        return faces

    def draw_faces(self, image: np.ndarray, faces: List[Face],
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw face bounding boxes on image.

        Args:
            image: Input image
            faces: List of faces to draw
            color: Box color (BGR)

        Returns:
            Image with drawn faces
        """
        output = image.copy()

        for face in faces:
            x, y, w, h = face.bbox

            # Different color for known faces
            box_color = (0, 255, 0) if face.is_known else color

            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)

            # Draw confidence
            label = f"Face {face.confidence:.2f}"
            if face.is_known:
                label = f"Known {face.face_id}: {face.confidence:.2f}"

            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Draw landmarks if available
            if face.landmarks is not None:
                for point in face.landmarks:
                    cv2.circle(output, tuple(point.astype(int)), 2, (0, 0, 255), -1)

        return output

    def release(self):
        """Release resources."""
        if self.method == "mediapipe" and self.detector:
            self.detector.close()
        self.face_cache.clear()


class FaceBlurrer:
    """Utility class for applying blur to faces."""

    @staticmethod
    def blur_faces(image: np.ndarray, faces: List[Face],
                  blur_type: str = "gaussian",
                  blur_strength: int = 21) -> np.ndarray:
        """Apply blur to detected faces.

        Args:
            image: Input image
            faces: List of faces to blur
            blur_type: Type of blur ('gaussian', 'pixelate', 'solid')
            blur_strength: Blur kernel size or pixelation factor

        Returns:
            Image with blurred faces
        """
        output = image.copy()

        for face in faces:
            if face.is_known:
                continue  # Skip known faces

            x, y, w, h = face.bbox

            # Ensure bounds are within image
            x = max(0, x)
            y = max(0, y)
            x_end = min(image.shape[1], x + w)
            y_end = min(image.shape[0], y + h)

            face_roi = output[y:y_end, x:x_end]

            if blur_type == "gaussian":
                # Apply Gaussian blur
                kernel_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
                blurred = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
                output[y:y_end, x:x_end] = blurred

            elif blur_type == "pixelate":
                # Pixelate effect
                temp = cv2.resize(face_roi,
                                (max(1, w // blur_strength), max(1, h // blur_strength)),
                                interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                output[y:y_end, x:x_end] = pixelated

            elif blur_type == "solid":
                # Solid color fill
                output[y:y_end, x:x_end] = (128, 128, 128)  # Gray fill

        return output