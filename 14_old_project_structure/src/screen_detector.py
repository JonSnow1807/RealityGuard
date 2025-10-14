"""
Screen Detection Module
Detects computer screens, phones, and other displays in images
"""

import cv2
import numpy as np
from typing import List, Tuple


class ScreenDetector:
    """Detects screens and displays in images."""

    def __init__(self):
        """Initialize screen detector."""
        self.min_area = 5000  # Minimum area for screen detection
        self.brightness_threshold = 200  # Threshold for bright regions

    def detect_screens(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect screens in image.

        Args:
            image: Input image (BGR)

        Returns:
            List of screen bounding boxes (x, y, width, height)
        """
        screens = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find bright regions (screens are usually bright)
        _, thresh = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio (screens usually have standard ratios)
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 3.0:  # Common screen aspect ratios

                # Additional validation: check if region is relatively uniform (screen-like)
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    std_dev = np.std(roi)
                    mean_val = np.mean(roi)

                    # Screens tend to have high brightness with moderate variation
                    if mean_val > 180 and std_dev < 80:
                        screens.append((x, y, w, h))

        # Merge overlapping screens
        screens = self._merge_overlapping(screens)

        return screens

    def _merge_overlapping(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes.

        Args:
            boxes: List of bounding boxes

        Returns:
            Merged bounding boxes
        """
        if not boxes:
            return boxes

        # Sort by area (larger first)
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue

            x1, y1, w1, h1 = box1
            merged_box = list(box1)

            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue

                x2, y2, w2, h2 = box2

                # Check for overlap
                if self._boxes_overlap(box1, box2):
                    # Merge boxes
                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1 + w1, x2 + w2)
                    y_max = max(y1 + h1, y2 + h2)

                    merged_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    used.add(j)

            merged.append(tuple(merged_box))
            used.add(i)

        return merged

    def _boxes_overlap(self, box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int]) -> bool:
        """Check if two boxes overlap.

        Args:
            box1: First bounding box (x, y, w, h)
            box2: Second bounding box (x, y, w, h)

        Returns:
            True if boxes overlap
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Check if boxes overlap
        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):
            return True

        return False

    def detect_screens_edge(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect screens using edge detection (alternative method).

        Args:
            image: Input image (BGR)

        Returns:
            List of screen bounding boxes
        """
        screens = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Look for rectangular shapes (4 vertices)
            if len(approx) == 4:
                area = cv2.contourArea(approx)

                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0

                    if 0.5 < aspect_ratio < 3.0:
                        screens.append((x, y, w, h))

        return screens

    def detect_glare_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect glare regions that might be reflective screens.

        Args:
            image: Input image (BGR)

        Returns:
            List of glare region bounding boxes
        """
        glare_regions = []

        # Convert to HSV for better glare detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract value channel (brightness)
        _, _, v = cv2.split(hsv)

        # Threshold for very bright regions
        _, bright = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Smaller threshold for glare
                x, y, w, h = cv2.boundingRect(contour)
                glare_regions.append((x, y, w, h))

        return glare_regions


def test_screen_detector():
    """Test screen detection."""
    print("Testing Screen Detector...")

    # Create test image with screens
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 50

    # Add bright rectangles (screens)
    cv2.rectangle(test_img, (50, 50), (250, 200), (230, 230, 230), -1)
    cv2.rectangle(test_img, (300, 100), (600, 400), (240, 240, 240), -1)

    # Add some text on screens
    cv2.putText(test_img, "SCREEN 1", (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_img, "DISPLAY", (400, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    detector = ScreenDetector()
    screens = detector.detect_screens(test_img)

    print(f"Detected {len(screens)} screens:")
    for i, (x, y, w, h) in enumerate(screens):
        print(f"  Screen {i+1}: x={x}, y={y}, width={w}, height={h}")

    # Visualize
    vis = test_img.copy()
    for x, y, w, h in screens:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite("screen_detection_test.png", vis)
    print("Saved visualization to screen_detection_test.png")


if __name__ == "__main__":
    test_screen_detector()