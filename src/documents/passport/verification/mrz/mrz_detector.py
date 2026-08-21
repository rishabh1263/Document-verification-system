import cv2
import numpy as np


class MRZDetector:
    """
    ICAO TD3 MRZ detector.

    Detects the Machine Readable Zone located at the bottom
    of passport biodata pages.
    """

    @staticmethod
    def detect(image_path: str):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to load image.")

        height, width = image.shape[:2]

        # Bottom 25% of the page
        start_y = int(height * 0.75)

        roi = image[start_y:height, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Improve text extraction
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

        # Merge characters into lines
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (25, 3)
        )

        merged = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / max(h, 1)

            area = w * h

            # Expected MRZ characteristics
            if (
                aspect_ratio > 8
                and
                w > width * 0.50
                and
                h > 20
                and
                area > 5000
            ):
                candidates.append((x, y, w, h))

        if not candidates:

            return {
                "passed": False,
                "confidence": 0.0,
                "bounding_box": None,
                "text_regions": 0,
                "reason": "MRZ region not detected."
            }

        # Select largest candidate
        x, y, w, h = max(
            candidates,
            key=lambda c: c[2] * c[3]
        )

        confidence = min(
            1.0,
            (w / width) * (h / (height * 0.25))
        )

        return {

            "passed": True,

            "confidence": round(confidence, 3),

            "bounding_box": {

                "x": int(x),

                "y": int(y + start_y),

                "width": int(w),

                "height": int(h)

            },

            "text_regions": len(candidates)

        }
