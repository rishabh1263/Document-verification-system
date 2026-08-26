from __future__ import annotations

import time

import cv2

from .model import get_face_app
from .schemas import (
    DetectedFace,
    FaceDetectionResult,
)


class FaceDetector:
    """
    Common Phase 1 face detection service.

    Responsibilities:
    - Detect all faces
    - Return face count
    - Return confidence for every face
    - Return bounding box for every face
    - Return face area
    - Measure detection latency

    Not responsible for:
    - Face embeddings
    - Face matching
    - Liveness detection
    - Deepfake detection
    """

    def __init__(self):
        """
        Load the shared InsightFace model.

        The actual FaceAnalysis instance is created only once
        per Python process by get_face_app().
        """

        self.app = get_face_app()

    def detect(
        self,
        image_path: str,
    ) -> FaceDetectionResult:
        """
        Detect all faces from an image file.

        Args:
            image_path:
                Path to the image file.

        Returns:
            FaceDetectionResult containing detected faces,
            count, and detection latency.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return self.detect_image(image)

    def detect_image(
        self,
        image,
    ) -> FaceDetectionResult:
        """
        Detect all faces from an OpenCV BGR image.

        Args:
            image:
                OpenCV BGR image.

        Returns:
            FaceDetectionResult.
        """

        start_time = time.perf_counter()

        faces = self.app.get(image)

        detection_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        detected_faces = []

        for face in faces:

            bbox = face.bbox

            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            confidence = float(
                face.det_score
            )

            detected_faces.append(
                DetectedFace(
                    bbox=[
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    confidence=confidence,
                    area=area,
                )
            )

        return FaceDetectionResult(
            faces=detected_faces,
            face_count=len(detected_faces),
            detection_time_ms=detection_time_ms,
        )