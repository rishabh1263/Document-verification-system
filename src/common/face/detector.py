from __future__ import annotations

import time

import cv2
from insightface.app import FaceAnalysis

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
        Load InsightFace once.

        The detector instance should be reused for
        multiple documents.
        """

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ],
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

    def detect(
        self,
        image_path: str,
    ) -> FaceDetectionResult:
        """
        Detect all faces from an image file.
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
            FaceDetectionResult
        """

        start_time = time.perf_counter()

        if image is None:
            raise ValueError(
                "Image is empty."
            )

        if len(image.shape) != 3:
            raise ValueError(
                "Image must be a 3-channel BGR image."
            )

        faces = self.app.get(image)

        processing_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        detected_faces = []

        height, width = image.shape[:2]

        for face in faces:

            x1, y1, x2, y2 = map(
                int,
                face.bbox,
            )

            # Keep bounding box inside image.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Calculate face dimensions.
            face_width = x2 - x1
            face_height = y2 - y1

            # Calculate face area.
            face_area = (
                face_width
                * face_height
            )

            detected_faces.append(
                DetectedFace(
                    confidence=float(
                        face.det_score
                    ),
                    bounding_box=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    area=face_area,
                )
            )

        return FaceDetectionResult(
            detected=(
                len(detected_faces) > 0
            ),
            face_count=len(
                detected_faces
            ),
            faces=detected_faces,
            processing_time_ms=(
                processing_time_ms
            ),
        )