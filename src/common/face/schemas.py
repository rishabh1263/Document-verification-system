from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DetectedFace:
    """
    Information about one detected face.
    """

    confidence: float
    bounding_box: Tuple[int, int, int, int]
    area: int


@dataclass
class FaceDetectionResult:
    """
    Result returned by the common face detector.
    """

    detected: bool
    face_count: int
    faces: List[DetectedFace]
    processing_time_ms: float = 0.0