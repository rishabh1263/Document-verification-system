"""
Signature validation decision layer.

This module combines:

    - blank detection
    - foreground features
    - basic geometry checks

It is intentionally lightweight.

IMPORTANT:
This is a Phase-1 deterministic validator.
It does NOT replace the ML signature classifier.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.documents.signature.validators.blank_detector import (
    detect_blank
)

from src.documents.signature.validators.signature_features import (
    extract_signature_features,
    SignatureFeatures
)


# =========================================================
# Decision
# =========================================================

class SignatureDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


# =========================================================
# Result
# =========================================================

@dataclass
class SignatureValidationResult:
    """
    Final Phase-1 validation result.
    """

    decision: SignatureDecision

    confidence: float

    reason_code: str

    message: str

    features: Optional[SignatureFeatures] = None


# =========================================================
# Configuration
# =========================================================

# Foreground density
#
# Your real signature:
#
#     0.004079
#
MIN_FOREGROUND_DENSITY = 0.0005

MAX_FOREGROUND_DENSITY = 0.20


# Candidate bounding box
MIN_BBOX_WIDTH = 20
MIN_BBOX_HEIGHT = 10

MAX_BBOX_IMAGE_RATIO = 0.85


# Signature geometry
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 12.0


# Occupancy
MIN_OCCUPANCY = 0.005
MAX_OCCUPANCY = 0.80


# Component count
MAX_COMPONENTS = 300


# =========================================================
# Helper
# =========================================================

def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0
) -> float:
    """
    Clamp a number between minimum and maximum.
    """

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# =========================================================
# Main validator
# =========================================================

def validate_signature(
    image: np.ndarray
) -> SignatureValidationResult:
    """
    Validate whether an image contains a plausible
    signature candidate.

    Parameters
    ----------
    image:
        OpenCV image as NumPy array.

    Returns
    -------
    SignatureValidationResult
    """

    # -----------------------------------------------------
    # Basic input validation
    # -----------------------------------------------------

    if image is None:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.99,
            reason_code="INVALID_IMAGE",
            message="Image is missing."
        )

    if not isinstance(
        image,
        np.ndarray
    ):

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.99,
            reason_code="INVALID_IMAGE",
            message="Image must be a NumPy array."
        )

    if image.size == 0:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.99,
            reason_code="EMPTY_IMAGE",
            message="Image is empty."
        )

    # -----------------------------------------------------
    # Blank detection
    # -----------------------------------------------------

    blank_result = detect_blank(
        image
    )

    if blank_result.is_blank:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.99,
            reason_code=blank_result.reason_code,
            message=blank_result.message
        )

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    try:

        features = extract_signature_features(
            image
        )

    except ValueError as exc:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.99,
            reason_code="FEATURE_EXTRACTION_FAILED",
            message=str(exc)
        )

    # -----------------------------------------------------
    # Foreground density
    # -----------------------------------------------------

    density = (
        features.foreground_density
    )

    if density < MIN_FOREGROUND_DENSITY:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.95,
            reason_code="INSUFFICIENT_FOREGROUND",
            message=(
                "No meaningful signature-like "
                "foreground was detected."
            ),
            features=features
        )

    if density > MAX_FOREGROUND_DENSITY:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.95,
            reason_code="EXCESSIVE_FOREGROUND",
            message=(
                "Image contains too much foreground "
                "content to be a normal signature."
            ),
            features=features
        )

    # -----------------------------------------------------
    # Bounding box
    # -----------------------------------------------------

    if (
        features.bbox_width is None
        or
        features.bbox_height is None
    ):

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.95,
            reason_code="NO_SIGNATURE_REGION",
            message=(
                "No meaningful foreground region "
                "was detected."
            ),
            features=features
        )

    bbox_width = features.bbox_width
    bbox_height = features.bbox_height

    if bbox_width < MIN_BBOX_WIDTH:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.90,
            reason_code="SIGNATURE_REGION_TOO_NARROW",
            message=(
                "Detected foreground region is too narrow."
            ),
            features=features
        )

    if bbox_height < MIN_BBOX_HEIGHT:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.90,
            reason_code="SIGNATURE_REGION_TOO_SMALL",
            message=(
                "Detected foreground region is too small."
            ),
            features=features
        )

    # -----------------------------------------------------
    # Region size relative to image
    # -----------------------------------------------------

    image_area = (
        features.image_width *
        features.image_height
    )

    bbox_area = (
        bbox_width *
        bbox_height
    )

    bbox_image_ratio = (
        bbox_area /
        image_area
    )

    if bbox_image_ratio > MAX_BBOX_IMAGE_RATIO:

        return SignatureValidationResult(
            decision=SignatureDecision.REJECT,
            confidence=0.95,
            reason_code="REGION_TOO_LARGE",
            message=(
                "Foreground region occupies an "
                "unusually large part of the image."
            ),
            features=features
        )

    # -----------------------------------------------------
    # Aspect ratio
    # -----------------------------------------------------

    aspect_ratio = (
        features.aspect_ratio
    )

    if aspect_ratio is None:

        return SignatureValidationResult(
            decision=SignatureDecision.REVIEW,
            confidence=0.55,
            reason_code="MISSING_ASPECT_RATIO",
            message=(
                "Signature geometry could not be "
                "determined confidently."
            ),
            features=features
        )

    if (
        aspect_ratio < MIN_ASPECT_RATIO
        or
        aspect_ratio > MAX_ASPECT_RATIO
    ):

        return SignatureValidationResult(
            decision=SignatureDecision.REVIEW,
            confidence=0.65,
            reason_code="UNUSUAL_ASPECT_RATIO",
            message=(
                "Foreground geometry is unusual "
                "for a signature."
            ),
            features=features
        )

    # -----------------------------------------------------
    # Occupancy
    # -----------------------------------------------------

    occupancy = (
        features.occupancy_ratio
    )

    if occupancy is None:

        return SignatureValidationResult(
            decision=SignatureDecision.REVIEW,
            confidence=0.55,
            reason_code="MISSING_OCCUPANCY",
            message=(
                "Signature occupancy could not "
                "be calculated."
            ),
            features=features
        )

    if occupancy < MIN_OCCUPANCY:

        return SignatureValidationResult(
            decision=SignatureDecision.REVIEW,
            confidence=0.65,
            reason_code="LOW_SIGNATURE_OCCUPANCY",
            message=(
                "Detected region contains very "
                "little foreground content."
            ),
            features=features
        )

    if occupancy > MAX_OCCUPANCY:

        return SignatureValidationResult(
            decision=SignatureDecision.REVIEW,
            confidence=0.65,
            reason_code="HIGH_SIGNATURE_OCCUPANCY",
            message=(
                "Detected region contains unusually "
                "dense foreground content."
            ),
            features=features
        )

    # -----------------------------------------------------
    # Connected components
    # -----------------------------------------------------

    components = (
        features.connected_components
    )

    if components > MAX_COMPONENTS:

        return SignatureValidationResult(
            decision=SignatureDecision.REVIEW,
            confidence=0.60,
            reason_code="TOO_MANY_COMPONENTS",
            message=(
                "Foreground is highly fragmented."
            ),
            features=features
        )

    # -----------------------------------------------------
    # Confidence calculation
    # -----------------------------------------------------
    #
    # This is a QUALITY confidence, not a probability
    # that the object is genuinely a human signature.
    #

    scores = []

    # Density score
    density_score = 1.0

    if density < 0.001:
        density_score = 0.65
    elif density > 0.05:
        density_score = 0.70

    scores.append(
        density_score
    )

    # Aspect score
    aspect_score = 1.0

    if (
        aspect_ratio < 0.5
        or
        aspect_ratio > 8.0
    ):
        aspect_score = 0.75

    scores.append(
        aspect_score
    )

    # Occupancy score
    occupancy_score = 1.0

    if occupancy < 0.02:
        occupancy_score = 0.75

    elif occupancy > 0.50:
        occupancy_score = 0.70

    scores.append(
        occupancy_score
    )

    # Component score
    component_score = 1.0

    if components > 100:
        component_score = 0.75

    scores.append(
        component_score
    )

    confidence = (
        sum(scores) /
        len(scores)
    )

    confidence = _clamp(
        confidence
    )

    # -----------------------------------------------------
    # Final Phase-1 decision
    # -----------------------------------------------------

    if confidence >= 0.80:

        return SignatureValidationResult(
            decision=SignatureDecision.ACCEPT,
            confidence=round(
                confidence,
                3
            ),
            reason_code="SIGNATURE_CANDIDATE",
            message=(
                "Image contains a strong "
                "signature-like candidate."
            ),
            features=features
        )

    return SignatureValidationResult(
        decision=SignatureDecision.REVIEW,
        confidence=round(
            confidence,
            3
        ),
        reason_code="BORDERLINE_SIGNATURE",
        message=(
            "Image contains foreground that may "
            "represent a signature but requires "
            "additional verification."
        ),
        features=features
    )