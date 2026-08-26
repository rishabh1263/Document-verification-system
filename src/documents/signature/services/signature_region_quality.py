"""
Signature region quality evaluation.

Purpose
-------

YOLOS tells us that a possible signature region exists.

This module determines how reliable that region is.

It does NOT make the final ACCEPT / REVIEW / REJECT
decision.

It classifies a YOLOS detection into:

    STRONG
    WEAK
    INVALID

The final signature validator combines this information
with the MobileNet classifier and whole-image context.

Design principle
----------------

A weak YOLOS localization should not automatically mean
that a signature does not exist.

Example:

    MobileNet:
        signature_probability = 1.00

    YOLOS:
        confidence = 0.254
        detection_count = 1

This is evidence of a signature, but the localization
is weak.

Therefore:

    WEAK

rather than:

    NOT_FOUND
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.documents.signature.services.signature_detector import (
    SignatureDetection,
)


# =========================================================
# Region quality
# =========================================================


class SignatureRegionQuality(str, Enum):

    STRONG = "STRONG"

    WEAK = "WEAK"

    INVALID = "INVALID"


# =========================================================
# Result
# =========================================================


@dataclass(frozen=True)
class SignatureRegionQualityResult:

    quality: SignatureRegionQuality

    confidence: float

    reason_code: str

    message: str

    detection_confidence: float

    area_ratio: float

    width_ratio: float

    height_ratio: float

    aspect_ratio: float

    touches_border: bool


# =========================================================
# Configuration
# =========================================================

# ---------------------------------------------------------
# YOLOS confidence
# ---------------------------------------------------------

STRONG_DETECTION_CONFIDENCE = 0.50

WEAK_DETECTION_CONFIDENCE = 0.20


# ---------------------------------------------------------
# Region geometry
# ---------------------------------------------------------

# A signature should normally occupy only a portion of
# the image.
#
# This is intentionally generous because uploaded images
# can be tightly cropped.
MAX_AREA_RATIO = 0.35

MAX_HEIGHT_RATIO = 0.80

MAX_WIDTH_RATIO = 0.90


# ---------------------------------------------------------
# Border tolerance
# ---------------------------------------------------------

BORDER_TOLERANCE_RATIO = 0.01


# =========================================================
# Helpers
# =========================================================


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# =========================================================
# Border detection
# =========================================================


def _touches_border(
    detection: SignatureDetection,
) -> bool:
    """
    Determine whether the detection touches the image
    boundary.

    A detection that effectively selects the entire image
    is not a useful signature localization.
    """

    width = max(
        1,
        detection.image_width,
    )

    height = max(
        1,
        detection.image_height,
    )

    x_tolerance = (
        width
        *
        BORDER_TOLERANCE_RATIO
    )

    y_tolerance = (
        height
        *
        BORDER_TOLERANCE_RATIO
    )

    return (
        detection.x1
        <= x_tolerance
        or
        detection.y1
        <= y_tolerance
        or
        detection.x2
        >= width - x_tolerance
        or
        detection.y2
        >= height - y_tolerance
    )


# =========================================================
# Aspect ratio
# =========================================================


def _calculate_aspect_ratio(
    detection: SignatureDetection,
) -> float:

    if detection.height <= 0:

        return 0.0

    return (
        detection.width
        /
        detection.height
    )


# =========================================================
# Main evaluator
# =========================================================


def evaluate_signature_region(
    detection: SignatureDetection,
) -> SignatureRegionQualityResult:
    """
    Evaluate one YOLOS signature detection.

    STRONG
    ------

    High detector confidence and reasonable geometry.

    WEAK
    ----

    Detector found a plausible region, but confidence
    and/or geometry are not strong enough for automatic
    acceptance.

    INVALID
    -------

    Region is unusable.
    """

    if detection is None:

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.INVALID
            ),

            confidence=0.0,

            reason_code=(
                "INVALID_SIGNATURE_REGION"
            ),

            message=(
                "No signature detection "
                "was provided."
            ),

            detection_confidence=0.0,

            area_ratio=0.0,

            width_ratio=0.0,

            height_ratio=0.0,

            aspect_ratio=0.0,

            touches_border=False,
        )

    # =====================================================
    # Read values
    # =====================================================

    confidence = _clamp(
        float(
            detection.confidence
        )
    )

    area_ratio = max(
        0.0,
        float(
            detection.area_ratio
        ),
    )

    width_ratio = max(
        0.0,
        float(
            detection.width_ratio
        ),
    )

    height_ratio = max(
        0.0,
        float(
            detection.height_ratio
        ),
    )

    aspect_ratio = (
        _calculate_aspect_ratio(
            detection
        )
    )

    touches_border = (
        _touches_border(
            detection
        )
    )

    # =====================================================
    # Basic invalid geometry
    # =====================================================

    if (
        detection.width <= 0
        or
        detection.height <= 0
        or
        detection.image_width <= 0
        or
        detection.image_height <= 0
    ):

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.INVALID
            ),

            confidence=0.0,

            reason_code=(
                "INVALID_SIGNATURE_REGION"
            ),

            message=(
                "Signature detection "
                "contains invalid geometry."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # Excessively large region
    # =====================================================

    if (
        area_ratio
        > MAX_AREA_RATIO
    ):

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.WEAK
            ),

            confidence=round(
                confidence,
                6,
            ),

            reason_code=(
                "LARGE_SIGNATURE_REGION"
            ),

            message=(
                "A signature candidate was "
                "detected, but the localized "
                "region is unusually large."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # Excessively tall region
    # =====================================================

    if (
        height_ratio
        > MAX_HEIGHT_RATIO
    ):

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.WEAK
            ),

            confidence=round(
                confidence,
                6,
            ),

            reason_code=(
                "TALL_SIGNATURE_REGION"
            ),

            message=(
                "A signature candidate was "
                "detected, but the localized "
                "region is unusually tall."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # Almost entire image
    # =====================================================

    if (
        width_ratio
        > MAX_WIDTH_RATIO
        or
        height_ratio
        > MAX_HEIGHT_RATIO
    ):

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.WEAK
            ),

            confidence=round(
                confidence,
                6,
            ),

            reason_code=(
                "OVERSIZED_SIGNATURE_REGION"
            ),

            message=(
                "The detected region is too "
                "large to be considered a "
                "strong signature localization."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # Border touching
    # =====================================================

    if touches_border:

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.WEAK
            ),

            confidence=round(
                confidence,
                6,
            ),

            reason_code=(
                "BORDER_TOUCHING_REGION"
            ),

            message=(
                "The signature candidate "
                "touches the image boundary."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # Strong detection
    # =====================================================

    if (
        confidence
        >= STRONG_DETECTION_CONFIDENCE
    ):

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.STRONG
            ),

            confidence=round(
                confidence,
                6,
            ),

            reason_code=(
                "STRONG_SIGNATURE_REGION"
            ),

            message=(
                "A high-confidence signature "
                "region was detected."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # Weak detection
    # =====================================================

    if (
        confidence
        >= WEAK_DETECTION_CONFIDENCE
    ):

        return SignatureRegionQualityResult(

            quality=(
                SignatureRegionQuality.WEAK
            ),

            confidence=round(
                confidence,
                6,
            ),

            reason_code=(
                "WEAK_SIGNATURE_REGION"
            ),

            message=(
                "A signature region was "
                "detected, but localization "
                "confidence is weak."
            ),

            detection_confidence=confidence,

            area_ratio=area_ratio,

            width_ratio=width_ratio,

            height_ratio=height_ratio,

            aspect_ratio=aspect_ratio,

            touches_border=touches_border,
        )

    # =====================================================
    # No reliable region
    # =====================================================

    return SignatureRegionQualityResult(

        quality=(
            SignatureRegionQuality.INVALID
        ),

        confidence=round(
            confidence,
            6,
        ),

        reason_code=(
            "SIGNATURE_REGION_NOT_FOUND"
        ),

        message=(
            "No reliable signature region "
            "was detected."
        ),

        detection_confidence=confidence,

        area_ratio=area_ratio,

        width_ratio=width_ratio,

        height_ratio=height_ratio,

        aspect_ratio=aspect_ratio,

        touches_border=touches_border,
    )