"""
Signature validation layer.

Pipeline
--------

FAST PATH:

    IMAGE
      |
      +-- Image validation
      |
      +-- OpenCV feature extraction
      |
      +-- MobileNetV3 classifier
      |
      +-- Graphic / logo guard
      |
      +-- Cheap geometry checks
      |
      +-- ACCEPT / REVIEW / REJECT

ESCALATION PATH:

    IMAGE
      |
      +-- OpenCV feature extraction
      |
      +-- MobileNetV3 classifier
      |
      +-- Suspicious / ambiguous
      |
      +-- YOLOS signature detection
      |
      +-- Whole-image context
      |
      +-- ACCEPT / REVIEW / REJECT

Important
---------

MobileNet classification alone must NEVER be treated as proof
that an image is a genuine standalone handwritten signature.

A logo, graphic, seal, printed signature illustration, or other
signature-like graphic can receive a very high MobileNet score.

Therefore automatic acceptance requires:

    classifier confidence
    +
    reasonable handwriting-like geometry
    +
    no strong graphic/logo signal
    +
    no obvious document signal

YOLOS is reserved for ambiguous/suspicious cases because it is
considerably slower than OpenCV + MobileNet.

Supported input modes:

    upload
    draw
    capture
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import time

import numpy as np

from src.documents.signature.validators.signature_features import (
    SignatureFeatures,
    extract_signature_features,
)

from src.documents.signature.services.image_context import (
    ImageContextResult,
    analyze_image_context,
)

from src.documents.signature.services.signature_classifier import (
    SignatureClassifierResult,
    classify_signature,
)

from src.documents.signature.services.signature_detector import (
    SignatureDetection,
    SignatureDetectionResult,
    detect_signatures,
)


# =========================================================
# Decisions
# =========================================================

class SignatureDecision(str, Enum):

    ACCEPT = "ACCEPT"

    REVIEW = "REVIEW"

    REJECT = "REJECT"


# =========================================================
# Input modes
# =========================================================

class SignatureInputMode(str, Enum):

    UPLOAD = "upload"

    DRAW = "draw"

    CAPTURE = "capture"


# =========================================================
# Validation result
# =========================================================

@dataclass
class SignatureValidationResult:

    decision: SignatureDecision

    confidence: float

    reason_code: str

    message: str

    features: Optional[
        SignatureFeatures
    ] = None

    classifier: Optional[
        SignatureClassifierResult
    ] = None

    context: Optional[
        ImageContextResult
    ] = None


# =========================================================
# Configuration
# =========================================================

# ---------------------------------------------------------
# MobileNet thresholds
# ---------------------------------------------------------

SIGNATURE_ACCEPT_THRESHOLD = 0.90

SIGNATURE_REVIEW_THRESHOLD = 0.50


# =========================================================
# FAST PATH
# =========================================================

FAST_ACCEPT_CLASSIFIER_THRESHOLD = 0.98

FAST_MAX_FOREGROUND_DENSITY = 0.08

FAST_MAX_COMPONENTS = 80

FAST_MIN_COMPONENTS = 2

FAST_MIN_OCCUPANCY_RATIO = 0.005

FAST_MAX_OCCUPANCY_RATIO = 0.65

FAST_MIN_ASPECT_RATIO = 0.20

FAST_MAX_ASPECT_RATIO = 8.00


# =========================================================
# General image safety
# =========================================================

MAX_FOREGROUND_DENSITY = 0.35

HIGH_COMPONENT_COUNT = 250

HIGH_EDGE_DENSITY = 0.08

DOCUMENT_CONTEXT_COMPONENTS = 150

SMALL_SIGNATURE_AREA_RATIO = 0.02


# =========================================================
# Graphic / logo protection
# =========================================================

"""
These rules are NOT designed to identify a specific website,
company, logo, or image.

They detect a general visual pattern:

    high classifier confidence
        +
    relatively large structured foreground
        +
    few connected components
        +
    dominant contour

That pattern is suspicious for a graphic/illustration.

This prevents a signature-like logo/graphic from receiving
automatic ACCEPT solely because MobileNet is confident.
"""

GRAPHIC_MIN_FOREGROUND_DENSITY = 0.055

GRAPHIC_MIN_OCCUPANCY_RATIO = 0.08

GRAPHIC_MAX_COMPONENTS = 15

GRAPHIC_DOMINANT_CONTOUR_RATIO = 0.55

GRAPHIC_LARGE_CONTOUR_IMAGE_RATIO = 0.040


# =========================================================
# Multiple-signature configuration
# =========================================================

MULTIPLE_SIGNATURE_IOU_THRESHOLD = 0.25

MULTIPLE_SIGNATURE_CONTAINMENT_THRESHOLD = 0.60


# =========================================================
# Timing
# =========================================================

ENABLE_TIMING_LOG = False


# =========================================================
# Image validation
# =========================================================

def _validate_input_image(
    image: np.ndarray,
) -> Optional[str]:
    """
    Validate input image.
    """

    if image is None:

        return "Image cannot be None."

    if not isinstance(
        image,
        np.ndarray,
    ):

        return (
            "Image must be a NumPy array."
        )

    if image.size == 0:

        return "Image cannot be empty."

    if len(image.shape) not in (
        2,
        3,
    ):

        return (
            "Image must be grayscale or BGR."
        )

    height, width = (
        image.shape[:2]
    )

    if height <= 0 or width <= 0:

        return (
            "Image dimensions must be positive."
        )

    return None


# =========================================================
# Invalid result
# =========================================================

def _invalid_result(
    message: str,
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=SignatureDecision.REJECT,

        confidence=0.95,

        reason_code="INVALID_IMAGE",

        message=message,

        features=None,

        classifier=None,

        context=None,
    )


# =========================================================
# Reject
# =========================================================

def _reject(
    reason_code: str,
    message: str,
    features: Optional[
        SignatureFeatures
    ],
    classifier: Optional[
        SignatureClassifierResult
    ],
    context: Optional[
        ImageContextResult
    ],
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=SignatureDecision.REJECT,

        confidence=0.95,

        reason_code=reason_code,

        message=message,

        features=features,

        classifier=classifier,

        context=context,
    )


# =========================================================
# Review
# =========================================================

def _review(
    reason_code: str,
    message: str,
    features: Optional[
        SignatureFeatures
    ],
    classifier: Optional[
        SignatureClassifierResult
    ],
    context: Optional[
        ImageContextResult
    ],
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=SignatureDecision.REVIEW,

        confidence=0.60,

        reason_code=reason_code,

        message=message,

        features=features,

        classifier=classifier,

        context=context,
    )


# =========================================================
# Accept
# =========================================================

def _accept(
    reason_code: str,
    message: str,
    features: SignatureFeatures,
    classifier: SignatureClassifierResult,
    context: Optional[
        ImageContextResult
    ],
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=SignatureDecision.ACCEPT,

        confidence=0.95,

        reason_code=reason_code,

        message=message,

        features=features,

        classifier=classifier,

        context=context,
    )


# =========================================================
# Box helpers
# =========================================================

def _box_from_detection(
    detection: SignatureDetection,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    return (
        float(detection.x1),
        float(detection.y1),
        float(detection.x2),
        float(detection.y2),
    )


def _box_area(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    x1, y1, x2, y2 = box

    return (
        max(
            0.0,
            x2 - x1,
        )
        *
        max(
            0.0,
            y2 - y1,
        )
    )


def _intersection_area(
    box_a: tuple[
        float,
        float,
        float,
        float,
    ],
    box_b: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return (
        width
        *
        height
    )


def _calculate_iou(
    box_a: tuple[
        float,
        float,
        float,
        float,
    ],
    box_b: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:

    intersection = _intersection_area(
        box_a,
        box_b,
    )

    if intersection <= 0:

        return 0.0

    area_a = _box_area(
        box_a
    )

    area_b = _box_area(
        box_b
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection
        /
        union
    )


# =========================================================
# Independent detections
# =========================================================

def _get_independent_detections(
    detection: SignatureDetectionResult,
) -> list[
    SignatureDetection
]:
    """
    Determine the number of independent signature regions.

    Overlapping/nested YOLOS boxes representing the same
    physical signature are treated as one.
    """

    if not detection.detections:

        return []

    ordered = sorted(
        detection.detections,
        key=lambda item: item.confidence,
        reverse=True,
    )

    independent: list[
        SignatureDetection
    ] = []

    for candidate in ordered:

        candidate_box = (
            _box_from_detection(
                candidate
            )
        )

        candidate_area = _box_area(
            candidate_box
        )

        if candidate_area <= 0:

            continue

        duplicate = False

        for accepted in independent:

            accepted_box = (
                _box_from_detection(
                    accepted
                )
            )

            accepted_area = _box_area(
                accepted_box
            )

            if accepted_area <= 0:

                continue

            intersection = (
                _intersection_area(
                    candidate_box,
                    accepted_box,
                )
            )

            if intersection <= 0:

                continue

            iou = _calculate_iou(
                candidate_box,
                accepted_box,
            )

            if (
                iou
                >= MULTIPLE_SIGNATURE_IOU_THRESHOLD
            ):

                duplicate = True

                break

            candidate_containment = (
                intersection
                /
                candidate_area
            )

            if (
                candidate_containment
                >= MULTIPLE_SIGNATURE_CONTAINMENT_THRESHOLD
            ):

                duplicate = True

                break

            accepted_containment = (
                intersection
                /
                accepted_area
            )

            if (
                accepted_containment
                >= MULTIPLE_SIGNATURE_CONTAINMENT_THRESHOLD
            ):

                duplicate = True

                break

        if not duplicate:

            independent.append(
                candidate
            )

    return independent


# =========================================================
# Graphic / logo detector
# =========================================================

def _looks_like_graphic_signature(
    features: SignatureFeatures,
) -> bool:
    """
    Detect a graphic/logo-like signature representation.

    This function does NOT identify a particular company or
    image.

    It uses general visual structure only.

    Returns True when the image has a suspicious combination
    of:

        - relatively high foreground density
        - large occupied region
        - few connected components
        - dominant contour

    This is a safety guard against automatically accepting
    signature-like graphics.
    """

    foreground = float(
        features.foreground_density
    )

    components = int(
        features.connected_components
    )

    occupancy = float(
        features.occupancy_ratio
    )

    largest_contour = float(
        features.largest_contour_area
    )

    total_contour = float(
        features.total_contour_area
    )

    image_width = int(
        features.image_width
    )

    image_height = int(
        features.image_height
    )

    # -----------------------------------------------------
    # Basic protection.
    # -----------------------------------------------------

    if (
        image_width <= 0
        or image_height <= 0
    ):

        return False

    if total_contour <= 0:

        return False

    image_area = (
        image_width
        *
        image_height
    )

    if image_area <= 0:

        return False

    # -----------------------------------------------------
    # Dominant contour ratio.
    # -----------------------------------------------------

    dominant_contour_ratio = (
        largest_contour
        /
        total_contour
    )

    # -----------------------------------------------------
    # Largest contour relative to image.
    # -----------------------------------------------------

    largest_contour_image_ratio = (
        largest_contour
        /
        image_area
    )

    # =====================================================
    # Rule 1
    # =====================================================

    dominant_contour = (
        dominant_contour_ratio
        >= GRAPHIC_DOMINANT_CONTOUR_RATIO
    )

    large_structured_region = (
        foreground
        >= GRAPHIC_MIN_FOREGROUND_DENSITY
        and
        occupancy
        >= GRAPHIC_MIN_OCCUPANCY_RATIO
        and
        components
        <= GRAPHIC_MAX_COMPONENTS
    )

    if (
        dominant_contour
        and
        large_structured_region
    ):

        return True

    # =====================================================
    # Rule 2
    # =====================================================

    very_large_contour = (
        largest_contour_image_ratio
        >= GRAPHIC_LARGE_CONTOUR_IMAGE_RATIO
    )

    if (
        very_large_contour
        and
        components
        <= GRAPHIC_MAX_COMPONENTS
        and
        occupancy
        >= GRAPHIC_MIN_OCCUPANCY_RATIO
    ):

        return True

    return False


# =========================================================
# Fast path candidate
# =========================================================

def _is_fast_path_candidate(
    features: SignatureFeatures,
    classifier: SignatureClassifierResult,
) -> bool:
    """
    Determine whether YOLOS can be skipped.

    Requirements:

        1. Very high MobileNet confidence.
        2. Low foreground density.
        3. Reasonable connected components.
        4. Reasonable occupancy.
        5. Reasonable aspect ratio.
        6. Not a graphic-like image.
    """

    if (
        classifier.signature_probability
        < FAST_ACCEPT_CLASSIFIER_THRESHOLD
    ):

        return False

    if (
        features.foreground_density
        > FAST_MAX_FOREGROUND_DENSITY
    ):

        return False

    if (
        features.connected_components
        > FAST_MAX_COMPONENTS
    ):

        return False

    if (
        features.connected_components
        < FAST_MIN_COMPONENTS
    ):

        return False

    occupancy = float(
        features.occupancy_ratio
    )

    if (
        occupancy
        < FAST_MIN_OCCUPANCY_RATIO
    ):

        return False

    if (
        occupancy
        > FAST_MAX_OCCUPANCY_RATIO
    ):

        return False

    aspect_ratio = float(
        features.aspect_ratio
    )

    if (
        aspect_ratio
        < FAST_MIN_ASPECT_RATIO
    ):

        return False

    if (
        aspect_ratio
        > FAST_MAX_ASPECT_RATIO
    ):

        return False

    # -----------------------------------------------------
    # IMPORTANT:
    # Never fast-accept a graphic-like image.
    # -----------------------------------------------------

    if _looks_like_graphic_signature(
        features
    ):

        return False

    return True


# =========================================================
# Suspicious OpenCV context
# =========================================================

def _has_suspicious_context(
    features: SignatureFeatures,
) -> bool:
    """
    Cheap OpenCV-only suspiciousness check.
    """

    if (
        features.foreground_density
        > MAX_FOREGROUND_DENSITY
    ):

        return True

    if (
        features.connected_components
        >= HIGH_COMPONENT_COUNT
    ):

        return True

    if (
        features.contour_count
        >= HIGH_COMPONENT_COUNT
    ):

        return True

    return False


# =========================================================
# Timing logger
# =========================================================

def _print_timing(
    total_start: float,
    feature_ms: float,
    classifier_ms: float,
    detection_ms: float = 0.0,
    context_ms: float = 0.0,
    yolos_used: bool = False,
) -> None:

    if not ENABLE_TIMING_LOG:

        return

    total_ms = (
        time.perf_counter()
        -
        total_start
    ) * 1000

    print(
        "\n"
        "========================================\n"
        "SIGNATURE TIMING\n"
        "========================================\n"
        f"Features   : {feature_ms:.2f} ms\n"
        f"Classifier : {classifier_ms:.2f} ms\n"
        f"YOLOS      : {detection_ms:.2f} ms\n"
        f"Context    : {context_ms:.2f} ms\n"
        f"Total      : {total_ms:.2f} ms\n"
        f"YOLOS used : {yolos_used}\n"
        "========================================"
    )


# =========================================================
# Main validator
# =========================================================

def validate_signature(
    image: np.ndarray,
    input_mode: str | SignatureInputMode = (
        SignatureInputMode.UPLOAD
    ),
) -> SignatureValidationResult:
    """
    Validate a signature image.

    Fast path:

        OpenCV
        +
        MobileNet
        +
        graphic guard

    Escalation path:

        OpenCV
        +
        MobileNet
        +
        YOLOS
        +
        context
    """

    total_start = time.perf_counter()

    # =====================================================
    # 1. Input mode
    # =====================================================

    try:

        if isinstance(
            input_mode,
            SignatureInputMode,
        ):

            normalized_input_mode = (
                input_mode.value
            )

        else:

            normalized_input_mode = str(
                input_mode
            ).lower().strip()

    except Exception:

        normalized_input_mode = "upload"

    if normalized_input_mode not in {
        "upload",
        "draw",
        "capture",
    }:

        return _invalid_result(
            (
                "Unsupported input mode. "
                "Allowed modes: "
                "upload, draw, capture."
            )
        )

    # =====================================================
    # 2. Validate image
    # =====================================================

    validation_error = (
        _validate_input_image(
            image
        )
    )

    if validation_error is not None:

        return _invalid_result(
            validation_error
        )

    # =====================================================
    # 3. OpenCV features
    # =====================================================

    feature_start = time.perf_counter()

    try:

        features = (
            extract_signature_features(
                image
            )
        )

    except Exception as exc:

        return _invalid_result(
            (
                "Feature extraction failed: "
                f"{exc}"
            )
        )

    feature_ms = (
        time.perf_counter()
        -
        feature_start
    ) * 1000

    # =====================================================
    # 4. Blank
    # =====================================================

    if (
        features.foreground_density
        <= 0
    ):

        return _reject(
            reason_code="BLANK_IMAGE",
            message=(
                "Image contains no usable "
                "foreground content."
            ),
            features=features,
            classifier=None,
            context=None,
        )

    # =====================================================
    # 5. Excessive foreground
    # =====================================================

    if (
        features.foreground_density
        > MAX_FOREGROUND_DENSITY
    ):

        return _reject(
            reason_code="EXCESSIVE_FOREGROUND",
            message=(
                "Image contains an unusually "
                "large amount of foreground "
                "content."
            ),
            features=features,
            classifier=None,
            context=None,
        )

    # =====================================================
    # 6. MobileNet
    # =====================================================

    classifier_start = time.perf_counter()

    try:

        classifier = classify_signature(
            image
        )

    except Exception as exc:

        return _review(
            reason_code="CLASSIFIER_ERROR",
            message=(
                "Signature classifier failed: "
                f"{exc}"
            ),
            features=features,
            classifier=None,
            context=None,
        )

    classifier_ms = (
        time.perf_counter()
        -
        classifier_start
    ) * 1000

    # =====================================================
    # 7. Non-signature
    # =====================================================

    if not classifier.is_signature:

        return _reject(
            reason_code="NO_SIGNATURE",
            message=(
                "The image was classified as "
                "non-signature."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 8. Weak confidence
    # =====================================================

    if (
        classifier.signature_probability
        < SIGNATURE_REVIEW_THRESHOLD
    ):

        return _reject(
            reason_code="NO_SIGNATURE",
            message=(
                "The image does not contain "
                "enough evidence of a signature."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 9. Medium confidence
    # =====================================================

    if (
        classifier.signature_probability
        < SIGNATURE_ACCEPT_THRESHOLD
    ):

        return _review(
            reason_code=(
                "WEAK_SIGNATURE_CLASSIFICATION"
            ),
            message=(
                "The image appears signature-like "
                "but classifier confidence is "
                "not high enough for automatic "
                "acceptance."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 10. GRAPHIC / LOGO GUARD
    # =====================================================

    """
    This is intentionally BEFORE the fast acceptance.

    A model confidence of 0.999+ is not enough.

    If the image has a strong graphic-like structure,
    reject it instead of allowing MobileNet to override
    the structural evidence.
    """

    if _looks_like_graphic_signature(
        features
    ):

        return _reject(
            reason_code="GRAPHIC_SIGNATURE_REJECTED",
            message=(
                "The image is strongly classified "
                "as signature-like, but its visual "
                "structure is more consistent with "
                "a graphic, logo, or non-handwritten "
                "signature image."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 11. Suspicious OpenCV context
    # =====================================================

    suspicious = _has_suspicious_context(
        features
    )

    # =====================================================
    # 12. FAST PATH
    # =====================================================

    fast_path = _is_fast_path_candidate(
        features=features,
        classifier=classifier,
    )

    if (
        fast_path
        and not suspicious
    ):

        _print_timing(
            total_start=total_start,
            feature_ms=feature_ms,
            classifier_ms=classifier_ms,
            yolos_used=False,
        )

        return _accept(
            reason_code="SIGNATURE_CANDIDATE",
            message=(
                "Image contains a high-confidence "
                "signature candidate."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 13. YOLOS
    # =====================================================

    detection_start = time.perf_counter()

    detection: Optional[
        SignatureDetectionResult
    ] = None

    try:

        detection = detect_signatures(
            image=image
        )

    except Exception as exc:

        return _review(
            reason_code="SIGNATURE_DETECTION_ERROR",
            message=(
                "Signature region detection failed: "
                f"{exc}"
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    detection_ms = (
        time.perf_counter()
        -
        detection_start
    ) * 1000

    # =====================================================
    # 14. No region
    # =====================================================

    if (
        detection is None
        or not detection.detected
        or not detection.detections
    ):

        _print_timing(
            total_start=total_start,
            feature_ms=feature_ms,
            classifier_ms=classifier_ms,
            detection_ms=detection_ms,
            yolos_used=True,
        )

        return _review(
            reason_code="SIGNATURE_REGION_NOT_FOUND",
            message=(
                "The classifier detected a signature, "
                "but no reliable signature region "
                "was detected."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 15. Independent detections
    # =====================================================

    independent_detections = (
        _get_independent_detections(
            detection
        )
    )

    independent_count = len(
        independent_detections
    )

    # =====================================================
    # 16. Multiple signatures
    # =====================================================

    if independent_count > 1:

        _print_timing(
            total_start=total_start,
            feature_ms=feature_ms,
            classifier_ms=classifier_ms,
            detection_ms=detection_ms,
            yolos_used=True,
        )

        return _review(
            reason_code="MULTIPLE_SIGNATURES",
            message=(
                "Multiple independent "
                "signature-like regions "
                "were detected."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 17. Whole-image context
    # =====================================================

    context_start = time.perf_counter()

    try:

        context = analyze_image_context(
            image=image,
            signature_result=detection,
        )

    except Exception as exc:

        return _review(
            reason_code="CONTEXT_ANALYSIS_ERROR",
            message=(
                "Whole-image context analysis "
                f"failed: {exc}"
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    context_ms = (
        time.perf_counter()
        -
        context_start
    ) * 1000

    # =====================================================
    # 18. Document context
    # =====================================================

    document_signals = 0

    if (
        context.connected_components
        >= HIGH_COMPONENT_COUNT
    ):

        document_signals += 1

    if (
        context.edge_density
        >= HIGH_EDGE_DENSITY
    ):

        document_signals += 1

    if (
        context.connected_components
        >= DOCUMENT_CONTEXT_COMPONENTS
        and
        context.signature_area_ratio
        < SMALL_SIGNATURE_AREA_RATIO
    ):

        document_signals += 1

    if context.document_like:

        document_signals += 1

    # =====================================================
    # 19. Strong document context
    # =====================================================

    if document_signals >= 2:

        _print_timing(
            total_start=total_start,
            feature_ms=feature_ms,
            classifier_ms=classifier_ms,
            detection_ms=detection_ms,
            context_ms=context_ms,
            yolos_used=True,
        )

        return _review(
            reason_code="DOCUMENT_CONTEXT",
            message=(
                "The image contains a "
                "high-confidence signature "
                "but also shows strong "
                "document context."
            ),
            features=features,
            classifier=classifier,
            context=context,
        )

    # =====================================================
    # 20. High confidence after YOLOS
    # =====================================================

    if (
        classifier.signature_probability
        >= SIGNATURE_ACCEPT_THRESHOLD
    ):

        _print_timing(
            total_start=total_start,
            feature_ms=feature_ms,
            classifier_ms=classifier_ms,
            detection_ms=detection_ms,
            context_ms=context_ms,
            yolos_used=True,
        )

        return _accept(
            reason_code="SIGNATURE_CANDIDATE",
            message=(
                "Image contains a high-confidence "
                "signature candidate."
            ),
            features=features,
            classifier=classifier,
            context=context,
        )

    # =====================================================
    # 21. Fallback
    # =====================================================

    _print_timing(
        total_start=total_start,
        feature_ms=feature_ms,
        classifier_ms=classifier_ms,
        detection_ms=detection_ms,
        context_ms=context_ms,
        yolos_used=True,
    )

    return _review(
        reason_code="REVIEW_REQUIRED",
        message=(
            "Image appears signature-like but "
            "requires manual review."
        ),
        features=features,
        classifier=classifier,
        context=context,
    )