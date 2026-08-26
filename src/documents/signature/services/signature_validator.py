"""
Signature validation layer.

Pipeline:

    IMAGE
      |
      +-- Image validation
      |
      +-- OpenCV feature extraction
      |
      +-- MobileNetV3 classifier
      |
      +-- YOLOS signature detection
      |
      +-- Region reliability filtering
      |
      +-- OpenCV fallback region validation
      |
      +-- Whole-image context
      |
      +-- Conservative decision logic
      |
      +-- ACCEPT / REVIEW / REJECT

MobileNetV3:
    Primary signature classifier.

YOLOS:
    Signature-region detector.

OpenCV:
    Geometry and image-structure fallback.

Important:
    YOLOS detections are NOT automatically trusted.

    A very large YOLOS box, for example a box covering
    most of the image, is considered an unreliable region.

Supported input modes:

    upload
    draw
    capture
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

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

SIGNATURE_ACCEPT_THRESHOLD = 0.90

SIGNATURE_REVIEW_THRESHOLD = 0.50

MAX_FOREGROUND_DENSITY = 0.35

HIGH_COMPONENT_COUNT = 250

HIGH_EDGE_DENSITY = 0.08

DOCUMENT_CONTEXT_COMPONENTS = 150

SMALL_SIGNATURE_AREA_RATIO = 0.02

ACCEPT_CONFIDENCE = 0.95

REVIEW_CONFIDENCE = 0.60

REJECT_CONFIDENCE = 0.95


# =========================================================
# YOLOS configuration
# =========================================================

# Normal YOLOS detection.
YOLOS_PRIMARY_THRESHOLD = 0.50

# Used only when MobileNet strongly believes the image
# is a signature but primary YOLOS cannot find a region.
YOLOS_FALLBACK_THRESHOLD = 0.10

# A YOLOS box occupying more than this percentage of the
# entire image is not considered a reliable signature box.
MAX_RELIABLE_REGION_AREA_RATIO = 0.45

# A signature should not normally occupy almost the complete
# width and height of the submitted image.
MAX_RELIABLE_REGION_WIDTH_RATIO = 0.90

MAX_RELIABLE_REGION_HEIGHT_RATIO = 0.90

# Minimum confidence for a fallback YOLOS detection.
YOLOS_FALLBACK_MIN_CONFIDENCE = 0.12


# =========================================================
# Multiple-signature configuration
# =========================================================

MULTIPLE_SIGNATURE_IOU_THRESHOLD = 0.25

MULTIPLE_SIGNATURE_CONTAINMENT_THRESHOLD = 0.60

# Two regions that are very close vertically and have
# significant horizontal overlap are often different
# detections of the same physical signature.
VERTICAL_PROXIMITY_THRESHOLD = 0.04

CENTER_VERTICAL_RATIO_THRESHOLD = 0.75

# Combined region should not look like a full document.
MAX_COMBINED_AREA_RATIO = 0.45


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
# Invalid image
# =========================================================

def _invalid_result(
    message: str,
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=SignatureDecision.REJECT,

        confidence=REJECT_CONFIDENCE,

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

        confidence=REJECT_CONFIDENCE,

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

        confidence=REVIEW_CONFIDENCE,

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

        confidence=ACCEPT_CONFIDENCE,

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


def _box_center(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    float,
    float,
]:

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def _combined_box(
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
) -> tuple[
    float,
    float,
    float,
    float,
]:

    return (
        min(
            box_a[0],
            box_b[0],
        ),
        min(
            box_a[1],
            box_b[1],
        ),
        max(
            box_a[2],
            box_b[2],
        ),
        max(
            box_a[3],
            box_b[3],
        ),
    )


# =========================================================
# Reliable YOLOS detection
# =========================================================

def _is_reliable_detection(
    detection: SignatureDetection,
) -> bool:
    """
    Determine whether a YOLOS detection looks like a
    plausible signature region.

    This prevents pathological detections such as:

        area_ratio = 0.706
        width_ratio = 0.95
        height_ratio = 0.74

    from being treated as a signature region.
    """

    if detection.confidence <= 0:

        return False

    if detection.area_ratio <= 0:

        return False

    if (
        detection.area_ratio
        > MAX_RELIABLE_REGION_AREA_RATIO
    ):

        return False

    if (
        detection.width_ratio
        > MAX_RELIABLE_REGION_WIDTH_RATIO
    ):

        return False

    if (
        detection.height_ratio
        > MAX_RELIABLE_REGION_HEIGHT_RATIO
    ):

        return False

    return True


def _get_reliable_detections(
    detection: Optional[
        SignatureDetectionResult
    ],
    *,
    fallback: bool = False,
) -> list[
    SignatureDetection
]:
    """
    Filter YOLOS detections by geometry.

    Primary detections only need to pass geometry.

    Fallback detections additionally require a minimum
    confidence.
    """

    if (
        detection is None
        or not detection.detections
    ):

        return []

    reliable = []

    for item in detection.detections:

        if not _is_reliable_detection(
            item
        ):

            continue

        if fallback:

            if (
                item.confidence
                < YOLOS_FALLBACK_MIN_CONFIDENCE
            ):

                continue

        reliable.append(
            item
        )

    return reliable


# =========================================================
# Independent signature detection
# =========================================================

def _get_independent_detections(
    detection: SignatureDetectionResult,
) -> list[
    SignatureDetection
]:
    """
    Determine how many independent signature regions exist.

    Overlapping/nested boxes are treated as one physical
    signature.

    Additional proximity logic handles cases where YOLOS
    returns two adjacent boxes belonging to the same
    handwritten signature.
    """

    reliable = _get_reliable_detections(
        detection
    )

    if not reliable:

        return []

    ordered = sorted(
        reliable,
        key=lambda item: item.confidence,
        reverse=True,
    )

    independent: list[
        SignatureDetection
    ] = []

    image_width = float(
        ordered[0].image_width
    )

    image_height = float(
        ordered[0].image_height
    )

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

            if intersection > 0:

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

            # -------------------------------------------------
            # Additional proximity logic.
            #
            # Useful when two boxes are separate but represent
            # adjacent parts of one handwritten signature.
            # -------------------------------------------------

            candidate_center = _box_center(
                candidate_box
            )

            accepted_center = _box_center(
                accepted_box
            )

            horizontal_gap = max(
                0.0,
                max(
                    candidate_box[0],
                    accepted_box[0],
                )
                -
                min(
                    candidate_box[2],
                    accepted_box[2],
                ),
            )

            vertical_gap = max(
                0.0,
                max(
                    candidate_box[1],
                    accepted_box[1],
                )
                -
                min(
                    candidate_box[3],
                    accepted_box[3],
                ),
            )

            horizontal_overlap = (
                min(
                    candidate_box[2],
                    accepted_box[2],
                )
                -
                max(
                    candidate_box[0],
                    accepted_box[0],
                )
            )

            horizontal_overlap_ratio = 0.0

            if horizontal_overlap > 0:

                smaller_width = min(
                    candidate_box[2]
                    -
                    candidate_box[0],
                    accepted_box[2]
                    -
                    accepted_box[0],
                )

                if smaller_width > 0:

                    horizontal_overlap_ratio = (
                        horizontal_overlap
                        /
                        smaller_width
                    )

            vertical_gap_ratio = (
                vertical_gap
                /
                image_height
            )

            if (
                vertical_gap_ratio
                <= VERTICAL_PROXIMITY_THRESHOLD
            ):

                center_y_difference = abs(
                    candidate_center[1]
                    -
                    accepted_center[1]
                )

                average_height = (
                    (
                        candidate_box[3]
                        -
                        candidate_box[1]
                    )
                    +
                    (
                        accepted_box[3]
                        -
                        accepted_box[1]
                    )
                ) / 2.0

                if average_height > 0:

                    center_y_ratio = (
                        center_y_difference
                        /
                        average_height
                    )

                    if (
                        center_y_ratio
                        <= CENTER_VERTICAL_RATIO_THRESHOLD
                    ):

                        if (
                            horizontal_overlap_ratio
                            >= 0.20
                        ):

                            duplicate = True
                            break

                        # -------------------------------------------------
                        # Horizontally adjacent parts.
                        #
                        # If their combined area is reasonable, they can
                        # be parts of the same signature.
                        # -------------------------------------------------

                        combined = _combined_box(
                            candidate_box,
                            accepted_box,
                        )

                        combined_area = _box_area(
                            combined
                        )

                        image_area = (
                            image_width
                            *
                            image_height
                        )

                        combined_area_ratio = (
                            combined_area
                            /
                            image_area
                            if image_area > 0
                            else 1.0
                        )

                        if (
                            combined_area_ratio
                            <= MAX_COMBINED_AREA_RATIO
                        ):

                            if (
                                horizontal_gap
                                <= (
                                    0.08
                                    *
                                    image_width
                                )
                            ):

                                duplicate = True
                                break

            # Avoid unused variable warnings and keep the
            # geometric calculation explicit.
            _ = horizontal_gap

        if not duplicate:

            independent.append(
                candidate
            )

    return independent


# =========================================================
# OpenCV fallback region plausibility
# =========================================================

def _opencv_region_fallback(
    features: SignatureFeatures,
) -> bool:
    """
    Determine whether the OpenCV feature geometry itself
    provides enough evidence for a plausible signature region.

    This is deliberately conservative.

    It does NOT claim that the image is definitely a
    signature. It only allows the pipeline to continue when
    MobileNet is already highly confident and the image
    geometry looks signature-like.
    """

    foreground = float(
        features.foreground_density
    )

    occupancy = float(
        features.occupancy_ratio
    )

    aspect_ratio = float(
        features.aspect_ratio
    )

    components = int(
        features.connected_components
    )

    contour_count = int(
        features.contour_count
    )

    largest_contour = float(
        features.largest_contour_area
    )

    total_contour = float(
        features.total_contour_area
    )

    # -----------------------------------------------------
    # Basic sanity.
    # -----------------------------------------------------

    if foreground <= 0:

        return False

    if foreground > 0.15:

        return False

    if components <= 0:

        return False

    if contour_count <= 0:

        return False

    if largest_contour <= 0:

        return False

    if total_contour <= 0:

        return False

    # -----------------------------------------------------
    # Occupancy.
    #
    # Very tiny or almost full-image occupancy is suspicious.
    # -----------------------------------------------------

    if occupancy < 0.005:

        return False

    if occupancy > 0.50:

        return False

    # -----------------------------------------------------
    # Aspect ratio.
    #
    # Signatures can vary considerably, therefore this is
    # intentionally broad.
    # -----------------------------------------------------

    if aspect_ratio <= 0:

        return False

    if aspect_ratio > 8.0:

        return False

    # -----------------------------------------------------
    # Reasonable contour structure.
    # -----------------------------------------------------

    if components > HIGH_COMPONENT_COUNT:

        return False

    if contour_count > HIGH_COMPONENT_COUNT:

        return False

    # -----------------------------------------------------
    # Large amount of dense contour area is suspicious.
    # -----------------------------------------------------

    image_area = (
        float(features.image_width)
        *
        float(features.image_height)
    )

    if image_area <= 0:

        return False

    contour_area_ratio = (
        total_contour
        /
        image_area
    )

    if contour_area_ratio > 0.20:

        return False

    # -----------------------------------------------------
    # If all checks pass, geometry is plausible.
    # -----------------------------------------------------

    return True


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

    MobileNetV3:
        Primary classifier.

    YOLOS:
        Region detection.

    OpenCV:
        Geometry fallback.

    Input modes:

        upload
        draw
        capture
    """

    # =====================================================
    # 1. Normalize input mode
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
    # 3. Extract OpenCV features
    # =====================================================

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

    # =====================================================
    # 4. Blank image
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
    # 6. MobileNet classifier
    # =====================================================

    classifier = None

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
    # 8. Weak signature confidence
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
    # 9. Medium signature confidence
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
    # 10. Primary YOLOS detection
    # =====================================================

    detection: Optional[
        SignatureDetectionResult
    ] = None

    try:

        detection = detect_signatures(
            image=image,
            threshold=YOLOS_PRIMARY_THRESHOLD,
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

    # =====================================================
    # 11. Filter primary YOLOS detections
    # =====================================================

    primary_reliable = (
        _get_reliable_detections(
            detection
        )
    )

    # =====================================================
    # 12. Fallback YOLOS detection
    #
    # Only execute when MobileNet is extremely confident
    # and the primary YOLOS detector found no reliable region.
    # =====================================================

    if (
        not primary_reliable
        and
        classifier.signature_probability
        >= 0.95
    ):

        try:

            fallback_detection = (
                detect_signatures(
                    image=image,
                    threshold=(
                        YOLOS_FALLBACK_THRESHOLD
                    ),
                )
            )

            fallback_reliable = (
                _get_reliable_detections(
                    fallback_detection,
                    fallback=True,
                )
            )

            if fallback_reliable:

                detection = (
                    fallback_detection
                )

                primary_reliable = (
                    fallback_reliable
                )

        except Exception:
            # Do not fail the whole validation because the
            # fallback detector failed.
            pass

    # =====================================================
    # 13. No reliable YOLOS region
    # =====================================================

    if not primary_reliable:

        # -------------------------------------------------
        # OpenCV geometry fallback.
        # -------------------------------------------------

        if _opencv_region_fallback(
            features
        ):

            try:

                context = analyze_image_context(
                    image=image,
                    signature_result=(
                        detection
                        if detection is not None
                        else SignatureDetectionResult(
                            detected=False,
                            detection_count=0,
                            highest_score=0.0,
                            largest_area_ratio=0.0,
                            multiple_signatures=False,
                            detections=[],
                        )
                    ),
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

            # -------------------------------------------------
            # Strong document context still overrides.
            # -------------------------------------------------

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

            if document_signals >= 2:

                return _review(
                    reason_code="DOCUMENT_CONTEXT",
                    message=(
                        "The image contains a "
                        "high-confidence signature "
                        "classification but also "
                        "shows strong document context."
                    ),
                    features=features,
                    classifier=classifier,
                    context=context,
                )

            return _accept(
                reason_code="SIGNATURE_CANDIDATE",
                message=(
                    "Image contains a high-confidence "
                    "signature candidate supported by "
                    "image geometry."
                ),
                features=features,
                classifier=classifier,
                context=context,
            )

        # -------------------------------------------------
        # Geometry fallback did not support it.
        # -------------------------------------------------

        return _review(
            reason_code="SIGNATURE_REGION_NOT_FOUND",
            message=(
                "The classifier detected a signature, "
                "but no reliable signature region "
                "was detected by the region detector."
            ),
            features=features,
            classifier=classifier,
            context=None,
        )

    # =====================================================
    # 14. Independent signature analysis
    # =====================================================

    # Construct a clean detection result containing only
    # reliable detections.
    reliable_result = (
        SignatureDetectionResult(
            detected=True,
            detection_count=len(
                primary_reliable
            ),
            highest_score=max(
                item.confidence
                for item in primary_reliable
            ),
            largest_area_ratio=max(
                item.area_ratio
                for item in primary_reliable
            ),
            multiple_signatures=(
                len(primary_reliable) > 1
            ),
            detections=primary_reliable,
        )
    )

    independent_detections = (
        _get_independent_detections(
            reliable_result
        )
    )

    independent_count = len(
        independent_detections
    )

    # =====================================================
    # 15. Multiple signatures
    # =====================================================

    if independent_count > 1:

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
    # 16. Whole-image context
    # =====================================================

    try:

        context = analyze_image_context(
            image=image,
            signature_result=reliable_result,
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

    # =====================================================
    # 17. Document context
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
    # 18. Strong document context
    # =====================================================

    if document_signals >= 2:

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
    # 19. High-confidence signature
    # =====================================================

    if (
        classifier.signature_probability
        >= SIGNATURE_ACCEPT_THRESHOLD
    ):

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
    # 20. Fallback
    # =====================================================

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