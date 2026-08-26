"""
Final signature validation layer.

Supports:

    1. upload
    2. draw
    3. capture

Pipeline:

    RAW IMAGE
        |
        +--> input validation
        |
        +--> OpenCV signature features
        |
        +--> YOLOS signature detection
        |
        +--> whole-image context
        |
        +--> input-mode aware fallback
        |
        +--> conservative decision
        |
        +--> ACCEPT / REVIEW / REJECT

Important:
YOLOS detection alone is NOT sufficient for ACCEPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.documents.signature.validators.signature_features import (
    SignatureFeatures,
    extract_signature_features,
)

from src.documents.signature.services.signature_detector import (
    SignatureDetectionResult,
    detect_signatures,
)

from src.documents.signature.services.image_context import (
    ImageContextResult,
    analyze_image_context,
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
# Result
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

    detection: Optional[
        SignatureDetectionResult
    ] = None

    context: Optional[
        ImageContextResult
    ] = None


# =========================================================
# Configuration
# =========================================================

YOLOS_MIN_SCORE = 0.50

STRONG_YOLOS_SCORE = 0.90

WEAK_YOLOS_SCORE = 0.85


# ---------------------------------------------------------
# Detector region sanity
# ---------------------------------------------------------

MAX_SIGNATURE_REGION_RATIO = 0.35

MAX_SIGNATURE_WIDTH_RATIO = 0.75

MAX_SIGNATURE_HEIGHT_RATIO = 0.75


# ---------------------------------------------------------
# Whole-image foreground sanity
# ---------------------------------------------------------

MAX_FOREGROUND_DENSITY = 0.35

MAX_BBOX_WIDTH_RATIO = 0.85

MAX_BBOX_HEIGHT_RATIO = 0.85


# ---------------------------------------------------------
# Context thresholds
# ---------------------------------------------------------

HIGH_COMPONENT_COUNT = 250

HIGH_EDGE_DENSITY = 0.08

DOCUMENT_CONTEXT_COMPONENTS = 150

SMALL_SIGNATURE_AREA_RATIO = 0.02


# ---------------------------------------------------------
# Input-mode fallback thresholds
# ---------------------------------------------------------

# A cropped uploaded signature can be nearly square.
#
# Therefore aspect ratio is NOT used as a hard rejection
# for upload mode.
#
# These values are deliberately conservative.

UPLOAD_MIN_FOREGROUND_DENSITY = 0.003

UPLOAD_MAX_FOREGROUND_DENSITY = 0.15

UPLOAD_MIN_COMPONENTS = 1

UPLOAD_MAX_COMPONENTS = 120

UPLOAD_MAX_BBOX_WIDTH_RATIO = 0.95

UPLOAD_MAX_BBOX_HEIGHT_RATIO = 0.95

UPLOAD_MIN_OCCUPANCY = 0.015


# Drawn signatures generally contain fewer connected
# components and are usually isolated from document context.

DRAW_MIN_FOREGROUND_DENSITY = 0.002

DRAW_MAX_FOREGROUND_DENSITY = 0.20

DRAW_MIN_COMPONENTS = 1

DRAW_MAX_COMPONENTS = 80

DRAW_MAX_BBOX_WIDTH_RATIO = 0.98

DRAW_MAX_BBOX_HEIGHT_RATIO = 0.98

DRAW_MIN_OCCUPANCY = 0.008


# Camera capture is stricter because the image can contain
# documents, tables, handwriting and background content.

CAPTURE_MIN_FOREGROUND_DENSITY = 0.003

CAPTURE_MAX_FOREGROUND_DENSITY = 0.15

CAPTURE_MAX_COMPONENTS = 120

CAPTURE_MAX_BBOX_WIDTH_RATIO = 0.90

CAPTURE_MAX_BBOX_HEIGHT_RATIO = 0.90

CAPTURE_MIN_OCCUPANCY = 0.015


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------

ACCEPT_CONFIDENCE = 0.95

REVIEW_CONFIDENCE = 0.60

REJECT_CONFIDENCE = 0.95


# =========================================================
# Input-mode normalization
# =========================================================

def _normalize_input_mode(
    input_mode: str | SignatureInputMode,
) -> SignatureInputMode:
    """
    Normalize input mode to SignatureInputMode.

    Unknown values safely fall back to upload.
    """

    if isinstance(
        input_mode,
        SignatureInputMode,
    ):

        return input_mode

    try:

        return SignatureInputMode(
            str(input_mode).strip().lower()
        )

    except ValueError:

        return SignatureInputMode.UPLOAD


# =========================================================
# Image validation
# =========================================================

def _validate_input_image(
    image: np.ndarray,
) -> Optional[str]:
    """
    Validate input without raising.

    The public validator returns a structured result
    for invalid images.
    """

    if image is None:

        return (
            "Image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):

        return (
            "Image must be a NumPy array."
        )

    if image.size == 0:

        return (
            "Image cannot be empty."
        )

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
# PIL conversion
# =========================================================

def _to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert OpenCV image to PIL.
    """

    if len(image.shape) == 2:

        return Image.fromarray(
            image
        )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(
        rgb
    )


# =========================================================
# Invalid result
# =========================================================

def _invalid_result(
    message: str,
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=(
            SignatureDecision.REJECT
        ),

        confidence=REJECT_CONFIDENCE,

        reason_code="INVALID_IMAGE",

        message=message,

        features=None,

        detection=None,

        context=None,
    )


# =========================================================
# Reject helper
# =========================================================

def _reject(
    reason_code: str,
    message: str,
    features: Optional[
        SignatureFeatures
    ],
    detection: Optional[
        SignatureDetectionResult
    ],
    context: Optional[
        ImageContextResult
    ],
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=(
            SignatureDecision.REJECT
        ),

        confidence=REJECT_CONFIDENCE,

        reason_code=reason_code,

        message=message,

        features=features,

        detection=detection,

        context=context,
    )


# =========================================================
# Review helper
# =========================================================

def _review(
    reason_code: str,
    message: str,
    features: Optional[
        SignatureFeatures
    ],
    detection: Optional[
        SignatureDetectionResult
    ],
    context: Optional[
        ImageContextResult
    ],
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=(
            SignatureDecision.REVIEW
        ),

        confidence=REVIEW_CONFIDENCE,

        reason_code=reason_code,

        message=message,

        features=features,

        detection=detection,

        context=context,
    )


# =========================================================
# Accept helper
# =========================================================

def _accept(
    reason_code: str,
    message: str,
    features: SignatureFeatures,
    detection: Optional[
        SignatureDetectionResult
    ],
    context: Optional[
        ImageContextResult
    ],
) -> SignatureValidationResult:

    return SignatureValidationResult(

        decision=(
            SignatureDecision.ACCEPT
        ),

        confidence=ACCEPT_CONFIDENCE,

        reason_code=reason_code,

        message=message,

        features=features,

        detection=detection,

        context=context,
    )


# =========================================================
# Foreground ratios
# =========================================================

def _get_bbox_ratios(
    features: SignatureFeatures,
) -> tuple[float, float]:
    """
    Return foreground bounding-box width/height ratios.
    """

    if (
        features.bbox_width is None
        or
        features.bbox_height is None
        or
        features.image_width <= 0
        or
        features.image_height <= 0
    ):

        return (
            0.0,
            0.0,
        )

    width_ratio = (
        features.bbox_width /
        features.image_width
    )

    height_ratio = (
        features.bbox_height /
        features.image_height
    )

    return (
        width_ratio,
        height_ratio,
    )


# =========================================================
# Upload fallback
# =========================================================

def _is_upload_signature_candidate(
    features: SignatureFeatures,
) -> bool:
    """
    Determine whether an uploaded image has enough
    signature-like geometric evidence to move to REVIEW
    when YOLOS does not detect a signature.

    This intentionally does NOT use aspect ratio as a
    hard requirement.

    Reason:

    A cropped signature downloaded from the internet may
    be nearly square because of whitespace, transparent
    padding, or the original crop.
    """

    width_ratio, height_ratio = (
        _get_bbox_ratios(
            features
        )
    )

    density = (
        features.foreground_density
    )

    components = (
        features.connected_components
    )

    occupancy = (
        features.occupancy_ratio
        or 0.0
    )

    if density <= 0:

        return False

    if density < (
        UPLOAD_MIN_FOREGROUND_DENSITY
    ):

        return False

    if density > (
        UPLOAD_MAX_FOREGROUND_DENSITY
    ):

        return False

    if components < (
        UPLOAD_MIN_COMPONENTS
    ):

        return False

    if components > (
        UPLOAD_MAX_COMPONENTS
    ):

        return False

    if width_ratio > (
        UPLOAD_MAX_BBOX_WIDTH_RATIO
    ):

        return False

    if height_ratio > (
        UPLOAD_MAX_BBOX_HEIGHT_RATIO
    ):

        return False

    if occupancy < (
        UPLOAD_MIN_OCCUPANCY
    ):

        return False

    return True


# =========================================================
# Draw fallback
# =========================================================

def _is_draw_signature_candidate(
    features: SignatureFeatures,
) -> bool:
    """
    Determine whether a drawn signature has enough
    geometric evidence to move to REVIEW.
    """

    width_ratio, height_ratio = (
        _get_bbox_ratios(
            features
        )
    )

    density = (
        features.foreground_density
    )

    components = (
        features.connected_components
    )

    occupancy = (
        features.occupancy_ratio
        or 0.0
    )

    if density <= 0:

        return False

    if density < (
        DRAW_MIN_FOREGROUND_DENSITY
    ):

        return False

    if density > (
        DRAW_MAX_FOREGROUND_DENSITY
    ):

        return False

    if components < (
        DRAW_MIN_COMPONENTS
    ):

        return False

    if components > (
        DRAW_MAX_COMPONENTS
    ):

        return False

    if width_ratio > (
        DRAW_MAX_BBOX_WIDTH_RATIO
    ):

        return False

    if height_ratio > (
        DRAW_MAX_BBOX_HEIGHT_RATIO
    ):

        return False

    if occupancy < (
        DRAW_MIN_OCCUPANCY
    ):

        return False

    return True


# =========================================================
# Capture fallback
# =========================================================

def _is_capture_signature_candidate(
    features: SignatureFeatures,
) -> bool:
    """
    Determine whether a camera-captured signature has
    enough geometric evidence to move to REVIEW.

    Capture is intentionally stricter than upload/draw.
    """

    width_ratio, height_ratio = (
        _get_bbox_ratios(
            features
        )
    )

    density = (
        features.foreground_density
    )

    components = (
        features.connected_components
    )

    occupancy = (
        features.occupancy_ratio
        or 0.0
    )

    if density <= 0:

        return False

    if density < (
        CAPTURE_MIN_FOREGROUND_DENSITY
    ):

        return False

    if density > (
        CAPTURE_MAX_FOREGROUND_DENSITY
    ):

        return False

    if components > (
        CAPTURE_MAX_COMPONENTS
    ):

        return False

    if width_ratio > (
        CAPTURE_MAX_BBOX_WIDTH_RATIO
    ):

        return False

    if height_ratio > (
        CAPTURE_MAX_BBOX_HEIGHT_RATIO
    ):

        return False

    if occupancy < (
        CAPTURE_MIN_OCCUPANCY
    ):

        return False

    return True


# =========================================================
# Generic geometric fallback
# =========================================================

def _is_geometric_signature_candidate(
    features: SignatureFeatures,
) -> bool:
    """
    Generic fallback for signature-like geometry.

    Aspect ratio is deliberately NOT mandatory.

    This prevents rejection of compact or square crops.
    """

    width_ratio, height_ratio = (
        _get_bbox_ratios(
            features
        )
    )

    density = (
        features.foreground_density
    )

    occupancy = (
        features.occupancy_ratio
        or 0.0
    )

    components = (
        features.connected_components
    )

    if density <= 0:

        return False

    if density > (
        MAX_FOREGROUND_DENSITY
    ):

        return False

    if width_ratio > 0.98:

        return False

    if height_ratio > 0.98:

        return False

    if components <= 0:

        return False

    if occupancy <= 0.005:

        return False

    return True


# =========================================================
# Input-mode fallback dispatcher
# =========================================================

def _is_input_mode_signature_candidate(
    features: SignatureFeatures,
    input_mode: SignatureInputMode,
) -> bool:
    """
    Apply input-mode specific fallback logic.
    """

    if input_mode == (
        SignatureInputMode.DRAW
    ):

        return _is_draw_signature_candidate(
            features
        )

    if input_mode == (
        SignatureInputMode.CAPTURE
    ):

        return _is_capture_signature_candidate(
            features
        )

    # Default/upload.
    return _is_upload_signature_candidate(
        features
    )


# =========================================================
# Main validator
# =========================================================

def validate_signature(
    image: np.ndarray,
    input_mode: str | SignatureInputMode = "upload",
) -> SignatureValidationResult:
    """
    Validate a signature image.

    Supported input modes:

        upload
        draw
        capture

    Decision philosophy:

        Strong YOLOS + clean context
            -> ACCEPT

        Weak YOLOS
            -> REVIEW

        No YOLOS + strong geometric evidence
            -> REVIEW

        No YOLOS + weak evidence
            -> REJECT

    No automatic ACCEPT is performed purely from OpenCV
    geometric features.
    """

    # =====================================================
    # 1. Normalize input mode
    # =====================================================

    mode = _normalize_input_mode(
        input_mode
    )

    # =====================================================
    # 2. Validate input
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
    # 3. Feature extraction
    # =====================================================

    try:

        features = (
            extract_signature_features(
                image
            )
        )

    except Exception as exc:

        return _invalid_result(
            f"Feature extraction failed: {exc}"
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
            detection=None,
            context=None,
        )

    # =====================================================
    # 5. Foreground bounding-box sanity
    # =====================================================

    (
        bbox_width_ratio,
        bbox_height_ratio,
    ) = _get_bbox_ratios(
        features
    )

    if (
        bbox_width_ratio
        >= MAX_BBOX_WIDTH_RATIO
        and
        bbox_height_ratio
        >= MAX_BBOX_HEIGHT_RATIO
    ):

        return _reject(
            reason_code="REGION_TOO_LARGE",
            message=(
                "Foreground region occupies "
                "most of the image."
            ),
            features=features,
            detection=None,
            context=None,
        )

    # =====================================================
    # 6. Excessive foreground
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
            detection=None,
            context=None,
        )

    # =====================================================
    # 7. YOLOS detection
    # =====================================================

    detection = None

    try:

        pil_image = _to_pil(
            image
        )

        detection = detect_signatures(
            pil_image,
            threshold=YOLOS_MIN_SCORE,
        )

    except Exception as exc:

        return _review(
            reason_code="DETECTOR_ERROR",
            message=(
                "Signature detector failed: "
                f"{exc}"
            ),
            features=features,
            detection=None,
            context=None,
        )

    # =====================================================
    # 8. No YOLOS detection
    # =====================================================

    if not detection.detected:

        geometric_candidate = (
            _is_input_mode_signature_candidate(
                features,
                mode,
            )
        )

        if geometric_candidate:

            return _review(
                reason_code="NO_SIGNATURE_DETECTION",
                message=(
                    "No YOLOS signature was detected, "
                    "but the image contains signature-like "
                    "geometric features. Manual review "
                    "is required."
                ),
                features=features,
                detection=detection,
                context=None,
            )

        # -------------------------------------------------
        # Generic fallback.
        #
        # This catches reasonable signature-like images
        # while still refusing blank/document-like images.
        # -------------------------------------------------

        if (
            mode == SignatureInputMode.UPLOAD
            and
            _is_geometric_signature_candidate(
                features
            )
        ):

            return _review(
                reason_code="NO_SIGNATURE_DETECTION",
                message=(
                    "No YOLOS signature was detected, "
                    "but the uploaded image contains "
                    "usable signature-like foreground. "
                    "Manual review is required."
                ),
                features=features,
                detection=detection,
                context=None,
            )

        return _reject(
            reason_code="NO_SIGNATURE",
            message=(
                "No signature-like region "
                "was detected."
            ),
            features=features,
            detection=detection,
            context=None,
        )

    # =====================================================
    # 9. Signature region sanity
    # =====================================================

    signature_area_ratio = (
        detection.largest_area_ratio
    )

    signature_width_ratio = (
        detection.largest_width_ratio
    )

    signature_height_ratio = (
        detection.largest_height_ratio
    )

    if (
        signature_area_ratio
        >= MAX_SIGNATURE_REGION_RATIO
    ):

        return _reject(
            reason_code="REGION_TOO_LARGE",
            message=(
                "Detected signature region "
                "occupies too much of the "
                "image."
            ),
            features=features,
            detection=detection,
            context=None,
        )

    if (
        signature_width_ratio
        >= MAX_SIGNATURE_WIDTH_RATIO
        or
        signature_height_ratio
        >= MAX_SIGNATURE_HEIGHT_RATIO
    ):

        return _reject(
            reason_code="SIGNATURE_REGION_TOO_LARGE",
            message=(
                "Detected signature region "
                "is too large for a normal "
                "standalone signature."
            ),
            features=features,
            detection=detection,
            context=None,
        )

    # =====================================================
    # 10. Whole-image context
    # =====================================================

    try:

        context = (
            analyze_image_context(
                image=image,
                signature_result=detection,
            )
        )

    except Exception as exc:

        return _review(
            reason_code="CONTEXT_ANALYSIS_ERROR",
            message=(
                "Whole-image context analysis "
                f"failed: {exc}"
            ),
            features=features,
            detection=detection,
            context=None,
        )

    # =====================================================
    # 11. Multiple signatures
    # =====================================================

    if (
        context.multiple_signatures
    ):

        return _review(
            reason_code="MULTIPLE_SIGNATURES",
            message=(
                "Multiple signature-like "
                "regions were detected."
            ),
            features=features,
            detection=detection,
            context=context,
        )

    # =====================================================
    # 12. Weak YOLOS detection
    # =====================================================

    if (
        detection.highest_score
        < WEAK_YOLOS_SCORE
    ):

        return _review(
            reason_code="WEAK_SIGNATURE_DETECTION",
            message=(
                "Signature detector confidence "
                "is too low for automatic "
                "acceptance."
            ),
            features=features,
            detection=detection,
            context=context,
        )

    # =====================================================
    # 13. Document-context scoring
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

    if (
        context.signature_count
        > 1
    ):

        document_signals += 1

    # =====================================================
    # 14. Strong document context
    # =====================================================

    if document_signals >= 2:

        return _review(
            reason_code="DOCUMENT_CONTEXT",
            message=(
                "The image contains a "
                "signature-like region but "
                "also shows strong document "
                "context."
            ),
            features=features,
            detection=detection,
            context=context,
        )

    # =====================================================
    # 15. Strong YOLOS candidate
    # =====================================================

    if (
        detection.highest_score
        >= STRONG_YOLOS_SCORE
    ):

        return _accept(
            reason_code="SIGNATURE_CANDIDATE",
            message=(
                "Image contains a strong "
                "signature-like candidate."
            ),
            features=features,
            detection=detection,
            context=context,
        )

    # =====================================================
    # 16. Final REVIEW
    # =====================================================

    return _review(
        reason_code="REVIEW_REQUIRED",
        message=(
            "Image contains a signature-like "
            "region but does not meet the "
            "confidence required for automatic "
            "acceptance."
        ),
        features=features,
        detection=detection,
        context=context,
    )