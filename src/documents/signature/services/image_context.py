"""
Whole-image context analysis for signature verification.

This module extracts contextual signals that help distinguish:

    signature-only image
        vs
    document containing a signature

It does NOT make the final ACCEPT / REJECT decision.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.documents.signature.services.signature_detector import (
    SignatureDetectionResult,
)

from src.documents.signature.validators.signature_features import (
    create_foreground_mask,
)


# =========================================================
# Configuration
# =========================================================

MIN_COMPONENT_AREA = 8

LARGE_RECTANGLE_MIN_AREA_RATIO = 0.20

HIGH_EDGE_DENSITY = 0.12

HIGH_COMPONENT_COUNT = 250


# =========================================================
# Result
# =========================================================

@dataclass
class ImageContextResult:

    image_width: int

    image_height: int

    foreground_density: float

    edge_density: float

    connected_components: int

    large_rectangle_count: int

    largest_rectangle_area_ratio: float

    document_like: bool

    signature_area_ratio: float

    signature_count: int

    multiple_signatures: bool


# =========================================================
# Edge density
# =========================================================

def _calculate_edge_density(
    image: np.ndarray,
) -> float:

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if image.size == 0:

        raise ValueError(
            "Image cannot be empty."
        )

    if len(image.shape) == 2:

        gray = image

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_pixels = cv2.countNonZero(
        edges
    )

    total_pixels = (
        edges.shape[0] *
        edges.shape[1]
    )

    if total_pixels == 0:

        return 0.0

    return round(
        edge_pixels /
        total_pixels,
        6,
    )


# =========================================================
# Connected components
# =========================================================

def _count_components(
    mask: np.ndarray,
) -> int:

    if mask is None:

        return 0

    if mask.size == 0:

        return 0

    number_of_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    count = 0

    for index in range(
        1,
        number_of_labels,
    ):

        area = stats[
            index,
            cv2.CC_STAT_AREA,
        ]

        if area >= MIN_COMPONENT_AREA:

            count += 1

    return count


# =========================================================
# Rectangle detection
# =========================================================

def _detect_large_rectangles(
    image: np.ndarray,
) -> tuple[int, float]:

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if image.size == 0:

        raise ValueError(
            "Image cannot be empty."
        )

    if len(image.shape) == 2:

        gray = image

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = (
        image.shape[0] *
        image.shape[1]
    )

    if image_area <= 0:

        return (
            0,
            0.0,
        )

    large_rectangle_count = 0

    largest_rectangle_ratio = 0.0

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approximation) != 4:
            continue

        contour_area = cv2.contourArea(
            contour
        )

        if contour_area <= 0:
            continue

        area_ratio = (
            contour_area /
            image_area
        )

        if (
            area_ratio >=
            LARGE_RECTANGLE_MIN_AREA_RATIO
        ):

            large_rectangle_count += 1

            largest_rectangle_ratio = max(
                largest_rectangle_ratio,
                area_ratio,
            )

    return (
        large_rectangle_count,
        round(
            largest_rectangle_ratio,
            6,
        ),
    )


# =========================================================
# Document-like heuristic
# =========================================================

def _is_document_like(
    connected_components: int,
    edge_density: float,
    large_rectangle_count: int,
) -> bool:
    """
    Conservative contextual heuristic.

    This is NOT a document classifier.
    """

    document_signals = 0

    if (
        connected_components >=
        HIGH_COMPONENT_COUNT
    ):

        document_signals += 1

    if (
        edge_density >=
        HIGH_EDGE_DENSITY
    ):

        document_signals += 1

    if large_rectangle_count > 0:

        document_signals += 1

    return (
        document_signals >= 2
    )


# =========================================================
# Main analyzer
# =========================================================

def analyze_image_context(
    image: np.ndarray,
    signature_result: SignatureDetectionResult,
) -> ImageContextResult:

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):

        raise ValueError(
            "Image must be a NumPy array."
        )

    if image.size == 0:

        raise ValueError(
            "Image cannot be empty."
        )

    if signature_result is None:

        raise ValueError(
            "signature_result cannot be None."
        )

    height, width = (
        image.shape[:2]
    )

    # -----------------------------------------------------
    # Reuse the shared signature foreground mask.
    # -----------------------------------------------------

    mask = create_foreground_mask(
        image
    )

    foreground_pixels = (
        cv2.countNonZero(
            mask
        )
    )

    total_pixels = (
        width *
        height
    )

    if total_pixels > 0:

        foreground_density = (
            foreground_pixels /
            total_pixels
        )

    else:

        foreground_density = 0.0

    foreground_density = round(
        foreground_density,
        6,
    )

    # -----------------------------------------------------
    # Edge density.
    # -----------------------------------------------------

    edge_density = (
        _calculate_edge_density(
            image
        )
    )

    # -----------------------------------------------------
    # Connected components.
    # -----------------------------------------------------

    connected_components = (
        _count_components(
            mask
        )
    )

    # -----------------------------------------------------
    # Large rectangles.
    # -----------------------------------------------------

    (
        large_rectangle_count,
        largest_rectangle_area_ratio,
    ) = _detect_large_rectangles(
        image
    )

    # -----------------------------------------------------
    # YOLOS features.
    # -----------------------------------------------------

    signature_area_ratio = 0.0

    if signature_result.detected:

        signature_area_ratio = (
            signature_result
            .largest_area_ratio
        )

    signature_count = (
        signature_result
        .detection_count
    )

    multiple_signatures = (
        signature_result
        .multiple_signatures
    )

    # -----------------------------------------------------
    # Context heuristic.
    # -----------------------------------------------------

    document_like = _is_document_like(
        connected_components=(
            connected_components
        ),
        edge_density=edge_density,
        large_rectangle_count=(
            large_rectangle_count
        ),
    )

    return ImageContextResult(

        image_width=width,

        image_height=height,

        foreground_density=(
            foreground_density
        ),

        edge_density=(
            edge_density
        ),

        connected_components=(
            connected_components
        ),

        large_rectangle_count=(
            large_rectangle_count
        ),

        largest_rectangle_area_ratio=(
            largest_rectangle_area_ratio
        ),

        document_like=(
            document_like
        ),

        signature_area_ratio=round(
            signature_area_ratio,
            6,
        ),

        signature_count=(
            signature_count
        ),

        multiple_signatures=(
            multiple_signatures
        ),
    )