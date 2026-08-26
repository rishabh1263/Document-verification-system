"""
Signature image feature extraction.

This module provides reusable image-processing functions for
signature verification.

The output is intentionally descriptive rather than a final
business decision. Final ACCEPT / REJECT / REVIEW logic belongs
to the signature validator.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# =========================================================
# Configuration
# =========================================================

# Foreground thresholding
DEFAULT_THRESHOLD = 200

# Small connected components are usually noise.
MIN_COMPONENT_AREA = 8

# Morphological processing
MORPH_KERNEL_SIZE = 3

# Document-line detection
HOUGH_THRESHOLD = 50

HOUGH_MIN_LINE_LENGTH_RATIO = 0.20

HOUGH_MAX_LINE_GAP = 20

DOCUMENT_LINE_THICKNESS = 3


# =========================================================
# Result structure
# =========================================================

@dataclass
class SignatureFeatures:
    """
    Extracted geometric/image features.
    """

    image_width: int

    image_height: int

    foreground_density: float

    bbox_x: Optional[int]

    bbox_y: Optional[int]

    bbox_width: Optional[int]

    bbox_height: Optional[int]

    aspect_ratio: Optional[float]

    occupancy_ratio: Optional[float]

    connected_components: int

    contour_count: int

    largest_contour_area: float

    total_contour_area: float


# =========================================================
# Validation helpers
# =========================================================

def _validate_image(
    image: np.ndarray,
) -> None:
    """
    Validate image input.
    """

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

    if len(image.shape) not in (2, 3):

        raise ValueError(
            "Image must be grayscale or BGR."
        )


# =========================================================
# Grayscale conversion
# =========================================================

def _to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert image to grayscale.

    Supports both grayscale and BGR input.
    """

    _validate_image(
        image
    )

    if len(image.shape) == 2:

        return image.copy()

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


# =========================================================
# Resize helper
# =========================================================

def _resize_for_processing(
    gray: np.ndarray,
    max_dimension: int = 1200,
) -> tuple[np.ndarray, float]:
    """
    Resize image for expensive OpenCV operations.

    Returns:
        resized image
        scale factor

    scale represents:

        resized = original * scale
    """

    height, width = gray.shape

    largest_dimension = max(
        height,
        width,
    )

    if largest_dimension <= max_dimension:

        return (
            gray.copy(),
            1.0,
        )

    scale = (
        max_dimension /
        largest_dimension
    )

    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return (
        resized,
        scale,
    )


# =========================================================
# Document line detection
# =========================================================

def _detect_document_lines(
    gray: np.ndarray,
) -> np.ndarray:
    """
    Detect long straight document/background lines.

    Hough line detection is used because photographed
    documents can contain slightly tilted lines.

    Returns:
        Binary mask containing detected long lines.
    """

    _validate_image(
        gray
    )

    if len(gray.shape) != 2:

        gray = _to_grayscale(
            gray
        )

    height, width = gray.shape

    (
        small,
        scale,
    ) = _resize_for_processing(
        gray,
        max_dimension=1000,
    )

    edges = cv2.Canny(
        small,
        50,
        150,
    )

    min_line_length = max(
        30,
        int(
            small.shape[1] *
            HOUGH_MIN_LINE_LENGTH_RATIO
        ),
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    line_mask = np.zeros_like(
        small,
        dtype=np.uint8,
    )

    if lines is None:

        return cv2.resize(
            line_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    # -----------------------------------------------------
    # IMPORTANT
    #
    # Depending on the OpenCV version/build,
    # HoughLinesP can return either:
    #
    #     (N, 1, 4)
    #
    # or:
    #
    #     (N, 4)
    #
    # Therefore we flatten each line instead of
    # assuming line[0] is always the coordinate array.
    # -----------------------------------------------------

    for line in lines:

        coordinates = np.asarray(
            line
        ).reshape(-1)

        if coordinates.size != 4:

            continue

        x1, y1, x2, y2 = (
            int(coordinates[0]),
            int(coordinates[1]),
            int(coordinates[2]),
            int(coordinates[3]),
        )

        cv2.line(
            line_mask,
            (x1, y1),
            (x2, y2),
            255,
            DOCUMENT_LINE_THICKNESS,
        )

    # Convert the resized mask back to original dimensions.
    line_mask = cv2.resize(
        line_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return line_mask


# =========================================================
# Remove document lines
# =========================================================

def _remove_document_lines(
    gray: np.ndarray,
) -> np.ndarray:
    """
    Remove long straight document/background lines.

    This is useful for forms and documents where horizontal
    or vertical lines could otherwise be interpreted as
    signature foreground.
    """

    line_mask = _detect_document_lines(
        gray
    )

    result = gray.copy()

    result[line_mask > 0] = 255

    return result


# =========================================================
# Foreground mask
# =========================================================

def create_foreground_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Create a binary foreground mask.

    White pixels represent foreground.

    Black pixels represent background.

    This function is shared by the signature feature
    extraction and image-context analysis pipelines.
    """

    _validate_image(
        image
    )

    gray = _to_grayscale(
        image
    )

    # -----------------------------------------------------
    # Normalize uneven lighting.
    # -----------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # -----------------------------------------------------
    # Remove long document/background lines.
    # -----------------------------------------------------

    line_removed = _remove_document_lines(
        gray
    )

    # -----------------------------------------------------
    # Adaptive thresholding.
    #
    # This handles photographed signatures better than
    # a single fixed threshold.
    # -----------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        line_removed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    # -----------------------------------------------------
    # Otsu threshold.
    # -----------------------------------------------------

    _, otsu = cv2.threshold(
        line_removed,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU,
    )

    # -----------------------------------------------------
    # Combine adaptive and Otsu masks.
    # -----------------------------------------------------

    mask = cv2.bitwise_and(
        adaptive,
        otsu,
    )

    # -----------------------------------------------------
    # Morphological cleanup.
    # -----------------------------------------------------

    kernel = np.ones(
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # -----------------------------------------------------
    # Remove very small connected components.
    # -----------------------------------------------------

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    for index in range(
        1,
        number_of_labels,
    ):

        area = stats[
            index,
            cv2.CC_STAT_AREA,
        ]

        if area >= MIN_COMPONENT_AREA:

            cleaned[
                labels == index
            ] = 255

    return cleaned


# =========================================================
# Foreground bounding box
# =========================================================

def _get_foreground_bbox(
    mask: np.ndarray,
) -> Optional[
    tuple[int, int, int, int]
]:
    """
    Get bounding box around all foreground pixels.
    """

    points = cv2.findNonZero(
        mask
    )

    if points is None:

        return None

    x, y, width, height = (
        cv2.boundingRect(
            points
        )
    )

    return (
        x,
        y,
        width,
        height,
    )


# =========================================================
# Connected components
# =========================================================

def _get_component_count(
    mask: np.ndarray,
) -> int:
    """
    Count meaningful connected components.
    """

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
# Contours
# =========================================================

def _get_contour_features(
    mask: np.ndarray,
) -> tuple[
    int,
    float,
    float,
]:
    """
    Calculate contour count and contour areas.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:

        return (
            0,
            0.0,
            0.0,
        )

    areas = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area > 0:

            areas.append(
                float(area)
            )

    if not areas:

        return (
            0,
            0.0,
            0.0,
        )

    return (
        len(areas),
        max(areas),
        sum(areas),
    )


# =========================================================
# Main feature extraction
# =========================================================

def extract_signature_features(
    image: np.ndarray,
) -> SignatureFeatures:
    """
    Extract signature-related geometric features.

    This function does not decide whether the image
    contains a valid signature.
    """

    _validate_image(
        image
    )

    height, width = (
        image.shape[:2]
    )

    if (
        height <= 0
        or
        width <= 0
    ):

        raise ValueError(
            "Image dimensions must be positive."
        )

    mask = create_foreground_mask(
        image
    )

    total_pixels = (
        width *
        height
    )

    foreground_pixels = (
        cv2.countNonZero(
            mask
        )
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
    # Bounding box
    # -----------------------------------------------------

    bbox = _get_foreground_bbox(
        mask
    )

    if bbox is None:

        bbox_x = None
        bbox_y = None
        bbox_width = None
        bbox_height = None

        aspect_ratio = None
        occupancy_ratio = None

    else:

        (
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
        ) = bbox

        if bbox_height > 0:

            aspect_ratio = round(
                bbox_width /
                bbox_height,
                4,
            )

        else:

            aspect_ratio = None

        bbox_area = (
            bbox_width *
            bbox_height
        )

        if bbox_area > 0:

            occupancy_ratio = round(
                foreground_pixels /
                bbox_area,
                6,
            )

        else:

            occupancy_ratio = None

    # -----------------------------------------------------
    # Components
    # -----------------------------------------------------

    connected_components = (
        _get_component_count(
            mask
        )
    )

    # -----------------------------------------------------
    # Contours
    # -----------------------------------------------------

    (
        contour_count,
        largest_contour_area,
        total_contour_area,
    ) = _get_contour_features(
        mask
    )

    return SignatureFeatures(
        image_width=width,
        image_height=height,

        foreground_density=(
            foreground_density
        ),

        bbox_x=bbox_x,
        bbox_y=bbox_y,

        bbox_width=bbox_width,
        bbox_height=bbox_height,

        aspect_ratio=aspect_ratio,

        occupancy_ratio=(
            occupancy_ratio
        ),

        connected_components=(
            connected_components
        ),

        contour_count=(
            contour_count
        ),

        largest_contour_area=round(
            largest_contour_area,
            4,
        ),

        total_contour_area=round(
            total_contour_area,
            4,
        ),
    )