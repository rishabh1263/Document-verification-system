"""
Foreground feature extraction for signature verification.

Pipeline:

    Image
      ↓
    Grayscale
      ↓
    Adaptive threshold
      ↓
    Remove long document lines
      ↓
    Remove border artifacts
      ↓
    Remove tiny noise
      ↓
    Group nearby foreground components
      ↓
    Signature candidate region
      ↓
    Extract geometric features

This module does NOT classify an image as a signature.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# =========================================================
# Configuration
# =========================================================

# Adaptive threshold
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# Hough line detection
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH_RATIO = 0.15
HOUGH_MAX_LINE_GAP = 20

# Lines close to horizontal are treated as document lines.
MAX_HORIZONTAL_ANGLE = 12.0

# Lines close to vertical near the image border are treated
# as document borders.
MAX_VERTICAL_ANGLE = 12.0

BORDER_MARGIN_RATIO = 0.05

# Small noise removal
MIN_COMPONENT_AREA = 8

# Component grouping.
#
# Nearby strokes belonging to the same signature will be
# connected temporarily using this dilation kernel.
GROUP_KERNEL_WIDTH = 21
GROUP_KERNEL_HEIGHT = 11


# =========================================================
# Result
# =========================================================

@dataclass
class SignatureFeatures:
    """Features extracted from the cleaned signature candidate."""

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
# Grayscale conversion
# =========================================================

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR, BGRA, or grayscale image to grayscale.
    """

    if len(image.shape) == 2:
        return image

    channels = image.shape[2]

    if channels == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if channels == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY
        )

    raise ValueError(
        "Unsupported image channel format."
    )


# =========================================================
# Raw foreground mask
# =========================================================

def _create_raw_foreground_mask(
    image: np.ndarray
) -> np.ndarray:
    """
    Create an initial foreground mask using adaptive
    thresholding.
    """

    gray = _to_grayscale(image)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    return binary


# =========================================================
# Detect document lines
# =========================================================

def _detect_document_lines(
    gray: np.ndarray
) -> np.ndarray:
    """
    Detect long straight document/background lines.

    Hough line detection is used because photographed
    documents can have lines that are slightly tilted.
    """

    height, width = gray.shape

    # Resize for faster processing.
    scale = min(
        1.0,
        1000.0 / max(height, width)
    )

    if scale < 1.0:
        small = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = gray.copy()

    edges = cv2.Canny(
        small,
        50,
        150
    )

    min_line_length = max(
        30,
        int(
            small.shape[1] *
            HOUGH_MIN_LINE_LENGTH_RATIO
        )
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    line_mask = np.zeros_like(
        small,
        dtype=np.uint8
    )

    if lines is None:
        return cv2.resize(
            line_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    for line in lines:

        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(
            np.arctan2(
                dy,
                dx
            )
        )

        absolute_angle = abs(angle)

        if absolute_angle > 90:
            absolute_angle = (
                180 -
                absolute_angle
            )

        line_length = np.sqrt(
            dx * dx +
            dy * dy
        )

        # -------------------------------------------------
        # Horizontal document lines
        # -------------------------------------------------

        if (
            absolute_angle <=
            MAX_HORIZONTAL_ANGLE
            and
            line_length >=
            min_line_length
        ):

            cv2.line(
                line_mask,
                (x1, y1),
                (x2, y2),
                255,
                3
            )

            continue

        # -------------------------------------------------
        # Vertical document borders
        # -------------------------------------------------

        distance_from_left = min(
            x1,
            x2
        )

        distance_from_right = min(
            small.shape[1] - x1,
            small.shape[1] - x2
        )

        near_left_border = (
            distance_from_left <
            small.shape[1] *
            BORDER_MARGIN_RATIO
        )

        near_right_border = (
            distance_from_right <
            small.shape[1] *
            BORDER_MARGIN_RATIO
        )

        if (
            abs(
                90 -
                absolute_angle
            ) <= MAX_VERTICAL_ANGLE
            and
            (
                near_left_border
                or
                near_right_border
            )
        ):

            cv2.line(
                line_mask,
                (x1, y1),
                (x2, y2),
                255,
                5
            )

    line_mask = cv2.resize(
        line_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    return line_mask


# =========================================================
# Remove document lines
# =========================================================

def _remove_document_lines(
    binary: np.ndarray,
    gray: np.ndarray
) -> np.ndarray:
    """
    Remove long document/background lines.
    """

    line_mask = _detect_document_lines(
        gray
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    line_mask = cv2.dilate(
        line_mask,
        kernel,
        iterations=1
    )

    cleaned = cv2.bitwise_and(
        binary,
        cv2.bitwise_not(line_mask)
    )

    return cleaned


# =========================================================
# Remove border artifacts
# =========================================================

def _remove_border_artifacts(
    binary: np.ndarray
) -> np.ndarray:
    """
    Remove foreground components touching the image border.

    This prevents page edges/borders from becoming the
    signature candidate.
    """

    height, width = binary.shape

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = binary.copy()

    for label in range(
        1,
        num_labels
    ):

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        touches_left = x <= 0
        touches_top = y <= 0

        touches_right = (
            x + w >= width
        )

        touches_bottom = (
            y + h >= height
        )

        if (
            touches_left
            or touches_top
            or touches_right
            or touches_bottom
        ):

            cleaned[
                labels == label
            ] = 0

    return cleaned


# =========================================================
# Remove tiny components
# =========================================================

def _remove_small_components(
    binary: np.ndarray
) -> np.ndarray:
    """
    Remove very small isolated foreground components.

    These are typically:
        - camera noise
        - JPEG artifacts
        - tiny dots
        - thresholding artifacts
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        binary
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_AREA:

            cleaned[
                labels == label
            ] = 255

    return cleaned


# =========================================================
# Group nearby components
# =========================================================

def _group_signature_components(
    binary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group nearby foreground components.

    A signature can consist of several disconnected strokes.
    We therefore temporarily dilate nearby components so that
    strokes belonging to the same signature become one
    candidate region.

    Returns
    -------
    grouped_mask:
        Temporary mask used to identify candidate regions.

    cleaned_binary:
        Original foreground pixels with tiny noise removed.
    """

    cleaned_binary = _remove_small_components(
        binary
    )

    grouping_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            GROUP_KERNEL_WIDTH,
            GROUP_KERNEL_HEIGHT
        )
    )

    grouped_mask = cv2.dilate(
        cleaned_binary,
        grouping_kernel,
        iterations=1
    )

    # Close small gaps between nearby strokes.
    grouped_mask = cv2.morphologyEx(
        grouped_mask,
        cv2.MORPH_CLOSE,
        grouping_kernel,
        iterations=1
    )

    return (
        grouped_mask,
        cleaned_binary
    )


# =========================================================
# Public foreground mask
# =========================================================

def create_foreground_mask(
    image: np.ndarray
) -> np.ndarray:
    """
    Create a cleaned foreground mask.

    Returns:

        0   = background
        255 = foreground
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise ValueError(
            "Image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Image cannot be empty."
        )

    gray = _to_grayscale(
        image
    )

    raw_mask = _create_raw_foreground_mask(
        image
    )

    line_removed = _remove_document_lines(
        raw_mask,
        gray
    )

    border_removed = _remove_border_artifacts(
        line_removed
    )

    cleaned = _remove_small_components(
        border_removed
    )

    return cleaned


# =========================================================
# Feature extraction
# =========================================================

def extract_signature_features(
    image: np.ndarray
) -> SignatureFeatures:
    """
    Extract geometric features from the foreground.

    Important:

    The largest individual stroke is NOT assumed to be
    the entire signature.

    Nearby components are grouped first and the resulting
    candidate region is used for the signature bounding box.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise ValueError(
            "Image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Image cannot be empty."
        )

    image_height, image_width = (
        image.shape[:2]
    )

    # -----------------------------------------------------
    # Create cleaned foreground
    # -----------------------------------------------------

    binary = create_foreground_mask(
        image
    )

    # -----------------------------------------------------
    # Foreground density
    # -----------------------------------------------------

    foreground_pixels = (
        cv2.countNonZero(
            binary
        )
    )

    total_pixels = (
        image_width *
        image_height
    )

    foreground_density = (
        foreground_pixels /
        total_pixels
    )

    # -----------------------------------------------------
    # Group nearby components
    # -----------------------------------------------------

    grouped_mask, cleaned_binary = (
        _group_signature_components(
            binary
        )
    )

    # -----------------------------------------------------
    # Find grouped candidate components
    # -----------------------------------------------------

    (
        grouped_labels,
        grouped_stats
    ) = _get_component_stats(
        grouped_mask
    )

    candidate_label = None

    if len(grouped_stats) > 1:

        # Ignore background label 0.
        candidate_areas = (
            grouped_stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )

        candidate_label = (
            int(
                np.argmax(
                    candidate_areas
                )
            ) + 1
        )

    # -----------------------------------------------------
    # Defaults
    # -----------------------------------------------------

    bbox_x = None
    bbox_y = None
    bbox_width = None
    bbox_height = None

    aspect_ratio = None
    occupancy_ratio = None

    # -----------------------------------------------------
    # Candidate bounding box
    # -----------------------------------------------------

    if candidate_label is not None:

        bbox_x = int(
            grouped_stats[
                candidate_label,
                cv2.CC_STAT_LEFT
            ]
        )

        bbox_y = int(
            grouped_stats[
                candidate_label,
                cv2.CC_STAT_TOP
            ]
        )

        bbox_width = int(
            grouped_stats[
                candidate_label,
                cv2.CC_STAT_WIDTH
            ]
        )

        bbox_height = int(
            grouped_stats[
                candidate_label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        if bbox_height > 0:

            aspect_ratio = (
                bbox_width /
                bbox_height
            )

        # -------------------------------------------------
        # Calculate occupancy using ORIGINAL foreground
        # pixels inside the grouped bounding box.
        # -------------------------------------------------

        x1 = bbox_x
        y1 = bbox_y

        x2 = bbox_x + bbox_width
        y2 = bbox_y + bbox_height

        roi = cleaned_binary[
            y1:y2,
            x1:x2
        ]

        roi_foreground_pixels = (
            cv2.countNonZero(
                roi
            )
        )

        bbox_area = (
            bbox_width *
            bbox_height
        )

        if bbox_area > 0:

            occupancy_ratio = (
                roi_foreground_pixels /
                bbox_area
            )

    # -----------------------------------------------------
    # Raw connected components
    # -----------------------------------------------------

    num_labels, _, _, _ = (
        cv2.connectedComponentsWithStats(
            cleaned_binary,
            connectivity=8
        )
    )

    connected_components = max(
        num_labels - 1,
        0
    )

    # -----------------------------------------------------
    # Contours
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        cleaned_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_count = len(
        contours
    )

    largest_contour_area = 0.0
    total_contour_area = 0.0

    if contours:

        contour_areas = [
            float(
                cv2.contourArea(
                    contour
                )
            )
            for contour in contours
        ]

        largest_contour_area = max(
            contour_areas
        )

        total_contour_area = sum(
            contour_areas
        )

    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return SignatureFeatures(
        image_width=image_width,
        image_height=image_height,

        foreground_density=round(
            foreground_density,
            6
        ),

        bbox_x=bbox_x,
        bbox_y=bbox_y,

        bbox_width=bbox_width,
        bbox_height=bbox_height,

        aspect_ratio=(
            round(
                aspect_ratio,
                4
            )
            if aspect_ratio is not None
            else None
        ),

        occupancy_ratio=(
            round(
                occupancy_ratio,
                6
            )
            if occupancy_ratio is not None
            else None
        ),

        connected_components=(
            connected_components
        ),

        contour_count=(
            contour_count
        ),

        largest_contour_area=round(
            largest_contour_area,
            2
        ),

        total_contour_area=round(
            total_contour_area,
            2
        )
    )


# =========================================================
# Connected component helper
# =========================================================

def _get_component_stats(
    binary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return connected-component labels and statistics.
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    return (
        labels,
        stats
    )