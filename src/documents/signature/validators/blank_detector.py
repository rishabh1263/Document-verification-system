"""
Blank image detection for signature verification.

This module determines whether an image contains meaningful
visual content.

It does NOT determine whether the foreground is a signature.
"""

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Adaptive threshold configuration
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# Very small foreground percentage = likely blank.
MIN_FOREGROUND_DENSITY = 0.001

# Very high foreground percentage = suspicious.
MAX_FOREGROUND_DENSITY = 0.85

# Nearly uniform images have very little useful information.
MIN_IMAGE_STD = 3.0


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

@dataclass
class BlankDetectionResult:
    is_blank: bool
    reason_code: str
    message: str
    foreground_density: float


# ---------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------

def detect_blank(
    image: np.ndarray
) -> BlankDetectionResult:
    """
    Detect whether an image is blank.

    Pipeline:

        Image
          ↓
        Grayscale
          ↓
        Uniform-image check
          ↓
        Gaussian blur
          ↓
        Adaptive threshold
          ↓
        Morphological cleanup
          ↓
        Foreground density
          ↓
        Blank / Non-blank
    """

    # -----------------------------------------------------
    # 1. Validate input
    # -----------------------------------------------------

    if image is None:
        return BlankDetectionResult(
            is_blank=True,
            reason_code="NO_IMAGE",
            message="No image was provided.",
            foreground_density=0.0
        )

    if not isinstance(image, np.ndarray):
        return BlankDetectionResult(
            is_blank=True,
            reason_code="INVALID_IMAGE",
            message="Invalid image data.",
            foreground_density=0.0
        )

    if image.size == 0:
        return BlankDetectionResult(
            is_blank=True,
            reason_code="EMPTY_IMAGE",
            message="Image contains no pixels.",
            foreground_density=0.0
        )

    # -----------------------------------------------------
    # 2. Convert to grayscale
    # -----------------------------------------------------

    try:
        gray = _to_grayscale(image)

    except ValueError:
        return BlankDetectionResult(
            is_blank=True,
            reason_code="UNSUPPORTED_CHANNELS",
            message="Unsupported image channel format.",
            foreground_density=0.0
        )

    # -----------------------------------------------------
    # 3. Detect nearly uniform images
    # -----------------------------------------------------

    image_std = float(np.std(gray))
    mean_intensity = float(np.mean(gray))

    if image_std < MIN_IMAGE_STD:

        # Almost completely black image
        if mean_intensity < 20:
            return BlankDetectionResult(
                is_blank=False,
                reason_code="SOLID_DARK_IMAGE",
                message="Image is almost completely dark.",
                foreground_density=1.0
            )

        # Almost completely white or uniform image
        return BlankDetectionResult(
            is_blank=True,
            reason_code="BLANK_IMAGE",
            message="Image contains almost no visual variation.",
            foreground_density=0.0
        )

    # -----------------------------------------------------
    # 4. Light blur
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # 5. Adaptive threshold
    # -----------------------------------------------------

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    # -----------------------------------------------------
    # 6. Morphological cleanup
    # -----------------------------------------------------

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # 7. Foreground density
    # -----------------------------------------------------

    foreground_pixels = cv2.countNonZero(binary)

    total_pixels = (
        binary.shape[0] *
        binary.shape[1]
    )

    if total_pixels == 0:
        return BlankDetectionResult(
            is_blank=True,
            reason_code="EMPTY_IMAGE",
            message="Image contains no usable pixels.",
            foreground_density=0.0
        )

    foreground_density = (
        foreground_pixels /
        total_pixels
    )

    # -----------------------------------------------------
    # 8. Blank detection
    # -----------------------------------------------------

    if foreground_density < MIN_FOREGROUND_DENSITY:
        return BlankDetectionResult(
            is_blank=True,
            reason_code="BLANK_IMAGE",
            message="Image contains almost no foreground content.",
            foreground_density=round(
                foreground_density,
                6
            )
        )

    # -----------------------------------------------------
    # 9. Excessive foreground
    # -----------------------------------------------------

    if foreground_density > MAX_FOREGROUND_DENSITY:
        return BlankDetectionResult(
            is_blank=False,
            reason_code="EXCESSIVE_FOREGROUND",
            message=(
                "Image contains an unusually large amount "
                "of foreground content."
            ),
            foreground_density=round(
                foreground_density,
                6
            )
        )

    # -----------------------------------------------------
    # 10. Non-blank image
    # -----------------------------------------------------

    return BlankDetectionResult(
        is_blank=False,
        reason_code="NON_BLANK_IMAGE",
        message="Image contains foreground content.",
        foreground_density=round(
            foreground_density,
            6
        )
    )