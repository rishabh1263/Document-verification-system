"""
Basic image validation for signature verification.

This module only checks whether the uploaded input is a
usable image. It does NOT determine whether the image
contains a signature.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MIN_WIDTH = 50
MIN_HEIGHT = 30

MAX_WIDTH = 10000
MAX_HEIGHT = 10000

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

@dataclass
class ImageValidationResult:
    valid: bool
    reason_code: str
    message: str
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None


# ---------------------------------------------------------
# Main validation function
# ---------------------------------------------------------

def validate_image(image_bytes: bytes) -> ImageValidationResult:
    """
    Validate whether raw bytes represent a usable image.

    This function performs only technical validation:
    - Empty input
    - File size
    - Image decoding
    - Image dimensions
    - Image channels

    It does NOT check whether the image is a signature.
    """

    # -----------------------------------------------------
    # 1. Empty input
    # -----------------------------------------------------

    if not image_bytes:
        return ImageValidationResult(
            valid=False,
            reason_code="EMPTY_FILE",
            message="No image data was provided."
        )

    # -----------------------------------------------------
    # 2. File size
    # -----------------------------------------------------

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        return ImageValidationResult(
            valid=False,
            reason_code="FILE_TOO_LARGE",
            message=f"Image size exceeds {MAX_FILE_SIZE_MB} MB."
        )

    # -----------------------------------------------------
    # 3. Convert bytes -> NumPy array
    # -----------------------------------------------------

    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    except Exception:
        return ImageValidationResult(
            valid=False,
            reason_code="INVALID_IMAGE_BYTES",
            message="Unable to process image data."
        )

    # -----------------------------------------------------
    # 4. Decode image
    # -----------------------------------------------------

    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    if image is None:
        return ImageValidationResult(
            valid=False,
            reason_code="INVALID_IMAGE",
            message="The uploaded file is not a valid image."
        )

    # -----------------------------------------------------
    # 5. Image dimensions
    # -----------------------------------------------------

    if len(image.shape) < 2:
        return ImageValidationResult(
            valid=False,
            reason_code="INVALID_IMAGE_SHAPE",
            message="Invalid image dimensions."
        )

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # 6. Minimum dimensions
    # -----------------------------------------------------

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return ImageValidationResult(
            valid=False,
            reason_code="IMAGE_TOO_SMALL",
            message=(
                f"Image resolution is too small. "
                f"Minimum size is {MIN_WIDTH}x{MIN_HEIGHT} pixels."
            ),
            width=width,
            height=height
        )

    # -----------------------------------------------------
    # 7. Maximum dimensions
    # -----------------------------------------------------

    if width > MAX_WIDTH or height > MAX_HEIGHT:
        return ImageValidationResult(
            valid=False,
            reason_code="IMAGE_TOO_LARGE",
            message=(
                f"Image resolution is too large. "
                f"Maximum size is {MAX_WIDTH}x{MAX_HEIGHT} pixels."
            ),
            width=width,
            height=height
        )

    # -----------------------------------------------------
    # 8. Number of channels
    # -----------------------------------------------------

    if len(image.shape) == 2:
        channels = 1

    elif len(image.shape) == 3:
        channels = image.shape[2]

        if channels not in (3, 4):
            return ImageValidationResult(
                valid=False,
                reason_code="UNSUPPORTED_CHANNELS",
                message="Image has an unsupported number of channels.",
                width=width,
                height=height,
                channels=channels
            )

    else:
        return ImageValidationResult(
            valid=False,
            reason_code="INVALID_IMAGE_SHAPE",
            message="Unsupported image structure.",
            width=width,
            height=height
        )

    # -----------------------------------------------------
    # 9. Everything passed
    # -----------------------------------------------------

    return ImageValidationResult(
        valid=True,
        reason_code="VALID_IMAGE",
        message="Image is valid and can be processed.",
        width=width,
        height=height,
        channels=channels
    )