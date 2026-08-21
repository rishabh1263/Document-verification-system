"""
Common image preprocessing utilities.

Reusable across:
    PAN
    Voter ID
    Aadhaar
    Passport
    Driving Licence
    Bank Statement
    Salary Slip
    ITR
    CIBIL
    CRIF
    and other document types.

Design principle:
    Do not blindly preprocess every image.

The original image should always remain available because
aggressive preprocessing can destroy OCR information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


ImageInput = Union[str, Path, bytes, np.ndarray]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(source: ImageInput) -> np.ndarray:
    """
    Load an image from:

        - file path
        - bytes
        - numpy.ndarray

    Returns:
        OpenCV BGR image.
    """

    if isinstance(source, np.ndarray):

        if source.size == 0:
            raise ValueError("Input image is empty.")

        return source.copy()

    if isinstance(source, (str, Path)):

        image = cv2.imread(str(source))

        if image is None:
            raise ValueError(
                f"Unable to read image: {source}"
            )

        return image

    if isinstance(source, bytes):

        buffer = np.frombuffer(
            source,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Unable to decode image bytes."
            )

        return image

    raise TypeError(
        "source must be a file path, bytes, "
        "or numpy.ndarray."
    )


# ============================================================
# GRAYSCALE
# ============================================================

def to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert BGR image to grayscale.

    If the image is already grayscale,
    return a copy unchanged.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if len(image.shape) == 2:
        return image.copy()

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


# ============================================================
# RESIZE
# ============================================================

def resize_image(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """
    Resize an image.

    You can provide:

        width
        height
        width + height
        scale

    Aspect ratio is preserved when only width,
    height, or scale is provided.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if scale is not None:

        if scale <= 0:
            raise ValueError(
                "scale must be greater than 0."
            )

        new_width = max(
            1,
            int(image.shape[1] * scale),
        )

        new_height = max(
            1,
            int(image.shape[0] * scale),
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )

    if width is None and height is None:
        return image.copy()

    if width is not None and width <= 0:
        raise ValueError(
            "width must be greater than 0."
        )

    if height is not None and height <= 0:
        raise ValueError(
            "height must be greater than 0."
        )

    original_height, original_width = image.shape[:2]

    # Both dimensions explicitly supplied.
    if width is not None and height is not None:

        return cv2.resize(
            image,
            (width, height),
            interpolation=interpolation,
        )

    # Width only.
    if width is not None:

        scale = width / original_width

        new_height = max(
            1,
            int(original_height * scale),
        )

        return cv2.resize(
            image,
            (width, new_height),
            interpolation=interpolation,
        )

    # Height only.
    scale = height / original_height

    new_width = max(
        1,
        int(original_width * scale),
    )

    return cv2.resize(
        image,
        (new_width, height),
        interpolation=interpolation,
    )


# ============================================================
# BRIGHTNESS
# ============================================================

def adjust_brightness(
    image: np.ndarray,
    value: int = 0,
) -> np.ndarray:
    """
    Adjust brightness.

    Positive value:
        brighter

    Negative value:
        darker
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    return cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=int(value),
    )


# ============================================================
# CONTRAST
# ============================================================

def adjust_contrast(
    image: np.ndarray,
    factor: float = 1.0,
) -> np.ndarray:
    """
    Adjust contrast.

    1.0:
        unchanged

    > 1.0:
        increase contrast

    < 1.0:
        decrease contrast
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if factor < 0:
        raise ValueError(
            "Contrast factor cannot be negative."
        )

    return cv2.convertScaleAbs(
        image,
        alpha=float(factor),
        beta=0,
    )


# ============================================================
# DENOISING
# ============================================================

def denoise(
    image: np.ndarray,
    strength: int = 10,
) -> np.ndarray:
    """
    Reduce image noise.

    For color images:
        fastNlMeansDenoisingColored

    For grayscale images:
        fastNlMeansDenoising
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    strength = max(
        1,
        int(strength),
    )

    if len(image.shape) == 2:

        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=strength,
        hColor=strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )


# ============================================================
# SHARPEN
# ============================================================

def sharpen(
    image: np.ndarray,
    amount: float = 1.0,
) -> np.ndarray:
    """
    Sharpen an image.

    Important:
        Sharpening cannot recover information that has
        already been destroyed by severe blur.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if amount < 0:
        raise ValueError(
            "amount cannot be negative."
        )

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0,
    )

    return cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )


# ============================================================
# UNSHARP MASK
# ============================================================

def unsharp_mask(
    image: np.ndarray,
    amount: float = 1.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Unsharp masking.

    Useful for improving text edges when an image
    is slightly soft.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if amount < 0:
        raise ValueError(
            "amount cannot be negative."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than 0."
        )

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
    )

    return cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )


# ============================================================
# STANDARD THRESHOLD
# ============================================================

def threshold(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255,
) -> np.ndarray:
    """
    Apply standard binary thresholding.

    Input is automatically converted to grayscale.
    """

    gray = to_grayscale(image)

    _, result = cv2.threshold(
        gray,
        threshold_value,
        max_value,
        cv2.THRESH_BINARY,
    )

    return result


# ============================================================
# ADAPTIVE THRESHOLD
# ============================================================

def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 31,
    constant: int = 10,
) -> np.ndarray:
    """
    Adaptive thresholding.

    Useful when lighting is uneven across a document.
    """

    gray = to_grayscale(image)

    block_size = max(
        3,
        int(block_size),
    )

    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )


# ============================================================
# AUTO BRIGHTNESS CORRECTION
# ============================================================

def auto_brightness(
    image: np.ndarray,
    dark_threshold: float = 70.0,
    bright_threshold: float = 220.0,
) -> np.ndarray:
    """
    Automatically correct extreme brightness.

    IMPORTANT:
        This function deliberately does nothing when
        brightness is already acceptable.

    This prevents the Voter sample from becoming
    unnecessarily brighter.
    """

    gray = to_grayscale(image)

    brightness = float(
        np.mean(gray)
    )

    if brightness < dark_threshold:

        # Move dark images toward a safer OCR range.
        value = int(
            min(
                50,
                dark_threshold - brightness,
            )
        )

        return adjust_brightness(
            image,
            value,
        )

    if brightness > bright_threshold:

        # Darken excessively bright images.
        value = -int(
            min(
                40,
                brightness - bright_threshold,
            )
        )

        return adjust_brightness(
            image,
            value,
        )

    return image.copy()


# ============================================================
# AUTO CONTRAST
# ============================================================

def auto_contrast(
    image: np.ndarray,
    low_threshold: float = 30.0,
    target_factor: float = 1.15,
) -> np.ndarray:
    """
    Increase contrast only when the image has
    genuinely low contrast.

    Good-contrast images are left untouched.
    """

    gray = to_grayscale(image)

    contrast = float(
        np.std(gray)
    )

    if contrast >= low_threshold:
        return image.copy()

    return adjust_contrast(
        image,
        target_factor,
    )


# ============================================================
# AUTO SHARPEN
# ============================================================

def auto_sharpen(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    amount: float = 0.5,
) -> np.ndarray:
    """
    Sharpen only when the image appears soft.

    Sharp images are left untouched.
    """

    gray = to_grayscale(image)

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    if blur_score >= blur_threshold:
        return image.copy()

    return sharpen(
        image,
        amount=amount,
    )


# ============================================================
# CONSERVATIVE OCR PREPARATION
# ============================================================

def prepare_for_ocr(
    image: ImageInput,
    target_width: int | None = 1200,
    grayscale: bool = True,
    auto_correct_brightness: bool = True,
    auto_correct_contrast: bool = True,
    auto_correct_sharpness: bool = True,
    denoise_image: bool = False,
) -> np.ndarray:
    """
    Conservative common OCR preprocessing pipeline.

    Order:

        1. Load image
        2. Resize if necessary
        3. Correct extreme brightness
        4. Correct low contrast
        5. Denoise if explicitly requested
        6. Sharpen only if image appears soft
        7. Convert to grayscale

    IMPORTANT:

    This function does NOT blindly apply every operation.

    The original image should always remain available to
    the document-specific OCR pipeline.
    """

    result = load_image(image)

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    if target_width is not None:

        current_width = result.shape[1]

        # Only enlarge small images.
        #
        # Do not unnecessarily resize already-large images.
        if current_width < target_width:

            result = resize_image(
                result,
                width=target_width,
            )

    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    if auto_correct_brightness:

        result = auto_brightness(
            result,
            dark_threshold=70.0,
            bright_threshold=220.0,
        )

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    if auto_correct_contrast:

        result = auto_contrast(
            result,
            low_threshold=30.0,
            target_factor=1.15,
        )

    # --------------------------------------------------------
    # DENOISING
    # --------------------------------------------------------

    if denoise_image:

        result = denoise(
            result,
            strength=8,
        )

    # --------------------------------------------------------
    # SHARPENING
    # --------------------------------------------------------

    if auto_correct_sharpness:

        result = auto_sharpen(
            result,
            blur_threshold=100.0,
            amount=0.5,
        )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if grayscale:

        result = to_grayscale(
            result
        )

    return result


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

def normalize_for_ocr(
    image: ImageInput,
    target_width: int | None = 1200,
    grayscale: bool = True,
    contrast: float | None = None,
    brightness: int | None = None,
    denoise_image: bool = False,
    sharpen_image: bool = False,
) -> np.ndarray:
    """
    Backward-compatible OCR preprocessing function.

    New code should prefer:

        prepare_for_ocr()

    This function remains available so existing code doesn't
    immediately break.
    """

    result = load_image(image)

    # Resize only when the image is smaller.
    if target_width is not None:

        if result.shape[1] < target_width:

            result = resize_image(
                result,
                width=target_width,
            )

    # Explicit brightness takes priority.
    if brightness is not None:

        result = adjust_brightness(
            result,
            brightness,
        )

    # Explicit contrast takes priority.
    if contrast is not None:

        result = adjust_contrast(
            result,
            contrast,
        )

    if denoise_image:

        result = denoise(
            result,
            strength=8,
        )

    if sharpen_image:

        result = sharpen(
            result,
            amount=0.5,
        )

    if grayscale:

        result = to_grayscale(
            result
        )

    return result