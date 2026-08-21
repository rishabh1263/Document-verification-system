"""
src/preprocessor.py

Lightweight image preprocessing for PaddleOCR.

Goals:
    1. Preserve small characters and numbers.
    2. Improve contrast when necessary.
    3. Avoid expensive preprocessing.
    4. Keep image in BGR format for PaddleOCR.
    5. Avoid unnecessary enlargement of already high-resolution PDFs.

Pipeline:

    OpenCV BGR image
        â†“
    Validate / normalize channels
        â†“
    Resize only if necessary
        â†“
    CLAHE on luminance channel
        â†“
    Mild sharpening
        â†“
    BGR image
        â†“
    PaddleOCR
"""

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Very small images can hurt OCR.
MIN_LONG_SIDE = 1200

# Huge PDF-rendered images dramatically increase OCR time.
MAX_LONG_SIDE = 2200

# CLAHE settings.
CLAHE_CLIP_LIMIT = 1.5
CLAHE_GRID_SIZE = (8, 8)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def _ensure_bgr(image):
    """
    Ensure that the input image is a normal uint8 BGR image.

    Supported:
        (H, W)       grayscale
        (H, W, 1)    grayscale
        (H, W, 3)    BGR
        (H, W, 4)    BGRA
    """

    if image is None:
        raise ValueError(
            "Cannot preprocess image: image is None."
        )

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image must be a numpy array."
        )

    if image.size == 0:
        raise ValueError(
            "Cannot preprocess an empty image."
        )

    # --------------------------------------------------------
    # UINT8
    # --------------------------------------------------------

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8
        )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    # --------------------------------------------------------
    # SINGLE CHANNEL
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[2] == 1
    ):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    # --------------------------------------------------------
    # BGRA
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )

    # --------------------------------------------------------
    # NORMAL BGR
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):

        pass

    else:

        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    return np.ascontiguousarray(
        image
    )


# ============================================================
# RESIZE
# ============================================================

def _resize_for_ocr(
    image,
    min_long_side=MIN_LONG_SIDE,
    max_long_side=MAX_LONG_SIDE,
):
    """
    Keep OCR image within a sensible resolution range.

    Small image:
        upscale

    Huge image:
        downscale

    Normal image:
        leave untouched

    This is important because OCR time increases substantially
    with image resolution.
    """

    height, width = image.shape[:2]

    long_side = max(
        height,
        width,
    )

    # --------------------------------------------------------
    # SMALL IMAGE
    # --------------------------------------------------------

    if long_side < min_long_side:

        scale = (
            min_long_side
            / long_side
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    # --------------------------------------------------------
    # HUGE IMAGE
    # --------------------------------------------------------

    elif long_side > max_long_side:

        scale = (
            max_long_side
            / long_side
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


# ============================================================
# CONTRAST ENHANCEMENT
# ============================================================

def _enhance_contrast(image):
    """
    Apply CLAHE only to the luminance channel.

    This improves local contrast while preserving the original
    colour information.

    Better suited to PaddleOCR than converting the entire image
    permanently to grayscale.
    """

    # BGR -> LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = (
        cv2.split(
            lab
        )
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE,
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    return enhanced


# ============================================================
# MILD SHARPENING
# ============================================================

def _sharpen(image):
    """
    Apply very mild sharpening.

    We intentionally avoid aggressive sharpening because it can
    turn characters such as:

        B -> 8
        O -> 0
        I -> 1

    into OCR errors.
    """

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0,
    )

    sharpened = cv2.addWeighted(
        image,
        1.15,
        blurred,
        -0.15,
        0,
    )

    return sharpened


# ============================================================
# OPTIONAL DESKEW
# ============================================================

def _estimate_skew_angle(image):
    """
    Estimate document skew.

    Returns:
        angle in degrees

    We only use this when rotation is meaningful.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Binary inverse image.
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        | cv2.THRESH_OTSU,
    )

    coordinates = np.column_stack(
        np.where(
            threshold > 0
        )
    )

    if len(coordinates) < 100:
        return 0.0

    angle = cv2.minAreaRect(
        coordinates
    )[-1]

    if angle < -45:

        angle = -(
            90 + angle
        )

    else:

        angle = -angle

    # Tiny rotation isn't worth resampling the image.
    if abs(angle) < 0.5:
        return 0.0

    # Large estimate is probably document graphics/background
    # confusing the estimator.
    if abs(angle) > 10:
        return 0.0

    return float(
        angle
    )


def _deskew_if_needed(image):
    """
    Deskew only when a meaningful small rotation is detected.
    """

    angle = _estimate_skew_angle(
        image
    )

    if angle == 0.0:
        return image

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


# ============================================================
# MAIN PREPROCESSOR
# ============================================================

def preprocess_image(
    image,
    enhance_contrast=True,
    sharpen=True,
    deskew=False,
):
    """
    Prepare a Driving Licence image for PaddleOCR.

    Default pipeline:

        BGR
         â†“
        resolution normalization
         â†“
        CLAHE
         â†“
        mild sharpening
         â†“
        PaddleOCR-ready BGR

    Deskew is disabled by default because scanned PDFs are normally
    already aligned and unnecessary rotation/resampling can hurt OCR.
    """

    # --------------------------------------------------------
    # NORMALIZE IMAGE
    # --------------------------------------------------------

    processed = _ensure_bgr(
        image
    )

    original_height, original_width = (
        processed.shape[:2]
    )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    processed = _resize_for_ocr(
        processed
    )

    # --------------------------------------------------------
    # OPTIONAL DESKEW
    # --------------------------------------------------------

    if deskew:

        processed = _deskew_if_needed(
            processed
        )

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    if enhance_contrast:

        processed = _enhance_contrast(
            processed
        )

    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    if sharpen:

        processed = _sharpen(
            processed
        )

    processed = np.ascontiguousarray(
        processed,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    final_height, final_width = (
        processed.shape[:2]
    )

    print(
        "[Preprocessor] "
        f"Original: {original_width}x{original_height} "
        f"-> OCR: {final_width}x{final_height}"
    )

    return processed


# ============================================================
# PADDLE OCR ENTRY POINT
# ============================================================

def to_ocr_ready(image):
    """
    Main preprocessing function used by dl_api.py.

    Returns:
        PaddleOCR-ready BGR uint8 image.
    """

    return preprocess_image(
        image,
        enhance_contrast=True,
        sharpen=True,
        deskew=False,
    )
