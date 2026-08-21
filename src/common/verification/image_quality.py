"""
Common document image-quality analysis.

Document-independent quality evidence shared by PAN, Aadhaar, Voter ID,
Passport, Driving Licence, passbook, ITR image pages and other documents.

IMPORTANT:
    Quality is evidence, not proof of authenticity.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


MIN_WIDTH = 300
MIN_HEIGHT = 180


def analyze_image_quality(
    image: np.ndarray,
) -> dict[str, Any]:

    if image is None or image.size == 0:

        return {
            "available": False,
            "resolution": "0x0",
            "width": 0,
            "height": 0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "aspect_ratio": 0.0,
            "resolution_ok": False,
            "aspect_ratio_ok": False,
        }

    height, width = image.shape[:2]

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = image

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    aspect_ratio = (
        float(width / height)
        if height
        else 0.0
    )

    return {
        "available": True,
        "resolution": f"{width}x{height}",
        "width": int(width),
        "height": int(height),
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "resolution_ok": bool(
            width >= MIN_WIDTH
            and height >= MIN_HEIGHT
        ),
        "aspect_ratio_ok": bool(
            aspect_ratio > 0
        ),
    }


def calculate_quality_score(
    metrics: dict[str, Any] | None,
) -> tuple[float, dict[str, Any]]:

    if not metrics:

        return 100.0, {}

    try:

        blur = float(
            metrics.get(
                "blur_score",
                0.0,
            )
        )

        brightness = float(
            metrics.get(
                "brightness",
                0.0,
            )
        )

        contrast = float(
            metrics.get(
                "contrast",
                0.0,
            )
        )

        width = int(
            metrics.get(
                "width",
                0,
            )
        )

        height = int(
            metrics.get(
                "height",
                0,
            )
        )

        aspect_ratio = float(
            metrics.get(
                "aspect_ratio",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.0, {
            "available": False,
            "error":
                "Invalid image-quality metrics.",
        }

    blur_component = (
        100.0
        if blur >= 300
        else 80.0
        if blur >= 100
        else 40.0
        if blur > 0
        else 0.0
    )

    brightness_component = (
        100.0
        if 30 <= brightness <= 245
        else 50.0
    )

    contrast_component = (
        100.0
        if contrast >= 20
        else 40.0
    )

    resolution_ok = (
        width >= MIN_WIDTH
        and height >= MIN_HEIGHT
    )

    aspect_ratio_ok = (
        aspect_ratio > 0
    )

    score = round(
        (
            blur_component
            + brightness_component
            + contrast_component
            + (
                100.0
                if resolution_ok
                else 50.0
            )
            + (
                100.0
                if aspect_ratio_ok
                else 50.0
            )
        )
        / 5.0,
        2,
    )

    details = {
        "available": True,
        "blur_score": round(
            blur,
            2,
        ),
        "brightness": round(
            brightness,
            2,
        ),
        "contrast": round(
            contrast,
            2,
        ),
        "width": width,
        "height": height,
        "aspect_ratio": round(
            aspect_ratio,
            3,
        ),
        "blur_score_component":
            blur_component,
        "brightness_component":
            brightness_component,
        "contrast_component":
            contrast_component,
        "resolution_ok":
            resolution_ok,
        "aspect_ratio_ok":
            aspect_ratio_ok,
    }

    return score, details