"""
Common OCR engine.

Currently uses EasyOCR because it is already part of the
document-verification system.

Document-specific modules should use this common layer
instead of creating their own EasyOCR reader repeatedly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .result import (
    OCRResult,
    filter_by_confidence,
    normalize_text,
)


# ============================================================
# OCR READER
# ============================================================

_reader = None


def get_ocr_reader(
    languages: Sequence[str] | None = None,
    gpu: bool = False,
):
    """
    Load and cache the EasyOCR reader.

    The reader is created only once per Python process.

    Args:
        languages:
            EasyOCR language list.

            Default:
                ["en", "hi", "mr"]

        gpu:
            Whether EasyOCR should use GPU.

            Default:
                False

    Returns:
        Cached EasyOCR Reader.
    """

    global _reader

    if _reader is not None:
        return _reader

    import easyocr

    if languages is None:
        languages = [
            "en",
            "hi",
            "mr",
        ]

    _reader = easyocr.Reader(
        list(languages),
        gpu=gpu,
    )

    return _reader


# ============================================================
# RAW OCR
# ============================================================

def run_ocr(
    image: np.ndarray,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    detail: int = 1,
) -> list[Any]:
    """
    Run EasyOCR directly.

    This function returns the raw EasyOCR output.

    Use extract_text() when you want standardized results.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    reader = get_ocr_reader(
        languages=languages,
        gpu=gpu,
    )

    return reader.readtext(
        image,
        detail=detail,
    )


# ============================================================
# STANDARDIZED OCR
# ============================================================

def extract_text(
    image: np.ndarray,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    minimum_confidence: float = 0.0,
) -> list[OCRResult]:
    """
    Run OCR and return standardized OCRResult objects.

    Each result contains:

        text
        confidence
        bounding box
    """

    raw_results = run_ocr(
        image=image,
        languages=languages,
        gpu=gpu,
        detail=1,
    )

    results: list[OCRResult] = []

    for item in raw_results:

        if not item:
            continue

        if len(item) < 3:
            continue

        bbox = item[0]
        text = normalize_text(
            str(item[1])
        )

        try:
            confidence = float(
                item[2]
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        if not text:
            continue

        results.append(
            OCRResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
            )
        )

    return filter_by_confidence(
        results,
        minimum_confidence,
    )


# ============================================================
# DICTIONARY OUTPUT
# ============================================================

def extract_text_dicts(
    image: np.ndarray,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    minimum_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """
    OCR output in the dictionary format currently used
    throughout the document project.

    Example:

        [
            {
                "text": "Name: Amit Pandey",
                "confidence": 0.92,
                "bbox": [...]
            }
        ]
    """

    results = extract_text(
        image=image,
        languages=languages,
        gpu=gpu,
        minimum_confidence=minimum_confidence,
    )

    return [
        result.to_dict()
        for result in results
    ]


# ============================================================
# TEXT ONLY
# ============================================================

def extract_text_lines(
    image: np.ndarray,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    minimum_confidence: float = 0.0,
) -> list[str]:
    """
    Return only OCR text lines.
    """

    results = extract_text(
        image=image,
        languages=languages,
        gpu=gpu,
        minimum_confidence=minimum_confidence,
    )

    return [
        result.normalized_text
        for result in results
    ]


# ============================================================
# FULL OCR TEXT
# ============================================================

def extract_full_text(
    image: np.ndarray,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    minimum_confidence: float = 0.0,
    separator: str = "\n",
) -> str:
    """
    Combine OCR results into one text string.
    """

    lines = extract_text_lines(
        image=image,
        languages=languages,
        gpu=gpu,
        minimum_confidence=minimum_confidence,
    )

    return separator.join(
        lines
    )