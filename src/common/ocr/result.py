"""
Common OCR result utilities.

Provides a standard OCR result format that can be reused
by every document verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OCRResult:
    """
    Standard representation of one OCR detection.
    """

    text: str
    confidence: float
    bbox: list[list[float]] | list[Any]

    @property
    def normalized_text(self) -> str:
        """
        Return normalized text for matching/parsing.
        """

        return " ".join(
            self.text.strip().split()
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert OCR result to a dictionary.
        """

        return {
            "text": self.text,
            "confidence": round(
                float(self.confidence),
                4,
            ),
            "bbox": self.bbox,
        }


def normalize_text(text: str) -> str:
    """
    Basic OCR text normalization.

    Does not aggressively modify characters because
    document-specific extraction may depend on the
    original OCR text.
    """

    if not text:
        return ""

    return " ".join(
        str(text).strip().split()
    )


def normalize_ocr_results(
    results: list[OCRResult],
) -> list[dict[str, Any]]:
    """
    Convert OCRResult objects to dictionaries.
    """

    return [
        result.to_dict()
        for result in results
        if result.text.strip()
    ]


def filter_by_confidence(
    results: list[OCRResult],
    minimum_confidence: float = 0.0,
) -> list[OCRResult]:
    """
    Keep OCR results above the requested confidence.
    """

    minimum_confidence = max(
        0.0,
        min(1.0, float(minimum_confidence)),
    )

    return [
        result
        for result in results
        if result.confidence >= minimum_confidence
    ]