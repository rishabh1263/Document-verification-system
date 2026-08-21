"""
Pytest test for the ITR Keyword Detector.

This file is intentionally pytest-compatible.

The previous version used:

    def test_pdf(file_path: str):

Pytest interprets every function argument as a fixture name.
There is no ``file_path`` fixture, so collection failed.

This version resolves the sample path inside the test.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from src.documents.itr.detection.keyword_detector import KeywordDetector


PDF_PATH = Path(__file__).resolve().parent / "Vedant ITR.pdf"


def _extract_text(pdf_path: Path) -> str:
    """Extract native PDF text for the keyword detector test."""
    document = fitz.open(str(pdf_path))

    try:
        return "\n".join(
            page.get_text("text")
            for page in document
        )
    finally:
        document.close()


def test_pdf() -> None:
    """Verify that the Vedant ITR produces meaningful ITR keywords."""
    assert PDF_PATH.exists(), f"Test PDF not found: {PDF_PATH}"

    text = _extract_text(PDF_PATH)

    assert text.strip(), "PDF text extraction returned empty text."

    result = KeywordDetector().analyze(text)

    assert result.total_keywords_found > 0, (
        "Expected at least one ITR keyword."
    )

    assert result.total_positive_score > 0, (
        "Expected positive ITR keyword evidence."
    )

    assert result.score > 0.0, (
        "Expected a positive keyword identity score."
    )

    matched = {
        item.keyword.lower()
        for item in result.matched_keywords
    }

    expected_keywords = {
        "income tax return",
        "assessment year",
        "pan",
        "tax paid",
    }

    assert matched.intersection(expected_keywords), (
        "Expected at least one strong ITR keyword, "
        f"but found: {sorted(matched)}"
    )