"""
Pytest tests for the ITR Content Validator.

The content layer verifies that the document contains the required
ITR identity and sections.

Tested:
    - Assessment year
    - PAN
    - Taxpayer information
    - Income information
    - Tax computation
    - Verification
    - Acknowledgement
    - Required-content flag
    - Overall content score
    - Missing-items handling
"""

from __future__ import annotations

from pathlib import Path

import fitz

from src.documents.itr.validation.content_validator import (
    ContentValidator,
)


PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "Vedant ITR.pdf"
)


def _extract_text(pdf_path: Path) -> str:
    """Extract all native PDF text for the content test."""
    document = fitz.open(str(pdf_path))

    try:
        return "\n".join(
            page.get_text("text")
            for page in document
        )
    finally:
        document.close()


def test_valid_itr_content() -> None:
    """A valid ITR must contain all required content indicators."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    text = _extract_text(PDF_PATH)

    assert text.strip(), (
        "PDF text extraction returned empty text."
    )

    validator = ContentValidator()

    result = validator.validate(text)

    # ------------------------------------------------------
    # Required content
    # ------------------------------------------------------

    assert result.required_content_present is True

    # ------------------------------------------------------
    # Individual required sections
    # ------------------------------------------------------

    assert result.assessment_year_present is True
    assert result.pan_present is True
    assert result.taxpayer_information_present is True
    assert result.income_information_present is True
    assert result.tax_computation_present is True
    assert result.verification_present is True
    assert result.acknowledgement_present is True

    # ------------------------------------------------------
    # Score and missing items
    # ------------------------------------------------------

    assert result.score == 1.0
    assert result.missing_items == []

    # ------------------------------------------------------
    # Reasons
    # ------------------------------------------------------

    assert result.reasons

    expected_reason_fragments = (
        "Assessment year",
        "PAN",
        "Taxpayer",
        "Income",
        "Tax computation",
        "Verification",
        "Acknowledgement",
    )

    reasons_text = " | ".join(result.reasons)

    for fragment in expected_reason_fragments:
        assert fragment.lower() in reasons_text.lower(), (
            f"Expected content reason containing "
            f"'{fragment}', got: {result.reasons}"
        )


def test_content_processing_time_is_recorded() -> None:
    """Content validation must record processing time."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    text = _extract_text(PDF_PATH)

    validator = ContentValidator()

    result = validator.validate(text)

    assert result.processing_time_ms >= 0.0