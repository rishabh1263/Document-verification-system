"""
Pytest tests for the ITR Consistency Validator.

The consistency layer verifies that repeated occurrences of the same
logical ITR information agree with each other.

Tested:
    - Assessment year consistency
    - PAN consistency
    - Income consistency
    - Tax consistency
    - Acknowledgement consistency
    - Overall consistency decision
    - Overall consistency score
    - Processing time

The Vedant ITR is an updated-return ITR, so different tax stages are
not compared against each other. The validator checks repeated
occurrences of the same logical field.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from src.documents.itr.validation.consistency_validator import (
    ConsistencyValidator,
)


PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "Vedant ITR.pdf"
)


def _extract_text(pdf_path: Path) -> str:
    """Extract all native PDF text."""
    document = fitz.open(str(pdf_path))

    try:
        return "\n".join(
            page.get_text("text")
            for page in document
        )
    finally:
        document.close()


def test_valid_itr_consistency() -> None:
    """A valid ITR must pass all consistency checks."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    text = _extract_text(PDF_PATH)

    assert text.strip(), (
        "PDF text extraction returned empty text."
    )

    validator = ConsistencyValidator()

    result = validator.validate(text)

    # ------------------------------------------------------
    # Overall result
    # ------------------------------------------------------

    assert result.consistent is True
    assert result.score == 1.0
    assert result.inconsistencies == []

    # ------------------------------------------------------
    # Individual consistency checks
    # ------------------------------------------------------

    assert result.assessment_year_consistent is True
    assert result.pan_consistent is True
    assert result.income_consistent is True
    assert result.tax_consistent is True
    assert result.acknowledgement_consistent is True

    # ------------------------------------------------------
    # Reasons
    # ------------------------------------------------------

    assert result.reasons

    reasons_text = " | ".join(result.reasons).lower()

    expected_reason_fragments = (
        "assessment year",
        "pan",
        "income",
        "tax",
        "acknowledgement",
    )

    for fragment in expected_reason_fragments:
        assert fragment in reasons_text, (
            f"Expected consistency reason containing "
            f"'{fragment}', got: {result.reasons}"
        )


def test_consistency_processing_time_is_recorded() -> None:
    """Consistency validation must record processing time."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    text = _extract_text(PDF_PATH)

    validator = ConsistencyValidator()

    result = validator.validate(text)

    assert result.processing_time_ms >= 0.0