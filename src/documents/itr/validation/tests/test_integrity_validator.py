"""
Pytest tests for the ITR Integrity Validator.

The integrity layer verifies technical document health only.

Tested:
    - File existence
    - File size
    - PDF validity
    - Encryption state
    - Page count
    - Readability
    - Corruption state
    - SHA-256 generation
    - Overall integrity score
"""

from __future__ import annotations

from pathlib import Path

from src.documents.itr.validation.integrity_validator import (
    IntegrityValidator,
)


PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "Vedant ITR.pdf"
)


def test_valid_itr_integrity() -> None:
    """A valid ITR PDF must pass all integrity checks."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    validator = IntegrityValidator()

    result = validator.validate(
        str(PDF_PATH)
    )

    # ------------------------------------------------------
    # Basic file state
    # ------------------------------------------------------

    assert result.valid_file is True
    assert result.valid_pdf is True
    assert result.readable is True

    # ------------------------------------------------------
    # Security / corruption state
    # ------------------------------------------------------

    assert result.encrypted is False
    assert result.corrupted is False

    # ------------------------------------------------------
    # PDF metadata
    # ------------------------------------------------------

    assert result.page_count == 5
    assert result.file_size > 0

    # ------------------------------------------------------
    # Hash
    # ------------------------------------------------------

    assert result.sha256
    assert len(result.sha256) == 64
    assert all(
        character in "0123456789abcdef"
        for character in result.sha256.lower()
    )

    # ------------------------------------------------------
    # Final score
    # ------------------------------------------------------

    assert result.score == 1.0

    # ------------------------------------------------------
    # Reasons
    # ------------------------------------------------------

    assert result.reasons
    assert any(
        "Valid PDF" in reason
        for reason in result.reasons
    )
    assert any(
        "readable" in reason.lower()
        for reason in result.reasons
    )


def test_integrity_processing_time_is_recorded() -> None:
    """Integrity validation must record processing time."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    validator = IntegrityValidator()

    result = validator.validate(
        str(PDF_PATH)
    )

    assert result.processing_time_ms >= 0.0