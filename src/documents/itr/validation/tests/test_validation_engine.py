"""
Pytest tests for the ITR Validation Engine.

This test verifies the complete validation orchestration for a known
valid ITR sample.

Tested layers:
    - Integrity
    - Required content
    - Internal consistency
    - Final validation decision
"""

from __future__ import annotations

from pathlib import Path

from src.documents.itr.validation.models import (
    ValidationDecision,
    ValidationStatus,
)
from src.documents.itr.validation.validation_engine import (
    ValidationEngine,
)


PDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "Vedant ITR.pdf"
)


def test_valid_itr_validation() -> None:
    """A valid ITR must pass the complete validation engine."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    engine = ValidationEngine()

    result = engine.validate_file(
        str(PDF_PATH),
        detection_confidence=0.771,
    )

    # ------------------------------------------------------
    # Final validation status
    # ------------------------------------------------------

    assert result.status == ValidationStatus.SUCCESS
    assert result.decision == ValidationDecision.VALID
    assert result.valid is True
    assert result.error is None

    # ------------------------------------------------------
    # Final confidence
    # ------------------------------------------------------

    assert 0.70 <= result.confidence <= 1.0

    # ------------------------------------------------------
    # Integrity
    # ------------------------------------------------------

    integrity = result.evidence.integrity

    assert integrity.valid_file is True
    assert integrity.valid_pdf is True
    assert integrity.readable is True
    assert integrity.encrypted is False
    assert integrity.corrupted is False
    assert integrity.page_count > 0
    assert integrity.file_size > 0
    assert integrity.sha256
    assert integrity.score == 1.0

    # ------------------------------------------------------
    # Required ITR content
    # ------------------------------------------------------

    content = result.evidence.content

    assert content.required_content_present is True
    assert content.assessment_year_present is True
    assert content.pan_present is True
    assert content.taxpayer_information_present is True
    assert content.income_information_present is True
    assert content.tax_computation_present is True
    assert content.verification_present is True
    assert content.acknowledgement_present is True
    assert content.missing_items == []
    assert content.score == 1.0

    # ------------------------------------------------------
    # Internal consistency
    # ------------------------------------------------------

    consistency = result.evidence.consistency

    assert consistency.consistent is True
    assert consistency.assessment_year_consistent is True
    assert consistency.pan_consistent is True
    assert consistency.income_consistent is True
    assert consistency.tax_consistent is True
    assert consistency.acknowledgement_consistent is True
    assert consistency.inconsistencies == []
    assert consistency.score == 1.0


def test_validation_result_contains_filename_and_hash() -> None:
    """The validation result must contain document identity metadata."""
    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    engine = ValidationEngine()

    result = engine.validate_file(
        str(PDF_PATH),
        detection_confidence=0.771,
    )

    assert result.filename == "Vedant ITR.pdf"
    assert len(result.document_hash) == 64
    assert all(
        character in "0123456789abcdef"
        for character in result.document_hash.lower()
    )