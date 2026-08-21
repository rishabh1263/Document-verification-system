"""
Tests for PDF integrity analysis.
"""

from __future__ import annotations

import fitz

from src.documents.itr.authenticity.pdf_integrity import (
    IntegritySeverity,
    IntegrityStatus,
    PDFIntegrityAnalyzer,
    analyze_pdf_integrity,
)


# ==========================================================
# HELPERS
# ==========================================================


def _create_pdf(
    text: str = "Indian Income Tax Return",
) -> bytes:
    """
    Create a minimal valid PDF in memory.
    """

    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        text,
    )

    pdf_bytes = document.tobytes()

    document.close()

    return pdf_bytes


# ==========================================================
# BASIC PDF
# ==========================================================


def test_valid_pdf_is_analyzed() -> None:
    pdf_bytes = _create_pdf()

    result = (
        PDFIntegrityAnalyzer().analyze(
            pdf_bytes
        )
    )

    assert result.success is True

    assert result.page_count == 1

    assert result.file_size > 0

    assert result.document_hash

    assert len(
        result.document_hash
    ) == 64


def test_sha256_is_deterministic() -> None:
    pdf_bytes = _create_pdf()

    first = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    second = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    assert (
        first.document_hash
        ==
        second.document_hash
    )


# ==========================================================
# EMPTY INPUT
# ==========================================================


def test_empty_pdf_bytes_are_rejected() -> None:
    result = (
        analyze_pdf_integrity(
            b""
        )
    )

    assert result.success is False

    assert (
        result.risk_level
        == IntegritySeverity.CRITICAL
    )

    assert (
        result.status
        == IntegrityStatus.HIGH_RISK
    )

    assert result.findings

    assert (
        result.findings[0].rule_id
        == "PDF_EMPTY"
    )


# ==========================================================
# INVALID PDF
# ==========================================================


def test_invalid_pdf_is_rejected() -> None:
    result = (
        analyze_pdf_integrity(
            b"this is not a pdf"
        )
    )

    assert result.success is False

    assert result.findings

    assert (
        result.findings[0].rule_id
        == "PDF_OPEN_FAILED"
    )

    assert (
        result.risk_level
        == IntegritySeverity.CRITICAL
    )


# ==========================================================
# PAGE ANALYSIS
# ==========================================================


def test_page_information_is_extracted() -> None:
    pdf_bytes = _create_pdf(
        "Name: Vedant Ashish Sinagare"
    )

    result = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    assert result.success is True

    assert len(
        result.pages
    ) == 1

    page = result.pages[0]

    assert page.page_number == 1

    assert page.has_text is True

    assert page.text_characters > 0

    assert page.word_count > 0


# ==========================================================
# SERIALIZATION
# ==========================================================


def test_result_to_dict() -> None:
    pdf_bytes = _create_pdf()

    result = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    payload = (
        result.to_dict()
    )

    assert (
        payload["success"]
        is True
    )

    assert (
        "document_hash"
        in payload
    )

    assert (
        "risk_level"
        in payload
    )

    assert (
        "risk_score"
        in payload
    )

    assert (
        "findings"
        in payload
    )

    assert isinstance(
        payload["findings"],
        list,
    )


# ==========================================================
# METADATA SIGNAL
# ==========================================================


def test_editor_metadata_creates_signal() -> None:
    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "ITR document",
    )

    # PyMuPDF supports setting PDF metadata.
    document.set_metadata(
        {
            "format": "PDF 1.7",

            "title": "ITR",

            "author": "Test",

            "subject": "",

            "keywords": "",

            "creator": "Adobe Photoshop",

            "producer": "Adobe Photoshop",
        }
    )

    pdf_bytes = document.tobytes()

    document.close()

    result = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    matches = [
        finding
        for finding in result.findings
        if finding.rule_id
        == "PDF_EDITOR_METADATA"
    ]

    assert matches

    assert (
        matches[0].severity
        == IntegritySeverity.MEDIUM
    )


# ==========================================================
# MULTI-PAGE PDF
# ==========================================================


def test_multi_page_document() -> None:
    document = fitz.open()

    for index in range(3):

        page = document.new_page()

        page.insert_text(
            (72, 72),
            f"ITR Page {index + 1}",
        )

    pdf_bytes = document.tobytes()

    document.close()

    result = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    assert result.success is True

    assert result.page_count == 3

    assert len(
        result.pages
    ) == 3

    assert [
        page.page_number
        for page in result.pages
    ] == [
        1,
        2,
        3,
    ]


# ==========================================================
# REASON
# ==========================================================


def test_reason_is_always_present() -> None:
    pdf_bytes = _create_pdf()

    result = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    assert (
        isinstance(
            result.reason,
            str,
        )
    )

    assert result.reason.strip()

    assert (
        isinstance(
            result.summary,
            str,
        )
    )

    assert result.summary.strip()


# ==========================================================
# NO FALSE "FAKE" CLAIM
# ==========================================================


def test_clean_pdf_is_not_declared_fake() -> None:
    pdf_bytes = _create_pdf(
        """
        Name: Vedant Ashish Sinagare
        PAN: MCVPS7350E
        Assessment Year: 2024-25
        """
    )

    result = (
        analyze_pdf_integrity(
            pdf_bytes
        )
    )

    assert result.success is True

    # A clean technical PDF does not prove authenticity.
    # This analyzer must therefore not expose a "fake" verdict.
    assert (
        result.status
        in {
            IntegrityStatus.CLEAN,
            IntegrityStatus.LOW_RISK,
            IntegrityStatus.MEDIUM_RISK,
        }
    )

    assert (
        "fake"
        not in result.reason.lower()
    )