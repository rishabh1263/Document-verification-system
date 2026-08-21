"""
Common PDF loading utilities.

Used by the common document-processing engine for:

- PDF validation
- page counting
- native PDF text extraction

Uses the modern PyMuPDF import instead of the deprecated
`fitz` import.
"""

from __future__ import annotations

import pymupdf


def validate_pdf(
    file_bytes: bytes,
) -> None:
    """
    Check whether the supplied bytes contain a readable PDF.
    """

    if not file_bytes:
        raise ValueError(
            "PDF file is empty."
        )

    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )

        document.close()

    except Exception as exc:

        raise ValueError(
            f"Invalid or unreadable PDF: {exc}"
        ) from exc


def get_pdf_page_count(
    file_bytes: bytes,
) -> int:
    """
    Return the number of pages in a PDF.
    """

    validate_pdf(
        file_bytes
    )

    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf",
    )

    try:

        return len(document)

    finally:

        document.close()


def extract_pdf_text(
    file_bytes: bytes,
) -> list[str]:
    """
    Extract native text from every PDF page.

    This is the FAST path for digital PDFs.

    Scanned PDFs may return little or no native text.
    Those PDFs should be sent to the OCR pipeline.
    """

    validate_pdf(
        file_bytes
    )

    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf",
    )

    try:

        return [
            page.get_text("text")
            for page in document
        ]

    finally:

        document.close()