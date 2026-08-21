"""
Native PDF Text Extractor.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- extract embedded text from digital PDFs
- preserve page boundaries
- provide page-level extraction statistics
- produce a generic standardized result

Important:
This extractor is bank-independent.

It does NOT:
- perform OCR
- parse transactions
- identify account numbers
- identify bank names
- detect tampering
- calculate fraud/risk scores
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(frozen=True)
class NativePDFPage:
    page_number: int
    text: str
    char_count: int
    line_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NativePDFExtractionResult:
    filename: str
    extraction_method: str
    page_count: int
    text: str
    text_char_count: int
    pages: tuple[NativePDFPage, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class NativePDFExtractor:
    """
    Extract embedded text directly from a digital PDF.

    This is the preferred fast path for PDFs that already
    contain a usable text layer.
    """

    def extract(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> NativePDFExtractionResult:

        if not file_bytes:
            raise ValueError("PDF bytes are required.")

        if not filename or not filename.strip():
            raise ValueError("Filename is required.")

        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError(
                "NativePDFExtractor received a non-PDF file."
            )

        try:
            reader = PdfReader(BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(
                f"Unable to open PDF: {exc}"
            ) from exc

        if len(reader.pages) == 0:
            raise ValueError(
                "PDF contains no readable pages."
            )

        extracted_pages: list[NativePDFPage] = []
        document_parts: list[str] = []

        for page_index, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                raise ValueError(
                    f"Native text extraction failed on "
                    f"page {page_index}: {exc}"
                ) from exc

            text = self._normalize_page_text(raw_text)

            lines = [
                line
                for line in text.splitlines()
                if line.strip()
            ]

            page_result = NativePDFPage(
                page_number=page_index,
                text=text,
                char_count=len(text),
                line_count=len(lines),
            )

            extracted_pages.append(page_result)

            if text:
                document_parts.append(text)

        document_text = "\n\n".join(document_parts)

        if not document_text.strip():
            raise ValueError(
                "PDF contains no extractable native text. "
                "Route this document to OCR."
            )

        return NativePDFExtractionResult(
            filename=filename,
            extraction_method="native_pdf",
            page_count=len(extracted_pages),
            text=document_text,
            text_char_count=len(document_text),
            pages=tuple(extracted_pages),
        )

    # --------------------------------------------------------
    # Lightweight normalization only
    # --------------------------------------------------------

    @staticmethod
    def _normalize_page_text(
        text: str,
    ) -> str:
        """
        Perform conservative normalization.

        Do not aggressively alter whitespace here because
        later transaction parsing may depend on line structure.
        """

        if not text:
            return ""

        normalized_lines = []

        for line in text.splitlines():
            cleaned = line.replace(
                "\x00",
                "",
            ).rstrip()

            normalized_lines.append(cleaned)

        return "\n".join(
            normalized_lines
        ).strip()


native_pdf_extractor = NativePDFExtractor()