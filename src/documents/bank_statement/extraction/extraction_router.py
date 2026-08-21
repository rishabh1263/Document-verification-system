"""
Bank Statement Extraction Router.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- reuse Phase 1 physical file detection
- reject unsupported or extension-spoofed files
- route image documents to OCR
- inspect PDFs to determine whether native text extraction is usable
- avoid unnecessary OCR for digital PDFs

Important:
This module does NOT:
- perform OCR
- parse transactions
- extract bank-specific fields
- detect tampering
- calculate fraud/risk scores
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

from pypdf import PdfReader

from src.documents.bank_statement.detection.file_detector import (
    FileDetectionResult,
    file_detector,
)


@dataclass(frozen=True)
class ExtractionRouteResult:
    filename: str
    detected_type: str
    extraction_method: str
    reason: str
    page_count: int | None
    text_char_count: int
    native_text_available: bool
    requires_ocr: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ExtractionRouter:
    """
    Select the appropriate extraction strategy.

    Routes:
        PDF with usable embedded text -> native_pdf
        PDF without usable text       -> ocr_pdf
        JPEG/PNG                      -> ocr_image
    """

    # Conservative threshold.
    # This is only used for routing, not document classification.
    MIN_NATIVE_TEXT_CHARS = 50

    def route(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> ExtractionRouteResult:

        detection = file_detector.detect(
            file_bytes=file_bytes,
            filename=filename,
        )

        self._validate_detection(detection)

        if detection.detected_type == "pdf":
            return self._route_pdf(
                file_bytes=file_bytes,
                detection=detection,
            )

        if detection.detected_type in {"jpeg", "png"}:
            return ExtractionRouteResult(
                filename=detection.filename,
                detected_type=detection.detected_type,
                extraction_method="ocr_image",
                reason="Image documents require OCR extraction.",
                page_count=1,
                text_char_count=0,
                native_text_available=False,
                requires_ocr=True,
            )

        # Defensive fallback. Normally unreachable because
        # unsupported types are rejected above.
        raise ValueError(
            f"No extraction route available for "
            f"{detection.detected_type}."
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_detection(
        detection: FileDetectionResult,
    ) -> None:

        if not detection.supported:
            raise ValueError(
                f"Unsupported physical file type: "
                f"{detection.detected_type}"
            )

        if not detection.signature_valid:
            raise ValueError(
                "File extension does not match the actual "
                "physical file type."
            )

    # --------------------------------------------------------
    # PDF routing
    # --------------------------------------------------------

    def _route_pdf(
        self,
        file_bytes: bytes,
        detection: FileDetectionResult,
    ) -> ExtractionRouteResult:

        try:
            reader = PdfReader(BytesIO(file_bytes))

            page_count = len(reader.pages)

            if page_count == 0:
                raise ValueError(
                    "PDF contains no readable pages."
                )

            total_text_chars = 0

            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                total_text_chars += len(text.strip())

            native_text_available = (
                total_text_chars >= self.MIN_NATIVE_TEXT_CHARS
            )

            if native_text_available:
                return ExtractionRouteResult(
                    filename=detection.filename,
                    detected_type="pdf",
                    extraction_method="native_pdf",
                    reason=(
                        "PDF contains sufficient embedded text; "
                        "native extraction selected."
                    ),
                    page_count=page_count,
                    text_char_count=total_text_chars,
                    native_text_available=True,
                    requires_ocr=False,
                )

            return ExtractionRouteResult(
                filename=detection.filename,
                detected_type="pdf",
                extraction_method="ocr_pdf",
                reason=(
                    "PDF does not contain sufficient usable "
                    "embedded text; OCR fallback required."
                ),
                page_count=page_count,
                text_char_count=total_text_chars,
                native_text_available=False,
                requires_ocr=True,
            )

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                f"Unable to inspect PDF for extraction routing: {exc}"
            ) from exc


extraction_router = ExtractionRouter()