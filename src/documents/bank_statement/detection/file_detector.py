"""
Generic File Detector.

First stage of the bank-statement detection pipeline.

Responsibilities:
- validate that uploaded bytes exist
- identify supported file type
- verify extension against actual file signature
- reject unsupported or obviously mismatched files

Important:
This module does NOT:
- perform OCR
- decide whether the document is a bank statement
- detect tampering
- parse transactions
- perform loan eligibility logic
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileDetectionResult:
    filename: str
    extension: str
    detected_type: str
    mime_type: str
    size_bytes: int
    signature_valid: bool
    supported: bool

    def to_dict(self) -> dict:
        return asdict(self)


class FileDetector:
    """
    Detect and validate the physical file type using file signatures.

    We deliberately do not trust the filename extension alone.

    Example:
        fake.pdf containing JPEG bytes

    must not automatically be treated as a valid PDF.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
    }

    MAX_FILE_SIZE = 25 * 1024 * 1024

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def detect(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> FileDetectionResult:

        if not filename or not filename.strip():
            raise ValueError("Filename is required.")

        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        size_bytes = len(file_bytes)

        if size_bytes > self.MAX_FILE_SIZE:
            raise ValueError(
                "File exceeds the maximum allowed size of 25 MB."
            )

        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension or 'none'}"
            )

        detected_type, mime_type = self._detect_signature(
            file_bytes
        )

        supported = detected_type in {
            "pdf",
            "jpeg",
            "png",
        }

        expected_type = self._extension_to_type(
            extension
        )

        signature_valid = (
            supported
            and expected_type == detected_type
        )

        return FileDetectionResult(
            filename=Path(filename).name,
            extension=extension,
            detected_type=detected_type,
            mime_type=mime_type,
            size_bytes=size_bytes,
            signature_valid=signature_valid,
            supported=supported,
        )

    # --------------------------------------------------------
    # File signature detection
    # --------------------------------------------------------

    @staticmethod
    def _detect_signature(
        file_bytes: bytes,
    ) -> tuple[str, str]:

        # PDF
        if file_bytes.startswith(b"%PDF-"):
            return "pdf", "application/pdf"

        # JPEG
        if file_bytes.startswith(
            b"\xFF\xD8\xFF"
        ):
            return "jpeg", "image/jpeg"

        # PNG
        if file_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            return "png", "image/png"

        return "unknown", "application/octet-stream"

    # --------------------------------------------------------
    # Extension normalization
    # --------------------------------------------------------

    @staticmethod
    def _extension_to_type(
        extension: str,
    ) -> str:

        mapping = {
            ".pdf": "pdf",
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
        }

        return mapping.get(
            extension,
            "unknown",
        )


file_detector = FileDetector()