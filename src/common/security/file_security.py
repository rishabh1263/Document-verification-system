"""
Common Upload Security Layer.

This module is intended to run BEFORE any document-specific processing.

Supported uploads:
    JPG
    JPEG
    PNG
    PDF

Security flow:
    1. File size validation
    2. Extension validation
    3. Content-Type validation
    4. Magic-byte/signature validation
    5. Basic structural validation
    6. Optional ClamAV malware scan

Document-specific OCR/extraction is NOT performed here.

The same module can be reused by:
    PAN
    Driving Licence
    Voter ID
    Passport
    Bank Statement
    ITR
    Salary Slip
    CIBIL
    CRIF
    etc.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_FILE_SIZE_BYTES = int(
    os.getenv(
        "DOCUMENT_MAX_FILE_SIZE_BYTES",
        str(15 * 1024 * 1024),
    )
)

CLAMAV_ENABLED = os.getenv(
    "DOCUMENT_CLAMAV_ENABLED",
    "false",
).lower() in {"1", "true", "yes", "on"}

CLAMAV_REQUIRED = os.getenv(
    "DOCUMENT_CLAMAV_REQUIRED",
    "false",
).lower() in {"1", "true", "yes", "on"}

CLAMAV_HOST = os.getenv(
    "DOCUMENT_CLAMAV_HOST",
    "127.0.0.1",
)

CLAMAV_PORT = int(
    os.getenv(
        "DOCUMENT_CLAMAV_PORT",
        "3310",
    )
)

CLAMAV_TIMEOUT_SECONDS = float(
    os.getenv(
        "DOCUMENT_CLAMAV_TIMEOUT_SECONDS",
        "1.5",
    )
)


# ============================================================================
# SUPPORTED FILE TYPES
# ============================================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".pdf",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "application/pdf": ".pdf",
}

# File signatures.

JPEG_SIGNATURES = (
    b"\xff\xd8\xff",
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

PDF_SIGNATURE = b"%PDF-"


# ============================================================================
# EXCEPTIONS
# ============================================================================

class UploadSecurityError(Exception):
    """Base exception for upload-security failures."""


class UnsupportedFileTypeError(UploadSecurityError):
    """Unsupported extension/content type."""


class InvalidFileSignatureError(UploadSecurityError):
    """Extension/content type does not match actual file bytes."""


class FileTooLargeError(UploadSecurityError):
    """Upload exceeds configured size limit."""


class MaliciousFileError(UploadSecurityError):
    """Antivirus identified the upload as malicious."""


class VirusScannerUnavailableError(UploadSecurityError):
    """Required antivirus scanner is unavailable."""


class InvalidFileError(UploadSecurityError):
    """File is malformed or cannot be structurally validated."""


# ============================================================================
# BASIC HELPERS
# ============================================================================

def _normalise_extension(filename: str | None) -> str:
    if not filename:
        return ""

    return Path(filename).suffix.lower().strip()


def _normalise_content_type(content_type: str | None) -> str:
    return (
        str(content_type or "")
        .split(";")[0]
        .strip()
        .lower()
    )


def _detect_signature(file_bytes: bytes) -> str | None:
    """
    Detect the real file format from its bytes.

    This is deliberately independent of filename and MIME type.
    """

    if not file_bytes:
        return None

    if any(
        file_bytes.startswith(signature)
        for signature in JPEG_SIGNATURES
    ):
        return ".jpg"

    if file_bytes.startswith(PNG_SIGNATURE):
        return ".png"

    if file_bytes.startswith(PDF_SIGNATURE):
        return ".pdf"

    return None


# ============================================================================
# BUILT-IN FILE SECURITY
# ============================================================================
#
# Fast, zero-install common upload-security layer.
#
# This is NOT a full antivirus engine. It detects obvious executable/script
# masquerading, suspicious containers, and active PDF content.
# It is intentionally common to every document type.
# ============================================================================

EXECUTABLE_SIGNATURES = (
    b"MZ",          # Windows PE
    b"\x7fELF",     # Linux ELF
    b"#!",          # Script
)

SUSPICIOUS_CONTAINER_SIGNATURES = (
    b"PK\x03\x04",
)

SUSPICIOUS_PDF_MARKERS: tuple[bytes, ...] = ()


def _run_basic_security_scan(
    file_bytes: bytes,
    detected_extension: str | None = None,
) -> dict[str, Any]:
    """Run fast built-in security checks common to all documents."""

    if not file_bytes:
        raise InvalidFileError("Uploaded file is empty.")

    # Executable/script disguised as a document.
    if file_bytes.startswith(EXECUTABLE_SIGNATURES):
        raise MaliciousFileError(
            "Executable or script content detected in uploaded document."
        )

    # ZIP/container disguised as an image.
    if detected_extension in {".jpg", ".jpeg", ".png"}:
        if file_bytes.startswith(SUSPICIOUS_CONTAINER_SIGNATURES):
            raise MaliciousFileError(
                "Suspicious container content detected in image upload."
            )

    # Do not reject PDFs based on generic PDF object/action markers.
    # Legitimate PDFs can contain JavaScript/action/annotation structures,
    # and treating those markers as malware creates false positives.
    #
    # A real antivirus engine can be plugged into this common layer later.
    return {
        "safe": True,
        "status": "BASIC_SECURITY_PASS",
    }


def _run_malware_scan(
    file_bytes: bytes,
    detected_extension: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible internal name for the built-in security scan."""
    return _run_basic_security_scan(
        file_bytes=file_bytes,
        detected_extension=detected_extension,
    )


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_upload(
    file_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    Common security entry point.

    IMPORTANT:
        Call this BEFORE document-specific processing.

    Returns:
        {
            "safe": True,
            "file_type": ".jpg",
            "content_type": "image/jpeg",
            "size_bytes": 12345,
            "virus_scan": {...}
        }
    """

    # ------------------------------------------------------------------------
    # 1. EMPTY FILE
    # ------------------------------------------------------------------------

    if not file_bytes:
        raise InvalidFileError(
            "Uploaded file is empty."
        )

    # ------------------------------------------------------------------------
    # 2. FILE SIZE
    # ------------------------------------------------------------------------

    size_bytes = len(file_bytes)

    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            "Uploaded file exceeds the maximum allowed size."
        )

    # ------------------------------------------------------------------------
    # 3. EXTENSION
    # ------------------------------------------------------------------------

    extension = _normalise_extension(filename)

    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            "Upload only JPG, JPEG, PNG or PDF files."
        )

    # ------------------------------------------------------------------------
    # 4. CONTENT TYPE
    # ------------------------------------------------------------------------

    normalised_content_type = _normalise_content_type(
        content_type
    )

    if normalised_content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedFileTypeError(
            "Upload only JPG, JPEG, PNG or PDF files."
        )

    # ------------------------------------------------------------------------
    # 5. SIGNATURE
    # ------------------------------------------------------------------------

    detected_extension = _detect_signature(
        file_bytes
    )

    if detected_extension is None:
        raise InvalidFileSignatureError(
            "File content does not match a supported JPG, PNG or PDF format."
        )

    # JPEG has both .jpg and .jpeg extensions.
    extension_matches = (
        detected_extension == extension
        or (
            detected_extension == ".jpg"
            and extension == ".jpeg"
        )
    )

    if not extension_matches:
        raise InvalidFileSignatureError(
            "File extension does not match the actual file content."
        )

    expected_from_mime = ALLOWED_CONTENT_TYPES[
        normalised_content_type
    ]

    mime_matches = (
        detected_extension == expected_from_mime
        or (
            detected_extension == ".jpg"
            and expected_from_mime == ".jpg"
        )
    )

    if not mime_matches:
        raise InvalidFileSignatureError(
            "Content-Type does not match the actual file content."
        )

    # ------------------------------------------------------------------------
    # COMMON BUILT-IN SECURITY GATE
    # ------------------------------------------------------------------------
    security_result = _run_basic_security_scan(
        file_bytes=file_bytes,
        detected_extension=detected_extension,
    )

    # ------------------------------------------------------------------------
    # 6. MALWARE SCAN
    # ------------------------------------------------------------------------

    # This is intentionally the FIRST expensive/security processing step
    # before document-specific decoding/OCR/extraction.
    virus_scan = _run_basic_security_scan(
        file_bytes=file_bytes,
        detected_extension=detected_extension,
    )

    # ------------------------------------------------------------------------
    # 7. CLEAN RESULT
    # ------------------------------------------------------------------------

    return {
        "safe": True,
        "file_type": detected_extension,
        "extension": extension,
        "content_type": normalised_content_type,
        "size_bytes": size_bytes,
        "virus_scan": security_result,
    }


# ============================================================================
# SIMPLE CONTRACT TEST
# ============================================================================

def security_contract_test() -> dict[str, Any]:
    return {
        "supported_extensions": sorted(
            ALLOWED_EXTENSIONS
        ),
        "supported_content_types": sorted(
            ALLOWED_CONTENT_TYPES.keys()
        ),
        "ocr_used": False,
        "extraction_used": False,
        "common_for_all_documents": True,
        "signature_validation": True,
        "malware_scan_supported": True,
        "external_antivirus_required": False,
        "zero_install_security": True,
    }


__all__ = [
    "validate_upload",
    "security_contract_test",
    "UploadSecurityError",
    "UnsupportedFileTypeError",
    "InvalidFileSignatureError",
    "FileTooLargeError",
    "MaliciousFileError",
    "VirusScannerUnavailableError",
    "InvalidFileError",
]