"""
==============================================================
ITR Integrity Validator
==============================================================

Purpose
-------
Validate the technical integrity of an ITR document.

Checks:
    1. File exists
    2. File is a regular file
    3. File is a supported PDF
    4. File size is valid
    5. PDF can be opened
    6. PDF is not encrypted
    7. PDF is not corrupted
    8. Page count is valid
    9. PDF pages are readable
    10. SHA-256 document hash

IMPORTANT
---------
This validator does NOT decide whether the document is an ITR.

Detection is handled by:
    src.documents.itr.detection

This validator only determines whether the document is
technically usable for further ITR validation.

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

import fitz

from .constants import (
    MAX_FILE_SIZE_BYTES,
    MAX_REASONABLE_PAGES,
    MIN_FILE_SIZE_BYTES,
    MIN_PAGES,
    REASON_CORRUPTED,
    REASON_FILE_EXISTS,
    REASON_FILE_MISSING,
    REASON_INVALID_FILE_SIZE,
    REASON_INVALID_PAGE_COUNT,
    REASON_INVALID_PDF,
    REASON_NOT_CORRUPTED,
    REASON_NOT_ENCRYPTED,
    REASON_READABLE,
    REASON_UNREADABLE,
    REASON_VALID_FILE_SIZE,
    REASON_VALID_PAGE_COUNT,
    REASON_VALID_PDF,
    REASON_ENCRYPTED,
)

from .models import IntegrityResult


logger = logging.getLogger(__name__)


class IntegrityValidator:
    """
    Validate technical integrity of an ITR document.
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self) -> None:
        """
        Initialize the integrity validator.
        """

        logger.debug(
            "IntegrityValidator initialized"
        )

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def validate(
        self,
        file_path: str,
    ) -> IntegrityResult:
        """
        Validate document integrity.

        Parameters
        ----------
        file_path:
            Absolute or relative document path.

        Returns
        -------
        IntegrityResult
            Complete technical integrity result.
        """

        start_time = perf_counter()

        result = IntegrityResult()

        try:
            # --------------------------------------------------
            # PATH
            # --------------------------------------------------

            path = Path(file_path)

            logger.info(
                "Starting integrity validation: %s",
                path,
            )

            # --------------------------------------------------
            # FILE EXISTENCE
            # --------------------------------------------------

            if not path.exists():

                result.reasons.append(
                    REASON_FILE_MISSING
                )

                return self._finalize(
                    result,
                    start_time,
                )

            result.valid_file = True

            result.reasons.append(
                REASON_FILE_EXISTS
            )

            # --------------------------------------------------
            # REGULAR FILE
            # --------------------------------------------------

            if not path.is_file():

                result.valid_file = False

                result.reasons.append(
                    "Path is not a regular file"
                )

                return self._finalize(
                    result,
                    start_time,
                )

            # --------------------------------------------------
            # EXTENSION
            # --------------------------------------------------

            extension = path.suffix.lower()

            if extension != ".pdf":

                result.reasons.append(
                    "Unsupported document format"
                )

                return self._finalize(
                    result,
                    start_time,
                )

            # --------------------------------------------------
            # FILE SIZE
            # --------------------------------------------------

            try:

                result.file_size = (
                    path.stat().st_size
                )

            except OSError as exc:

                logger.exception(
                    "Unable to read file size: %s",
                    path,
                )

                result.reasons.append(
                    f"Unable to read file size: {exc}"
                )

                return self._finalize(
                    result,
                    start_time,
                )

            if (
                MIN_FILE_SIZE_BYTES
                <= result.file_size
                <= MAX_FILE_SIZE_BYTES
            ):

                result.reasons.append(
                    REASON_VALID_FILE_SIZE
                )

            else:

                result.reasons.append(
                    REASON_INVALID_FILE_SIZE
                )

                return self._finalize(
                    result,
                    start_time,
                )

            # --------------------------------------------------
            # SHA-256
            # --------------------------------------------------

            try:

                result.sha256 = (
                    self._calculate_sha256(
                        path
                    )
                )

            except Exception as exc:

                logger.exception(
                    "SHA-256 calculation failed: %s",
                    path,
                )

                result.reasons.append(
                    f"Unable to calculate SHA-256: {exc}"
                )

            # --------------------------------------------------
            # PDF OPEN
            # --------------------------------------------------

            document: Optional[
                fitz.Document
            ] = None

            try:

                document = fitz.open(
                    str(path)
                )

                # ----------------------------------------------
                # PDF VALID
                # ----------------------------------------------

                result.valid_pdf = True

                result.reasons.append(
                    REASON_VALID_PDF
                )

                # ----------------------------------------------
                # ENCRYPTION
                # ----------------------------------------------

                if document.is_encrypted:

                    result.encrypted = True

                    result.reasons.append(
                        REASON_ENCRYPTED
                    )

                    return self._finalize(
                        result,
                        start_time,
                    )

                result.encrypted = False

                result.reasons.append(
                    REASON_NOT_ENCRYPTED
                )

                # ----------------------------------------------
                # PAGE COUNT
                # ----------------------------------------------

                result.page_count = len(
                    document
                )

                if (
                    MIN_PAGES
                    <= result.page_count
                    <= MAX_REASONABLE_PAGES
                ):

                    result.reasons.append(
                        REASON_VALID_PAGE_COUNT
                    )

                else:

                    result.reasons.append(
                        REASON_INVALID_PAGE_COUNT
                    )

                    return self._finalize(
                        result,
                        start_time,
                    )

                # ----------------------------------------------
                # READABILITY
                # ----------------------------------------------

                readable = (
                    self._is_readable(
                        document
                    )
                )

                if readable:

                    result.readable = True

                    result.reasons.append(
                        REASON_READABLE
                    )

                else:

                    result.readable = False

                    result.reasons.append(
                        REASON_UNREADABLE
                    )

                    return self._finalize(
                        result,
                        start_time,
                    )

            except Exception as exc:

                logger.exception(
                    "PDF integrity validation failed: %s",
                    path,
                )

                result.valid_pdf = False

                result.corrupted = True

                result.reasons.append(
                    REASON_INVALID_PDF
                )

                result.reasons.append(
                    REASON_CORRUPTED
                )

                result.reasons.append(
                    f"PDF error: {exc}"
                )

                return self._finalize(
                    result,
                    start_time,
                )

            finally:

                if document is not None:

                    try:

                        document.close()

                    except Exception:

                        logger.warning(
                            "Failed to close PDF: %s",
                            path,
                        )

            # --------------------------------------------------
            # NOT CORRUPTED
            # --------------------------------------------------

            result.corrupted = False

            result.reasons.append(
                REASON_NOT_CORRUPTED
            )

            # --------------------------------------------------
            # FINALIZE
            # --------------------------------------------------

            return self._finalize(
                result,
                start_time,
            )

        except Exception as exc:

            logger.exception(
                "Unexpected integrity validation error"
            )

            result.reasons.append(
                f"Unexpected validation error: {exc}"
            )

            return self._finalize(
                result,
                start_time,
            )

    # ==========================================================
    # SHA-256
    # ==========================================================

    @staticmethod
    def _calculate_sha256(
        file_path: Path,
    ) -> str:
        """
        Calculate SHA-256 hash of the complete document.
        """

        sha256 = hashlib.sha256()

        with file_path.open(
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                sha256.update(
                    chunk
                )

        return sha256.hexdigest()

    # ==========================================================
    # READABILITY
    # ==========================================================

    @staticmethod
    def _is_readable(
        document: fitz.Document,
    ) -> bool:
        """
        Verify that PDF pages can actually be accessed.

        IMPORTANT:
        A scanned PDF can have zero native text and still be
        perfectly readable.

        Therefore this method does NOT require text content.

        It verifies that pages can be loaded and basic page
        properties can be accessed successfully.
        """

        try:

            page_count = len(
                document
            )

            if page_count <= 0:

                return False

            # --------------------------------------------------
            # Check first three pages at most.
            # --------------------------------------------------

            pages_to_check = min(
                3,
                page_count,
            )

            for page_number in range(
                pages_to_check
            ):

                page = document.load_page(
                    page_number
                )

                # Access page geometry.

                _ = page.rect

                # Access native text layer.

                # Empty text is acceptable because the PDF may
                # be scanned.

                _ = page.get_text(
                    "text"
                )

            return True

        except Exception as exc:

            logger.warning(
                "PDF readability check failed: %s",
                exc,
            )

            return False

    # ==========================================================
    # SCORE
    # ==========================================================

    @staticmethod
    def _calculate_score(
        result: IntegrityResult,
    ) -> float:
        """
        Calculate technical integrity score.

        This score represents document usability.

        It does NOT represent:
            - ITR confidence
            - authenticity
            - tax correctness
            - PAN correctness
        """

        checks = [
            result.valid_file,
            result.valid_pdf,
            result.readable,
            not result.encrypted,
            not result.corrupted,
            (
                MIN_FILE_SIZE_BYTES
                <= result.file_size
                <= MAX_FILE_SIZE_BYTES
            ),
            (
                MIN_PAGES
                <= result.page_count
                <= MAX_REASONABLE_PAGES
            ),
        ]

        if not checks:

            return 0.0

        passed = sum(
            1
            for check in checks
            if check
        )

        return round(
            passed / len(checks),
            3,
        )

    # ==========================================================
    # FINALIZE
    # ==========================================================

    def _finalize(
        self,
        result: IntegrityResult,
        start_time: float,
    ) -> IntegrityResult:
        """
        Finalize integrity result.
        """

        result.score = (
            self._calculate_score(
                result
            )
        )

        result.processing_time_ms = round(
            (
                perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        return result


# ==========================================================
# OPTIONAL FUNCTION API
# ==========================================================


def validate_integrity(
    file_path: str,
) -> IntegrityResult:
    """
    Convenience function for integrity validation.

    Example
    -------
    result = validate_integrity(
        "sample.pdf"
    )
    """

    validator = IntegrityValidator()

    return validator.validate(
        file_path
    )


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python integrity_validator.py <pdf_path>"
        )

        raise SystemExit(1)

    pdf_path = sys.argv[1]

    validator = IntegrityValidator()

    validation_result = (
        validator.validate(
            pdf_path
        )
    )

    print(
        "=" * 80
    )

    print(
        "ITR INTEGRITY VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        validation_result.model_dump()
    )

    print(
        "=" * 80
    )