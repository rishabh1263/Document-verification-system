"""
==============================================================
ITR Metadata Detector
==============================================================

Purpose
-------
Perform file-level and PDF-level detection evidence.

Responsibilities
----------------
- File existence
- Extension validation
- File size validation
- SHA256 generation
- PDF opening
- PDF corruption detection
- Encryption detection
- Page count
- PDF metadata
- Digital / scanned / mixed mode detection
- Metadata evidence scoring

Does NOT perform:
- OCR
- PAN extraction
- Name extraction
- DOB extraction
- Field validation
- Tampering validation

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fitz

from ..config import CONFIG
from ..models import DocumentMode, MetadataResult
from .utils import (
    current_time,
    elapsed_time,
    file_exists,
    generate_sha256,
    get_extension,
    get_file_size,
)


logger = logging.getLogger(__name__)


class MetadataDetector:
    """
    Detect file and PDF-level metadata evidence.
    """

    # ==========================================================
    # PUBLIC
    # ==========================================================

    def analyze(
        self,
        file_path: str,
    ) -> MetadataResult:
        """
        Analyze a document at file/PDF metadata level.
        """

        start_time = current_time()

        result = MetadataResult()

        path = Path(file_path)

        # ------------------------------------------------------
        # File existence
        # ------------------------------------------------------

        if not file_exists(file_path):

            result.valid_file = False
            result.is_supported = False
            result.is_valid_pdf = False

            self._add_reason(
                result,
                "File not found",
            )

            result.processing_time_ms = elapsed_time(
                start_time
            )

            return result

        result.valid_file = True

        self._add_reason(
            result,
            "File exists",
        )

        # ------------------------------------------------------
        # Extension
        # ------------------------------------------------------

        result.extension = get_extension(
            file_path
        )

        result.is_supported = (
            result.extension
            in CONFIG.supported_extensions
        )

        if result.is_supported:

            self._add_reason(
                result,
                "Supported extension",
            )

        else:

            self._add_reason(
                result,
                "Unsupported extension",
            )

            result.processing_time_ms = elapsed_time(
                start_time
            )

            return result

        # ------------------------------------------------------
        # File size
        # ------------------------------------------------------

        try:

            result.file_size = get_file_size(
                file_path
            )

        except OSError as exc:

            logger.exception(
                "Unable to read file size"
            )

            self._add_reason(
                result,
                f"Unable to read file size: {exc}",
            )

            result.processing_time_ms = elapsed_time(
                start_time
            )

            return result

        if result.file_size <= 0:

            self._add_reason(
                result,
                "Empty file",
            )

            result.processing_time_ms = elapsed_time(
                start_time
            )

            return result

        self._add_reason(
            result,
            "Valid file size",
        )

        # ------------------------------------------------------
        # SHA256
        # ------------------------------------------------------

        try:

            result.sha256 = generate_sha256(
                file_path
            )

        except OSError:

            logger.exception(
                "SHA256 generation failed"
            )

            self._add_reason(
                result,
                "Unable to generate SHA256",
            )

        # ------------------------------------------------------
        # Image input
        # ------------------------------------------------------

        if result.extension in {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
        }:

            result.is_valid_pdf = False

            result.mode = DocumentMode.IMAGE

            result.score = self._calculate_score(
                result
            )

            self._add_reason(
                result,
                "Image document detected",
            )

            result.processing_time_ms = elapsed_time(
                start_time
            )

            return result

        # ------------------------------------------------------
        # PDF
        # ------------------------------------------------------

        return self._process_pdf(
            file_path=file_path,
            result=result,
            start_time=start_time,
        )

    # ==========================================================
    # PDF PROCESSING
    # ==========================================================

    def _process_pdf(
        self,
        file_path: str,
        result: MetadataResult,
        start_time: float,
    ) -> MetadataResult:
        """
        Open and analyze PDF.
        """

        pdf: Optional[fitz.Document] = None

        try:

            # --------------------------------------------------
            # Open PDF
            # --------------------------------------------------

            pdf = fitz.open(
                file_path
            )

            result.is_valid_pdf = True

            self._add_reason(
                result,
                "Valid PDF",
            )

            # --------------------------------------------------
            # Encryption
            # --------------------------------------------------

            if pdf.needs_pass:

                result.encrypted = True

                self._add_reason(
                    result,
                    "Password protected PDF",
                )

                result.score = self._calculate_score(
                    result
                )

                return result

            # --------------------------------------------------
            # Page count
            # --------------------------------------------------

            result.page_count = len(pdf)

            if result.page_count < CONFIG.min_pages:

                self._add_reason(
                    result,
                    "Invalid page count",
                )

                result.score = self._calculate_score(
                    result
                )

                return result

            if result.page_count > CONFIG.max_pages:

                self._add_reason(
                    result,
                    "Page count exceeds reasonable limit",
                )

            else:

                self._add_reason(
                    result,
                    "Valid page count",
                )

            # --------------------------------------------------
            # PDF metadata
            # --------------------------------------------------

            result.metadata = (
                self._extract_pdf_metadata(
                    pdf
                )
            )

            metadata_values = [
                value
                for value in result.metadata.values()
                if value
            ]

            if metadata_values:

                self._add_reason(
                    result,
                    "PDF metadata available",
                )

            # --------------------------------------------------
            # Digital / scanned / mixed
            # --------------------------------------------------

            self._detect_mode(
                pdf=pdf,
                result=result,
            )

            # --------------------------------------------------
            # Score
            # --------------------------------------------------

            result.score = self._calculate_score(
                result
            )

            return result

        except Exception:

            logger.exception(
                "Unable to process PDF: %s",
                file_path,
            )

            result.corrupted = True
            result.is_valid_pdf = False

            self._add_reason(
                result,
                "Corrupted PDF",
            )

            result.score = self._calculate_score(
                result
            )

            return result

        finally:

            if pdf is not None:

                pdf.close()

            result.processing_time_ms = elapsed_time(
                start_time
            )

    # ==========================================================
    # MODE DETECTION
    # ==========================================================

    def _detect_mode(
        self,
        pdf: fitz.Document,
        result: MetadataResult,
    ) -> None:
        """
        Detect PDF mode:

        DIGITAL
        SCANNED
        MIXED
        """

        digital_pages = 0

        scanned_pages = 0

        total_pages = len(pdf)

        pages_to_check = min(
            CONFIG.max_pages_to_analyze,
            total_pages,
        )

        if pages_to_check <= 0:

            result.mode = DocumentMode.UNKNOWN

            return

        for page_number in range(
            pages_to_check
        ):

            page = pdf.load_page(
                page_number
            )

            words = page.get_text(
                "words"
            )

            blocks = page.get_text(
                "blocks"
            )

            images = page.get_images(
                full=True
            )

            word_count = len(words)

            block_count = len(blocks)

            image_count = len(images)

            # --------------------------------------------------
            # Digital
            # --------------------------------------------------

            if (
                word_count >= 30
                and block_count >= 5
            ):

                digital_pages += 1

                continue

            # --------------------------------------------------
            # Scanned
            # --------------------------------------------------

            if (
                image_count > 0
                and word_count < 10
            ):

                scanned_pages += 1

                continue

            # --------------------------------------------------
            # Fallback
            # --------------------------------------------------

            if word_count > 0:

                digital_pages += 1

            else:

                scanned_pages += 1

        # ------------------------------------------------------
        # Final mode
        # ------------------------------------------------------

        if digital_pages == pages_to_check:

            result.mode = DocumentMode.DIGITAL

            self._add_reason(
                result,
                "Digital PDF detected",
            )

        elif scanned_pages == pages_to_check:

            result.mode = DocumentMode.SCANNED

            self._add_reason(
                result,
                "Scanned PDF detected",
            )

        else:

            result.mode = DocumentMode.MIXED

            self._add_reason(
                result,
                "Mixed PDF detected",
            )

    # ==========================================================
    # METADATA EXTRACTION
    # ==========================================================

    @staticmethod
    def _extract_pdf_metadata(
        pdf: fitz.Document,
    ) -> dict:
        """
        Extract useful PDF metadata.
        """

        metadata = pdf.metadata or {}

        return {
            "title": metadata.get(
                "title"
            ),

            "author": metadata.get(
                "author"
            ),

            "subject": metadata.get(
                "subject"
            ),

            "creator": metadata.get(
                "creator"
            ),

            "producer": metadata.get(
                "producer"
            ),

            "creation_date": metadata.get(
                "creationDate"
            ),

            "modification_date": metadata.get(
                "modDate"
            ),

            "format": metadata.get(
                "format"
            ),
        }

    # ==========================================================
    # SCORE
    # ==========================================================

    @staticmethod
    def _calculate_score(
        result: MetadataResult,
    ) -> float:
        """
        Calculate metadata confidence.

        This is NOT final document confidence.

        It represents how trustworthy/usable the PDF is
        for downstream detection.
        """

        score = 0.0

        # ------------------------------------------------------
        # File
        # ------------------------------------------------------

        if result.valid_file:

            score += 0.10

        # ------------------------------------------------------
        # Supported extension
        # ------------------------------------------------------

        if result.is_supported:

            score += 0.10

        # ------------------------------------------------------
        # File size
        # ------------------------------------------------------

        if result.file_size > 0:

            score += 0.05

        # ------------------------------------------------------
        # Valid PDF
        # ------------------------------------------------------

        if result.is_valid_pdf:

            score += 0.15

        # ------------------------------------------------------
        # Page count
        # ------------------------------------------------------

        if (
            result.page_count
            >= CONFIG.min_pages
            and
            result.page_count
            <= CONFIG.max_pages
        ):

            score += 0.10

        # ------------------------------------------------------
        # Mode
        # ------------------------------------------------------

        if result.mode == DocumentMode.DIGITAL:

            score += 0.20

        elif result.mode == DocumentMode.MIXED:

            score += 0.15

        elif result.mode == DocumentMode.SCANNED:

            score += 0.10

        # ------------------------------------------------------
        # Metadata
        # ------------------------------------------------------

        if result.metadata:

            if any(
                value
                for value in result.metadata.values()
            ):

                score += 0.05

        # ------------------------------------------------------
        # Penalties
        # ------------------------------------------------------

        if result.encrypted:

            score -= 0.10

        if result.corrupted:

            score -= 0.30

        # ------------------------------------------------------
        # Normalize
        # ------------------------------------------------------

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        return round(
            score,
            3,
        )

    # ==========================================================
    # REASON HELPER
    # ==========================================================

    @staticmethod
    def _add_reason(
        result: MetadataResult,
        reason: str,
    ) -> None:
        """
        Add unique detection reason.
        """

        if reason not in result.reasons:

            result.reasons.append(
                reason
            )