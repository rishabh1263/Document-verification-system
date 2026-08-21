"""
ITR Detection Engine
====================

Purpose
-------
Orchestrate the complete ITR detection pipeline.

Pipeline
--------
1. Validate input
2. Run metadata detection
3. Extract detection text
4. Run keyword detection
5. Run layout detection
6. Run structure detection
7. Combine all evidence
8. Calculate final confidence
9. Return DetectionResult

Detection Evidence
------------------
The detector combines four independent evidence sources:

    Metadata
        ↓
    Keyword
        ↓
    Layout
        ↓
    Structure
        ↓
    Final Confidence

This module performs DETECTION only.

It does NOT perform:
- PAN extraction
- Name extraction
- DOB extraction
- Field matching
- Validation
- Tampering validation
- Business-rule validation

Author
------
SBFC Document Intelligence
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import fitz

from ..models import (
    DetectionEvidence,
    DetectionInput,
    DetectionResult,
    DetectionStatus,
)
from .confidence import ConfidenceEngine
from .keyword_detector import KeywordDetector
from .layout_detector import LayoutDetector
from .metadata_detector import MetadataDetector
from .structure_detector import StructureDetector


logger = logging.getLogger(__name__)


class ITRDetector:
    """
    Main ITR detection orchestrator.

    Responsible only for determining whether the supplied
    document is an ITR and producing evidence-backed confidence.
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self) -> None:
        """
        Initialize all detection components.
        """

        self.metadata_detector = MetadataDetector()

        self.keyword_detector = KeywordDetector()

        self.layout_detector = LayoutDetector()

        self.structure_detector = StructureDetector()

        self.confidence_engine = ConfidenceEngine()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def detect(
        self,
        file_path: str,
    ) -> DetectionResult:
        """
        Detect whether a document is an ITR.

        Parameters
        ----------
        file_path:
            Absolute or project-relative document path.

        Returns
        -------
        DetectionResult
            Complete detection response containing:

            - detection status
            - detected flag
            - document type
            - document mode
            - confidence
            - page count
            - metadata evidence
            - keyword evidence
            - layout evidence
            - structure evidence
            - processing time
        """

        start_time = perf_counter()

        try:

            # --------------------------------------------------
            # 1. VALIDATE INPUT
            # --------------------------------------------------

            detection_input = DetectionInput(
                file_path=file_path
            )

            path = Path(
                detection_input.file_path
            )

            logger.info(
                "Starting ITR detection: %s",
                path,
            )

            # --------------------------------------------------
            # 2. FILE EXISTENCE
            # --------------------------------------------------

            if not path.exists():

                logger.warning(
                    "File not found: %s",
                    path,
                )

                return self._error_result(
                    message="File not found",
                    start_time=start_time,
                )

            if not path.is_file():

                logger.warning(
                    "Path is not a file: %s",
                    path,
                )

                return self._error_result(
                    message="Path is not a file",
                    start_time=start_time,
                )

            # --------------------------------------------------
            # 3. METADATA DETECTION
            # --------------------------------------------------

            logger.info(
                "Running metadata detection"
            )

            metadata_result = (
                self.metadata_detector.analyze(
                    str(path)
                )
            )

            # --------------------------------------------------
            # STOP FOR UNUSABLE DOCUMENT
            # --------------------------------------------------

            if (
                metadata_result.corrupted
                or metadata_result.encrypted
            ):

                logger.warning(
                    "Document cannot be analysed: "
                    "corrupted=%s encrypted=%s",
                    metadata_result.corrupted,
                    metadata_result.encrypted,
                )

                evidence = DetectionEvidence(
                    metadata=metadata_result
                )

                result = (
                    self.confidence_engine.calculate(
                        evidence
                    )
                )

                return self._finalize_result(
                    result=result,
                    file_path=path,
                    start_time=start_time,
                )

            # --------------------------------------------------
            # 4. EXTRACT DETECTION TEXT
            # --------------------------------------------------

            logger.info(
                "Extracting detection text"
            )

            text = self._extract_detection_text(
                path
            )

            logger.info(
                "Detection text extracted: %d characters",
                len(text),
            )

            # --------------------------------------------------
            # 5. KEYWORD DETECTION
            # --------------------------------------------------

            logger.info(
                "Running keyword detection"
            )

            keyword_result = (
                self.keyword_detector.analyze(
                    text
                )
            )

            logger.info(
                "Keyword detection completed: score=%.3f",
                keyword_result.score,
            )

            # --------------------------------------------------
            # 6. LAYOUT DETECTION
            # --------------------------------------------------

            logger.info(
                "Running layout detection"
            )

            layout_result = (
                self.layout_detector.analyze(
                    text
                )
            )

            logger.info(
                "Layout detection completed: score=%.3f",
                layout_result.score,
            )

            # --------------------------------------------------
            # 7. STRUCTURE DETECTION
            # --------------------------------------------------

            logger.info(
                "Running structure detection"
            )

            structure_result = (
                self.structure_detector.analyze(
                    text
                )
            )

            logger.info(
                "Structure detection completed: score=%.3f",
                structure_result.score,
            )

            # --------------------------------------------------
            # 8. COMBINE EVIDENCE
            # --------------------------------------------------

            evidence = DetectionEvidence(
                metadata=metadata_result,
                keyword=keyword_result,
                layout=layout_result,
                structure=structure_result,
                raw_text_length=len(text),
                first_page_only=False,
            )

            # --------------------------------------------------
            # 9. FINAL CONFIDENCE
            # --------------------------------------------------

            logger.info(
                "Calculating final ITR confidence"
            )

            result = (
                self.confidence_engine.calculate(
                    evidence
                )
            )

            # --------------------------------------------------
            # 10. FINALIZE RESULT
            # --------------------------------------------------

            return self._finalize_result(
                result=result,
                file_path=path,
                start_time=start_time,
            )

        except Exception as exc:

            logger.exception(
                "ITR detection failed: %s",
                file_path,
            )

            return self._error_result(
                message=str(exc),
                start_time=start_time,
            )

    # ==========================================================
    # TEXT EXTRACTION
    # ==========================================================

    def _extract_detection_text(
        self,
        file_path: Path,
    ) -> str:
        """
        Extract text used by detection detectors.

        Current strategy
        ----------------
        - Native PDF text extraction
        - Maximum first 3 pages
        - OCR is NOT performed here
        - Scanned documents therefore return empty text

        This is intentional for the current detection phase.

        OCR fallback can be added later without changing the
        detector orchestration architecture.
        """

        extension = (
            file_path.suffix.lower()
        )

        # ------------------------------------------------------
        # NON-PDF
        # ------------------------------------------------------

        if extension != ".pdf":

            logger.info(
                "Skipping native PDF extraction: %s",
                file_path,
            )

            return ""

        # ------------------------------------------------------
        # OPEN PDF
        # ------------------------------------------------------

        document = fitz.open(
            str(file_path)
        )

        try:

            # --------------------------------------------------
            # LIMIT DETECTION EXTRACTION
            # --------------------------------------------------

            max_pages = min(
                3,
                len(document),
            )

            text_parts: list[str] = []

            # --------------------------------------------------
            # PAGE LOOP
            # --------------------------------------------------

            for page_number in range(
                max_pages
            ):

                try:

                    page = document.load_page(
                        page_number
                    )

                    page_text = page.get_text(
                        "text"
                    )

                    if page_text:

                        text_parts.append(
                            page_text
                        )

                except Exception as exc:

                    logger.warning(
                        "Unable to extract page %d "
                        "from %s: %s",
                        page_number + 1,
                        file_path,
                        exc,
                    )

                    continue

            # --------------------------------------------------
            # COMBINE TEXT
            # --------------------------------------------------

            return "\n".join(
                text_parts
            )

        finally:

            document.close()

    # ==========================================================
    # FINALIZE RESULT
    # ==========================================================

    @staticmethod
    def _finalize_result(
        result: DetectionResult,
        file_path: Path,
        start_time: float,
    ) -> DetectionResult:
        """
        Add final response metadata.

        Adds:
        - filename
        - SHA-256 document hash
        - total processing time
        """

        # ------------------------------------------------------
        # FILENAME
        # ------------------------------------------------------

        result.filename = (
            file_path.name
        )

        # ------------------------------------------------------
        # DOCUMENT HASH
        # ------------------------------------------------------

        if (
            result.evidence
            and result.evidence.metadata
        ):

            result.document_hash = (
                result.evidence.metadata.sha256
            )

        # ------------------------------------------------------
        # PROCESSING TIME
        # ------------------------------------------------------

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
    # ERROR RESULT
    # ==========================================================

    @staticmethod
    def _error_result(
        message: str,
        start_time: float,
    ) -> DetectionResult:
        """
        Create a safe DetectionResult for unexpected
        or invalid input errors.
        """

        processing_time = round(
            (
                perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        return DetectionResult(
            status=DetectionStatus.ERROR,
            detected=False,
            confidence=0.0,
            processing_time_ms=processing_time,
            error=message,
        )