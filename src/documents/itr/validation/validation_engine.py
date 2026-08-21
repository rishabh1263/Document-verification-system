"""
ITR Validation Engine

Production-oriented orchestration layer for ITR validation.

Validation stages:
    1. Document integrity
    2. Required ITR content
    3. Internal consistency
    4. Final decision

Important:
    - This engine does NOT perform document detection.
    - This engine does NOT perform OCR.
    - This engine does NOT perform external PAN verification.
    - Integrity validation is implemented locally so this module does not
      depend on a non-existent ".integrity._validator" module.
    - Internal consistency is delegated to the existing consistency validator.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

from .models import (
    ContentResult,
    ConsistencyResult,
    IntegrityResult,
    ValidationDecision,
    ValidationEvidence,
    ValidationInput,
    ValidationResult,
    ValidationStatus,
)

# Consistency validator lives directly beside this file:
# src/documents/itr/validation/consistency_validator.py
try:
    from .consistency_validator import ConsistencyValidator
except ImportError:
    from src.documents.itr.validation.consistency_validator import (
        ConsistencyValidator,
    )

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Main orchestration engine for ITR validation.

    The engine deliberately keeps each validation stage independent.
    A failure in one stage does not prevent the remaining stages from
    running when enough information is available.
    """

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def validate(
        self,
        validation_input: ValidationInput,
    ) -> ValidationResult:
        """
        Validate an ITR document from a ValidationInput object.
        """
        start_time = perf_counter()

        file_path = Path(validation_input.file_path)

        try:
            integrity = self._validate_integrity(file_path)

            if not integrity.valid_file or not integrity.valid_pdf:
                content = ContentResult(
                    required_content_present=False,
                    score=0.0,
                    missing_items=["Readable ITR document"],
                    reasons=[
                        "Content validation skipped because the PDF "
                        "could not be validated."
                    ],
                )

                consistency = ConsistencyResult(
                    consistent=False,
                    score=0.0,
                    inconsistencies=[
                        "Consistency validation skipped because the "
                        "document could not be read."
                    ],
                    reasons=[
                        "Document text was unavailable."
                    ],
                )

                return self._build_result(
                    validation_input=validation_input,
                    integrity=integrity,
                    content=content,
                    consistency=consistency,
                    status=ValidationStatus.FAILED,
                    start_time=start_time,
                )

            text = self._extract_pdf_text(file_path)

            content = self._validate_content(text)
            consistency = self._validate_consistency(text)

            status = ValidationStatus.SUCCESS

            return self._build_result(
                validation_input=validation_input,
                integrity=integrity,
                content=content,
                consistency=consistency,
                status=status,
                start_time=start_time,
            )

        except Exception as exc:
            logger.exception(
                "ITR validation engine failed for %s",
                file_path,
            )

            processing_time_ms = round(
                (perf_counter() - start_time) * 1000,
                2,
            )

            return ValidationResult(
                status=ValidationStatus.ERROR,
                decision=ValidationDecision.UNKNOWN,
                valid=False,
                filename=file_path.name,
                confidence=0.0,
                processing_time_ms=processing_time_ms,
                evidence=ValidationEvidence(),
                reasons=[
                    f"Validation engine error: {exc}"
                ],
                error=str(exc),
            )

    def validate_file(
        self,
        file_path: str,
        detection_confidence: float = 0.0,
    ) -> ValidationResult:
        """
        Convenience API for callers that have a file path directly.
        """
        request = ValidationInput(
            file_path=file_path,
            detection_confidence=detection_confidence,
        )
        return self.validate(request)

    def run(
        self,
        file_path: str,
        detection_confidence: float = 0.0,
    ) -> ValidationResult:
        """
        Backward-compatible alias for validate_file().
        """
        return self.validate_file(
            file_path=file_path,
            detection_confidence=detection_confidence,
        )

    # ==========================================================
    # INTEGRITY
    # ==========================================================

    def _validate_integrity(
        self,
        file_path: Path,
    ) -> IntegrityResult:
        """
        Validate technical health of the input document.

        Checks:
            - file exists
            - file is a regular file
            - file is a PDF
            - PDF can be opened
            - PDF is not encrypted
            - PDF has at least one page
            - PDF pages can be read
            - SHA-256 is calculated
        """
        start_time = perf_counter()

        reasons: List[str] = []

        valid_file = False
        valid_pdf = False
        readable = False
        encrypted = False
        corrupted = False
        page_count = 0
        file_size = 0
        sha256 = ""

        try:
            if not file_path.exists():
                reasons.append("File does not exist")
                return self._integrity_result(
                    valid_file=False,
                    valid_pdf=False,
                    readable=False,
                    encrypted=False,
                    corrupted=False,
                    page_count=0,
                    file_size=0,
                    sha256="",
                    score=0.0,
                    reasons=reasons,
                    start_time=start_time,
                )

            if not file_path.is_file():
                reasons.append("Path is not a regular file")
                return self._integrity_result(
                    valid_file=False,
                    valid_pdf=False,
                    readable=False,
                    encrypted=False,
                    corrupted=False,
                    page_count=0,
                    file_size=0,
                    sha256="",
                    score=0.0,
                    reasons=reasons,
                    start_time=start_time,
                )

            valid_file = True
            file_size = file_path.stat().st_size

            if file_size <= 0:
                reasons.append("File is empty")
                return self._integrity_result(
                    valid_file=True,
                    valid_pdf=False,
                    readable=False,
                    encrypted=False,
                    corrupted=True,
                    page_count=0,
                    file_size=file_size,
                    sha256="",
                    score=0.0,
                    reasons=reasons,
                    start_time=start_time,
                )

            sha256 = self._calculate_sha256(file_path)

            if file_path.suffix.lower() != ".pdf":
                reasons.append("File extension is not PDF")
                return self._integrity_result(
                    valid_file=True,
                    valid_pdf=False,
                    readable=False,
                    encrypted=False,
                    corrupted=False,
                    page_count=0,
                    file_size=file_size,
                    sha256=sha256,
                    score=0.25,
                    reasons=reasons,
                    start_time=start_time,
                )

            if fitz is None:
                reasons.append(
                    "PyMuPDF is not installed; PDF validation unavailable"
                )
                return self._integrity_result(
                    valid_file=True,
                    valid_pdf=False,
                    readable=False,
                    encrypted=False,
                    corrupted=False,
                    page_count=0,
                    file_size=file_size,
                    sha256=sha256,
                    score=0.25,
                    reasons=reasons,
                    start_time=start_time,
                )

            try:
                document = fitz.open(str(file_path))
            except Exception as exc:
                corrupted = True
                reasons.append(
                    f"PDF could not be opened: {exc}"
                )
                return self._integrity_result(
                    valid_file=True,
                    valid_pdf=False,
                    readable=False,
                    encrypted=False,
                    corrupted=True,
                    page_count=0,
                    file_size=file_size,
                    sha256=sha256,
                    score=0.25,
                    reasons=reasons,
                    start_time=start_time,
                )

            valid_pdf = True
            encrypted = bool(getattr(document, "is_encrypted", False))
            page_count = len(document)

            if encrypted:
                reasons.append("PDF is encrypted")
                document.close()

                return self._integrity_result(
                    valid_file=True,
                    valid_pdf=True,
                    readable=False,
                    encrypted=True,
                    corrupted=False,
                    page_count=page_count,
                    file_size=file_size,
                    sha256=sha256,
                    score=0.50,
                    reasons=reasons,
                    start_time=start_time,
                )

            if page_count <= 0:
                corrupted = True
                reasons.append("PDF contains no pages")
                document.close()

                return self._integrity_result(
                    valid_file=True,
                    valid_pdf=True,
                    readable=False,
                    encrypted=False,
                    corrupted=True,
                    page_count=0,
                    file_size=file_size,
                    sha256=sha256,
                    score=0.50,
                    reasons=reasons,
                    start_time=start_time,
                )

            # Force page access so a damaged PDF is not treated as healthy
            # merely because fitz.open() succeeded.
            for page_index in range(page_count):
                page = document.load_page(page_index)
                _ = page.rect
                _ = page.get_text("text")

            document.close()

            readable = True

            reasons.extend(
                [
                    "File exists and is readable",
                    "PDF structure is valid",
                    f"PDF contains {page_count} page(s)",
                    "SHA-256 hash calculated successfully",
                ]
            )

            score = 1.0

            return self._integrity_result(
                valid_file=valid_file,
                valid_pdf=valid_pdf,
                readable=readable,
                encrypted=encrypted,
                corrupted=corrupted,
                page_count=page_count,
                file_size=file_size,
                sha256=sha256,
                score=score,
                reasons=reasons,
                start_time=start_time,
            )

        except Exception as exc:
            logger.exception(
                "Integrity validation failed for %s",
                file_path,
            )

            corrupted = True
            reasons.append(
                f"Integrity validation error: {exc}"
            )

            return self._integrity_result(
                valid_file=valid_file,
                valid_pdf=valid_pdf,
                readable=readable,
                encrypted=encrypted,
                corrupted=corrupted,
                page_count=page_count,
                file_size=file_size,
                sha256=sha256,
                score=0.0,
                reasons=reasons,
                start_time=start_time,
            )

    @staticmethod
    def _calculate_sha256(
        file_path: Path,
    ) -> str:
        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    @classmethod
    def _integrity_result(
        cls,
        *,
        valid_file: bool,
        valid_pdf: bool,
        readable: bool,
        encrypted: bool,
        corrupted: bool,
        page_count: int,
        file_size: int,
        sha256: str,
        score: float,
        reasons: List[str],
        start_time: float,
    ) -> IntegrityResult:
        return IntegrityResult(
            valid_file=valid_file,
            valid_pdf=valid_pdf,
            readable=readable,
            encrypted=encrypted,
            corrupted=corrupted,
            page_count=page_count,
            file_size=file_size,
            sha256=sha256,
            score=max(0.0, min(1.0, score)),
            processing_time_ms=round(
                (perf_counter() - start_time) * 1000,
                2,
            ),
            reasons=reasons,
        )

    # ==========================================================
    # PDF TEXT EXTRACTION
    # ==========================================================

    @staticmethod
    def _extract_pdf_text(
        file_path: Path,
    ) -> str:
        """
        Extract native PDF text.

        OCR is intentionally NOT performed here.
        OCR belongs to the extraction/preprocessing layer.
        """
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is required for ITR validation"
            )

        document = fitz.open(str(file_path))

        try:
            pages: List[str] = []

            for page in document:
                pages.append(
                    page.get_text("text")
                )

            return "\n".join(pages)

        finally:
            document.close()

    # ==========================================================
    # CONTENT
    # ==========================================================

    @classmethod
    def _validate_content(
        cls,
        text: str,
    ) -> ContentResult:
        """
        Validate the presence of essential ITR information.

        This is a presence check, not an authenticity check.
        """
        start_time = perf_counter()

        normalized = cls._normalize_text(text)

        checks = [
            (
                "Assessment Year",
                cls._contains_any(
                    normalized,
                    [
                        r"\bassessment\s+year\b",
                        r"\basst\.?\s+year\b",
                        r"\ba\.?\s*y\.?\b",
                    ],
                ),
            ),
            (
                "PAN",
                bool(
                    re.search(
                        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                        normalized,
                        re.IGNORECASE,
                    )
                ),
            ),
            (
                "Taxpayer information",
                cls._contains_any(
                    normalized,
                    [
                        r"\bname\b",
                        r"\btaxpayer\b",
                        r"\bassessee\b",
                    ],
                ),
            ),
            (
                "Income information",
                cls._contains_any(
                    normalized,
                    [
                        r"\btotal\s+income\b",
                        r"\bgross\s+total\s+income\b",
                        r"\bincome\s+from\b",
                    ],
                ),
            ),
            (
                "Tax computation",
                cls._contains_any(
                    normalized,
                    [
                        r"\btax\s+on\s+total\s+income\b",
                        r"\btax\s+computation\b",
                        r"\bnet\s+tax\b",
                        r"\btax\s+payable\b",
                    ],
                ),
            ),
            (
                "Verification",
                cls._contains_any(
                    normalized,
                    [
                        r"\bverification\b",
                        r"\bverified\b",
                        r"\bverification\s+under\b",
                    ],
                ),
            ),
            (
                "Acknowledgement",
                cls._contains_any(
                    normalized,
                    [
                        r"\backnowledgement\b",
                        r"\be[-\s]?filing\s+acknowledgement\b",
                    ],
                ),
            ),
        ]

        missing_items = [
            name
            for name, present in checks
            if not present
        ]

        present_count = sum(
            1
            for _, present in checks
            if present
        )

        score = (
            present_count / len(checks)
            if checks
            else 0.0
        )

        required_content_present = (
            len(missing_items) == 0
        )

        reasons: List[str] = []

        if required_content_present:
            reasons.append(
                "All required ITR content indicators are present"
            )
        else:
            reasons.append(
                "Some required ITR content indicators are missing"
            )

            for item in missing_items:
                reasons.append(
                    f"Missing content: {item}"
                )

        return ContentResult(
            required_content_present=required_content_present,
            assessment_year_present=checks[0][1],
            pan_present=checks[1][1],
            taxpayer_information_present=checks[2][1],
            income_information_present=checks[3][1],
            tax_computation_present=checks[4][1],
            verification_present=checks[5][1],
            acknowledgement_present=checks[6][1],
            score=round(score, 3),
            missing_items=missing_items,
            reasons=reasons,
            processing_time_ms=round(
                (perf_counter() - start_time) * 1000,
                2,
            ),
        )

    @staticmethod
    def _contains_any(
        text: str,
        patterns: List[str],
    ) -> bool:
        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\xa0", " ")
        text = (
            text.replace("–", "-")
            .replace("—", "-")
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        return text.strip()

    # ==========================================================
    # CONSISTENCY
    # ==========================================================

    @staticmethod
    def _validate_consistency(
        text: str,
    ) -> ConsistencyResult:
        """
        Delegate internal consistency validation to the existing
        consistency validator.

        The consistency validator already handles:
            - Assessment Year
            - taxpayer PAN
            - income
            - tax
            - acknowledgement

        and returns the structured ConsistencyResult expected by
        the validation models.
        """
        validator = ConsistencyValidator()
        return validator.validate(text)

    # ==========================================================
    # FINAL DECISION
    # ==========================================================

    @classmethod
    def _build_result(
        cls,
        *,
        validation_input: ValidationInput,
        integrity: IntegrityResult,
        content: ContentResult,
        consistency: ConsistencyResult,
        status: ValidationStatus,
        start_time: float,
    ) -> ValidationResult:
        """
        Combine all validation evidence into one final result.
        """
        reasons: List[str] = []

        # ------------------------------------------------------
        # Integrity
        # ------------------------------------------------------

        if integrity.valid_file:
            reasons.append(
                "File is present"
            )
        else:
            reasons.append(
                "File is invalid or missing"
            )

        if integrity.valid_pdf:
            reasons.append(
                "PDF structure is valid"
            )
        else:
            reasons.append(
                "PDF structure is invalid"
            )

        if integrity.readable:
            reasons.append(
                "Document is readable"
            )

        # ------------------------------------------------------
        # Content
        # ------------------------------------------------------

        if content.required_content_present:
            reasons.append(
                "Required ITR content is present"
            )
        else:
            reasons.append(
                "Required ITR content is incomplete"
            )

        # ------------------------------------------------------
        # Consistency
        # ------------------------------------------------------

        if consistency.consistent:
            reasons.append(
                "Internal ITR information is consistent"
            )
        else:
            reasons.append(
                "Internal ITR information is inconsistent"
            )

        # ------------------------------------------------------
        # Decision
        # ------------------------------------------------------

        decision = cls._make_decision(
            integrity=integrity,
            content=content,
            consistency=consistency,
            detection_confidence=validation_input.detection_confidence,
        )

        valid = (
            decision == ValidationDecision.VALID
        )

        confidence = cls._calculate_confidence(
            integrity=integrity,
            content=content,
            consistency=consistency,
            detection_confidence=validation_input.detection_confidence,
        )

        processing_time_ms = round(
            (perf_counter() - start_time) * 1000,
            2,
        )

        return ValidationResult(
            status=status,
            decision=decision,
            valid=valid,
            filename=Path(
                validation_input.file_path
            ).name,
            document_hash=integrity.sha256,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            evidence=ValidationEvidence(
                integrity=integrity,
                content=content,
                consistency=consistency,
            ),
            reasons=reasons,
            error=None,
        )

    @staticmethod
    def _make_decision(
        *,
        integrity: IntegrityResult,
        content: ContentResult,
        consistency: ConsistencyResult,
        detection_confidence: float,
    ) -> ValidationDecision:
        """
        Conservative final decision.

        VALID:
            technically healthy document,
            required ITR content present,
            internally consistent.

        INVALID:
            technically invalid/corrupted document, or clearly
            incomplete content.

        REVIEW:
            technically healthy but internal consistency/content
            evidence is not strong enough for automatic approval.
        """
        if (
            not integrity.valid_file
            or not integrity.valid_pdf
            or integrity.corrupted
            or integrity.encrypted
        ):
            return ValidationDecision.INVALID

        if not integrity.readable:
            return ValidationDecision.INVALID

        if not content.required_content_present:
            return ValidationDecision.REVIEW

        if not consistency.consistent:
            return ValidationDecision.REVIEW

        # Detection confidence is evidence from the Detection Engine,
        # not a replacement for validation.
        if detection_confidence > 0.0 and detection_confidence < 0.50:
            return ValidationDecision.REVIEW

        return ValidationDecision.VALID

    @staticmethod
    def _calculate_confidence(
        *,
        integrity: IntegrityResult,
        content: ContentResult,
        consistency: ConsistencyResult,
        detection_confidence: float,
    ) -> float:
        """
        Calculate a normalized confidence score.

        Weighting:
            Integrity       20%
            Content         25%
            Consistency     35%
            Detection       20%

        If detection confidence is unavailable (0.0), the remaining
        evidence is normalized rather than artificially penalized.
        """
        integrity_score = integrity.score
        content_score = content.score
        consistency_score = consistency.score

        if detection_confidence > 0.0:
            confidence = (
                integrity_score * 0.20
                + content_score * 0.25
                + consistency_score * 0.35
                + detection_confidence * 0.20
            )
        else:
            denominator = 0.20 + 0.25 + 0.35

            confidence = (
                integrity_score * 0.20
                + content_score * 0.25
                + consistency_score * 0.35
            ) / denominator

        return round(
            max(
                0.0,
                min(1.0, confidence),
            ),
            3,
        )


# ============================================================
# MODULE INSTANCE
# ============================================================

validation_engine = ValidationEngine()