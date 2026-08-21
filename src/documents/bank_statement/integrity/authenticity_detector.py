"""
Generic Bank Statement Authenticity Detector.

Evaluates whether a detected bank-statement PDF has structural
characteristics consistent with a normally generated statement.

Consumes evidence from:
- PDFIntegrityAnalyzer
- PageConsistencyAnalyzer
- TamperDetector

Important:
- No OCR.
- No bank-specific templates.
- No transaction extraction.
- No claim that the document is legally/forensically genuine.
- authenticity_score is a heuristic evidence score,
  NOT a probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .page_consistency_analyzer import PageConsistencyResult
from .pdf_integrity_analyzer import PDFIntegrityResult
from .tamper_detector import TamperDetectionResult


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True)
class AuthenticityDetectionResult:
    assessment: str
    authenticity_score: int

    positive_signals: tuple[str, ...]
    suspicious_signals: tuple[str, ...]
    warnings: tuple[str, ...]

    digital_text_present: bool
    consistent_page_structure: bool
    consistent_font_structure: bool

    suspicious_metadata: bool
    tamper_evidence_present: bool

    manual_review_required: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Detector
# ============================================================


class AuthenticityDetector:
    """
    Generic authenticity-evidence evaluator.

    The detector evaluates the overall document proportionally.

    Example:

        1 unusual page in 84 pages

    should not be treated the same as:

        4 unusual pages in 5 pages.
    """

    # ========================================================
    # Positive evidence
    # ========================================================

    SCORE_DIGITAL_TEXT = 20
    SCORE_ALL_PAGES_TEXT = 10

    SCORE_PAGE_CONSISTENCY = 20
    SCORE_FONT_CONSISTENCY = 15

    SCORE_NO_EMBEDDED_FILES = 5
    SCORE_NO_ANNOTATIONS = 5

    SCORE_LOW_TAMPER_RISK = 20

    # ========================================================
    # Outlier ratio thresholds
    # ========================================================

    MINOR_OUTLIER_RATIO = 0.05
    SIGNIFICANT_OUTLIER_RATIO = 0.20

    # ========================================================
    # Public API
    # ========================================================

    def detect(
        self,
        integrity: PDFIntegrityResult,
        consistency: PageConsistencyResult,
        tamper: TamperDetectionResult,
    ) -> AuthenticityDetectionResult:

        self._validate_inputs(
            integrity,
            consistency,
            tamper,
        )

        score = 0

        positive_signals: list[str] = []
        suspicious_signals: list[str] = []
        warnings: list[str] = []

        total_pages = max(
            consistency.pages_analyzed,
            1,
        )

        # ====================================================
        # 1. DIGITAL TEXT
        # ====================================================

        digital_text_present = (
            integrity.pages_with_text > 0
        )

        if digital_text_present:

            score += self.SCORE_DIGITAL_TEXT

            positive_signals.append(
                "Embedded digital text detected."
            )

        if (
            integrity.page_count > 0
            and integrity.pages_with_text
            == integrity.page_count
        ):

            score += self.SCORE_ALL_PAGES_TEXT

            positive_signals.append(
                "Digital text is present on all pages."
            )

        elif integrity.pages_without_text > 0:

            warnings.append(
                f"{integrity.pages_without_text} page(s) "
                "contain no embedded text."
            )

        # ====================================================
        # 2. STRUCTURAL CONSISTENCY
        # ====================================================

        structural_count = len(
            consistency.structural_outlier_pages
        )

        structural_ratio = (
            structural_count / total_pages
        )

        consistent_page_structure = (
            structural_ratio
            < self.MINOR_OUTLIER_RATIO
        )

        if structural_count == 0:

            score += self.SCORE_PAGE_CONSISTENCY

            positive_signals.append(
                "Page structure is consistent."
            )

        elif (
            structural_ratio
            < self.MINOR_OUTLIER_RATIO
        ):

            # Minor variation.
            # Still award most of the consistency evidence.

            score += 15

            positive_signals.append(
                "Page structure is highly consistent."
            )

            warnings.append(
                f"{structural_count} minor structural "
                "outlier page(s) detected."
            )

        elif (
            structural_ratio
            < self.SIGNIFICANT_OUTLIER_RATIO
        ):

            score += 5

            suspicious_signals.append(
                "Multiple structural page variations detected."
            )

        else:

            suspicious_signals.append(
                "Significant structural inconsistency detected."
            )

        # ====================================================
        # 3. FONT CONSISTENCY
        # ====================================================

        font_count = len(
            consistency.font_outlier_pages
        )

        font_ratio = (
            font_count / total_pages
        )

        consistent_font_structure = (
            font_ratio
            < self.MINOR_OUTLIER_RATIO
        )

        if font_count == 0:

            score += self.SCORE_FONT_CONSISTENCY

            positive_signals.append(
                "Font usage is consistent."
            )

        elif (
            font_ratio
            < self.MINOR_OUTLIER_RATIO
        ):

            # Small font variation can occur on headers,
            # cover pages, summary pages, or ending pages.

            score += 10

            positive_signals.append(
                "Font usage is broadly consistent."
            )

            warnings.append(
                f"Minor font variation detected on "
                f"{font_count} page(s)."
            )

        elif (
            font_ratio
            < self.SIGNIFICANT_OUTLIER_RATIO
        ):

            score += 5

            suspicious_signals.append(
                "Font variation detected across multiple pages."
            )

        else:

            suspicious_signals.append(
                "Significant font inconsistency detected."
            )

        # ====================================================
        # 4. EMBEDDED FILES
        # ====================================================

        if integrity.embedded_file_count == 0:

            score += self.SCORE_NO_EMBEDDED_FILES

            positive_signals.append(
                "No embedded files detected."
            )

        else:

            suspicious_signals.append(
                f"{integrity.embedded_file_count} embedded "
                "file(s) detected."
            )

        # ====================================================
        # 5. ANNOTATIONS
        # ====================================================

        if integrity.annotation_count == 0:

            score += self.SCORE_NO_ANNOTATIONS

            positive_signals.append(
                "No PDF annotations detected."
            )

        else:

            warnings.append(
                f"{integrity.annotation_count} PDF "
                "annotation(s) detected."
            )

        # ====================================================
        # 6. METADATA
        # ====================================================

        suspicious_metadata = (
            self._has_suspicious_metadata(
                tamper
            )
        )

        if suspicious_metadata:

            suspicious_signals.append(
                "Suspicious metadata indicator detected."
            )

        elif integrity.metadata_available:

            positive_signals.append(
                "PDF metadata is available."
            )

        else:

            warnings.append(
                "Descriptive PDF metadata is absent."
            )

        # Missing metadata is intentionally not penalized.

        # ====================================================
        # 7. TAMPER ASSESSMENT
        # ====================================================

        tamper_evidence_present = (
            tamper.tamper_suspected
        )

        if tamper.risk_level == "LOW":

            score += self.SCORE_LOW_TAMPER_RISK

            positive_signals.append(
                "No material tampering indicators detected."
            )

        elif tamper.risk_level == "MEDIUM":

            suspicious_signals.append(
                "Tamper analysis returned MEDIUM risk."
            )

            score -= 20

        elif tamper.risk_level == "HIGH":

            suspicious_signals.append(
                "Tamper analysis returned HIGH risk."
            )

            score -= 40

        # ====================================================
        # 8. CROSS-ANALYZER CHECK
        # ====================================================

        if (
            integrity.page_count
            != consistency.page_count
        ):

            suspicious_signals.append(
                "Analyzer page-count mismatch detected."
            )

            score -= 20

        # ====================================================
        # 9. SCORE BOUNDARY
        # ====================================================

        score = max(
            0,
            min(
                int(score),
                100,
            ),
        )

        # ====================================================
        # 10. ASSESSMENT
        # ====================================================

        assessment = self._assessment(
            score=score,
            tamper=tamper,
            structural_ratio=structural_ratio,
            font_ratio=font_ratio,
        )

        manual_review_required = (
            assessment
            in {
                "WEAK",
                "REVIEW_REQUIRED",
            }
        )

        # ====================================================
        # 11. LIMITATION
        # ====================================================

        warnings.append(
            "Authenticity assessment reflects structural "
            "evidence only and does not prove bank issuance."
        )

        # ====================================================
        # Result
        # ====================================================

        return AuthenticityDetectionResult(

            assessment=assessment,

            authenticity_score=score,

            positive_signals=tuple(
                positive_signals
            ),

            suspicious_signals=tuple(
                suspicious_signals
            ),

            warnings=tuple(
                self._deduplicate(
                    warnings
                )
            ),

            digital_text_present=(
                digital_text_present
            ),

            consistent_page_structure=(
                consistent_page_structure
            ),

            consistent_font_structure=(
                consistent_font_structure
            ),

            suspicious_metadata=(
                suspicious_metadata
            ),

            tamper_evidence_present=(
                tamper_evidence_present
            ),

            manual_review_required=(
                manual_review_required
            ),
        )

    # ========================================================
    # Assessment
    # ========================================================

    @staticmethod
    def _assessment(
        *,
        score: int,
        tamper: TamperDetectionResult,
        structural_ratio: float,
        font_ratio: float,
    ) -> str:

        # Material tamper evidence takes priority.

        if tamper.risk_level == "HIGH":
            return "REVIEW_REQUIRED"

        if tamper.tamper_suspected:
            return "REVIEW_REQUIRED"

        # Large structural inconsistency also requires review.

        if structural_ratio >= 0.20:
            return "REVIEW_REQUIRED"

        # ----------------------------------------------------
        # Evidence classification
        # ----------------------------------------------------

        if score >= 80:
            return "STRONG"

        if score >= 60:
            return "MODERATE"

        return "WEAK"

    # ========================================================
    # Metadata Evidence
    # ========================================================

    @staticmethod
    def _has_suspicious_metadata(
        tamper: TamperDetectionResult,
    ) -> bool:

        keywords = (
            "editing software",
            "photoshop",
            "gimp",
            "inkscape",
            "illustrator",
            "microsoft word",
            "libreoffice",
            "modification timestamp is earlier",
        )

        for signal in tamper.signals:

            normalized = signal.lower()

            if any(
                keyword in normalized
                for keyword in keywords
            ):
                return True

        return False

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_inputs(
        integrity,
        consistency,
        tamper,
    ) -> None:

        if not isinstance(
            integrity,
            PDFIntegrityResult,
        ):
            raise TypeError(
                "integrity must be a PDFIntegrityResult."
            )

        if not isinstance(
            consistency,
            PageConsistencyResult,
        ):
            raise TypeError(
                "consistency must be a "
                "PageConsistencyResult."
            )

        if not isinstance(
            tamper,
            TamperDetectionResult,
        ):
            raise TypeError(
                "tamper must be a TamperDetectionResult."
            )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _deduplicate(
        values: list[str],
    ) -> list[str]:

        return list(
            dict.fromkeys(
                values
            )
        )


# ============================================================
# Default Instance
# ============================================================


authenticity_detector = AuthenticityDetector()