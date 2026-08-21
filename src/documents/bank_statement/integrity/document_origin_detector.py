"""
Generic Document Origin Detector.

Detects explicit indicators that a PDF is a synthetic, sample,
demo, specimen, test, training, or otherwise non-production
document.

This detector uses embedded PDF text only.

Important:
- No OCR.
- No bank-specific rules.
- No transaction extraction.
- No tamper detection.
- No claim of bank issuance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import re

import fitz


# ============================================================
# Result Model
# ============================================================


@dataclass(frozen=True)
class DocumentOriginResult:
    """
    Result of document-origin analysis.

    origin_status:

        NORMAL
        SUSPICIOUS
        NON_PRODUCTION

    non_production_document:
        True when strong explicit evidence indicates that the
        document is synthetic/sample/test/specimen/demo data.

    confidence:
        Heuristic confidence in the origin classification.
        This is NOT a fraud probability.
    """

    origin_status: str

    non_production_document: bool

    confidence: float

    matched_indicators: tuple[str, ...]

    signals: tuple[str, ...]

    pages_checked: int

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Detector
# ============================================================


class DocumentOriginDetector:
    """
    Detect explicit non-production document indicators.

    This detector intentionally focuses on strong phrases,
    rather than isolated generic words.

    Example:

        "test"

    alone is too broad.

    But:

        "test document"
        "synthetic statement"
        "not a real bank statement"

    are much stronger signals.
    """

    MAX_PAGES_TO_CHECK = 10

    # ========================================================
    # Strong Indicators
    # ========================================================

    STRONG_PATTERNS = {

        "synthetic_test_document": (
            r"\bsynthetic\s+test\s+document\b"
        ),

        "synthetic_document": (
            r"\bsynthetic\s+document\b"
        ),

        "synthetic_bank_statement": (
            r"\bsynthetic\s+(?:bank\s+)?statement\b"
        ),

        "not_real_bank_statement": (
            r"\bnot\s+(?:a\s+)?real\s+bank\s+statement\b"
        ),

        "not_real_statement": (
            r"\bnot\s+(?:a\s+)?real\s+statement\b"
        ),

        "sample_bank_statement": (
            r"\bsample\s+bank\s+statement\b"
        ),

        "sample_statement": (
            r"\bsample\s+(?:account\s+)?statement\b"
        ),

        "specimen_document": (
            r"\bspecimen\s+document\b"
        ),

        "specimen_statement": (
            r"\bspecimen\s+(?:bank\s+)?statement\b"
        ),

        "demo_document": (
            r"\bdemo(?:nstration)?\s+document\b"
        ),

        "demo_statement": (
            r"\bdemo(?:nstration)?\s+(?:bank\s+)?statement\b"
        ),

        "test_document": (
            r"\btest\s+document\b"
        ),

        "test_statement": (
            r"\btest\s+(?:bank\s+)?statement\b"
        ),

        "training_document": (
            r"\btraining\s+document\b"
        ),

        "training_sample": (
            r"\btraining\s+sample\b"
        ),

        "not_valid": (
            r"\bnot\s+valid\b"
        ),

        "for_testing_only": (
            r"\bfor\s+(?:testing|test)\s+(?:purposes?\s+)?only\b"
        ),

        "qa_only": (
            r"\b(?:for\s+)?qa\s+(?:purposes?\s+)?only\b"
        ),

        "detector_qa": (
            r"\bdetector\s+qa\b"
        ),

        "fictional_institution": (
            r"\bfictional\s+(?:bank|financial\s+institution|institution)\b"
        ),
    }

    # ========================================================
    # Weaker contextual indicators
    # ========================================================

    WEAK_PATTERNS = {

        "dummy_data": (
            r"\bdummy\s+data\b"
        ),

        "dummy_account": (
            r"\bdummy\s+account\b"
        ),

        "example_document": (
            r"\bexample\s+document\b"
        ),

        "example_statement": (
            r"\bexample\s+(?:bank\s+)?statement\b"
        ),

        "mock_document": (
            r"\bmock\s+document\b"
        ),

        "mock_statement": (
            r"\bmock\s+(?:bank\s+)?statement\b"
        ),

        "generated_for_testing": (
            r"\bgenerated\s+for\s+testing\b"
        ),
    }

    # ========================================================
    # Public API
    # ========================================================

    def detect(
        self,
        file_bytes: bytes,
    ) -> DocumentOriginResult:

        if not file_bytes:

            raise ValueError(
                "PDF bytes are required."
            )

        # ====================================================
        # Open PDF
        # ====================================================

        try:

            document = fitz.open(
                stream=BytesIO(file_bytes),
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                "Unable to open PDF for document-origin "
                "analysis."
            ) from exc

        try:

            # ------------------------------------------------
            # Password protected
            # ------------------------------------------------

            if document.needs_pass:

                return DocumentOriginResult(

                    origin_status="SUSPICIOUS",

                    non_production_document=False,

                    confidence=0.0,

                    matched_indicators=(),

                    signals=(
                        "Document origin could not be "
                        "evaluated because the PDF requires "
                        "a password.",
                    ),

                    pages_checked=0,
                )

            # ------------------------------------------------
            # Empty PDF
            # ------------------------------------------------

            if document.page_count <= 0:

                return DocumentOriginResult(

                    origin_status="SUSPICIOUS",

                    non_production_document=False,

                    confidence=0.0,

                    matched_indicators=(),

                    signals=(
                        "Document contains no pages.",
                    ),

                    pages_checked=0,
                )

            # =================================================
            # Extract embedded text
            # =================================================

            text, pages_checked = (
                self._extract_text(
                    document
                )
            )

            normalized = (
                self._normalize_text(
                    text
                )
            )

            # =================================================
            # Strong matches
            # =================================================

            strong_matches = (
                self._find_matches(
                    normalized,
                    self.STRONG_PATTERNS,
                )
            )

            # =================================================
            # Weak matches
            # =================================================

            weak_matches = (
                self._find_matches(
                    normalized,
                    self.WEAK_PATTERNS,
                )
            )

            # =================================================
            # Classification
            # =================================================

            signals: list[str] = []

            matched = (
                strong_matches
                + weak_matches
            )

            # ------------------------------------------------
            # Strong explicit evidence
            # ------------------------------------------------

            if strong_matches:

                signals.append(
                    "Explicit non-production document "
                    "indicator detected."
                )

                if len(
                    strong_matches
                ) >= 2:

                    signals.append(
                        "Multiple independent synthetic/test "
                        "document indicators detected."
                    )

                return DocumentOriginResult(

                    origin_status=(
                        "NON_PRODUCTION"
                    ),

                    non_production_document=True,

                    confidence=self._strong_confidence(
                        len(
                            strong_matches
                        )
                    ),

                    matched_indicators=tuple(
                        matched
                    ),

                    signals=tuple(
                        signals
                    ),

                    pages_checked=pages_checked,
                )

            # ------------------------------------------------
            # Multiple weaker indicators
            # ------------------------------------------------

            if len(
                weak_matches
            ) >= 2:

                signals.append(
                    "Multiple non-production contextual "
                    "indicators detected."
                )

                return DocumentOriginResult(

                    origin_status=(
                        "SUSPICIOUS"
                    ),

                    non_production_document=False,

                    confidence=0.60,

                    matched_indicators=tuple(
                        weak_matches
                    ),

                    signals=tuple(
                        signals
                    ),

                    pages_checked=pages_checked,
                )

            # ------------------------------------------------
            # Single weak indicator
            # ------------------------------------------------

            if len(
                weak_matches
            ) == 1:

                signals.append(
                    "A possible non-production contextual "
                    "indicator was detected."
                )

                return DocumentOriginResult(

                    origin_status=(
                        "SUSPICIOUS"
                    ),

                    non_production_document=False,

                    confidence=0.35,

                    matched_indicators=tuple(
                        weak_matches
                    ),

                    signals=tuple(
                        signals
                    ),

                    pages_checked=pages_checked,
                )

            # ------------------------------------------------
            # No indicators
            # ------------------------------------------------

            return DocumentOriginResult(

                origin_status="NORMAL",

                non_production_document=False,

                confidence=0.0,

                matched_indicators=(),

                signals=(
                    "No explicit synthetic, sample, test, "
                    "demo, or specimen indicators detected.",
                ),

                pages_checked=pages_checked,
            )

        finally:

            document.close()

    # ========================================================
    # Embedded Text Extraction
    # ========================================================

    def _extract_text(
        self,
        document: fitz.Document,
    ) -> tuple[str, int]:

        pages_to_check = min(
            document.page_count,
            self.MAX_PAGES_TO_CHECK,
        )

        extracted: list[str] = []

        for index in range(
            pages_to_check
        ):

            try:

                text = document[
                    index
                ].get_text(
                    "text"
                )

            except Exception:

                text = ""

            if text:

                extracted.append(
                    text
                )

        return (
            "\n".join(
                extracted
            ),
            pages_to_check,
        )

    # ========================================================
    # Pattern Matching
    # ========================================================

    @staticmethod
    def _find_matches(
        text: str,
        patterns: dict[str, str],
    ) -> list[str]:

        matches: list[str] = []

        for name, pattern in (
            patterns.items()
        ):

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                matches.append(
                    name
                )

        return matches

    # ========================================================
    # Text Normalization
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        if not text:

            return ""

        text = (
            text
            .replace("\x00", " ")
            .replace("\r", "\n")
            .lower()
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ========================================================
    # Confidence
    # ========================================================

    @staticmethod
    def _strong_confidence(
        match_count: int,
    ) -> float:

        if match_count >= 4:
            return 1.0

        if match_count == 3:
            return 0.95

        if match_count == 2:
            return 0.90

        return 0.80


# ============================================================
# Default Reusable Instance
# ============================================================


document_origin_detector = (
    DocumentOriginDetector()
)