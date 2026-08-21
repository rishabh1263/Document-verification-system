"""
Generic Bank Statement Detector.

Second stage of the bank-statement detection pipeline.

Responsibilities:
- inspect embedded text from a valid PDF
- determine whether enough digital text exists
- identify generic bank-statement terminology
- calculate a lightweight detection confidence
- decide whether OCR is required

Important:
This module does NOT:
- perform OCR
- detect tampering
- verify authenticity
- extract transactions
- identify a specific bank
- perform loan eligibility logic

Expected pipeline:

    FileDetector
        ↓
    BankStatementDetector
        ↓
    Integrity / Tamper / Authenticity
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
class BankStatementDetectionResult:
    """
    Result of generic bank-statement detection.

    confidence:
        Heuristic document-classification confidence from
        0.0 to 1.0.

    document_mode:
        digital_pdf
        scanned_or_image_pdf

    requires_ocr:
        True when embedded PDF text is insufficient for
        reliable classification.
    """

    is_bank_statement: bool | None
    confidence: float

    document_mode: str
    requires_ocr: bool

    signals: tuple[str, ...]

    text_length: int
    pages_checked: int

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Detector
# ============================================================


class BankStatementDetector:
    """
    Detect whether a PDF appears to be a bank statement using
    embedded digital text only.

    This detector deliberately avoids:
    - OCR
    - bank-specific templates
    - account-number extraction
    - transaction parsing

    The goal is lightweight routing, not deep document
    understanding.
    """

    # ========================================================
    # Configuration
    # ========================================================

    # Only inspect the first few pages during document
    # classification. This keeps detection fast even for
    # statements containing hundreds of pages.

    MAX_PAGES_TO_CHECK = 5

    # If extracted embedded text is below this threshold,
    # classification is considered unreliable and OCR should
    # be requested by a later pipeline stage.

    MIN_DIGITAL_TEXT_LENGTH = 120

    # Minimum confidence required to classify the document as
    # a bank statement.

    BANK_STATEMENT_THRESHOLD = 0.70

    # ========================================================
    # Generic signal groups
    # ========================================================

    # Each group represents one type of evidence.
    #
    # Multiple matching words inside the same group do not
    # repeatedly increase confidence. This prevents documents
    # containing repeated words such as "balance" from
    # artificially receiving a high score.

    SIGNAL_GROUPS = {

        # ----------------------------------------------------
        # Explicit statement terminology
        # ----------------------------------------------------

        "statement": {
            "weight": 0.30,
            "message": (
                "Statement terminology detected."
            ),
            "patterns": (
                r"\baccount\s+statement\b",
                r"\bbank\s+statement\b",
                r"\bstatement\s+of\s+account\b",
                r"\bstatement\s+period\b",
                r"\baccount\s+summary\b",
            ),
        },

        # ----------------------------------------------------
        # Account terminology
        # ----------------------------------------------------

        "account": {
            "weight": 0.20,
            "message": (
                "Account-related terminology detected."
            ),
            "patterns": (
                r"\baccount\s*(?:number|no\.?|#)\b",
                r"\ba\/c\s*(?:number|no\.?|#)\b",
                r"\baccount\s+type\b",
                r"\baccount\s+name\b",
                r"\baccount\s+holder\b",
                r"\bcustomer\s+id\b",
            ),
        },

        # ----------------------------------------------------
        # Transaction terminology
        # ----------------------------------------------------

        "transaction": {
            "weight": 0.20,
            "message": (
                "Transaction terminology detected."
            ),
            "patterns": (
                r"\btransaction\s+date\b",
                r"\btransaction\s+details\b",
                r"\btransaction\s+description\b",
                r"\btransactions?\b",
                r"\bparticulars\b",
                r"\bnarration\b",
                r"\bwithdrawals?\b",
                r"\bdeposits?\b",
                r"\bdebit\b",
                r"\bcredit\b",
            ),
        },

        # ----------------------------------------------------
        # Balance terminology
        # ----------------------------------------------------

        "balance": {
            "weight": 0.20,
            "message": (
                "Balance terminology detected."
            ),
            "patterns": (
                r"\bopening\s+balance\b",
                r"\bclosing\s+balance\b",
                r"\bavailable\s+balance\b",
                r"\bcurrent\s+balance\b",
                r"\bbalance\b",
            ),
        },

        # ----------------------------------------------------
        # General banking terminology
        # ----------------------------------------------------

        "banking": {
            "weight": 0.10,
            "message": (
                "Banking terminology detected."
            ),
            "patterns": (
                r"\bifsc\b",
                r"\bbranch\b",
                r"\bcheque\b",
                r"\bupi\b",
                r"\bneft\b",
                r"\brtgs\b",
                r"\bimps\b",
                r"\batm\b",
                r"\binterest\b",
            ),
        },
    }

    # ========================================================
    # NON-BANK-STATEMENT EXCLUSION SIGNALS
    # ========================================================

    NON_BANK_STATEMENT_GROUPS = {
        "income_tax_return": (
            r"\bincome\s+tax\b", r"\bincome\s+tax\s+return\b",
            r"\bassessment\s+year\b", r"\bprevious\s+year\b",
            r"\btotal\s+income\b", r"\bgross\s+total\s+income\b",
            r"\btaxable\s+income\b", r"\btax\s+payable\b",
            r"\backnowledg(?:e?ment|ement)\s*(?:no|number)\b",
            r"\bitr\s*(?:1|2|3|4|5|6|7)\b", r"\bform\s+itr\b",
        ),
        "form_16": (r"\bform\s*16\b", r"\btds\b", r"\btax\s+deducted\s+at\s+source\b"),
        "gst": (r"\bgoods\s+and\s+services\s+tax\b", r"\bgstin\b", r"\binput\s+tax\s+credit\b"),
    }

    # ========================================================
    # Public API
    # ========================================================

    def detect(
        self,
        file_bytes: bytes,
    ) -> BankStatementDetectionResult:
        """
        Determine whether PDF bytes appear to represent a bank
        statement.

        FileDetector should normally run before this method.
        """

        if not file_bytes:
            raise ValueError(
                "PDF bytes are required."
            )

        # ----------------------------------------------------
        # Open PDF
        # ----------------------------------------------------

        try:
            document = fitz.open(
                stream=BytesIO(file_bytes),
                filetype="pdf",
            )

        except Exception as exc:
            raise ValueError(
                "Unable to open PDF for bank-statement "
                "detection."
            ) from exc

        try:

            # ------------------------------------------------
            # Password-protected PDF
            # ------------------------------------------------

            if document.needs_pass:

                return BankStatementDetectionResult(
                    is_bank_statement=None,
                    confidence=0.0,
                    document_mode=(
                        "scanned_or_image_pdf"
                    ),
                    requires_ocr=False,
                    signals=(
                        "PDF requires a password before "
                        "document classification can continue.",
                    ),
                    text_length=0,
                    pages_checked=0,
                )

            # ------------------------------------------------
            # Empty PDF
            # ------------------------------------------------

            if document.page_count <= 0:

                return BankStatementDetectionResult(
                    is_bank_statement=False,
                    confidence=0.0,
                    document_mode="digital_pdf",
                    requires_ocr=False,
                    signals=(
                        "PDF contains no pages.",
                    ),
                    text_length=0,
                    pages_checked=0,
                )

            # ------------------------------------------------
            # Extract embedded text
            # ------------------------------------------------

            text, pages_checked = (
                self._extract_embedded_text(
                    document
                )
            )

            normalized_text = (
                self._normalize_text(
                    text
                )
            )

            text_length = len(
                normalized_text
            )

            # =================================================
            # OCR ROUTING
            # =================================================

            if (
                text_length
                < self.MIN_DIGITAL_TEXT_LENGTH
            ):

                return BankStatementDetectionResult(
                    is_bank_statement=None,
                    confidence=0.0,
                    document_mode=(
                        "scanned_or_image_pdf"
                    ),
                    requires_ocr=True,
                    signals=(
                        "Insufficient embedded PDF text "
                        "for reliable classification.",
                    ),
                    text_length=text_length,
                    pages_checked=pages_checked,
                )

            # =================================================
            # SIGNAL DETECTION
            # =================================================

            confidence, signals = (
                self._calculate_confidence(
                    normalized_text
                )
            )

            negative_groups = self._detect_negative_groups(
                normalized_text
            )

            if negative_groups:
                signals.append(
                    "Non-bank financial document markers detected: "
                    + ", ".join(negative_groups) + "."
                )

            composite_pass = self._passes_composite_gate(
                normalized_text
            )

            if not composite_pass:
                signals.append(
                    "Required bank-statement evidence combination was not satisfied."
                )

            if negative_groups:
                is_bank_statement = False
                confidence = min(confidence, 0.39)
            else:
                is_bank_statement = (
                    composite_pass
                    and confidence >= self.BANK_STATEMENT_THRESHOLD
                )

            return BankStatementDetectionResult(
                is_bank_statement=(
                    is_bank_statement
                ),

                confidence=round(
                    confidence,
                    2,
                ),

                document_mode="digital_pdf",

                requires_ocr=False,

                signals=tuple(
                    signals
                ),

                text_length=text_length,

                pages_checked=pages_checked,
            )

        finally:
            document.close()

    # ========================================================
    # Embedded Text Extraction
    # ========================================================

    def _extract_embedded_text(
        self,
        document: fitz.Document,
    ) -> tuple[str, int]:
        """
        Extract embedded text from a limited number of pages.

        This is NOT OCR.

        PyMuPDF simply reads text already stored inside the PDF.
        """

        pages_to_check = min(
            document.page_count,
            self.MAX_PAGES_TO_CHECK,
        )

        extracted_text: list[str] = []

        pages_checked = 0

        for page_index in range(
            pages_to_check
        ):

            try:
                page = document[
                    page_index
                ]

                page_text = page.get_text(
                    "text"
                )

            except Exception:
                page_text = ""

            if page_text:
                extracted_text.append(
                    page_text
                )

            pages_checked += 1

        return (
            "\n".join(
                extracted_text
            ),
            pages_checked,
        )

    # ========================================================
    # Text Normalization
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize embedded text before pattern matching.

        We preserve useful punctuation while reducing
        whitespace inconsistencies.
        """

        if not text:
            return ""

        normalized = (
            text
            .replace("\x00", " ")
            .replace("\r", "\n")
            .lower()
        )

        normalized = re.sub(
            r"[ \t]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\n+",
            "\n",
            normalized,
        )

        return normalized.strip()

    # ========================================================
    # Confidence Calculation
    # ========================================================

    def _detect_negative_groups(
        self,
        text: str,
    ) -> list[str]:
        matched_groups: list[str] = []
        for name, patterns in self.NON_BANK_STATEMENT_GROUPS.items():
            if any(
                re.search(pattern, text, flags=re.IGNORECASE) is not None
                for pattern in patterns
            ):
                matched_groups.append(name)
        return matched_groups

    def _passes_composite_gate(
        self,
        text: str,
    ) -> bool:
        statement = self._group_matches(self.SIGNAL_GROUPS["statement"], text)
        account = self._group_matches(self.SIGNAL_GROUPS["account"], text)
        transaction = self._group_matches(self.SIGNAL_GROUPS["transaction"], text)
        balance = self._group_matches(self.SIGNAL_GROUPS["balance"], text)
        banking = self._group_matches(self.SIGNAL_GROUPS["banking"], text)
        account_or_balance = account or balance
        path_a = statement and transaction and account_or_balance
        path_b = banking and transaction and account_or_balance and (statement or balance)
        return bool(path_a or path_b)

    @staticmethod
    def _group_matches(group: dict, text: str) -> bool:
        return any(
            re.search(pattern, text, flags=re.IGNORECASE) is not None
            for pattern in group["patterns"]
        )

    def _calculate_confidence(
        self,
        text: str,
    ) -> tuple[float, list[str]]:
        """
        Evaluate independent bank-statement signal groups.

        A signal group contributes its weight at most once.

        Example:

            500 occurrences of "balance"

        should not produce a higher score than one valid
        balance signal.
        """

        confidence = 0.0

        signals: list[str] = []

        for group in self.SIGNAL_GROUPS.values():

            patterns = group[
                "patterns"
            ]

            matched = any(
                re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )
                is not None
                for pattern in patterns
            )

            if not matched:
                continue

            confidence += float(
                group["weight"]
            )

            signals.append(
                str(
                    group["message"]
                )
            )

        return (
            min(
                confidence,
                1.0,
            ),
            signals,
        )


# ============================================================
# Default Reusable Instance
# ============================================================


bank_statement_detector = BankStatementDetector()