"""
page1_statement_detector.py

Bank Statement V2
Phase 1 - Page-1 Bank Statement Detection

Purpose
-------
Determine whether Page 1 is genuinely consistent with a bank statement.

Design principles
-----------------
1. Generic and bank-independent.
2. Uses multiple independent signals.
3. Does NOT classify from one keyword.
4. Uses metadata already extracted by page1_metadata_extractor.
5. Uses transaction/header structure from Page-1 text.
6. Returns confidence, decision and explainable signals.
7. Resistant to simple keyword stuffing / adversarial text.
8. Conservative around ambiguous documents.

Important
---------
This module does NOT determine PDF mode
(digital_pdf / scanned_pdf / hybrid_pdf).

Mode belongs to the Page-1 text/document analysis layer and will be
integrated separately after detector testing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# RESULT MODELS
# ============================================================


@dataclass
class DetectionSignal:
    detected: bool
    score: float
    max_score: float
    confidence: float
    evidence: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    is_bank_statement: bool
    confidence: float
    decision: str
    score: float
    max_score: float
    signals_detected: int
    total_signals: int
    strong_signals_detected: int
    signals: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# CONFIGURATION
# ============================================================


class DetectionConfig:
    """
    Scores intentionally sum to 100.

    Identity and statement-period evidence receive the largest
    weights because they are substantially stronger than generic
    words such as "balance" or "transaction".
    """

    ACCOUNT_IDENTITY_WEIGHT = 18.0
    IFSC_WEIGHT = 13.0
    STATEMENT_PERIOD_WEIGHT = 18.0

    TRANSACTION_TABLE_WEIGHT = 16.0
    BALANCE_TERMS_WEIGHT = 10.0
    BANKING_METADATA_WEIGHT = 10.0

    STATEMENT_CONTEXT_WEIGHT = 8.0
    TRANSACTION_ACTIVITY_WEIGHT = 7.0

    MAX_SCORE = 100.0

    # --------------------------------------------------------
    # Decision thresholds
    # --------------------------------------------------------

    HIGH_CONFIDENCE_THRESHOLD = 70.0
    REVIEW_THRESHOLD = 50.0

    # A positive decision should not be produced from a single
    # unusually strong signal.
    MIN_SIGNALS_FOR_POSITIVE = 3

    # At least one strong structural/identity signal should exist.
    MIN_STRONG_SIGNALS_FOR_POSITIVE = 1


# ============================================================
# TEXT UTILITIES
# ============================================================


class TextUtils:

    @staticmethod
    def normalize_for_matching(text: str) -> str:
        """
        Conservative normalization for detection only.

        File 1 remains responsible for canonical Page-1
        normalization. We do not mutate the original extraction.
        """

        text = text or ""

        text = text.replace("\u00a0", " ")

        text = (
            text.replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def flat(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text or "",
        ).strip()

    @staticmethod
    def evidence(
        text: str,
        start: int,
        end: int,
        radius: int = 60,
    ) -> str:
        left = max(
            0,
            start - radius,
        )

        right = min(
            len(text),
            end + radius,
        )

        return TextUtils.flat(
            text[left:right]
        )

    @staticmethod
    def first_evidence(
        text: str,
        patterns: Sequence[str],
        radius: int = 60,
    ) -> Optional[str]:
        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return TextUtils.evidence(
                    text,
                    match.start(),
                    match.end(),
                    radius=radius,
                )

        return None


# ============================================================
# METADATA UTILITIES
# ============================================================


class MetadataUtils:

    @staticmethod
    def fields(
        metadata_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(
            metadata_result,
            dict,
        ):
            return {}

        fields = metadata_result.get(
            "fields",
            {}
        )

        if not isinstance(fields, dict):
            return {}

        return fields

    @classmethod
    def field(
        cls,
        metadata_result: Optional[Dict[str, Any]],
        field_name: str,
    ) -> Dict[str, Any]:
        fields = cls.fields(
            metadata_result
        )

        field = fields.get(
            field_name,
            {}
        )

        if not isinstance(field, dict):
            return {}

        return field

    @classmethod
    def value(
        cls,
        metadata_result: Optional[Dict[str, Any]],
        field_name: str,
    ) -> Any:
        return cls.field(
            metadata_result,
            field_name,
        ).get("value")

    @classmethod
    def confidence(
        cls,
        metadata_result: Optional[Dict[str, Any]],
        field_name: str,
    ) -> float:
        value = cls.field(
            metadata_result,
            field_name,
        ).get(
            "confidence",
            0.0,
        )

        try:
            return float(value or 0.0)

        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def evidence(
        cls,
        metadata_result: Optional[Dict[str, Any]],
        field_name: str,
    ) -> Optional[str]:
        value = cls.field(
            metadata_result,
            field_name,
        ).get("evidence")

        if value is None:
            return None

        return str(value)


# ============================================================
# SIGNAL 1 - ACCOUNT IDENTITY
# ============================================================


class AccountIdentitySignal:
    """
    Strong signal.

    Primary evidence:
    - account number extracted successfully

    Supporting evidence:
    - account-holder name
    - customer ID / CIF / CRN
    - account type
    """

    WEIGHT = (
        DetectionConfig.ACCOUNT_IDENTITY_WEIGHT
    )

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        account_number = MetadataUtils.value(
            metadata,
            "account_number",
        )

        if not account_number:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                source="metadata",
            )

        account_confidence = (
            MetadataUtils.confidence(
                metadata,
                "account_number",
            )
        )

        supporting = 0

        for field_name in (
            "account_holder_name",
            "customer_id",
            "account_type",
        ):
            if MetadataUtils.value(
                metadata,
                field_name,
            ) is not None:
                supporting += 1

        confidence = min(
            1.0,
            max(
                0.75,
                account_confidence,
            )
            + supporting * 0.02,
        )

        score = round(
            cls.WEIGHT * confidence,
            2,
        )

        return DetectionSignal(
            detected=True,
            score=score,
            max_score=cls.WEIGHT,
            confidence=round(
                confidence,
                4,
            ),
            evidence=(
                MetadataUtils.evidence(
                    metadata,
                    "account_number",
                )
                or str(account_number)
            ),
            source="metadata.account_number",
        )


# ============================================================
# SIGNAL 2 - IFSC
# ============================================================


class IFSCSignal:
    """
    Strong Indian-bank identity signal.

    IFSC alone is not enough to classify a document as a bank
    statement, but when combined with account/period/transaction
    evidence it is highly useful.
    """

    WEIGHT = DetectionConfig.IFSC_WEIGHT

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        value = MetadataUtils.value(
            metadata,
            "ifsc",
        )

        if not value:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                source="metadata",
            )

        confidence = max(
            0.80,
            MetadataUtils.confidence(
                metadata,
                "ifsc",
            ),
        )

        confidence = min(
            confidence,
            1.0,
        )

        return DetectionSignal(
            detected=True,
            score=round(
                cls.WEIGHT * confidence,
                2,
            ),
            max_score=cls.WEIGHT,
            confidence=round(
                confidence,
                4,
            ),
            evidence=(
                MetadataUtils.evidence(
                    metadata,
                    "ifsc",
                )
                or str(value)
            ),
            source="metadata.ifsc",
        )


# ============================================================
# SIGNAL 3 - STATEMENT PERIOD
# ============================================================


class StatementPeriodSignal:
    """
    Strong signal.

    Both start and end dates must have been extracted.
    """

    WEIGHT = (
        DetectionConfig.STATEMENT_PERIOD_WEIGHT
    )

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        start_date = MetadataUtils.value(
            metadata,
            "statement_start_date",
        )

        end_date = MetadataUtils.value(
            metadata,
            "statement_end_date",
        )

        if not start_date or not end_date:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                source="metadata",
            )

        start_confidence = (
            MetadataUtils.confidence(
                metadata,
                "statement_start_date",
            )
        )

        end_confidence = (
            MetadataUtils.confidence(
                metadata,
                "statement_end_date",
            )
        )

        confidence = min(
            start_confidence,
            end_confidence,
        )

        confidence = max(
            confidence,
            0.75,
        )

        confidence = min(
            confidence,
            1.0,
        )

        evidence = (
            MetadataUtils.evidence(
                metadata,
                "statement_start_date",
            )
            or (
                f"{start_date} -> "
                f"{end_date}"
            )
        )

        return DetectionSignal(
            detected=True,
            score=round(
                cls.WEIGHT * confidence,
                2,
            ),
            max_score=cls.WEIGHT,
            confidence=round(
                confidence,
                4,
            ),
            evidence=evidence,
            source="metadata.statement_period",
        )


# ============================================================
# SIGNAL 4 - TRANSACTION TABLE
# ============================================================


class TransactionTableSignal:
    """
    Structural signal.

    Detects combinations of transaction-table column concepts.

    We deliberately require multiple concepts rather than merely
    seeing the word "transaction".
    """

    WEIGHT = (
        DetectionConfig.TRANSACTION_TABLE_WEIGHT
    )

    COLUMN_GROUPS: Dict[
        str,
        Sequence[str],
    ] = {
        "date": (
            r"\bdate\b",
            r"\bvalue\s+date\b",
            r"\bpost\s+date\b",
        ),

        "description": (
            r"\bdescription\b",
            r"\bparticulars?\b",
            r"\bnarration\b",
            r"\btransaction\s+details?\b",
            r"\bremarks?\b",
        ),

        "debit": (
            r"\bdebit\b",
            r"\bwithdrawals?\b",
            r"\bwithdrawal\s*\(dr\.?\)",
            r"\bdr\.?\b",
        ),

        "credit": (
            r"\bcredit\b",
            r"\bdeposits?\b",
            r"\bdeposit\s*\(cr\.?\)",
            r"\bcr\.?\b",
        ),

        "balance": (
            r"\bbalance\b",
            r"\brunning\s+balance\b",
        ),

        "reference": (
            r"\bchq(?:ue)?\b",
            r"\bcheque\b",
            r"\bref(?:erence)?\.?\s*(?:no\.?)?\b",
            r"\bchq/ref\b",
            r"\btransaction\s+id\b",
        ),
    }

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        detected_groups: List[str] = []
        evidence_parts: List[str] = []

        for group_name, patterns in (
            cls.COLUMN_GROUPS.items()
        ):
            evidence = TextUtils.first_evidence(
                text,
                patterns,
                radius=35,
            )

            if evidence:
                detected_groups.append(
                    group_name
                )

                if len(evidence_parts) < 2:
                    evidence_parts.append(
                        evidence
                    )

        group_count = len(
            detected_groups
        )

        # Need at least three distinct transaction-table concepts.
        if group_count < 3:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                evidence=(
                    ", ".join(
                        detected_groups
                    )
                    if detected_groups
                    else None
                ),
                source="page1_transaction_structure",
            )

        if group_count >= 5:
            confidence = 0.98

        elif group_count == 4:
            confidence = 0.90

        else:
            confidence = 0.78

        # Date + balance is particularly useful.
        if (
            "date" in detected_groups
            and "balance" in detected_groups
        ):
            confidence = min(
                1.0,
                confidence + 0.02,
            )

        evidence = (
            "Detected columns: "
            + ", ".join(
                detected_groups
            )
        )

        if evidence_parts:
            evidence += (
                " | "
                + " | ".join(
                    evidence_parts
                )
            )

        return DetectionSignal(
            detected=True,
            score=round(
                cls.WEIGHT * confidence,
                2,
            ),
            max_score=cls.WEIGHT,
            confidence=round(
                confidence,
                4,
            ),
            evidence=evidence,
            source="page1_transaction_structure",
        )


# ============================================================
# SIGNAL 5 - BALANCE TERMINOLOGY
# ============================================================


class BalanceTermsSignal:

    WEIGHT = (
        DetectionConfig.BALANCE_TERMS_WEIGHT
    )

    PATTERN_GROUPS: Dict[
        str,
        Sequence[str],
    ] = {
        "opening_balance": (
            r"\bopening\s+balance\b",
            r"\bbrought\s+forward\b",
            r"\bbalance\s+brought\s+forward\b",
            r"\bb/f\b",
        ),

        "closing_balance": (
            r"\bclosing\s+balance\b",
            r"\bcarried\s+forward\b",
            r"\bbalance\s+carried\s+forward\b",
            r"\bc/f\b",
        ),

        "available_balance": (
            r"\bavailable\s+balance\b",
            r"\bclear(?:ed)?\s+balance\b",
            r"\bledger\s+balance\b",
        ),

        "generic_balance": (
            r"\bbalance\b",
        ),
    }

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        detected_groups: List[str] = []

        evidence = None

        for group_name, patterns in (
            cls.PATTERN_GROUPS.items()
        ):
            current_evidence = (
                TextUtils.first_evidence(
                    text,
                    patterns,
                    radius=45,
                )
            )

            if current_evidence:
                detected_groups.append(
                    group_name
                )

                if evidence is None:
                    evidence = current_evidence

        specific_groups = [
            group
            for group in detected_groups
            if group != "generic_balance"
        ]

        if not detected_groups:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                source="page1_balance_terms",
            )

        if len(specific_groups) >= 2:
            confidence = 0.95

        elif len(specific_groups) == 1:
            confidence = 0.84

        else:
            # Generic "balance" by itself is weak.
            confidence = 0.45

        return DetectionSignal(
            detected=True,
            score=round(
                cls.WEIGHT * confidence,
                2,
            ),
            max_score=cls.WEIGHT,
            confidence=confidence,
            evidence=evidence,
            source="page1_balance_terms",
        )


# ============================================================
# SIGNAL 6 - BANKING METADATA
# ============================================================


class BankingMetadataSignal:
    """
    Supporting signal based on several header fields.

    No single field is sufficient.
    """

    WEIGHT = (
        DetectionConfig.BANKING_METADATA_WEIGHT
    )

    FIELD_NAMES = (
        "bank_name",
        "branch",
        "branch_code",
        "customer_id",
        "account_type",
        "account_holder_name",
    )

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        detected_fields = [
            field_name
            for field_name in cls.FIELD_NAMES
            if MetadataUtils.value(
                metadata,
                field_name,
            ) is not None
        ]

        count = len(
            detected_fields
        )

        if count < 2:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                evidence=(
                    ", ".join(
                        detected_fields
                    )
                    if detected_fields
                    else None
                ),
                source="metadata.header_fields",
            )

        if count >= 4:
            confidence = 0.95

        elif count == 3:
            confidence = 0.85

        else:
            confidence = 0.72

        evidence = (
            "Detected metadata: "
            + ", ".join(
                detected_fields
            )
        )

        return DetectionSignal(
            detected=True,
            score=round(
                cls.WEIGHT * confidence,
                2,
            ),
            max_score=cls.WEIGHT,
            confidence=confidence,
            evidence=evidence,
            source="metadata.header_fields",
        )


# ============================================================
# SIGNAL 7 - STATEMENT CONTEXT
# ============================================================


class StatementContextSignal:

    WEIGHT = (
        DetectionConfig.STATEMENT_CONTEXT_WEIGHT
    )

    STRONG_PATTERNS = (
        r"\baccount\s+statement\b",
        r"\bstatement\s+of\s+account\b",
        r"\bbank\s+statement\b",
        r"\bstatement\s+for\s+(?:a/c|account)\b",
        r"\bstatement\s+from\b",
        r"\bstatement\s+period\b",
    )

    WEAK_PATTERNS = (
        r"\bstatement\b",
        r"\btransaction\s+statement\b",
    )

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        evidence = TextUtils.first_evidence(
            text,
            cls.STRONG_PATTERNS,
            radius=55,
        )

        if evidence:
            confidence = 0.96

            return DetectionSignal(
                detected=True,
                score=round(
                    cls.WEIGHT
                    * confidence,
                    2,
                ),
                max_score=cls.WEIGHT,
                confidence=confidence,
                evidence=evidence,
                source="page1_statement_context",
            )

        evidence = TextUtils.first_evidence(
            text,
            cls.WEAK_PATTERNS,
            radius=55,
        )

        if evidence:
            confidence = 0.55

            return DetectionSignal(
                detected=True,
                score=round(
                    cls.WEIGHT
                    * confidence,
                    2,
                ),
                max_score=cls.WEIGHT,
                confidence=confidence,
                evidence=evidence,
                source="page1_statement_context",
            )

        return DetectionSignal(
            detected=False,
            score=0.0,
            max_score=cls.WEIGHT,
            confidence=0.0,
            source="page1_statement_context",
        )


# ============================================================
# SIGNAL 8 - TRANSACTION ACTIVITY
# ============================================================


class TransactionActivitySignal:
    """
    Supporting transaction evidence.

    Detects actual transaction-like banking activity rather than
    merely a transaction-table heading.
    """

    WEIGHT = (
        DetectionConfig.TRANSACTION_ACTIVITY_WEIGHT
    )

    ACTIVITY_PATTERNS = {
        "upi": (
            r"\bUPI\b",
        ),

        "neft": (
            r"\bNEFT\b",
        ),

        "imps": (
            r"\bIMPS\b",
        ),

        "rtgs": (
            r"\bRTGS\b",
        ),

        "atm": (
            r"\bATM\b",
            r"\bATM\s+WDL\b",
            r"\bATM\s+CASH\b",
        ),

        "transfer": (
            r"\btransfer\b",
            r"\bTFR\b",
            r"\bDEP\s+TFR\b",
        ),

        "cheque": (
            r"\bcheque\b",
            r"\bchq\b",
        ),

        "debit_credit": (
            r"\bdebit\b",
            r"\bcredit\b",
            r"\bwithdrawal\b",
            r"\bdeposit\b",
        ),
    }

    @classmethod
    def evaluate(
        cls,
        text: str,
        metadata: Dict[str, Any],
    ) -> DetectionSignal:

        detected_types: List[str] = []
        first_evidence = None

        for activity, patterns in (
            cls.ACTIVITY_PATTERNS.items()
        ):
            evidence = TextUtils.first_evidence(
                text,
                patterns,
                radius=45,
            )

            if evidence:
                detected_types.append(
                    activity
                )

                if first_evidence is None:
                    first_evidence = evidence

        count = len(
            detected_types
        )

        if count < 2:
            return DetectionSignal(
                detected=False,
                score=0.0,
                max_score=cls.WEIGHT,
                confidence=0.0,
                evidence=first_evidence,
                source="page1_transaction_activity",
            )

        if count >= 4:
            confidence = 0.95

        elif count == 3:
            confidence = 0.85

        else:
            confidence = 0.72

        evidence = (
            "Transaction activity: "
            + ", ".join(
                detected_types
            )
        )

        if first_evidence:
            evidence += (
                " | "
                + first_evidence
            )

        return DetectionSignal(
            detected=True,
            score=round(
                cls.WEIGHT * confidence,
                2,
            ),
            max_score=cls.WEIGHT,
            confidence=confidence,
            evidence=evidence,
            source="page1_transaction_activity",
        )


# ============================================================
# CONTRADICTION / FALSE-POSITIVE GUARD
# ============================================================


class FalsePositiveGuard:
    """
    Detects document contexts that commonly contain bank details
    without necessarily being bank statements.

    This is deliberately generic.

    Examples:
    - invoice
    - salary slip
    - loan application
    - cancelled cheque
    - KYC form
    - payment receipt

    These do NOT automatically reject the document. They reduce
    confidence when statement structure is weak.
    """

    NON_STATEMENT_PATTERNS: Dict[
        str,
        Sequence[str],
    ] = {
        "invoice": (
            r"\binvoice\b",
            r"\btax\s+invoice\b",
            r"\binvoice\s+number\b",
            r"\binvoice\s+no\.?\b",
        ),

        "salary_slip": (
            r"\bsalary\s+slip\b",
            r"\bpayslip\b",
            r"\bpay\s+slip\b",
            r"\bgross\s+salary\b",
            r"\bnet\s+salary\b",
        ),

        "loan_form": (
            r"\bloan\s+application\b",
            r"\bapplication\s+form\b",
            r"\bloan\s+agreement\b",
        ),

        "kyc_form": (
            r"\bkyc\s+form\b",
            r"\bknow\s+your\s+customer\b",
        ),

        "cancelled_cheque": (
            r"\bcancelled\s+cheque\b",
            r"\bcanceled\s+cheque\b",
        ),

        "receipt": (
            r"\bpayment\s+receipt\b",
            r"\breceipt\s+number\b",
            r"\breceipt\s+no\.?\b",
        ),
    }

    @classmethod
    def detect(
        cls,
        text: str,
    ) -> Tuple[
        List[str],
        Optional[str],
    ]:

        detected: List[str] = []
        evidence = None

        for category, patterns in (
            cls.NON_STATEMENT_PATTERNS.items()
        ):
            current_evidence = (
                TextUtils.first_evidence(
                    text,
                    patterns,
                    radius=45,
                )
            )

            if current_evidence:
                detected.append(
                    category
                )

                if evidence is None:
                    evidence = current_evidence

        return detected, evidence


# ============================================================
# MAIN DETECTOR
# ============================================================


class Page1StatementDetector:

    STRONG_SIGNAL_NAMES = {
        "account_identity",
        "statement_period",
        "transaction_table",
    }

    def _evaluate_signals(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, DetectionSignal]:

        return {
            "account_identity":
                AccountIdentitySignal.evaluate(
                    text,
                    metadata,
                ),

            "ifsc":
                IFSCSignal.evaluate(
                    text,
                    metadata,
                ),

            "statement_period":
                StatementPeriodSignal.evaluate(
                    text,
                    metadata,
                ),

            "transaction_table":
                TransactionTableSignal.evaluate(
                    text,
                    metadata,
                ),

            "balance_terms":
                BalanceTermsSignal.evaluate(
                    text,
                    metadata,
                ),

            "banking_metadata":
                BankingMetadataSignal.evaluate(
                    text,
                    metadata,
                ),

            "statement_context":
                StatementContextSignal.evaluate(
                    text,
                    metadata,
                ),

            "transaction_activity":
                TransactionActivitySignal.evaluate(
                    text,
                    metadata,
                ),
        }

    @staticmethod
    def _calculate_base_score(
        signals: Dict[
            str,
            DetectionSignal,
        ],
    ) -> float:

        return round(
            sum(
                signal.score
                for signal
                in signals.values()
            ),
            2,
        )

    @staticmethod
    def _signals_detected(
        signals: Dict[
            str,
            DetectionSignal,
        ],
    ) -> int:

        return sum(
            1
            for signal
            in signals.values()
            if signal.detected
        )

    @classmethod
    def _strong_signals_detected(
        cls,
        signals: Dict[
            str,
            DetectionSignal,
        ],
    ) -> int:

        return sum(
            1
            for name, signal
            in signals.items()
            if (
                name
                in cls.STRONG_SIGNAL_NAMES
                and signal.detected
            )
        )

    @staticmethod
    def _apply_false_positive_guard(
        score: float,
        text: str,
        signals: Dict[
            str,
            DetectionSignal,
        ],
    ) -> Tuple[
        float,
        List[str],
        Optional[str],
        float,
    ]:

        categories, evidence = (
            FalsePositiveGuard.detect(
                text
            )
        )

        if not categories:
            return (
                score,
                categories,
                evidence,
                0.0,
            )

        transaction_table = signals[
            "transaction_table"
        ].detected

        statement_period = signals[
            "statement_period"
        ].detected

        account_identity = signals[
            "account_identity"
        ].detected

        strong_statement_structure = (
            transaction_table
            and statement_period
            and account_identity
        )

        # If strong statement structure is present, a word such
        # as "receipt" inside a transaction narration should not
        # heavily punish the document.
        if strong_statement_structure:
            penalty = min(
                5.0,
                len(categories) * 2.0,
            )

        else:
            penalty = min(
                25.0,
                len(categories) * 10.0,
            )

        adjusted = max(
            0.0,
            score - penalty,
        )

        return (
            round(
                adjusted,
                2,
            ),
            categories,
            evidence,
            penalty,
        )

    @staticmethod
    def _confidence_from_score(
        score: float,
    ) -> float:
        """
        Score already represents weighted evidence out of 100.

        Convert to 0-1 confidence without pretending it is a
        statistically calibrated probability.
        """

        return round(
            max(
                0.0,
                min(
                    1.0,
                    score / 100.0,
                ),
            ),
            4,
        )

    @staticmethod
    def _decision(
        score: float,
        signals_detected: int,
        strong_signals_detected: int,
    ) -> Tuple[bool, str]:

        if (
            score
            >= DetectionConfig.HIGH_CONFIDENCE_THRESHOLD
            and signals_detected
            >= DetectionConfig.MIN_SIGNALS_FOR_POSITIVE
            and strong_signals_detected
            >= DetectionConfig.MIN_STRONG_SIGNALS_FOR_POSITIVE
        ):
            return (
                True,
                "bank_statement",
            )

        if (
            score
            >= DetectionConfig.REVIEW_THRESHOLD
        ):
            return (
                False,
                "review",
            )

        return (
            False,
            "not_bank_statement",
        )

    def detect(
        self,
        text: str,
        metadata_result: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        text:
            Normalized Page-1 text from page1_text_provider.

        metadata_result:
            Full result returned by
            page1_metadata_extractor.extract(text).

        Returns
        -------
        dict
            Explainable bank-statement detection result.
        """

        normalized_text = (
            TextUtils.normalize_for_matching(
                text
            )
        )

        metadata = (
            metadata_result
            if isinstance(
                metadata_result,
                dict,
            )
            else {}
        )

        signals = self._evaluate_signals(
            normalized_text,
            metadata,
        )

        base_score = (
            self._calculate_base_score(
                signals
            )
        )

        signals_detected = (
            self._signals_detected(
                signals
            )
        )

        strong_signals_detected = (
            self._strong_signals_detected(
                signals
            )
        )

        (
            final_score,
            contradictions,
            contradiction_evidence,
            penalty,
        ) = self._apply_false_positive_guard(
            base_score,
            normalized_text,
            signals,
        )

        (
            is_bank_statement,
            decision,
        ) = self._decision(
            final_score,
            signals_detected,
            strong_signals_detected,
        )

        confidence = (
            self._confidence_from_score(
                final_score
            )
        )

        signal_dict = {
            name: signal.to_dict()
            for name, signal
            in signals.items()
        }

        return {
            "is_bank_statement":
                is_bank_statement,

            "confidence":
                confidence,

            "decision":
                decision,

            "score":
                final_score,

            "max_score":
                DetectionConfig.MAX_SCORE,

            "base_score":
                base_score,

            "penalty":
                penalty,

            "signals_detected":
                signals_detected,

            "total_signals":
                len(signals),

            "strong_signals_detected":
                strong_signals_detected,

            "signals":
                signal_dict,

            "contradictions": {
                "detected":
                    bool(contradictions),

                "categories":
                    contradictions,

                "evidence":
                    contradiction_evidence,
            },
        }


# ============================================================
# SINGLETON
# ============================================================


page1_statement_detector = (
    Page1StatementDetector()
)