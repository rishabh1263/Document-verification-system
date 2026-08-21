"""
Transaction Anomaly Analyzer
============================

Phase 4 component for generic transaction-level behavioral analysis.

Current detectors
-----------------
1. Unusual transaction amounts
2. Duplicate-like transactions
3. Transaction activity bursts

Critical design rule
--------------------
Behavioral anomaly != document fraud.

Examples:
- A large NEFT/RTGS transaction may be completely legitimate.
- Repeated EMI/UPI/ATM transactions may be legitimate.
- Many transactions on one date may be normal customer behavior.

Therefore this analyzer detects and explains behavioral anomalies,
but DOES NOT independently add fraud/document-tampering risk points.

Actual positive document-risk evidence should come from stronger
verification evidence such as:

- balance-chain mismatches
- statement consistency failures
- structural validation failures
- integrity / tampering evidence

This analyzer remains useful for:
- explainability
- behavioral profiling
- downstream analytics
- contextual evidence
- future compound-risk rules

The analyzer is:
- bank-independent
- deterministic
- statistical
- modular
- extraction-format tolerant
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any, Iterable, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
    TransactionAnomaly,
    TransactionAnomalySummary,
)


class TransactionAnomalyAnalyzer:
    """
    Generic transaction-level behavioral anomaly analyzer.

    Important:
    Behavioral anomalies generated here have zero direct risk score.

    This prevents legitimate customer behavior from being interpreted
    as document manipulation or fraud.
    """

    # =========================================================
    # CONFIGURATION
    # =========================================================

    MIN_AMOUNT_SAMPLE_SIZE = 10

    LARGE_AMOUNT_MULTIPLIER = Decimal("8")

    MIN_LARGE_AMOUNT = Decimal("10000")

    BURST_TRANSACTION_COUNT = 8

    # ---------------------------------------------------------
    # Behavioral anomaly scores
    # ---------------------------------------------------------
    #
    # These intentionally remain ZERO.
    #
    # Detection is preserved for explainability, but these
    # observations do not independently increase document risk.
    # ---------------------------------------------------------

    DUPLICATE_TIMELESS_SCORE = 0.0

    LARGE_AMOUNT_SCORE = 0.0

    BURST_SCORE = 0.0

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        transactions: Iterable[Any],
    ) -> TransactionAnomalySummary:
        """
        Analyze standardized bank transactions.

        Parameters
        ----------
        transactions:
            Iterable containing dictionaries or standardized
            transaction objects.

        Returns
        -------
        TransactionAnomalySummary
        """

        if transactions is None:
            raise ValueError(
                "transactions cannot be None."
            )

        transaction_list = list(
            transactions
        )

        usable_transactions = [
            transaction
            for transaction
            in transaction_list
            if self._is_analyzable(
                transaction
            )
        ]

        anomalies: list[
            TransactionAnomaly
        ] = []

        signals: list[
            RiskSignal
        ] = []

        # =====================================================
        # 1. UNUSUAL AMOUNT ANALYSIS
        # =====================================================

        unusual_anomalies, unusual_signals = (
            self._detect_unusual_amounts(
                usable_transactions
            )
        )

        anomalies.extend(
            unusual_anomalies
        )

        signals.extend(
            unusual_signals
        )

        # =====================================================
        # 2. DUPLICATE-LIKE ANALYSIS
        # =====================================================

        duplicate_anomalies, duplicate_signals = (
            self._detect_duplicate_like_transactions(
                usable_transactions
            )
        )

        anomalies.extend(
            duplicate_anomalies
        )

        signals.extend(
            duplicate_signals
        )

        # =====================================================
        # 3. TRANSACTION BURST ANALYSIS
        # =====================================================

        burst_anomalies, burst_signals = (
            self._detect_transaction_bursts(
                usable_transactions
            )
        )

        anomalies.extend(
            burst_anomalies
        )

        signals.extend(
            burst_signals
        )

        # =====================================================
        # SUMMARY
        # =====================================================

        unusual_amount_count = sum(
            1
            for anomaly in anomalies
            if anomaly.anomaly_type
            == "unusual_amount"
        )

        duplicate_like_count = sum(
            1
            for anomaly in anomalies
            if anomaly.anomaly_type
            == "duplicate_like"
        )

        burst_count = sum(
            1
            for anomaly in anomalies
            if anomaly.anomaly_type
            == "transaction_burst"
        )

        confidence = (
            self._calculate_confidence(
                transaction_count=len(
                    transaction_list
                ),
                analyzed_count=len(
                    usable_transactions
                ),
            )
        )

        return TransactionAnomalySummary(
            transaction_count=len(
                transaction_list
            ),

            analyzed_count=len(
                usable_transactions
            ),

            anomaly_count=len(
                anomalies
            ),

            unusual_amount_count=(
                unusual_amount_count
            ),

            duplicate_like_count=(
                duplicate_like_count
            ),

            burst_count=(
                burst_count
            ),

            anomalies=tuple(
                anomalies
            ),

            signals=tuple(
                signals
            ),

            confidence=confidence,
        )

    # =========================================================
    # UNUSUAL AMOUNT DETECTION
    # =========================================================

    def _detect_unusual_amounts(
        self,
        transactions: list[Any],
    ) -> tuple[
        list[TransactionAnomaly],
        list[RiskSignal],
    ]:
        """
        Detect unusually large transaction amounts.

        This is a BEHAVIORAL detector.

        It does not imply:
            fraud
            document manipulation
            tampering
            invalid statement

        Uses a robust median-based threshold:

            max(
                median_amount * LARGE_AMOUNT_MULTIPLIER,
                MIN_LARGE_AMOUNT
            )

        Median is useful because bank transaction distributions
        are normally highly skewed.

        However, a small median can make legitimate transactions
        appear statistically extreme. Therefore these detections
        remain informational and carry zero direct risk points.
        """

        amount_records: list[
            tuple[Any, Decimal]
        ] = []

        for transaction in transactions:

            amount = self._transaction_amount(
                transaction
            )

            if (
                amount is not None
                and amount > 0
            ):
                amount_records.append(
                    (
                        transaction,
                        amount,
                    )
                )

        if (
            len(amount_records)
            < self.MIN_AMOUNT_SAMPLE_SIZE
        ):
            return [], []

        amounts = [
            amount
            for _transaction, amount
            in amount_records
        ]

        median_amount = Decimal(
            str(
                median(
                    amounts
                )
            )
        )

        threshold = max(
            median_amount
            * self.LARGE_AMOUNT_MULTIPLIER,

            self.MIN_LARGE_AMOUNT,
        )

        anomalies: list[
            TransactionAnomaly
        ] = []

        signals: list[
            RiskSignal
        ] = []

        for transaction, amount in amount_records:

            if amount <= threshold:
                continue

            sequence = self._get(
                transaction,
                "sequence",
            )

            transaction_date = self._get(
                transaction,
                "date",
            )

            description = self._get(
                transaction,
                "description",
            )

            ratio = (
                amount / median_amount
                if median_amount > 0
                else Decimal("0")
            )

            confidence = min(
                1.0,
                0.65
                + min(
                    float(ratio) / 100.0,
                    0.30,
                ),
            )

            severity = (
                self._amount_severity(
                    ratio
                )
            )

            evidence = {
                "amount":
                    float(
                        amount
                    ),

                "median_amount":
                    float(
                        median_amount
                    ),

                "threshold":
                    float(
                        threshold
                    ),

                "median_multiple":
                    round(
                        float(
                            ratio
                        ),
                        4,
                    ),

                "behavioral_only":
                    True,

                "direct_risk_contribution":
                    0.0,
            }

            anomaly = TransactionAnomaly(
                sequence=sequence,

                anomaly_type=(
                    "unusual_amount"
                ),

                severity=severity,

                message=(
                    "Transaction amount is unusually "
                    "large relative to the statement's "
                    "typical transaction amount."
                ),

                score=(
                    self.LARGE_AMOUNT_SCORE
                ),

                confidence=confidence,

                amount=float(
                    amount
                ),

                transaction_date=(
                    transaction_date
                ),

                description=(
                    description
                ),

                evidence=evidence,
            )

            anomalies.append(
                anomaly
            )

            signals.append(
                RiskSignal(
                    code=(
                        "UNUSUAL_TRANSACTION_AMOUNT"
                    ),

                    category="transaction",

                    severity="info",

                    message=(
                        "Statistically unusual transaction "
                        "amount detected. This is behavioral "
                        "context and does not independently "
                        "indicate document fraud."
                    ),

                    score=0.0,

                    confidence=confidence,

                    transaction_sequence=(
                        sequence
                    ),

                    field_name="amount",

                    expected=(
                        f"<= {float(threshold)}"
                    ),

                    actual=float(
                        amount
                    ),

                    evidence=evidence,

                    source=(
                        "transaction_anomaly_analyzer"
                    ),
                )
            )

        return (
            anomalies,
            signals,
        )

    # =========================================================
    # DUPLICATE-LIKE DETECTION
    # =========================================================

    def _detect_duplicate_like_transactions(
        self,
        transactions: list[Any],
    ) -> tuple[
        list[TransactionAnomaly],
        list[RiskSignal],
    ]:
        """
        Detect transactions sharing:

            date
            amount
            normalized description

        This is intentionally called DUPLICATE-LIKE.

        Same date + amount + description does NOT prove that a
        transaction was duplicated fraudulently.

        Legitimate examples include:
            repeated ATM withdrawals
            repeated merchant payments
            recurring transfers
            EMI transactions
            UPI payments

        Therefore this detector produces behavioral evidence only.

        Actual duplicate validation remains the responsibility of
        Phase 3 statement validation and stronger future compound
        rules.
        """

        groups: dict[
            tuple[str, str, str],
            list[Any],
        ] = defaultdict(
            list
        )

        for transaction in transactions:

            date = self._normalize_text(
                self._get(
                    transaction,
                    "date",
                )
            )

            description = (
                self._normalize_description(
                    self._get(
                        transaction,
                        "description",
                    )
                )
            )

            amount = self._transaction_amount(
                transaction
            )

            if (
                not date
                or not description
                or amount is None
            ):
                continue

            key = (
                date,
                str(
                    amount
                ),
                description,
            )

            groups[
                key
            ].append(
                transaction
            )

        anomalies: list[
            TransactionAnomaly
        ] = []

        signals: list[
            RiskSignal
        ] = []

        for (
            date,
            amount_text,
            description,
        ), group in groups.items():

            if len(
                group
            ) <= 1:
                continue

            amount = self._to_decimal(
                amount_text
            )

            sequences = [
                self._get(
                    transaction,
                    "sequence",
                )
                for transaction
                in group
            ]

            evidence = {
                "occurrence_count":
                    len(
                        group
                    ),

                "sequences":
                    sequences,

                "date":
                    date,

                "amount": (
                    float(
                        amount
                    )
                    if amount is not None
                    else None
                ),

                "normalized_description":
                    description,

                "behavioral_only":
                    True,

                "direct_risk_contribution":
                    0.0,
            }

            for transaction in group:

                sequence = self._get(
                    transaction,
                    "sequence",
                )

                anomaly = TransactionAnomaly(
                    sequence=sequence,

                    anomaly_type=(
                        "duplicate_like"
                    ),

                    severity="low",

                    message=(
                        "Transaction resembles another "
                        "transaction with the same date, "
                        "amount and description."
                    ),

                    score=0.0,

                    confidence=0.75,

                    amount=(
                        float(
                            amount
                        )
                        if amount is not None
                        else None
                    ),

                    transaction_date=(
                        self._get(
                            transaction,
                            "date",
                        )
                    ),

                    description=(
                        self._get(
                            transaction,
                            "description",
                        )
                    ),

                    evidence=evidence,
                )

                anomalies.append(
                    anomaly
                )

                signals.append(
                    RiskSignal(
                        code=(
                            "DUPLICATE_LIKE_TRANSACTION"
                        ),

                        category="transaction",

                        severity="info",

                        message=(
                            "Duplicate-like behavioral "
                            "transaction pattern detected. "
                            "The pattern alone does not "
                            "establish duplication or fraud."
                        ),

                        score=0.0,

                        confidence=0.75,

                        transaction_sequence=(
                            sequence
                        ),

                        evidence=evidence,

                        source=(
                            "transaction_anomaly_analyzer"
                        ),
                    )
                )

        return (
            anomalies,
            signals,
        )

    # =========================================================
    # BURST DETECTION
    # =========================================================

    def _detect_transaction_bursts(
        self,
        transactions: list[Any],
    ) -> tuple[
        list[TransactionAnomaly],
        list[RiskSignal],
    ]:
        """
        Detect dense transaction days.

        Without timestamps, we cannot distinguish:

            10 transactions in 5 minutes

        from:

            10 transactions across an entire day.

        Therefore date-level transaction density is behavioral
        context only and must not independently increase document
        fraud risk.
        """

        by_date: dict[
            str,
            list[Any],
        ] = defaultdict(
            list
        )

        for transaction in transactions:

            date = self._normalize_text(
                self._get(
                    transaction,
                    "date",
                )
            )

            if not date:
                continue

            by_date[
                date
            ].append(
                transaction
            )

        daily_counts = Counter(
            {
                date: len(
                    items
                )
                for date, items
                in by_date.items()
            }
        )

        anomalies: list[
            TransactionAnomaly
        ] = []

        signals: list[
            RiskSignal
        ] = []

        for date, count in daily_counts.items():

            if (
                count
                < self.BURST_TRANSACTION_COUNT
            ):
                continue

            day_transactions = (
                by_date[
                    date
                ]
            )

            sequences = [
                self._get(
                    transaction,
                    "sequence",
                )
                for transaction
                in day_transactions
            ]

            evidence = {
                "date":
                    date,

                "transaction_count":
                    count,

                "burst_threshold":
                    self.BURST_TRANSACTION_COUNT,

                "sequences":
                    sequences,

                "behavioral_only":
                    True,

                "direct_risk_contribution":
                    0.0,
            }

            representative = (
                day_transactions[
                    0
                ]
            )

            representative_sequence = (
                self._get(
                    representative,
                    "sequence",
                )
            )

            anomalies.append(
                TransactionAnomaly(
                    sequence=(
                        representative_sequence
                    ),

                    anomaly_type=(
                        "transaction_burst"
                    ),

                    severity="low",

                    message=(
                        "High transaction activity "
                        "detected on a single date."
                    ),

                    score=0.0,

                    confidence=0.70,

                    transaction_date=date,

                    description=None,

                    evidence=evidence,
                )
            )

            signals.append(
                RiskSignal(
                    code=(
                        "TRANSACTION_ACTIVITY_BURST"
                    ),

                    category="transaction",

                    severity="info",

                    message=(
                        "High transaction activity detected "
                        "on a single date. Date-level activity "
                        "density is behavioral context and does "
                        "not independently indicate fraud."
                    ),

                    score=0.0,

                    confidence=0.70,

                    transaction_sequence=(
                        representative_sequence
                    ),

                    evidence=evidence,

                    source=(
                        "transaction_anomaly_analyzer"
                    ),
                )
            )

        return (
            anomalies,
            signals,
        )

    # =========================================================
    # AMOUNT HELPERS
    # =========================================================

    def _transaction_amount(
        self,
        transaction: Any,
    ) -> Decimal | None:
        """
        Resolve transaction amount from debit/credit.

        Absolute magnitude is used for behavioral analysis.

        Signed accounting semantics remain the responsibility
        of Phase 3 balance validation.
        """

        debit = self._to_decimal(
            self._get(
                transaction,
                "debit",
            )
        )

        credit = self._to_decimal(
            self._get(
                transaction,
                "credit",
            )
        )

        if (
            debit is None
            and credit is None
        ):
            return None

        if (
            debit is not None
            and credit is None
        ):
            return abs(
                debit
            )

        if (
            credit is not None
            and debit is None
        ):
            return abs(
                credit
            )

        # If both are present, use the larger magnitude only
        # for behavioral analysis.
        #
        # Phase 3 remains responsible for structural conflict
        # detection.

        return max(
            abs(
                debit
            ),
            abs(
                credit
            ),
        )

    # =========================================================
    # BEHAVIORAL SEVERITY
    # =========================================================

    @staticmethod
    def _amount_severity(
        median_multiple: Decimal,
    ) -> str:
        """
        Severity of the statistical deviation.

        This severity belongs to the anomaly object for analytics
        and explainability.

        The corresponding RiskSignal remains informational with
        score=0 because statistical extremeness is not equivalent
        to document fraud.
        """

        if median_multiple >= Decimal(
            "50"
        ):
            return "high"

        if median_multiple >= Decimal(
            "20"
        ):
            return "medium"

        return "low"

    # =========================================================
    # ANALYZABILITY
    # =========================================================

    def _is_analyzable(
        self,
        transaction: Any,
    ) -> bool:

        amount = self._transaction_amount(
            transaction
        )

        date = self._get(
            transaction,
            "date",
        )

        description = self._get(
            transaction,
            "description",
        )

        return bool(
            amount is not None
            or date
            or description
        )

    # =========================================================
    # CONFIDENCE
    # =========================================================

    @staticmethod
    def _calculate_confidence(
        transaction_count: int,
        analyzed_count: int,
    ) -> float:

        if transaction_count <= 0:
            return 0.0

        confidence = (
            analyzed_count
            / transaction_count
        )

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        return round(
            confidence,
            4,
        )

    # =========================================================
    # GENERIC OBJECT ACCESS
    # =========================================================

    @staticmethod
    def _get(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            obj,
            Mapping,
        ):
            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return str(
            value
        ).strip()

    @staticmethod
    def _normalize_description(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = " ".join(
            str(
                value
            )
            .upper()
            .split()
        )

        return text

    # =========================================================
    # DECIMAL
    # =========================================================

    @staticmethod
    def _to_decimal(
        value: Any,
    ) -> Decimal | None:

        if value is None:
            return None

        if isinstance(
            value,
            Decimal,
        ):
            return value

        if isinstance(
            value,
            bool,
        ):
            return None

        raw = (
            str(
                value
            )
            .replace(
                ",",
                "",
            )
            .replace(
                "₹",
                "",
            )
            .strip()
        )

        if not raw:
            return None

        try:
            return Decimal(
                raw
            )

        except (
            InvalidOperation,
            ValueError,
            TypeError,
        ):
            return None


transaction_anomaly_analyzer = (
    TransactionAnomalyAnalyzer()
)