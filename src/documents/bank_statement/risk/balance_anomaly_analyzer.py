"""
Balance Anomaly Analyzer
========================

Phase 4 adapter/analyzer for balance-chain anomalies.

This component consumes Phase 3 balance-chain validation output.

It does NOT:
- recalculate running balances
- modify transactions
- infer fraud directly
- contain bank-specific rules

Its job is to translate validated accounting inconsistencies into
explainable Phase 4 RiskSignal objects.

Important
---------
A balance mismatch is evidence of inconsistency, not automatically
evidence of fraud.

Unverifiable rows primarily reduce assessment confidence and are
therefore reported separately from arithmetic mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
)


@dataclass(frozen=True)
class BalanceAnomalySummary:
    """
    Phase 4 summary for balance-chain evidence.
    """

    transaction_count: int

    checked_count: int
    reconciled_count: int

    mismatch_count: int
    unverifiable_count: int

    mismatch_rate: float
    coverage: float

    signal_count: int

    signals: tuple[
        RiskSignal,
        ...,
    ]

    confidence: float

    def to_dict(self) -> dict:

        return {
            "transaction_count":
                self.transaction_count,

            "checked_count":
                self.checked_count,

            "reconciled_count":
                self.reconciled_count,

            "mismatch_count":
                self.mismatch_count,

            "unverifiable_count":
                self.unverifiable_count,

            "mismatch_rate":
                round(
                    self.mismatch_rate,
                    4,
                ),

            "coverage":
                round(
                    self.coverage,
                    4,
                ),

            "signal_count":
                self.signal_count,

            "confidence":
                round(
                    self.confidence,
                    4,
                ),

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
        }


class BalanceAnomalyAnalyzer:
    """
    Convert Phase 3 balance validation evidence into Phase 4 signals.
    """

    # Risk contribution per individual mismatch.
    #
    # Final aggregation will cap/group repeated signals so large
    # statements cannot accumulate unlimited risk simply because
    # they contain many rows.
    MISMATCH_SCORE = 8.0

    # Unverifiable evidence is NOT treated as fraud risk.
    # It produces an informational signal with zero risk score.
    UNVERIFIABLE_SCORE = 0.0

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        balance_validation: Any,
    ) -> BalanceAnomalySummary:
        """
        Analyze Phase 3 balance-chain validation output.

        Accepted input:
        - dictionary returned by to_dict()
        - BalanceChainValidationResult-like object
        """

        if balance_validation is None:
            raise ValueError(
                "balance_validation cannot be None."
            )

        transaction_count = self._to_int(
            self._get(
                balance_validation,
                "transaction_count",
                0,
            )
        )

        checked_count = self._to_int(
            self._get(
                balance_validation,
                "checked_count",
                0,
            )
        )

        reconciled_count = self._to_int(
            self._get(
                balance_validation,
                "reconciled_count",
                0,
            )
        )

        mismatch_count = self._to_int(
            self._get(
                balance_validation,
                "mismatch_count",
                0,
            )
        )

        unverifiable_count = self._to_int(
            self._get(
                balance_validation,
                "unverifiable_count",
                0,
            )
        )

        checks = self._get(
            balance_validation,
            "checks",
            (),
        ) or ()

        signals: list[
            RiskSignal
        ] = []

        # -----------------------------------------------------
        # Individual arithmetic mismatch signals
        # -----------------------------------------------------

        for check in checks:

            status = str(
                self._get(
                    check,
                    "status",
                    "",
                )
            ).strip().lower()

            if status != "mismatch":
                continue

            sequence = self._get(
                check,
                "sequence",
            )

            previous_balance = self._get(
                check,
                "previous_balance",
            )

            debit = self._get(
                check,
                "debit",
            )

            credit = self._get(
                check,
                "credit",
            )

            current_balance = self._get(
                check,
                "current_balance",
            )

            expected_balance = self._get(
                check,
                "expected_balance",
            )

            difference = self._get(
                check,
                "difference",
            )

            severity = self._mismatch_severity(
                difference=difference,
                expected_balance=expected_balance,
            )

            confidence = self._mismatch_confidence(
                check
            )

            evidence = {
                "previous_balance":
                    previous_balance,

                "debit":
                    debit,

                "credit":
                    credit,

                "current_balance":
                    current_balance,

                "expected_balance":
                    expected_balance,

                "difference":
                    difference,

                "phase3_status":
                    status,
            }

            signals.append(
                RiskSignal(
                    code="BALANCE_CHAIN_MISMATCH",

                    category="balance",

                    severity=severity,

                    message=(
                        "Running balance does not reconcile "
                        "with the transaction amount."
                    ),

                    score=self.MISMATCH_SCORE,

                    confidence=confidence,

                    transaction_sequence=(
                        sequence
                    ),

                    field_name="balance",

                    expected=expected_balance,

                    actual=current_balance,

                    evidence=evidence,

                    source=(
                        "balance_anomaly_analyzer"
                    ),
                )
            )

        # -----------------------------------------------------
        # Missing balance evidence
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # This is assessment uncertainty, not fraud risk.
        #
        # One aggregate signal is enough. We deliberately do
        # not create one zero-score signal for every missing row.
        # -----------------------------------------------------

        if unverifiable_count > 0:

            coverage = self._safe_ratio(
                checked_count,
                transaction_count,
            )

            signals.append(
                RiskSignal(
                    code="BALANCE_EVIDENCE_INCOMPLETE",

                    category="balance",

                    severity=self._coverage_severity(
                        coverage
                    ),

                    message=(
                        "Some transaction balance transitions "
                        "could not be mathematically verified."
                    ),

                    score=self.UNVERIFIABLE_SCORE,

                    confidence=1.0,

                    evidence={
                        "transaction_count":
                            transaction_count,

                        "checked_count":
                            checked_count,

                        "unverifiable_count":
                            unverifiable_count,

                        "coverage":
                            round(
                                coverage,
                                4,
                            ),
                    },

                    source=(
                        "balance_anomaly_analyzer"
                    ),
                )
            )

        mismatch_rate = self._safe_ratio(
            mismatch_count,
            checked_count,
        )

        coverage = self._safe_ratio(
            checked_count,
            transaction_count,
        )

        phase3_confidence = self._to_float(
            self._get(
                balance_validation,
                "confidence",
                0.0,
            )
        )

        # Assessment confidence incorporates both:
        #
        # 1. Phase 3 arithmetic agreement
        # 2. how much of the statement could actually be checked
        #
        # This prevents:
        #
        #   40/44 reconciled
        #
        # from looking equivalent to:
        #
        #   40/70 fully evidenced
        #
        confidence = (
            phase3_confidence
            * coverage
        )

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        return BalanceAnomalySummary(
            transaction_count=(
                transaction_count
            ),

            checked_count=(
                checked_count
            ),

            reconciled_count=(
                reconciled_count
            ),

            mismatch_count=(
                mismatch_count
            ),

            unverifiable_count=(
                unverifiable_count
            ),

            mismatch_rate=(
                mismatch_rate
            ),

            coverage=(
                coverage
            ),

            signal_count=len(
                signals
            ),

            signals=tuple(
                signals
            ),

            confidence=round(
                confidence,
                4,
            ),
        )

    # =========================================================
    # SEVERITY
    # =========================================================

    def _mismatch_severity(
        self,
        difference: Any,
        expected_balance: Any,
    ) -> str:
        """
        Severity based on relative balance discrepancy.

        Absolute amount alone is a poor measure because a
        ₹5,000 discrepancy has different significance on a
        ₹10,000 balance versus a ₹10,000,000 balance.
        """

        diff = abs(
            self._to_float(
                difference
            )
        )

        expected = abs(
            self._to_float(
                expected_balance
            )
        )

        if diff <= 0:
            return "low"

        if expected <= 0:
            if diff >= 10000:
                return "high"

            if diff >= 1000:
                return "medium"

            return "low"

        ratio = (
            diff / expected
        )

        if ratio >= 0.25:
            return "high"

        if ratio >= 0.05:
            return "medium"

        return "low"

    @staticmethod
    def _coverage_severity(
        coverage: float,
    ) -> str:

        if coverage < 0.50:
            return "medium"

        if coverage < 0.80:
            return "low"

        return "info"

    # =========================================================
    # CONFIDENCE
    # =========================================================

    def _mismatch_confidence(
        self,
        check: Any,
    ) -> float:
        """
        Confidence that the mismatch signal is supported by
        sufficient accounting evidence.
        """

        required_fields = (
            "previous_balance",
            "current_balance",
            "expected_balance",
            "difference",
        )

        available = sum(
            1
            for field_name
            in required_fields
            if self._get(
                check,
                field_name,
            ) is not None
        )

        confidence = (
            available
            / len(
                required_fields
            )
        )

        return round(
            max(
                0.0,
                min(
                    confidence,
                    1.0,
                ),
            ),
            4,
        )

    # =========================================================
    # HELPERS
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

    @staticmethod
    def _safe_ratio(
        numerator: int,
        denominator: int,
    ) -> float:

        if denominator <= 0:
            return 0.0

        return max(
            0.0,
            min(
                numerator / denominator,
                1.0,
            ),
        )

    @staticmethod
    def _to_int(
        value: Any,
    ) -> int:

        try:
            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float:

        if value is None:
            return 0.0

        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0


balance_anomaly_analyzer = (
    BalanceAnomalyAnalyzer()
)