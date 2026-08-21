"""
Risk Aggregator
===============

Phase 4 risk aggregation layer.

Responsibilities
----------------
1. Collect RiskSignal objects from Phase 4 analyzers.
2. Separate informational/reliability signals from positive-risk signals.
3. Prevent repeated signals from exploding the final score.
4. Apply per-code caps.
5. Apply per-category caps.
6. Weight risk by signal confidence.
7. Produce a normalized 0-100 risk score.
8. Produce an explainable risk level.
9. Keep risk score separate from assessment confidence.

Important
---------
Risk score answers:

    "How much suspicious evidence exists?"

Assessment confidence answers:

    "How reliable is our ability to make that assessment?"

These must NOT be treated as the same quantity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
)


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass(frozen=True)
class RiskAggregationResult:
    """
    Final output of Phase 4 risk aggregation.
    """

    raw_risk_score: float

    risk_score: float

    risk_level: str

    assessment_confidence: float

    positive_signal_count: int

    informational_signal_count: int

    total_signal_count: int

    category_scores: dict[str, float]

    code_scores: dict[str, float]

    signals: tuple[RiskSignal, ...]

    def to_dict(self) -> dict[str, Any]:

        return {
            "raw_risk_score":
                round(
                    self.raw_risk_score,
                    4,
                ),

            "risk_score":
                round(
                    self.risk_score,
                    2,
                ),

            "risk_level":
                self.risk_level,

            "assessment_confidence":
                round(
                    self.assessment_confidence,
                    4,
                ),

            "positive_signal_count":
                self.positive_signal_count,

            "informational_signal_count":
                self.informational_signal_count,

            "total_signal_count":
                self.total_signal_count,

            "category_scores": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in self.category_scores.items()
            },

            "code_scores": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in self.code_scores.items()
            },

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
        }


# ============================================================
# AGGREGATOR
# ============================================================


class RiskAggregator:
    """
    Aggregate Phase 4 signals into a bounded risk score.
    """

    # --------------------------------------------------------
    # Per-code caps
    # --------------------------------------------------------
    #
    # Repeated occurrences are meaningful, but should not
    # accumulate without limit.
    #
    # Example:
    # 100 duplicate-like transactions must NOT automatically
    # produce 100 * 3 = 300 risk points.
    # --------------------------------------------------------

    CODE_CAPS = {
        "UNUSUAL_TRANSACTION_AMOUNT": 12.0,
        "DUPLICATE_LIKE_TRANSACTION": 10.0,
        "TRANSACTION_BURST": 8.0,

        "BALANCE_CHAIN_MISMATCH": 24.0,

        "TRANSACTION_OUTSIDE_PERIOD": 10.0,
        "TRANSACTION_ORDERING_INCONSISTENCY": 8.0,
        "DUPLICATE_TRANSACTION": 10.0,
        "INVALID_STATEMENT_PERIOD": 10.0,

        "INTEGRITY_OBSERVATION": 20.0,
        "INTEGRITY_TAMPERING_FLAG": 30.0,
        "FILE_EXTENSION_MISMATCH": 20.0,
        "INTEGRITY_SUSPICION_FLAG": 15.0,
    }

    DEFAULT_CODE_CAP = 15.0

    # --------------------------------------------------------
    # Category caps
    # --------------------------------------------------------

    CATEGORY_CAPS = {
        "transaction": 25.0,
        "balance": 30.0,
        "statement": 20.0,
        "integrity": 40.0,
        "extraction": 0.0,
    }

    DEFAULT_CATEGORY_CAP = 20.0

    # ========================================================
    # PUBLIC API
    # ========================================================

    def aggregate(
        self,
        signals: Iterable[RiskSignal] | None,
        reliability_scores: Iterable[float] | None = None,
    ) -> RiskAggregationResult:
        """
        Aggregate signals.

        Parameters
        ----------
        signals:
            Phase 4 RiskSignal objects.

        reliability_scores:
            Independent assessment-quality values in [0, 1].

            Examples:
            - extraction reliability
            - balance validation confidence
            - statement validation confidence
            - integrity analysis confidence

            These affect assessment confidence, NOT risk score.
        """

        signal_list = tuple(
            signal
            for signal in (signals or ())
            if signal is not None
        )

        positive_signals = [
            signal
            for signal in signal_list
            if self._to_float(
                signal.score
            ) > 0.0
        ]

        informational_signals = [
            signal
            for signal in signal_list
            if self._to_float(
                signal.score
            ) <= 0.0
        ]

        # ----------------------------------------------------
        # STEP 1:
        # Confidence-weight each positive signal.
        # ----------------------------------------------------

        weighted_by_code: dict[
            str,
            float,
        ] = defaultdict(
            float
        )

        code_category: dict[
            str,
            str,
        ] = {}

        for signal in positive_signals:

            code = self._normalize_code(
                signal.code
            )

            category = self._normalize_category(
                signal.category
            )

            score = max(
                0.0,
                self._to_float(
                    signal.score
                ),
            )

            confidence = self._probability(
                signal.confidence
            )

            weighted_score = (
                score
                * confidence
            )

            weighted_by_code[
                code
            ] += weighted_score

            code_category[
                code
            ] = category

        # ----------------------------------------------------
        # STEP 2:
        # Apply per-code caps.
        # ----------------------------------------------------

        capped_code_scores: dict[
            str,
            float,
        ] = {}

        for code, score in weighted_by_code.items():

            cap = self.CODE_CAPS.get(
                code,
                self.DEFAULT_CODE_CAP,
            )

            capped_code_scores[
                code
            ] = min(
                score,
                cap,
            )

        # ----------------------------------------------------
        # STEP 3:
        # Group capped code scores by category.
        # ----------------------------------------------------

        category_raw_scores: dict[
            str,
            float,
        ] = defaultdict(
            float
        )

        for code, score in capped_code_scores.items():

            category = code_category.get(
                code,
                "other",
            )

            category_raw_scores[
                category
            ] += score

        # ----------------------------------------------------
        # STEP 4:
        # Apply category caps.
        # ----------------------------------------------------

        category_scores: dict[
            str,
            float,
        ] = {}

        for category, score in category_raw_scores.items():

            cap = self.CATEGORY_CAPS.get(
                category,
                self.DEFAULT_CATEGORY_CAP,
            )

            category_scores[
                category
            ] = min(
                score,
                cap,
            )

        raw_risk_score = sum(
            category_scores.values()
        )

        # Overall score is bounded 0-100.
        risk_score = max(
            0.0,
            min(
                raw_risk_score,
                100.0,
            ),
        )

        assessment_confidence = (
            self._assessment_confidence(
                reliability_scores
            )
        )

        risk_level = self._risk_level(
            risk_score=risk_score,
            signals=positive_signals,
        )

        return RiskAggregationResult(
            raw_risk_score=round(
                raw_risk_score,
                4,
            ),

            risk_score=round(
                risk_score,
                2,
            ),

            risk_level=risk_level,

            assessment_confidence=round(
                assessment_confidence,
                4,
            ),

            positive_signal_count=len(
                positive_signals
            ),

            informational_signal_count=len(
                informational_signals
            ),

            total_signal_count=len(
                signal_list
            ),

            category_scores=dict(
                sorted(
                    category_scores.items()
                )
            ),

            code_scores=dict(
                sorted(
                    capped_code_scores.items()
                )
            ),

            signals=signal_list,
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def _risk_level(
        self,
        risk_score: float,
        signals: list[RiskSignal],
    ) -> str:
        """
        Convert numerical score into an operational risk level.

        Explicit critical integrity evidence receives a minimum
        HIGH classification even if the numeric score is low
        because of confidence weighting.
        """

        has_critical_integrity = any(
            self._normalize_category(
                signal.category
            ) == "integrity"
            and str(
                signal.severity
            ).strip().lower()
            == "critical"
            for signal in signals
        )

        if risk_score >= 60.0:
            return "critical"

        if (
            risk_score >= 35.0
            or has_critical_integrity
        ):
            return "high"

        if risk_score >= 15.0:
            return "medium"

        if risk_score > 0.0:
            return "low"

        return "minimal"

    # ========================================================
    # ASSESSMENT CONFIDENCE
    # ========================================================

    def _assessment_confidence(
        self,
        reliability_scores: Iterable[float] | None,
    ) -> float:
        """
        Calculate independent assessment confidence.

        We use the harmonic mean rather than the arithmetic mean.

        Why?
        ----
        A very weak pipeline component should meaningfully reduce
        overall assessment confidence.

        Example:
            extraction = 0.50
            balance    = 1.00
            statement  = 1.00

        Arithmetic mean would hide too much of the weak extraction.

        Harmonic mean penalizes weak links more appropriately.
        """

        values = [
            self._probability(
                value
            )
            for value
            in (reliability_scores or ())
            if value is not None
        ]

        if not values:
            return 0.0

        if any(
            value <= 0.0
            for value in values
        ):
            return 0.0

        denominator = sum(
            1.0 / value
            for value
            in values
        )

        if denominator <= 0:
            return 0.0

        harmonic_mean = (
            len(
                values
            )
            / denominator
        )

        return max(
            0.0,
            min(
                harmonic_mean,
                1.0,
            ),
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize_code(
        value: Any,
    ) -> str:

        return str(
            value
            or "UNKNOWN"
        ).strip().upper()

    @staticmethod
    def _normalize_category(
        value: Any,
    ) -> str:

        return str(
            value
            or "other"
        ).strip().lower()

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float:

        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @classmethod
    def _probability(
        cls,
        value: Any,
    ) -> float:

        number = cls._to_float(
            value
        )

        return max(
            0.0,
            min(
                number,
                1.0,
            ),
        )


risk_aggregator = RiskAggregator()