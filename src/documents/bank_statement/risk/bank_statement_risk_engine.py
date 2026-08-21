"""
Bank Statement Risk Engine
==========================

Phase 4 orchestration layer.

This engine combines:

Phase 1
    Integrity evidence

Phase 2
    Extraction evidence

Phase 3
    Validation evidence

Phase 4
    Transaction anomaly analysis
    Balance anomaly analysis
    Statement consistency analysis
    Extraction reliability analysis
    Integrity signal adaptation
    Risk aggregation

The engine does NOT modify extraction or validation results.

It produces:
    - explainable risk signals
    - bounded risk score
    - risk level
    - assessment confidence
    - component-level summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
)

from src.documents.bank_statement.risk.transaction_anomaly_analyzer import (
    transaction_anomaly_analyzer,
)

from src.documents.bank_statement.risk.balance_anomaly_analyzer import (
    balance_anomaly_analyzer,
)

from src.documents.bank_statement.risk.statement_consistency_analyzer import (
    statement_consistency_analyzer,
)

from src.documents.bank_statement.risk.extraction_reliability_analyzer import (
    extraction_reliability_analyzer,
)

from src.documents.bank_statement.risk.integrity_signal_adapter import (
    integrity_signal_adapter,
)

from src.documents.bank_statement.risk.risk_aggregator import (
    risk_aggregator,
)


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass(frozen=True)
class BankStatementRiskResult:
    """
    Final Phase 4 bank-statement risk result.
    """

    filename: str | None

    transaction_count: int

    risk_score: float

    risk_level: str

    assessment_confidence: float

    positive_signal_count: int

    informational_signal_count: int

    total_signal_count: int

    category_scores: dict[str, float]

    transaction_analysis: dict[str, Any]

    balance_analysis: dict[str, Any]

    statement_analysis: dict[str, Any]

    extraction_reliability: dict[str, Any]

    integrity_analysis: dict[str, Any]

    signals: tuple[RiskSignal, ...]

    def to_dict(self) -> dict[str, Any]:

        return {
            "filename":
                self.filename,

            "transaction_count":
                self.transaction_count,

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

            "components": {
                "transaction_analysis":
                    self.transaction_analysis,

                "balance_analysis":
                    self.balance_analysis,

                "statement_analysis":
                    self.statement_analysis,

                "extraction_reliability":
                    self.extraction_reliability,

                "integrity_analysis":
                    self.integrity_analysis,
            },

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
        }


# ============================================================
# ENGINE
# ============================================================


class BankStatementRiskEngine:
    """
    Phase 4 bank-statement risk orchestration engine.
    """

    # ========================================================
    # PUBLIC API
    # ========================================================

    def analyze(
        self,
        extraction: Any,
        validation: Any,
        integrity: Any | None = None,
    ) -> BankStatementRiskResult:
        """
        Run complete Phase 4 analysis.

        Parameters
        ----------
        extraction:
            Phase 2 extraction result or dictionary.

        validation:
            Phase 3 bank-statement validation result or dictionary.

        integrity:
            Optional Phase 1 integrity result.
        """

        if extraction is None:
            raise ValueError(
                "extraction cannot be None."
            )

        if validation is None:
            raise ValueError(
                "validation cannot be None."
            )

        filename = self._normalize_optional_text(
            self._first(
                extraction,
                (
                    "filename",
                    "file_name",
                ),
            )
        )

        if filename is None:

            filename = self._normalize_optional_text(
                self._first(
                    validation,
                    (
                        "filename",
                        "file_name",
                    ),
                )
            )

        transactions = self._get(
            extraction,
            "transactions",
            (),
        ) or ()

        transaction_count = self._to_int(
            self._get(
                extraction,
                "transaction_count",
                0,
            )
        )

        if transaction_count <= 0:

            try:
                transaction_count = len(
                    transactions
                )

            except TypeError:
                transaction_count = 0

        # ----------------------------------------------------
        # Phase 3 component results
        # ----------------------------------------------------

        transaction_validation = self._first(
            validation,
            (
                "transaction_validation",
                "transactions",
            ),
            default=None,
        )

        balance_validation = self._first(
            validation,
            (
                "balance_chain_validation",
                "balance_chain",
            ),
            default=None,
        )

        statement_validation = self._first(
            validation,
            (
                "statement_validation",
                "statement",
            ),
            default=None,
        )

        # ----------------------------------------------------
        # 1. Transaction anomaly analysis
        # ----------------------------------------------------

        transaction_result = (
            transaction_anomaly_analyzer.analyze(
                transactions
            )
        )

        # ----------------------------------------------------
        # 2. Balance anomaly analysis
        # ----------------------------------------------------

        if balance_validation is not None:

            balance_result = (
                balance_anomaly_analyzer.analyze(
                    balance_validation
                )
            )

        else:

            balance_result = None

        # ----------------------------------------------------
        # 3. Statement consistency
        # ----------------------------------------------------

        if statement_validation is not None:

            statement_result = (
                statement_consistency_analyzer.analyze(
                    statement_validation
                )
            )

        else:

            statement_result = None

        # ----------------------------------------------------
        # 4. Extraction reliability
        # ----------------------------------------------------

        extraction_result = (
            extraction_reliability_analyzer.analyze(
                extraction=extraction,
                transaction_validation=(
                    transaction_validation
                ),
            )
        )

        # ----------------------------------------------------
        # 5. Integrity evidence
        # ----------------------------------------------------

        integrity_result = (
            integrity_signal_adapter.analyze(
                integrity
            )
        )

        # ----------------------------------------------------
        # 6. Collect all signals
        # ----------------------------------------------------

        all_signals: list[
            RiskSignal
        ] = []

        all_signals.extend(
            self._signals_from_result(
                transaction_result
            )
        )

        if balance_result is not None:

            all_signals.extend(
                self._signals_from_result(
                    balance_result
                )
            )

        if statement_result is not None:

            all_signals.extend(
                self._signals_from_result(
                    statement_result
                )
            )

        all_signals.extend(
            self._signals_from_result(
                extraction_result
            )
        )

        all_signals.extend(
            self._signals_from_result(
                integrity_result
            )
        )

        # ----------------------------------------------------
        # 7. Assessment reliability evidence
        # ----------------------------------------------------

        reliability_scores: list[
            float
        ] = [
            extraction_result.reliability_score,
        ]

        if balance_result is not None:

            reliability_scores.append(
                balance_result.confidence
            )

        if statement_result is not None:

            reliability_scores.append(
                statement_result.confidence
            )

        # Integrity confidence is included only when Phase 1
        # integrity evidence was actually supplied.
        #
        # Missing optional integrity input should not force
        # assessment confidence to zero.
        if integrity is not None:

            integrity_confidence = (
                integrity_result.confidence
            )

            if integrity_confidence > 0:

                reliability_scores.append(
                    integrity_confidence
                )

        # ----------------------------------------------------
        # 8. Aggregate
        # ----------------------------------------------------

        aggregate = risk_aggregator.aggregate(
            signals=all_signals,
            reliability_scores=(
                reliability_scores
            ),
        )

        # ----------------------------------------------------
        # 9. Component dictionaries
        # ----------------------------------------------------

        transaction_dict = self._to_dict(
            transaction_result
        )

        balance_dict = (
            self._to_dict(
                balance_result
            )
            if balance_result is not None
            else {}
        )

        statement_dict = (
            self._to_dict(
                statement_result
            )
            if statement_result is not None
            else {}
        )

        extraction_dict = self._to_dict(
            extraction_result
        )

        integrity_dict = self._to_dict(
            integrity_result
        )

        return BankStatementRiskResult(
            filename=filename,

            transaction_count=(
                transaction_count
            ),

            risk_score=(
                aggregate.risk_score
            ),

            risk_level=(
                aggregate.risk_level
            ),

            assessment_confidence=(
                aggregate.assessment_confidence
            ),

            positive_signal_count=(
                aggregate.positive_signal_count
            ),

            informational_signal_count=(
                aggregate.informational_signal_count
            ),

            total_signal_count=(
                aggregate.total_signal_count
            ),

            category_scores=dict(
                aggregate.category_scores
            ),

            transaction_analysis=(
                transaction_dict
            ),

            balance_analysis=(
                balance_dict
            ),

            statement_analysis=(
                statement_dict
            ),

            extraction_reliability=(
                extraction_dict
            ),

            integrity_analysis=(
                integrity_dict
            ),

            signals=tuple(
                aggregate.signals
            ),
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _signals_from_result(
        result: Any,
    ) -> list[RiskSignal]:

        if result is None:
            return []

        signals = getattr(
            result,
            "signals",
            None,
        )

        if signals is None:

            if isinstance(
                result,
                Mapping,
            ):
                signals = result.get(
                    "signals",
                    (),
                )

        if not signals:
            return []

        # The Phase 4 analyzers return RiskSignal objects.
        # Ignore unexpected values rather than crashing the
        # complete risk engine.
        return [
            signal
            for signal in signals
            if isinstance(
                signal,
                RiskSignal,
            )
        ]

    @staticmethod
    def _to_dict(
        result: Any,
    ) -> dict[str, Any]:

        if result is None:
            return {}

        if hasattr(
            result,
            "to_dict",
        ):

            value = result.to_dict()

            if isinstance(
                value,
                Mapping,
            ):
                return dict(
                    value
                )

        if isinstance(
            result,
            Mapping,
        ):
            return dict(
                result
            )

        return {}

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

    def _first(
        self,
        obj: Any,
        keys: tuple[str, ...],
        default: Any = None,
    ) -> Any:

        for key in keys:

            value = self._get(
                obj,
                key,
                None,
            )

            if value is not None:
                return value

        return default

    @staticmethod
    def _normalize_optional_text(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        text = str(
            value
        ).strip()

        return (
            text
            if text
            else None
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


bank_statement_risk_engine = (
    BankStatementRiskEngine()
)