"""
Extraction Reliability Analyzer
===============================

Phase 4 reliability analyzer for Phase 2 extraction evidence.

Purpose
-------
Estimate how trustworthy/complete the extracted transaction evidence is.

This module deliberately separates:

    extraction uncertainty

from:

    fraud / tampering risk

Examples
--------
- OCR usage does NOT mean fraud.
- Missing transaction amounts do NOT mean fraud.
- Unresolved debit/credit direction does NOT mean fraud.
- Low parser confidence does NOT automatically mean fraud.

These conditions reduce assessment confidence and generate
informational reliability signals.

The final RiskAggregator will use this reliability result when
calculating assessment confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
)


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass(frozen=True)
class ExtractionReliabilitySummary:
    """
    Standard Phase 4 extraction reliability result.
    """

    extraction_method: str | None

    ocr_used: bool

    ocr_engine: str | None

    transaction_count: int

    parser_confidence: float

    complete_count: int

    incomplete_count: int

    unresolved_direction_count: int

    completeness_rate: float

    direction_resolution_rate: float

    signal_count: int

    signals: tuple[
        RiskSignal,
        ...,
    ]

    reliability_score: float

    def to_dict(self) -> dict[str, Any]:

        return {
            "extraction_method":
                self.extraction_method,

            "ocr_used":
                self.ocr_used,

            "ocr_engine":
                self.ocr_engine,

            "transaction_count":
                self.transaction_count,

            "parser_confidence":
                round(
                    self.parser_confidence,
                    4,
                ),

            "complete_count":
                self.complete_count,

            "incomplete_count":
                self.incomplete_count,

            "unresolved_direction_count":
                self.unresolved_direction_count,

            "completeness_rate":
                round(
                    self.completeness_rate,
                    4,
                ),

            "direction_resolution_rate":
                round(
                    self.direction_resolution_rate,
                    4,
                ),

            "signal_count":
                self.signal_count,

            "reliability_score":
                round(
                    self.reliability_score,
                    4,
                ),

            "signals": [
                signal.to_dict()
                for signal
                in self.signals
            ],
        }


# ============================================================
# ANALYZER
# ============================================================


class ExtractionReliabilityAnalyzer:
    """
    Analyze extraction reliability using Phase 2 + Phase 3 evidence.

    Risk score contributions from this analyzer are intentionally zero.

    Reliability problems affect assessment confidence, not fraud risk.
    """

    # ========================================================
    # PUBLIC API
    # ========================================================

    def analyze(
        self,
        extraction: Any,
        transaction_validation: Any | None = None,
    ) -> ExtractionReliabilitySummary:
        """
        Analyze extraction reliability.

        Parameters
        ----------
        extraction:
            Phase 2 extraction result or extraction dictionary.

        transaction_validation:
            Optional Phase 3 transaction validation result.

            When available, it provides stronger evidence for:
                complete_count
                incomplete_count
                unresolved_direction_count
        """

        if extraction is None:
            raise ValueError(
                "extraction cannot be None."
            )

        # ----------------------------------------------------
        # Extraction metadata
        # ----------------------------------------------------

        extraction_method = self._normalize_optional_text(
            self._first(
                extraction,
                (
                    "extraction_method",
                    "method",
                ),
            )
        )

        ocr_used = bool(
            self._get(
                extraction,
                "ocr_used",
                False,
            )
        )

        ocr_engine = self._normalize_optional_text(
            self._get(
                extraction,
                "ocr_engine",
            )
        )

        transactions = self._get(
            extraction,
            "transactions",
            (),
        ) or ()

        extraction_transaction_count = self._to_int(
            self._get(
                extraction,
                "transaction_count",
                0,
            )
        )

        if (
            extraction_transaction_count <= 0
            and self._has_length(
                transactions
            )
        ):
            extraction_transaction_count = len(
                transactions
            )

        transaction_count = (
            extraction_transaction_count
        )

        # ----------------------------------------------------
        # Parser confidence
        # ----------------------------------------------------

        parser_confidence = self._to_probability(
            self._first(
                extraction,
                (
                    "transaction_parser_confidence",
                    "parser_confidence",
                ),
                default=0.0,
            )
        )

        # ----------------------------------------------------
        # Phase 3 completeness evidence
        # ----------------------------------------------------

        if transaction_validation is not None:

            phase3_transaction_count = self._to_int(
                self._get(
                    transaction_validation,
                    "transaction_count",
                    0,
                )
            )

            if phase3_transaction_count > 0:
                transaction_count = (
                    phase3_transaction_count
                )

            complete_count = self._to_int(
                self._get(
                    transaction_validation,
                    "complete_count",
                    0,
                )
            )

            incomplete_count = self._to_int(
                self._get(
                    transaction_validation,
                    "incomplete_count",
                    0,
                )
            )

            unresolved_direction_count = self._to_int(
                self._get(
                    transaction_validation,
                    "unresolved_direction_count",
                    0,
                )
            )

        else:

            (
                complete_count,
                incomplete_count,
                unresolved_direction_count,
            ) = self._derive_transaction_quality(
                transactions
            )

        # ----------------------------------------------------
        # Defensive normalization
        # ----------------------------------------------------

        complete_count = max(
            0,
            complete_count,
        )

        incomplete_count = max(
            0,
            incomplete_count,
        )

        unresolved_direction_count = max(
            0,
            unresolved_direction_count,
        )

        transaction_count = max(
            0,
            transaction_count,
        )

        if transaction_count > 0:

            complete_count = min(
                complete_count,
                transaction_count,
            )

            incomplete_count = min(
                incomplete_count,
                transaction_count,
            )

            unresolved_direction_count = min(
                unresolved_direction_count,
                transaction_count,
            )

        # ----------------------------------------------------
        # Rates
        # ----------------------------------------------------

        completeness_rate = self._safe_ratio(
            complete_count,
            transaction_count,
        )

        resolved_direction_count = max(
            0,
            transaction_count
            - unresolved_direction_count,
        )

        direction_resolution_rate = self._safe_ratio(
            resolved_direction_count,
            transaction_count,
        )

        # ----------------------------------------------------
        # Reliability signals
        # ----------------------------------------------------

        signals: list[
            RiskSignal
        ] = []

        # OCR is context, not risk.
        if ocr_used:

            signals.append(
                RiskSignal(
                    code="OCR_EXTRACTION_USED",

                    category="extraction",

                    severity="info",

                    message=(
                        "Statement required OCR-based "
                        "text extraction."
                    ),

                    score=0.0,

                    confidence=1.0,

                    evidence={
                        "extraction_method":
                            extraction_method,

                        "ocr_engine":
                            ocr_engine,
                    },

                    source=(
                        "extraction_reliability_analyzer"
                    ),
                )
            )

        # Low parser confidence.
        if parser_confidence < 0.80:

            signals.append(
                RiskSignal(
                    code="LOW_PARSER_CONFIDENCE",

                    category="extraction",

                    severity=(
                        self._parser_confidence_severity(
                            parser_confidence
                        )
                    ),

                    message=(
                        "Transaction parser confidence is "
                        "below the preferred reliability "
                        "threshold."
                    ),

                    score=0.0,

                    confidence=1.0,

                    field_name=(
                        "transaction_parser_confidence"
                    ),

                    expected=">= 0.80",

                    actual=parser_confidence,

                    evidence={
                        "parser_confidence":
                            parser_confidence,
                    },

                    source=(
                        "extraction_reliability_analyzer"
                    ),
                )
            )

        # Incomplete transaction evidence.
        if incomplete_count > 0:

            signals.append(
                RiskSignal(
                    code="INCOMPLETE_TRANSACTION_EVIDENCE",

                    category="extraction",

                    severity=(
                        self._coverage_severity(
                            completeness_rate
                        )
                    ),

                    message=(
                        "Some extracted transactions do not "
                        "contain all fields required for full "
                        "verification."
                    ),

                    score=0.0,

                    confidence=1.0,

                    evidence={
                        "transaction_count":
                            transaction_count,

                        "complete_count":
                            complete_count,

                        "incomplete_count":
                            incomplete_count,

                        "completeness_rate":
                            round(
                                completeness_rate,
                                4,
                            ),
                    },

                    source=(
                        "extraction_reliability_analyzer"
                    ),
                )
            )

        # Unresolved debit/credit direction.
        if unresolved_direction_count > 0:

            signals.append(
                RiskSignal(
                    code="UNRESOLVED_TRANSACTION_DIRECTION",

                    category="extraction",

                    severity=(
                        self._direction_severity(
                            direction_resolution_rate
                        )
                    ),

                    message=(
                        "Debit or credit direction could not "
                        "be resolved for some transactions."
                    ),

                    score=0.0,

                    confidence=1.0,

                    evidence={
                        "transaction_count":
                            transaction_count,

                        "unresolved_direction_count":
                            unresolved_direction_count,

                        "direction_resolution_rate":
                            round(
                                direction_resolution_rate,
                                4,
                            ),
                    },

                    source=(
                        "extraction_reliability_analyzer"
                    ),
                )
            )

        # No usable transaction evidence.
        if transaction_count <= 0:

            signals.append(
                RiskSignal(
                    code="NO_TRANSACTION_EVIDENCE",

                    category="extraction",

                    severity="high",

                    message=(
                        "No transaction evidence is available "
                        "for risk assessment."
                    ),

                    score=0.0,

                    confidence=1.0,

                    source=(
                        "extraction_reliability_analyzer"
                    ),
                )
            )

        # ----------------------------------------------------
        # Reliability score
        # ----------------------------------------------------
        #
        # Weighted evidence quality:
        #
        # parser quality       = 40%
        # transaction complete = 35%
        # direction resolved   = 25%
        #
        # OCR itself receives NO penalty.
        #
        # A perfectly extracted OCR document can therefore
        # still achieve reliability 1.0.
        # ----------------------------------------------------

        if transaction_count <= 0:

            reliability_score = 0.0

        else:

            reliability_score = (
                parser_confidence
                * 0.40
                + completeness_rate
                * 0.35
                + direction_resolution_rate
                * 0.25
            )

        reliability_score = max(
            0.0,
            min(
                reliability_score,
                1.0,
            ),
        )

        return ExtractionReliabilitySummary(
            extraction_method=(
                extraction_method
            ),

            ocr_used=(
                ocr_used
            ),

            ocr_engine=(
                ocr_engine
            ),

            transaction_count=(
                transaction_count
            ),

            parser_confidence=(
                parser_confidence
            ),

            complete_count=(
                complete_count
            ),

            incomplete_count=(
                incomplete_count
            ),

            unresolved_direction_count=(
                unresolved_direction_count
            ),

            completeness_rate=(
                completeness_rate
            ),

            direction_resolution_rate=(
                direction_resolution_rate
            ),

            signal_count=len(
                signals
            ),

            signals=tuple(
                signals
            ),

            reliability_score=round(
                reliability_score,
                4,
            ),
        )

    # ========================================================
    # FALLBACK TRANSACTION QUALITY
    # ========================================================

    def _derive_transaction_quality(
        self,
        transactions: Any,
    ) -> tuple[int, int, int]:
        """
        Derive basic quality metrics directly from transactions
        when Phase 3 validation is unavailable.

        This is only a fallback. Phase 3 evidence is preferred.
        """

        if not transactions:
            return 0, 0, 0

        complete_count = 0
        incomplete_count = 0
        unresolved_count = 0

        for transaction in transactions:

            date = self._get(
                transaction,
                "date",
            )

            description = self._get(
                transaction,
                "description",
            )

            debit = self._get(
                transaction,
                "debit",
            )

            credit = self._get(
                transaction,
                "credit",
            )

            balance = self._get(
                transaction,
                "balance",
            )

            has_amount = (
                debit is not None
                or credit is not None
            )

            direction_resolved = (
                (debit is not None)
                ^ (credit is not None)
            )

            complete = bool(
                date
                and description
                and has_amount
                and balance is not None
                and direction_resolved
            )

            if complete:
                complete_count += 1

            else:
                incomplete_count += 1

            if not direction_resolved:
                unresolved_count += 1

        return (
            complete_count,
            incomplete_count,
            unresolved_count,
        )

    # ========================================================
    # SEVERITY
    # ========================================================

    @staticmethod
    def _parser_confidence_severity(
        confidence: float,
    ) -> str:

        if confidence < 0.40:
            return "high"

        if confidence < 0.60:
            return "medium"

        if confidence < 0.80:
            return "low"

        return "info"

    @staticmethod
    def _coverage_severity(
        completeness_rate: float,
    ) -> str:

        if completeness_rate < 0.50:
            return "high"

        if completeness_rate < 0.75:
            return "medium"

        if completeness_rate < 0.95:
            return "low"

        return "info"

    @staticmethod
    def _direction_severity(
        resolution_rate: float,
    ) -> str:

        if resolution_rate < 0.50:
            return "high"

        if resolution_rate < 0.75:
            return "medium"

        if resolution_rate < 0.95:
            return "low"

        return "info"

    # ========================================================
    # HELPERS
    # ========================================================

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
    def _to_probability(
        value: Any,
    ) -> float:

        try:
            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                number,
                1.0,
            ),
        )

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
    def _has_length(
        value: Any,
    ) -> bool:

        try:
            len(
                value
            )
            return True

        except TypeError:
            return False


extraction_reliability_analyzer = (
    ExtractionReliabilityAnalyzer()
)