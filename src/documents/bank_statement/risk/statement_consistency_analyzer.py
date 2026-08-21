"""
Statement Consistency Analyzer
==============================

Phase 4 analyzer/adapter for statement-level consistency evidence.

Consumes Phase 3 StatementValidator output.

It converts:
- transactions outside statement period
- transaction ordering inconsistencies
- duplicate transaction evidence
- invalid statement-period metadata

into explainable Phase 4 risk signals.

Design principles
-----------------
1. Phase 3 remains the source of deterministic validation truth.
2. Phase 4 interprets the risk significance of that evidence.
3. A validation issue is not automatically fraud evidence.
4. Small statement-boundary spillover is treated differently from
   material out-of-period activity.
5. No bank-specific rules are used.
6. Unknown evidence is preserved rather than silently discarded.

It does NOT:
- parse transaction tables
- recalculate Phase 3 validation
- automatically call anomalies fraud
- contain bank-specific logic
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping

from src.documents.bank_statement.risk.models import (
    RiskSignal,
)


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass(frozen=True)
class StatementConsistencySummary:
    """
    Phase 4 statement-consistency result.
    """

    transaction_count: int

    period_valid: bool

    outside_period_count: int

    ordering_issue_count: int

    duplicate_count: int

    issue_count: int

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

            "period_valid":
                self.period_valid,

            "outside_period_count":
                self.outside_period_count,

            "ordering_issue_count":
                self.ordering_issue_count,

            "duplicate_count":
                self.duplicate_count,

            "issue_count":
                self.issue_count,

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


# ============================================================
# ANALYZER
# ============================================================


class StatementConsistencyAnalyzer:
    """
    Translate Phase 3 statement validation into Phase 4 signals.

    Important distinction
    ---------------------
    Phase 3 answers:

        "Does this transaction violate the declared period?"

    Phase 4 answers:

        "How suspicious is that violation?"

    A transaction one day outside a statement boundary is still
    a Phase 3 validation issue, but it is weak evidence of document
    manipulation by itself.

    Material out-of-period activity remains positive risk evidence.
    """

    # ========================================================
    # RISK SCORES
    # ========================================================

    OUTSIDE_PERIOD_SCORE = 2.0

    ORDERING_SCORE = 2.0

    DUPLICATE_SCORE = 3.0

    INVALID_PERIOD_SCORE = 5.0

    # ========================================================
    # PERIOD BOUNDARY POLICY
    # ========================================================

    # A transaction exactly one calendar day outside the declared
    # period is considered near-boundary context.
    #
    # Phase 3 still reports it as outside-period.
    #
    # Phase 4 preserves the evidence but assigns zero direct risk.
    #
    # This is generic and bank-independent.
    NEAR_BOUNDARY_DAYS = 1

    # ========================================================
    # PUBLIC API
    # ========================================================

    def analyze(
        self,
        statement_validation: Any,
    ) -> StatementConsistencySummary:
        """
        Analyze Phase 3 statement-level validation evidence.
        """

        if statement_validation is None:
            raise ValueError(
                "statement_validation cannot be None."
            )

        transaction_count = self._to_int(
            self._get(
                statement_validation,
                "transaction_count",
                0,
            )
        )

        period_valid = bool(
            self._get(
                statement_validation,
                "period_valid",
                True,
            )
        )

        outside_period_count = self._to_int(
            self._get(
                statement_validation,
                "outside_period_count",
                0,
            )
        )

        ordering_issue_count = self._to_int(
            self._get(
                statement_validation,
                "ordering_issue_count",
                0,
            )
        )

        duplicate_count = self._to_int(
            self._get(
                statement_validation,
                "duplicate_count",
                0,
            )
        )

        issue_count = self._to_int(
            self._get(
                statement_validation,
                "issue_count",
                0,
            )
        )

        issues = self._get(
            statement_validation,
            "issues",
            (),
        ) or ()

        signals: list[
            RiskSignal
        ] = []

        # ----------------------------------------------------
        # Track issue categories represented by detailed
        # Phase 3 evidence.
        #
        # This prevents aggregate fallback signals from
        # duplicating detailed issue signals.
        # ----------------------------------------------------

        represented_categories: set[
            str
        ] = set()

        for issue in issues:

            signal = self._issue_to_signal(
                issue
            )

            if signal is None:
                continue

            signals.append(
                signal
            )

            represented_categories.add(
                signal.code
            )

        # ====================================================
        # INVALID PERIOD METADATA
        # ====================================================

        if (
            not period_valid
            and "INVALID_STATEMENT_PERIOD"
            not in represented_categories
        ):

            signals.append(
                RiskSignal(
                    code=(
                        "INVALID_STATEMENT_PERIOD"
                    ),

                    category="statement",

                    severity="medium",

                    message=(
                        "Statement period metadata is "
                        "internally inconsistent."
                    ),

                    score=(
                        self.INVALID_PERIOD_SCORE
                    ),

                    confidence=0.95,

                    source=(
                        "statement_consistency_analyzer"
                    ),
                )
            )

        # ====================================================
        # AGGREGATE FALLBACK SIGNALS
        # ====================================================
        #
        # Phase 3 normally provides detailed issues.
        #
        # These fallbacks exist only for summarized validation
        # dictionaries where detailed issue evidence is absent.
        #
        # Because summarized evidence does not contain actual
        # boundary distance, it cannot safely receive the
        # near-boundary exemption.
        # ====================================================

        if (
            outside_period_count > 0
            and not self._has_code_prefix(
                represented_categories,
                "TRANSACTION_OUTSIDE_PERIOD",
            )
            and not self._has_code_prefix(
                represented_categories,
                "TRANSACTION_BOUNDARY_CONTEXT",
            )
        ):

            signals.append(
                RiskSignal(
                    code=(
                        "TRANSACTION_OUTSIDE_PERIOD"
                    ),

                    category="statement",

                    severity="low",

                    message=(
                        "One or more transactions fall "
                        "outside the detected statement period."
                    ),

                    score=(
                        self.OUTSIDE_PERIOD_SCORE
                    ),

                    confidence=0.90,

                    evidence={
                        "outside_period_count":
                            outside_period_count,

                        "boundary_distance_known":
                            False,
                    },

                    source=(
                        "statement_consistency_analyzer"
                    ),
                )
            )

        if (
            ordering_issue_count > 0
            and not self._has_code_prefix(
                represented_categories,
                "TRANSACTION_ORDERING",
            )
        ):

            signals.append(
                RiskSignal(
                    code=(
                        "TRANSACTION_ORDERING_INCONSISTENCY"
                    ),

                    category="statement",

                    severity="low",

                    message=(
                        "Transaction dates contain one or "
                        "more ordering inconsistencies."
                    ),

                    score=(
                        self.ORDERING_SCORE
                    ),

                    confidence=0.85,

                    evidence={
                        "ordering_issue_count":
                            ordering_issue_count,
                    },

                    source=(
                        "statement_consistency_analyzer"
                    ),
                )
            )

        if (
            duplicate_count > 0
            and not self._has_code_prefix(
                represented_categories,
                "DUPLICATE_TRANSACTION",
            )
        ):

            signals.append(
                RiskSignal(
                    code=(
                        "DUPLICATE_TRANSACTION"
                    ),

                    category="statement",

                    severity="low",

                    message=(
                        "Phase 3 detected duplicate "
                        "transaction evidence."
                    ),

                    score=(
                        self.DUPLICATE_SCORE
                    ),

                    confidence=0.90,

                    evidence={
                        "duplicate_count":
                            duplicate_count,
                    },

                    source=(
                        "statement_consistency_analyzer"
                    ),
                )
            )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        phase3_confidence = self._to_float(
            self._get(
                statement_validation,
                "confidence",
                0.0,
            )
        )

        confidence = max(
            0.0,
            min(
                phase3_confidence,
                1.0,
            ),
        )

        return StatementConsistencySummary(
            transaction_count=(
                transaction_count
            ),

            period_valid=(
                period_valid
            ),

            outside_period_count=(
                outside_period_count
            ),

            ordering_issue_count=(
                ordering_issue_count
            ),

            duplicate_count=(
                duplicate_count
            ),

            issue_count=(
                issue_count
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

    # ========================================================
    # ISSUE ADAPTER
    # ========================================================

    def _issue_to_signal(
        self,
        issue: Any,
    ) -> RiskSignal | None:
        """
        Convert known Phase 3 issue types into Phase 4 signals.

        Unknown issues are preserved as informational evidence
        rather than silently discarded.
        """

        code = str(
            self._get(
                issue,
                "code",
                "",
            )
        ).strip().upper()

        if not code:
            return None

        sequence = self._get(
            issue,
            "transaction_sequence",
        )

        field_name = self._get(
            issue,
            "field_name",
        )

        expected = self._get(
            issue,
            "expected",
        )

        actual = self._get(
            issue,
            "actual",
        )

        message = str(
            self._get(
                issue,
                "message",
                code,
            )
        )

        # ====================================================
        # OUTSIDE STATEMENT PERIOD
        # ====================================================

        if (
            "BEFORE_PERIOD" in code
            or "AFTER_PERIOD" in code
            or "OUTSIDE_PERIOD" in code
        ):

            boundary_distance = (
                self._boundary_distance_days(
                    code=code,
                    expected=expected,
                    actual=actual,
                )
            )

            # ------------------------------------------------
            # NEAR-BOUNDARY CONTEXT
            # ------------------------------------------------
            #
            # Example:
            #
            # statement start = 2026-01-23
            # transaction     = 2026-01-22
            #
            # Phase 3:
            #     correctly reports BEFORE_PERIOD.
            #
            # Phase 4:
            #     preserves the observation but does not
            #     independently treat a one-day boundary
            #     spillover as document fraud evidence.
            # ------------------------------------------------

            if (
                boundary_distance is not None
                and boundary_distance
                <= self.NEAR_BOUNDARY_DAYS
            ):

                return RiskSignal(
                    code=(
                        "TRANSACTION_BOUNDARY_CONTEXT"
                    ),

                    category="statement",

                    severity="info",

                    message=(
                        "Transaction falls immediately outside "
                        "the declared statement-period boundary. "
                        "The validation issue is preserved as "
                        "context but does not independently "
                        "increase document risk."
                    ),

                    score=0.0,

                    confidence=0.90,

                    transaction_sequence=(
                        sequence
                    ),

                    field_name=(
                        field_name
                    ),

                    expected=(
                        expected
                    ),

                    actual=(
                        actual
                    ),

                    evidence={
                        "phase3_issue_code":
                            code,

                        "boundary_distance_days":
                            boundary_distance,

                        "near_boundary":
                            True,

                        "behavioral_context":
                            True,

                        "direct_risk_contribution":
                            0.0,
                    },

                    source=(
                        "statement_consistency_analyzer"
                    ),
                )

            # ------------------------------------------------
            # MATERIAL OUT-OF-PERIOD ACTIVITY
            # ------------------------------------------------

            return RiskSignal(
                code=(
                    "TRANSACTION_OUTSIDE_PERIOD"
                ),

                category="statement",

                severity="low",

                message=message,

                score=(
                    self.OUTSIDE_PERIOD_SCORE
                ),

                confidence=0.90,

                transaction_sequence=(
                    sequence
                ),

                field_name=(
                    field_name
                ),

                expected=(
                    expected
                ),

                actual=(
                    actual
                ),

                evidence={
                    "phase3_issue_code":
                        code,

                    "boundary_distance_days":
                        boundary_distance,

                    "near_boundary":
                        False,

                    "direct_risk_contribution":
                        self.OUTSIDE_PERIOD_SCORE,
                },

                source=(
                    "statement_consistency_analyzer"
                ),
            )

        # ====================================================
        # ORDERING
        # ====================================================

        if (
            "ORDER" in code
            or "CHRONOLOG" in code
        ):

            return RiskSignal(
                code=(
                    "TRANSACTION_ORDERING_INCONSISTENCY"
                ),

                category="statement",

                severity="low",

                message=message,

                score=(
                    self.ORDERING_SCORE
                ),

                confidence=0.85,

                transaction_sequence=(
                    sequence
                ),

                field_name=(
                    field_name
                ),

                expected=(
                    expected
                ),

                actual=(
                    actual
                ),

                evidence={
                    "phase3_issue_code":
                        code,
                },

                source=(
                    "statement_consistency_analyzer"
                ),
            )

        # ====================================================
        # DUPLICATE
        # ====================================================

        if "DUPLICATE" in code:

            return RiskSignal(
                code=(
                    "DUPLICATE_TRANSACTION"
                ),

                category="statement",

                severity="low",

                message=message,

                score=(
                    self.DUPLICATE_SCORE
                ),

                confidence=0.90,

                transaction_sequence=(
                    sequence
                ),

                field_name=(
                    field_name
                ),

                expected=(
                    expected
                ),

                actual=(
                    actual
                ),

                evidence={
                    "phase3_issue_code":
                        code,
                },

                source=(
                    "statement_consistency_analyzer"
                ),
            )

        # ====================================================
        # INVALID STATEMENT PERIOD
        # ====================================================

        if (
            "PERIOD" in code
            and "INVALID" in code
        ):

            return RiskSignal(
                code=(
                    "INVALID_STATEMENT_PERIOD"
                ),

                category="statement",

                severity="medium",

                message=message,

                score=(
                    self.INVALID_PERIOD_SCORE
                ),

                confidence=0.95,

                transaction_sequence=(
                    sequence
                ),

                field_name=(
                    field_name
                ),

                expected=(
                    expected
                ),

                actual=(
                    actual
                ),

                evidence={
                    "phase3_issue_code":
                        code,
                },

                source=(
                    "statement_consistency_analyzer"
                ),
            )

        # ====================================================
        # UNKNOWN STATEMENT ISSUE
        # ====================================================
        #
        # Preserve evidence but do not assign positive risk
        # without understanding its semantics.
        # ====================================================

        return RiskSignal(
            code=(
                "STATEMENT_VALIDATION_OBSERVATION"
            ),

            category="statement",

            severity="info",

            message=message,

            score=0.0,

            confidence=0.70,

            transaction_sequence=(
                sequence
            ),

            field_name=(
                field_name
            ),

            expected=(
                expected
            ),

            actual=(
                actual
            ),

            evidence={
                "phase3_issue_code":
                    code,
            },

            source=(
                "statement_consistency_analyzer"
            ),
        )

    # ========================================================
    # PERIOD BOUNDARY DISTANCE
    # ========================================================

    def _boundary_distance_days(
        self,
        code: str,
        expected: Any,
        actual: Any,
    ) -> int | None:
        """
        Calculate how far an outside-period transaction is from
        the expected statement boundary.

        Phase 3 examples:

            expected = ">= 2026-01-23"
            actual   = "2026-01-22"

        or:

            expected = "<= 2026-07-22"
            actual   = "2026-07-23"

        Returns
        -------
        int | None

        Absolute number of calendar days between the transaction
        date and the relevant boundary.

        None means the evidence could not be interpreted safely.
        """

        actual_date = self._parse_date(
            actual
        )

        boundary_date = self._extract_date(
            expected
        )

        if (
            actual_date is None
            or boundary_date is None
        ):
            return None

        difference = (
            actual_date
            - boundary_date
        ).days

        # BEFORE_PERIOD should normally be negative.
        if "BEFORE_PERIOD" in code:

            if difference >= 0:
                return None

            return abs(
                difference
            )

        # AFTER_PERIOD should normally be positive.
        if "AFTER_PERIOD" in code:

            if difference <= 0:
                return None

            return abs(
                difference
            )

        # Generic OUTSIDE_PERIOD.
        #
        # If Phase 3 supplies enough expected/actual evidence,
        # absolute boundary distance remains useful.
        if "OUTSIDE_PERIOD" in code:

            return abs(
                difference
            )

        return None

    # ========================================================
    # DATE HELPERS
    # ========================================================

    @classmethod
    def _extract_date(
        cls,
        value: Any,
    ) -> date | None:
        """
        Extract an ISO date from Phase 3 expected-value evidence.

        Supported examples:

            ">= 2026-01-23"
            "<= 2026-07-22"
            "2026-01-23"
        """

        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            return value.date()

        if isinstance(
            value,
            date,
        ):
            return value

        text = str(
            value
        ).strip()

        if not text:
            return None

        # Look for ISO YYYY-MM-DD anywhere in the expected text.
        for token in (
            text
            .replace(
                ">=",
                " ",
            )
            .replace(
                "<=",
                " ",
            )
            .replace(
                ">",
                " ",
            )
            .replace(
                "<",
                " ",
            )
            .replace(
                "=",
                " ",
            )
            .split()
        ):

            parsed = cls._parse_date(
                token
            )

            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _parse_date(
        value: Any,
    ) -> date | None:
        """
        Parse normalized Phase 3 date evidence.

        Phase 3 currently emits ISO dates, but a few safe
        alternatives are accepted for adapter robustness.
        """

        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            return value.date()

        if isinstance(
            value,
            date,
        ):
            return value

        text = str(
            value
        ).strip()

        if not text:
            return None

        formats = (
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d %b %Y",
        )

        for fmt in formats:

            try:
                return datetime.strptime(
                    text,
                    fmt,
                ).date()

            except ValueError:
                continue

        return None

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _has_code_prefix(
        codes: set[str],
        prefix: str,
    ) -> bool:

        return any(
            code.startswith(
                prefix
            )
            for code
            in codes
        )

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

        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0


statement_consistency_analyzer = (
    StatementConsistencyAnalyzer()
)