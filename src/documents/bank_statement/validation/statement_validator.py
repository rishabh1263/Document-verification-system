"""
Generic Bank Statement-Level Validator
======================================

Validates consistency at statement level rather than individual
transaction level.

Responsibilities
----------------
- Validate statement period.
- Validate transaction dates against statement period.
- Detect transaction date ordering problems.
- Detect duplicate transactions.
- Detect empty statements.
- Produce structured statement-level issues.

This module is bank-independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class StatementValidationIssue:
    """
    Statement-level validation issue.
    """

    code: str
    severity: str
    message: str

    transaction_sequence: int | None = None

    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StatementValidationResult:
    """
    Complete statement-level validation result.
    """

    transaction_count: int

    statement_start_date: str | None
    statement_end_date: str | None

    period_valid: bool | None

    dated_transaction_count: int
    undated_transaction_count: int

    outside_period_count: int
    ordering_issue_count: int
    duplicate_count: int

    issue_count: int
    issues: tuple[StatementValidationIssue, ...]

    is_valid: bool
    confidence: float

    def to_dict(self) -> dict:
        return {
            "transaction_count":
                self.transaction_count,

            "statement_start_date":
                self.statement_start_date,

            "statement_end_date":
                self.statement_end_date,

            "period_valid":
                self.period_valid,

            "dated_transaction_count":
                self.dated_transaction_count,

            "undated_transaction_count":
                self.undated_transaction_count,

            "outside_period_count":
                self.outside_period_count,

            "ordering_issue_count":
                self.ordering_issue_count,

            "duplicate_count":
                self.duplicate_count,

            "issue_count":
                self.issue_count,

            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],

            "is_valid":
                self.is_valid,

            "confidence":
                self.confidence,
        }


class StatementValidator:
    """
    Generic statement-level consistency validator.
    """

    DATE_FORMATS = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
    )

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def validate(
        self,
        transactions: Sequence[Any],
        statement_start_date: Any = None,
        statement_end_date: Any = None,
    ) -> StatementValidationResult:
        """
        Validate statement-level consistency.
        """

        if transactions is None:
            raise ValueError(
                "transactions cannot be None."
            )

        issues: list[
            StatementValidationIssue
        ] = []

        start_date = self._parse_date(
            statement_start_date
        )

        end_date = self._parse_date(
            statement_end_date
        )

        # -----------------------------------------------------
        # STATEMENT PERIOD
        # -----------------------------------------------------

        period_valid: bool | None = None

        if (
            statement_start_date is not None
            and start_date is None
        ):
            issues.append(
                StatementValidationIssue(
                    code="INVALID_STATEMENT_START_DATE",
                    severity="error",
                    message=(
                        "Statement start date could not "
                        "be parsed."
                    ),
                    actual=statement_start_date,
                )
            )

        if (
            statement_end_date is not None
            and end_date is None
        ):
            issues.append(
                StatementValidationIssue(
                    code="INVALID_STATEMENT_END_DATE",
                    severity="error",
                    message=(
                        "Statement end date could not "
                        "be parsed."
                    ),
                    actual=statement_end_date,
                )
            )

        if (
            start_date is not None
            and end_date is not None
        ):
            period_valid = (
                start_date <= end_date
            )

            if not period_valid:
                issues.append(
                    StatementValidationIssue(
                        code="INVALID_STATEMENT_PERIOD",
                        severity="error",
                        message=(
                            "Statement start date occurs "
                            "after statement end date."
                        ),
                        expected=(
                            "start_date <= end_date"
                        ),
                        actual=(
                            f"{start_date.isoformat()} > "
                            f"{end_date.isoformat()}"
                        ),
                    )
                )

        # -----------------------------------------------------
        # TRANSACTIONS
        # -----------------------------------------------------

        transaction_count = len(
            transactions
        )

        if transaction_count == 0:
            issues.append(
                StatementValidationIssue(
                    code="EMPTY_STATEMENT",
                    severity="error",
                    message=(
                        "Statement contains no "
                        "transactions."
                    ),
                )
            )

        dated_transaction_count = 0
        undated_transaction_count = 0
        outside_period_count = 0
        ordering_issue_count = 0

        previous_date: date | None = None

        # Used for exact standardized duplicate detection.
        seen_fingerprints: dict[
            tuple,
            int | None,
        ] = {}

        duplicate_count = 0

        for index, transaction in enumerate(
            transactions,
            start=1,
        ):
            sequence = self._get(
                transaction,
                "sequence",
                index,
            )

            raw_date = self._get(
                transaction,
                "date",
            )

            transaction_date = (
                self._parse_date(
                    raw_date
                )
            )

            # -----------------------------------------------
            # DATE PRESENCE / VALIDITY
            # -----------------------------------------------

            if transaction_date is None:
                undated_transaction_count += 1

                issues.append(
                    StatementValidationIssue(
                        code="UNUSABLE_TRANSACTION_DATE",
                        severity="warning",
                        message=(
                            "Transaction has no usable "
                            "date for statement-level "
                            "validation."
                        ),
                        transaction_sequence=sequence,
                        actual=raw_date,
                    )
                )

            else:
                dated_transaction_count += 1

                # -------------------------------------------
                # PERIOD BOUNDARY
                # -------------------------------------------

                if (
                    start_date is not None
                    and transaction_date < start_date
                ):
                    outside_period_count += 1

                    issues.append(
                        StatementValidationIssue(
                            code="TRANSACTION_BEFORE_PERIOD",
                            severity="error",
                            message=(
                                "Transaction date occurs "
                                "before statement period."
                            ),
                            transaction_sequence=sequence,
                            expected=(
                                f">= {start_date.isoformat()}"
                            ),
                            actual=(
                                transaction_date.isoformat()
                            ),
                        )
                    )

                elif (
                    end_date is not None
                    and transaction_date > end_date
                ):
                    outside_period_count += 1

                    issues.append(
                        StatementValidationIssue(
                            code="TRANSACTION_AFTER_PERIOD",
                            severity="error",
                            message=(
                                "Transaction date occurs "
                                "after statement period."
                            ),
                            transaction_sequence=sequence,
                            expected=(
                                f"<= {end_date.isoformat()}"
                            ),
                            actual=(
                                transaction_date.isoformat()
                            ),
                        )
                    )

                # -------------------------------------------
                # CHRONOLOGICAL ORDER
                # -------------------------------------------

                if (
                    previous_date is not None
                    and transaction_date < previous_date
                ):
                    ordering_issue_count += 1

                    issues.append(
                        StatementValidationIssue(
                            code="TRANSACTION_DATE_ORDER",
                            severity="warning",
                            message=(
                                "Transaction date moves "
                                "backward relative to the "
                                "previous dated transaction."
                            ),
                            transaction_sequence=sequence,
                            expected=(
                                f">= {previous_date.isoformat()}"
                            ),
                            actual=(
                                transaction_date.isoformat()
                            ),
                        )
                    )

                previous_date = (
                    transaction_date
                )

            # -----------------------------------------------
            # DUPLICATE DETECTION
            # -----------------------------------------------

            fingerprint = (
                transaction_date.isoformat()
                if transaction_date is not None
                else None,

                self._normalize_text(
                    self._get(
                        transaction,
                        "description",
                    )
                ),

                self._normalize_amount(
                    self._get(
                        transaction,
                        "debit",
                    )
                ),

                self._normalize_amount(
                    self._get(
                        transaction,
                        "credit",
                    )
                ),

                self._normalize_amount(
                    self._get(
                        transaction,
                        "balance",
                    )
                ),

                self._normalize_text(
                    self._get(
                        transaction,
                        "reference",
                    )
                ),
            )

            # Do not call a completely empty/incomplete row a
            # duplicate merely because all its fields are None.
            meaningful_fingerprint = any(
                value is not None
                and value != ""
                for value in fingerprint
            )

            if (
                meaningful_fingerprint
                and fingerprint
                in seen_fingerprints
            ):
                duplicate_count += 1

                original_sequence = (
                    seen_fingerprints[
                        fingerprint
                    ]
                )

                issues.append(
                    StatementValidationIssue(
                        code="DUPLICATE_TRANSACTION",
                        severity="warning",
                        message=(
                            "Transaction has the same "
                            "standardized fingerprint as "
                            "an earlier transaction."
                        ),
                        transaction_sequence=sequence,
                        expected=(
                            f"unique transaction; first "
                            f"seen at sequence "
                            f"{original_sequence}"
                        ),
                        actual=(
                            f"duplicate of sequence "
                            f"{original_sequence}"
                        ),
                    )
                )

            elif meaningful_fingerprint:
                seen_fingerprints[
                    fingerprint
                ] = sequence

        # -----------------------------------------------------
        # FINAL CLASSIFICATION
        # -----------------------------------------------------

        error_count = sum(
            1
            for issue in issues
            if issue.severity == "error"
        )

        is_valid = (
            error_count == 0
        )

        confidence = (
            self._calculate_confidence(
                transaction_count=(
                    transaction_count
                ),
                dated_transaction_count=(
                    dated_transaction_count
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
                error_count=error_count,
            )
        )

        return StatementValidationResult(
            transaction_count=transaction_count,

            statement_start_date=(
                start_date.isoformat()
                if start_date is not None
                else None
            ),

            statement_end_date=(
                end_date.isoformat()
                if end_date is not None
                else None
            ),

            period_valid=period_valid,

            dated_transaction_count=(
                dated_transaction_count
            ),

            undated_transaction_count=(
                undated_transaction_count
            ),

            outside_period_count=(
                outside_period_count
            ),

            ordering_issue_count=(
                ordering_issue_count
            ),

            duplicate_count=duplicate_count,

            issue_count=len(issues),

            issues=tuple(issues),

            is_valid=is_valid,

            confidence=confidence,
        )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        transaction_count: int,
        dated_transaction_count: int,
        outside_period_count: int,
        ordering_issue_count: int,
        duplicate_count: int,
        error_count: int,
    ) -> float:

        if transaction_count == 0:
            return 0.0

        date_coverage = (
            dated_transaction_count
            / transaction_count
        )

        score = date_coverage

        score -= min(
            outside_period_count
            / transaction_count,
            1.0,
        ) * 0.40

        score -= min(
            ordering_issue_count
            / transaction_count,
            1.0,
        ) * 0.15

        score -= min(
            duplicate_count
            / transaction_count,
            1.0,
        ) * 0.15

        if error_count:
            score -= min(
                error_count * 0.05,
                0.25,
            )

        score = max(
            0.0,
            min(
                score,
                1.0,
            ),
        )

        return round(
            score,
            4,
        )

    # ---------------------------------------------------------
    # DATE
    # ---------------------------------------------------------

    def _parse_date(
        self,
        value: Any,
    ) -> date | None:

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

        raw = str(value).strip()

        if not raw:
            return None

        for date_format in self.DATE_FORMATS:

            try:
                return datetime.strptime(
                    raw,
                    date_format,
                ).date()

            except ValueError:
                continue

        return None

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        raw = " ".join(
            str(value).split()
        ).strip()

        if not raw:
            return None

        return raw.casefold()

    @staticmethod
    def _normalize_amount(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        if isinstance(
            value,
            Decimal,
        ):
            return str(
                value.normalize()
            )

        raw = (
            str(value)
            .replace(",", "")
            .replace("₹", "")
            .strip()
        )

        if not raw:
            return None

        upper = raw.upper()

        if upper.endswith("CR"):
            raw = raw[:-2].strip()

        elif upper.endswith("DR"):
            raw = raw[:-2].strip()

            if raw and not raw.startswith("-"):
                raw = "-" + raw

        for prefix in (
            "RS.",
            "RS",
            "INR",
        ):
            if raw.upper().startswith(prefix):
                raw = raw[
                    len(prefix):
                ].strip()

                break

        try:
            return str(
                Decimal(raw).normalize()
            )

        except (
            InvalidOperation,
            ValueError,
        ):
            return raw.casefold()

    # ---------------------------------------------------------
    # GENERIC ACCESS
    # ---------------------------------------------------------

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


statement_validator = StatementValidator()