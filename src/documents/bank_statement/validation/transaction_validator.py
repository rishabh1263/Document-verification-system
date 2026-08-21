"""
Generic Bank Statement Transaction Validator
============================================

Validates standardized Phase 2 bank-statement transactions.

Responsibilities
----------------
- Validate transaction dates.
- Validate debit / credit values.
- Detect conflicting debit and credit values.
- Detect transactions with no monetary evidence.
- Validate balances.
- Detect missing descriptions.
- Detect unresolved transaction direction.
- Produce structured validation issues.

Important
---------
This module is bank-independent.

It must not contain SBI, Kotak, Canara, HDFC, ICICI, Axis,
or any other bank-specific rules.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .models import (
    TransactionValidationResult,
    TransactionValidationSummary,
    ValidationIssue,
)


class TransactionValidator:
    """
    Generic validator for standardized bank transactions.
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
    ) -> TransactionValidationSummary:
        """
        Validate a collection of standardized transactions.
        """

        if transactions is None:
            raise ValueError(
                "transactions cannot be None."
            )

        results: list[
            TransactionValidationResult
        ] = []

        for index, transaction in enumerate(
            transactions,
            start=1,
        ):
            result = self.validate_transaction(
                transaction=transaction,
                fallback_sequence=index,
            )

            results.append(result)

        transaction_count = len(results)

        valid_count = sum(
            1
            for result in results
            if result.is_valid
        )

        invalid_count = (
            transaction_count - valid_count
        )

        complete_count = sum(
            1
            for result in results
            if result.is_complete
        )

        incomplete_count = (
            transaction_count - complete_count
        )

        resolved_direction_count = sum(
            1
            for result in results
            if result.direction_resolved
        )

        unresolved_direction_count = (
            transaction_count
            - resolved_direction_count
        )

        issue_count = sum(
            result.issue_count
            for result in results
        )

        confidence = self._calculate_summary_confidence(
            results
        )

        return TransactionValidationSummary(
            transaction_count=transaction_count,
            valid_count=valid_count,
            invalid_count=invalid_count,
            complete_count=complete_count,
            incomplete_count=incomplete_count,
            resolved_direction_count=(
                resolved_direction_count
            ),
            unresolved_direction_count=(
                unresolved_direction_count
            ),
            issue_count=issue_count,
            transaction_results=tuple(results),
            confidence=confidence,
        )

    def validate_transaction(
        self,
        transaction: Any,
        fallback_sequence: int | None = None,
    ) -> TransactionValidationResult:
        """
        Validate one standardized transaction.
        """

        sequence = self._get(
            transaction,
            "sequence",
            fallback_sequence,
        )

        date = self._get(
            transaction,
            "date",
        )

        description = self._get(
            transaction,
            "description",
        )

        debit_raw = self._get(
            transaction,
            "debit",
        )

        credit_raw = self._get(
            transaction,
            "credit",
        )

        balance_raw = self._get(
            transaction,
            "balance",
        )

        issues: list[ValidationIssue] = []

        # -----------------------------------------------------
        # DATE
        # -----------------------------------------------------

        has_date = bool(
            date
            and str(date).strip()
        )

        if not has_date:
            issues.append(
                ValidationIssue(
                    code="MISSING_DATE",
                    severity="error",
                    message=(
                        "Transaction date is missing."
                    ),
                    transaction_sequence=sequence,
                    field_name="date",
                    actual=date,
                )
            )

        elif not self._is_valid_date(date):
            issues.append(
                ValidationIssue(
                    code="INVALID_DATE",
                    severity="error",
                    message=(
                        "Transaction date could not be "
                        "parsed using supported formats."
                    ),
                    transaction_sequence=sequence,
                    field_name="date",
                    actual=date,
                )
            )

        # -----------------------------------------------------
        # DESCRIPTION
        # -----------------------------------------------------

        has_description = bool(
            description
            and str(description).strip()
        )

        if not has_description:
            issues.append(
                ValidationIssue(
                    code="MISSING_DESCRIPTION",
                    severity="warning",
                    message=(
                        "Transaction description is missing."
                    ),
                    transaction_sequence=sequence,
                    field_name="description",
                    actual=description,
                )
            )

        # -----------------------------------------------------
        # AMOUNTS
        # -----------------------------------------------------

        debit = self._to_decimal(
            debit_raw
        )

        credit = self._to_decimal(
            credit_raw
        )

        balance = self._to_decimal(
            balance_raw
        )

        if (
            debit_raw is not None
            and debit is None
        ):
            issues.append(
                ValidationIssue(
                    code="INVALID_DEBIT",
                    severity="error",
                    message=(
                        "Debit value is not a valid "
                        "numeric amount."
                    ),
                    transaction_sequence=sequence,
                    field_name="debit",
                    actual=debit_raw,
                )
            )

        if (
            credit_raw is not None
            and credit is None
        ):
            issues.append(
                ValidationIssue(
                    code="INVALID_CREDIT",
                    severity="error",
                    message=(
                        "Credit value is not a valid "
                        "numeric amount."
                    ),
                    transaction_sequence=sequence,
                    field_name="credit",
                    actual=credit_raw,
                )
            )

        if (
            balance_raw is not None
            and balance is None
        ):
            issues.append(
                ValidationIssue(
                    code="INVALID_BALANCE",
                    severity="error",
                    message=(
                        "Balance value is not a valid "
                        "numeric amount."
                    ),
                    transaction_sequence=sequence,
                    field_name="balance",
                    actual=balance_raw,
                )
            )

        # -----------------------------------------------------
        # NEGATIVE TRANSACTION AMOUNTS
        # -----------------------------------------------------

        if (
            debit is not None
            and debit < Decimal("0")
        ):
            issues.append(
                ValidationIssue(
                    code="NEGATIVE_DEBIT",
                    severity="warning",
                    message=(
                        "Debit amount is negative."
                    ),
                    transaction_sequence=sequence,
                    field_name="debit",
                    actual=str(debit),
                )
            )

        if (
            credit is not None
            and credit < Decimal("0")
        ):
            issues.append(
                ValidationIssue(
                    code="NEGATIVE_CREDIT",
                    severity="warning",
                    message=(
                        "Credit amount is negative."
                    ),
                    transaction_sequence=sequence,
                    field_name="credit",
                    actual=str(credit),
                )
            )

        # -----------------------------------------------------
        # DIRECTION
        # -----------------------------------------------------

        has_debit = (
            debit is not None
            and debit != Decimal("0")
        )

        has_credit = (
            credit is not None
            and credit != Decimal("0")
        )

        has_amount = (
            has_debit
            or has_credit
        )

        if has_debit and has_credit:
            issues.append(
                ValidationIssue(
                    code="DEBIT_CREDIT_CONFLICT",
                    severity="error",
                    message=(
                        "Transaction contains both a "
                        "debit and credit amount."
                    ),
                    transaction_sequence=sequence,
                )
            )

        direction_resolved = (
            has_debit ^ has_credit
        )

        if not has_amount:
            issues.append(
                ValidationIssue(
                    code="MISSING_TRANSACTION_AMOUNT",
                    severity="warning",
                    message=(
                        "Transaction contains no resolved "
                        "debit or credit amount."
                    ),
                    transaction_sequence=sequence,
                )
            )

        elif not direction_resolved:
            issues.append(
                ValidationIssue(
                    code="UNRESOLVED_DIRECTION",
                    severity="error",
                    message=(
                        "Transaction direction could not "
                        "be uniquely resolved."
                    ),
                    transaction_sequence=sequence,
                )
            )

        # -----------------------------------------------------
        # BALANCE
        # -----------------------------------------------------

        has_balance = (
            balance is not None
        )

        if not has_balance:
            issues.append(
                ValidationIssue(
                    code="MISSING_BALANCE",
                    severity="warning",
                    message=(
                        "Running balance is missing."
                    ),
                    transaction_sequence=sequence,
                    field_name="balance",
                    actual=balance_raw,
                )
            )

        # -----------------------------------------------------
        # RESULT CLASSIFICATION
        # -----------------------------------------------------

        error_count = sum(
            1
            for issue in issues
            if issue.severity == "error"
        )

        is_valid = (
            error_count == 0
        )

        is_complete = all(
            (
                has_date,
                has_description,
                has_amount,
                has_balance,
                direction_resolved,
            )
        )

        confidence = self._calculate_transaction_confidence(
            has_date=has_date,
            has_description=has_description,
            has_amount=has_amount,
            has_balance=has_balance,
            direction_resolved=(
                direction_resolved
            ),
            issues=issues,
        )

        return TransactionValidationResult(
            sequence=sequence,
            is_valid=is_valid,
            is_complete=is_complete,
            has_date=has_date,
            has_amount=has_amount,
            has_balance=has_balance,
            has_description=has_description,
            direction_resolved=(
                direction_resolved
            ),
            issue_count=len(issues),
            issues=tuple(issues),
            confidence=confidence,
        )

    # ---------------------------------------------------------
    # DATE HELPERS
    # ---------------------------------------------------------

    def _is_valid_date(
        self,
        value: Any,
    ) -> bool:

        raw = str(value).strip()

        if not raw:
            return False

        for date_format in self.DATE_FORMATS:

            try:
                datetime.strptime(
                    raw,
                    date_format,
                )

                return True

            except ValueError:
                continue

        return False

    # ---------------------------------------------------------
    # NUMERIC HELPERS
    # ---------------------------------------------------------

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

        raw = str(value).strip()

        if not raw:
            return None

        raw = (
            raw
            .replace(",", "")
            .replace("₹", "")
            .strip()
        )

        upper = raw.upper()

        # Phase 2 may preserve CR / DR suffixes.
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
            return Decimal(raw)

        except (
            InvalidOperation,
            ValueError,
        ):
            return None

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_transaction_confidence(
        has_date: bool,
        has_description: bool,
        has_amount: bool,
        has_balance: bool,
        direction_resolved: bool,
        issues: Sequence[ValidationIssue],
    ) -> float:

        score = 0.0

        if has_date:
            score += 0.20

        if has_description:
            score += 0.15

        if has_amount:
            score += 0.25

        if has_balance:
            score += 0.20

        if direction_resolved:
            score += 0.20

        error_count = sum(
            1
            for issue in issues
            if issue.severity == "error"
        )

        warning_count = sum(
            1
            for issue in issues
            if issue.severity == "warning"
        )

        score -= (
            error_count * 0.15
        )

        score -= (
            warning_count * 0.05
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

    @staticmethod
    def _calculate_summary_confidence(
        results: Sequence[
            TransactionValidationResult
        ],
    ) -> float:

        if not results:
            return 0.0

        score = sum(
            result.confidence
            for result in results
        ) / len(results)

        return round(
            score,
            4,
        )

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


transaction_validator = TransactionValidator()