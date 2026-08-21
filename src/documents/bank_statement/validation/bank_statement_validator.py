"""
Bank Statement Validation Orchestrator
======================================

Phase 3 orchestration layer.

This module combines:

1. Transaction-level validation
2. Running-balance chain validation
3. Statement-level validation

It consumes the standardized Phase 2 extraction result and produces
one unified Phase 3 validation result.

Design goals
------------
- Bank-independent
- Extraction-independent
- No OCR logic
- No bank-specific rules
- Missing evidence != automatic fraud
- Deterministic validation
- Plug-in / plug-out architecture
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .transaction_validator import (
    transaction_validator,
)

from .balance_chain_validator import (
    balance_chain_validator,
)

from .statement_validator import (
    statement_validator,
)


@dataclass(frozen=True)
class BankStatementValidationResult:
    """
    Unified Phase 3 bank-statement validation result.
    """

    filename: str | None

    transaction_count: int

    transaction_validation: dict
    balance_chain_validation: dict
    statement_validation: dict

    error_count: int
    warning_count: int

    is_valid: bool

    validation_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class BankStatementValidator:
    """
    Orchestrates all Phase 3 validation components.
    """

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def validate(
        self,
        extraction_result: Any,
    ) -> BankStatementValidationResult:
        """
        Validate a standardized Phase 2 extraction result.

        Parameters
        ----------
        extraction_result:
            Either:

            - BankStatementExtractionResult
            - dictionary containing standardized extraction output

        Returns
        -------
        BankStatementValidationResult
            Unified Phase 3 validation result.
        """

        if extraction_result is None:
            raise ValueError(
                "extraction_result cannot be None."
            )

        transactions = self._get(
            extraction_result,
            "transactions",
            (),
        )

        if transactions is None:
            transactions = ()

        filename = self._get(
            extraction_result,
            "filename",
        )

        opening_balance = self._get(
            extraction_result,
            "opening_balance",
        )

        metadata = self._get(
            extraction_result,
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        # -----------------------------------------------------
        # 1. TRANSACTION VALIDATION
        # -----------------------------------------------------

        transaction_result = (
            transaction_validator.validate(
                transactions
            )
        )

        # -----------------------------------------------------
        # 2. BALANCE-CHAIN VALIDATION
        # -----------------------------------------------------

        balance_result = (
            balance_chain_validator.validate(
                transactions,
                opening_balance=opening_balance,
            )
        )

        # -----------------------------------------------------
        # 3. STATEMENT PERIOD
        # -----------------------------------------------------

        (
            statement_start_date,
            statement_end_date,
        ) = self._extract_statement_period(
            metadata
        )

        # -----------------------------------------------------
        # 4. STATEMENT VALIDATION
        # -----------------------------------------------------

        statement_result = (
            statement_validator.validate(
                transactions,
                statement_start_date=(
                    statement_start_date
                ),
                statement_end_date=(
                    statement_end_date
                ),
            )
        )

        # -----------------------------------------------------
        # 5. ISSUE COUNTS
        # -----------------------------------------------------

        transaction_dict = (
            transaction_result.to_dict()
        )

        balance_dict = (
            balance_result.to_dict()
        )

        statement_dict = (
            statement_result.to_dict()
        )

        (
            transaction_errors,
            transaction_warnings,
        ) = self._count_transaction_issues(
            transaction_dict
        )

        (
            statement_errors,
            statement_warnings,
        ) = self._count_statement_issues(
            statement_dict
        )

        # A mathematically inconsistent running balance is a
        # validation error.
        balance_errors = (
            balance_result.mismatch_count
        )

        # Unverifiable rows are evidence-quality warnings,
        # not automatic failures.
        balance_warnings = (
            balance_result.unverifiable_count
        )

        error_count = (
            transaction_errors
            + statement_errors
            + balance_errors
        )

        warning_count = (
            transaction_warnings
            + statement_warnings
            + balance_warnings
        )

        # -----------------------------------------------------
        # 6. FINAL VALIDITY
        # -----------------------------------------------------

        is_valid = (
            error_count == 0
            and transaction_result.invalid_count == 0
            and statement_result.is_valid
            and balance_result.mismatch_count == 0
        )

        # -----------------------------------------------------
        # 7. CONFIDENCE
        # -----------------------------------------------------

        validation_confidence = (
            self._calculate_confidence(
                transaction_confidence=(
                    transaction_result.confidence
                ),
                balance_confidence=(
                    balance_result.confidence
                ),
                statement_confidence=(
                    statement_result.confidence
                ),
                transaction_count=len(
                    transactions
                ),
                balance_checked_count=(
                    balance_result.checked_count
                ),
            )
        )

        return BankStatementValidationResult(
            filename=filename,

            transaction_count=len(
                transactions
            ),

            transaction_validation=(
                transaction_dict
            ),

            balance_chain_validation=(
                balance_dict
            ),

            statement_validation=(
                statement_dict
            ),

            error_count=error_count,
            warning_count=warning_count,

            is_valid=is_valid,

            validation_confidence=(
                validation_confidence
            ),
        )

    # ---------------------------------------------------------
    # STATEMENT PERIOD
    # ---------------------------------------------------------

    def _extract_statement_period(
        self,
        metadata: Any,
    ) -> tuple[
        Any,
        Any,
    ]:
        """
        Extract standardized statement period from Phase 2 metadata.

        Expected Phase 2 shape:

        metadata = {
            "statement_period": {
                "start_date": "...",
                "end_date": "...",
                ...
            }
        }

        Defensive support is also included for flatter dictionaries.
        """

        statement_period = self._get(
            metadata,
            "statement_period",
        )

        if statement_period is not None:

            start_date = self._get(
                statement_period,
                "start_date",
            )

            end_date = self._get(
                statement_period,
                "end_date",
            )

            return (
                start_date,
                end_date,
            )

        start_date = self._get(
            metadata,
            "statement_start_date",
        )

        end_date = self._get(
            metadata,
            "statement_end_date",
        )

        return (
            start_date,
            end_date,
        )

    # ---------------------------------------------------------
    # ISSUE COUNTS
    # ---------------------------------------------------------

    @staticmethod
    def _count_transaction_issues(
        result: dict,
    ) -> tuple[int, int]:

        errors = 0
        warnings = 0

        transaction_results = (
            result.get(
                "transaction_results",
                [],
            )
        )

        for transaction in transaction_results:

            for issue in transaction.get(
                "issues",
                [],
            ):
                severity = (
                    str(
                        issue.get(
                            "severity",
                            "",
                        )
                    )
                    .strip()
                    .lower()
                )

                if severity == "error":
                    errors += 1

                elif severity == "warning":
                    warnings += 1

        return (
            errors,
            warnings,
        )

    @staticmethod
    def _count_statement_issues(
        result: dict,
    ) -> tuple[int, int]:

        errors = 0
        warnings = 0

        for issue in result.get(
            "issues",
            [],
        ):
            severity = (
                str(
                    issue.get(
                        "severity",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            if severity == "error":
                errors += 1

            elif severity == "warning":
                warnings += 1

        return (
            errors,
            warnings,
        )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        transaction_confidence: float,
        balance_confidence: float,
        statement_confidence: float,
        transaction_count: int,
        balance_checked_count: int,
    ) -> float:
        """
        Calculate unified validation confidence.

        Transaction and statement validation are always applicable.

        Balance-chain confidence is included only when at least one
        mathematical balance transition could actually be checked.
        """

        scores = [
            float(
                transaction_confidence
                or 0.0
            ),
            float(
                statement_confidence
                or 0.0
            ),
        ]

        if balance_checked_count > 0:
            scores.append(
                float(
                    balance_confidence
                    or 0.0
                )
            )

        if transaction_count == 0:
            return 0.0

        if not scores:
            return 0.0

        score = (
            sum(scores)
            / len(scores)
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


bank_statement_validator = (
    BankStatementValidator()
)