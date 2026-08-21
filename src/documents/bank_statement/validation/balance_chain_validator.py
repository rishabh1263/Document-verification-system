"""
Running Balance Chain Validator
===============================

Phase 3 bank-statement validation component.

Validates transaction arithmetic using the generic accounting rule:

    expected_balance
        = previous_balance
        - debit
        + credit

Important
---------
Signed amounts are preserved.

Examples:

    debit = 6000
        -> subtract 6000

    debit = -6000
        -> subtract(-6000)
        -> add 6000

    credit = 6000
        -> add 6000

    credit = -6000
        -> add(-6000)
        -> subtract 6000

This allows reversal/refund/correction transactions to reconcile
without bank-specific narration rules.

The validator is:

- bank-independent
- deterministic
- tolerant of missing evidence
- Decimal-based
- independent from OCR/extraction implementation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BalanceChainCheck:
    """
    Validation result for one transaction's running-balance chain.
    """

    sequence: int | None

    previous_balance: Decimal | None

    debit: Decimal | None
    credit: Decimal | None

    current_balance: Decimal | None
    expected_balance: Decimal | None

    difference: Decimal | None

    status: str

    reason: str | None = None

    def to_dict(self) -> dict:

        data = asdict(self)

        for key in (
            "previous_balance",
            "debit",
            "credit",
            "current_balance",
            "expected_balance",
            "difference",
        ):
            value = data.get(key)

            if value is not None:
                data[key] = float(value)

        return data


@dataclass(frozen=True)
class BalanceChainValidationResult:
    """
    Summary of running-balance validation.
    """

    transaction_count: int

    checked_count: int
    reconciled_count: int

    mismatch_count: int
    unverifiable_count: int

    confidence: float

    checks: tuple[
        BalanceChainCheck,
        ...,
    ]

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

            "confidence":
                self.confidence,

            "checks": [
                check.to_dict()
                for check in self.checks
            ],
        }


class BalanceChainValidator:
    """
    Generic running-balance validator.
    """

    DEFAULT_TOLERANCE = Decimal("0.01")

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def validate(
        self,
        transactions: Iterable[Any],
        opening_balance:
            Decimal | float | int | str | None = None,
        tolerance:
            Decimal | float | int | str | None = None,
    ) -> BalanceChainValidationResult:
        """
        Validate the running balance across transactions.

        Parameters
        ----------
        transactions:
            Iterable containing standardized transaction objects
            or dictionaries.

        opening_balance:
            Optional statement opening balance.

        tolerance:
            Maximum allowed absolute arithmetic difference.

        Returns
        -------
        BalanceChainValidationResult
        """

        if transactions is None:
            raise ValueError(
                "transactions cannot be None."
            )

        transaction_list = list(
            transactions
        )

        resolved_tolerance = (
            self._resolve_tolerance(
                tolerance
            )
        )

        previous_balance = (
            self._to_decimal(
                opening_balance
            )
            if opening_balance is not None
            else None
        )

        checks: list[
            BalanceChainCheck
        ] = []

        for transaction in transaction_list:

            sequence = self._get(
                transaction,
                "sequence",
            )

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

            current_balance = (
                self._to_decimal(
                    self._get(
                        transaction,
                        "balance",
                    )
                )
            )

            check = self._validate_transaction(
                sequence=sequence,
                previous_balance=(
                    previous_balance
                ),
                debit=debit,
                credit=credit,
                current_balance=(
                    current_balance
                ),
                tolerance=(
                    resolved_tolerance
                ),
            )

            checks.append(
                check
            )

            # -------------------------------------------------
            # Advance balance chain whenever the current
            # transaction exposes a usable running balance.
            #
            # Even if a row cannot itself be mathematically
            # verified, its printed balance is still the best
            # available starting point for the next row.
            # -------------------------------------------------

            if current_balance is not None:
                previous_balance = (
                    current_balance
                )

        checked_count = sum(
            1
            for check in checks
            if check.status
            in {
                "reconciled",
                "mismatch",
            }
        )

        reconciled_count = sum(
            1
            for check in checks
            if check.status
            == "reconciled"
        )

        mismatch_count = sum(
            1
            for check in checks
            if check.status
            == "mismatch"
        )

        unverifiable_count = sum(
            1
            for check in checks
            if check.status
            == "unverifiable"
        )

        confidence = (
            self._calculate_confidence(
                checked_count=(
                    checked_count
                ),
                reconciled_count=(
                    reconciled_count
                ),
            )
        )

        return BalanceChainValidationResult(
            transaction_count=len(
                transaction_list
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

            confidence=(
                confidence
            ),

            checks=tuple(
                checks
            ),
        )

    # ---------------------------------------------------------
    # TRANSACTION VALIDATION
    # ---------------------------------------------------------

    def _validate_transaction(
        self,
        sequence: int | None,
        previous_balance: Decimal | None,
        debit: Decimal | None,
        credit: Decimal | None,
        current_balance: Decimal | None,
        tolerance: Decimal,
    ) -> BalanceChainCheck:
        """
        Validate one balance transition.
        """

        # -----------------------------------------------------
        # Current balance missing
        # -----------------------------------------------------

        if current_balance is None:

            return BalanceChainCheck(
                sequence=sequence,

                previous_balance=(
                    previous_balance
                ),

                debit=debit,
                credit=credit,

                current_balance=None,
                expected_balance=None,

                difference=None,

                status="unverifiable",

                reason=(
                    "current_balance_missing"
                ),
            )

        # -----------------------------------------------------
        # Previous balance unavailable
        # -----------------------------------------------------

        if previous_balance is None:

            return BalanceChainCheck(
                sequence=sequence,

                previous_balance=None,

                debit=debit,
                credit=credit,

                current_balance=(
                    current_balance
                ),

                expected_balance=None,
                difference=None,

                status="unverifiable",

                reason=(
                    "previous_balance_missing"
                ),
            )

        # -----------------------------------------------------
        # No resolved transaction direction/amount
        # -----------------------------------------------------

        if (
            debit is None
            and credit is None
        ):

            return BalanceChainCheck(
                sequence=sequence,

                previous_balance=(
                    previous_balance
                ),

                debit=None,
                credit=None,

                current_balance=(
                    current_balance
                ),

                expected_balance=None,
                difference=None,

                status="unverifiable",

                reason=(
                    "transaction_amount_missing"
                ),
            )

        # -----------------------------------------------------
        # Both debit and credit populated
        # -----------------------------------------------------
        #
        # This is not automatically invalid.
        #
        # Some statement formats may expose both components.
        # The generic accounting equation remains valid:
        #
        # previous - debit + credit
        #
        # -----------------------------------------------------

        resolved_debit = (
            debit
            if debit is not None
            else Decimal("0")
        )

        resolved_credit = (
            credit
            if credit is not None
            else Decimal("0")
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT abs() the values.
        #
        # Negative debit is mathematically a reversal:
        #
        #     previous - (-amount)
        #       = previous + amount
        #
        # Negative credit similarly reduces the balance.
        #
        # -----------------------------------------------------

        expected_balance = (
            previous_balance
            - resolved_debit
            + resolved_credit
        )

        difference = (
            current_balance
            - expected_balance
        )

        absolute_difference = abs(
            difference
        )

        if (
            absolute_difference
            <= tolerance
        ):

            status = "reconciled"
            reason = None

        else:

            status = "mismatch"

            reason = (
                "balance_arithmetic_mismatch"
            )

        return BalanceChainCheck(
            sequence=sequence,

            previous_balance=(
                previous_balance
            ),

            debit=debit,
            credit=credit,

            current_balance=(
                current_balance
            ),

            expected_balance=(
                expected_balance
            ),

            difference=(
                difference
            ),

            status=status,
            reason=reason,
        )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        checked_count: int,
        reconciled_count: int,
    ) -> float:
        """
        Confidence measures mathematical agreement among rows
        that were actually verifiable.

        Missing evidence is represented separately through
        unverifiable_count and does not become an arithmetic
        mismatch.
        """

        if checked_count <= 0:
            return 0.0

        confidence = (
            reconciled_count
            / checked_count
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

    # ---------------------------------------------------------
    # TOLERANCE
    # ---------------------------------------------------------

    def _resolve_tolerance(
        self,
        tolerance:
            Decimal | float | int | str | None,
    ) -> Decimal:

        if tolerance is None:
            return self.DEFAULT_TOLERANCE

        resolved = self._to_decimal(
            tolerance
        )

        if resolved is None:
            raise ValueError(
                "Invalid balance tolerance."
            )

        if resolved < 0:
            raise ValueError(
                "Balance tolerance cannot be negative."
            )

        return resolved

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

    # ---------------------------------------------------------
    # DECIMAL CONVERSION
    # ---------------------------------------------------------

    @staticmethod
    def _to_decimal(
        value: Any,
    ) -> Decimal | None:
        """
        Safely convert standardized numeric values to Decimal.

        Signed values are intentionally preserved.
        """

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
            str(value)
            .replace(",", "")
            .replace("₹", "")
            .strip()
        )

        if not raw:
            return None

        upper = raw.upper()

        # Defensive support in case validation receives
        # non-standardized DR/CR values directly.
        suffix = None

        if upper.endswith("CR"):
            suffix = "CR"
            raw = raw[:-2].strip()

        elif upper.endswith("DR"):
            suffix = "DR"
            raw = raw[:-2].strip()

        raw = (
            raw
            .replace("INR", "")
            .replace("Rs.", "")
            .replace("RS.", "")
            .strip()
        )

        try:
            parsed = Decimal(
                raw
            )

        except (
            InvalidOperation,
            ValueError,
            TypeError,
        ):
            return None

        # Running-balance DR suffix represents a negative
        # balance when raw non-standardized input reaches here.
        if (
            suffix == "DR"
            and parsed > 0
        ):
            parsed = -parsed

        return parsed


balance_chain_validator = (
    BalanceChainValidator()
)