"""
Bank Statement Validation Models
================================

Shared data models for Phase 3 bank-statement validation.

Phase 3 consumes the standardized transaction output produced by
Phase 2. These models deliberately contain no bank-specific logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    """
    A single validation issue discovered while checking a transaction.

    severity:
        info
        warning
        error

    code:
        Stable machine-readable identifier.

    message:
        Human-readable explanation.
    """

    code: str
    severity: str
    message: str

    transaction_sequence: int | None = None
    field_name: str | None = None

    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransactionValidationResult:
    """
    Validation result for one standardized transaction.
    """

    sequence: int | None

    is_valid: bool
    is_complete: bool

    has_date: bool
    has_amount: bool
    has_balance: bool
    has_description: bool

    direction_resolved: bool

    issue_count: int
    issues: tuple[ValidationIssue, ...]

    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "is_valid": self.is_valid,
            "is_complete": self.is_complete,
            "has_date": self.has_date,
            "has_amount": self.has_amount,
            "has_balance": self.has_balance,
            "has_description": self.has_description,
            "direction_resolved": self.direction_resolved,
            "issue_count": self.issue_count,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TransactionValidationSummary:
    """
    Aggregate result after validating all transactions.
    """

    transaction_count: int

    valid_count: int
    invalid_count: int

    complete_count: int
    incomplete_count: int

    resolved_direction_count: int
    unresolved_direction_count: int

    issue_count: int

    transaction_results: tuple[
        TransactionValidationResult,
        ...
    ] = field(default_factory=tuple)

    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_count": self.transaction_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "complete_count": self.complete_count,
            "incomplete_count": self.incomplete_count,
            "resolved_direction_count":
                self.resolved_direction_count,
            "unresolved_direction_count":
                self.unresolved_direction_count,
            "issue_count": self.issue_count,
            "confidence": self.confidence,
            "transactions": [
                result.to_dict()
                for result in self.transaction_results
            ],
        }