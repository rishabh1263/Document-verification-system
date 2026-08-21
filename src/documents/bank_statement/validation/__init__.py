"""
Bank Statement Phase 3 Validation Package
=========================================

Public validation API for bank-statement validation.

The package contains:

- transaction-level validation
- running-balance-chain validation
- statement-level validation
- unified validation orchestration

All components are designed to remain bank-independent.
"""

from .models import (
    ValidationIssue,
    TransactionValidationResult,
    TransactionValidationSummary,
)

from .transaction_validator import (
    TransactionValidator,
    transaction_validator,
)

from .balance_chain_validator import (
    BalanceChainCheck,
    BalanceChainValidationResult,
    BalanceChainValidator,
    balance_chain_validator,
)

from .statement_validator import (
    StatementValidationIssue,
    StatementValidationResult,
    StatementValidator,
    statement_validator,
)

from .bank_statement_validator import (
    BankStatementValidationResult,
    BankStatementValidator,
    bank_statement_validator,
)


__all__ = [
    # Transaction validation models
    "ValidationIssue",
    "TransactionValidationResult",
    "TransactionValidationSummary",

    # Transaction validator
    "TransactionValidator",
    "transaction_validator",

    # Balance-chain validation
    "BalanceChainCheck",
    "BalanceChainValidationResult",
    "BalanceChainValidator",
    "balance_chain_validator",

    # Statement validation
    "StatementValidationIssue",
    "StatementValidationResult",
    "StatementValidator",
    "statement_validator",

    # Phase 3 orchestrator
    "BankStatementValidationResult",
    "BankStatementValidator",
    "bank_statement_validator",
]