"""
Bank Statement Extraction - Standardized Output Schema

Purpose
-------
Provides the stable, bank-independent output contract for Phase 2
bank-statement extraction.

This module does NOT:
- extract PDF text
- perform OCR
- detect bank names
- parse bank-specific layouts
- infer fraud/tampering
- calculate risk scores

It only converts the internal BankStatementExtractionResult into a
clean, predictable structure for downstream consumers such as:

- Phase 3 validation / intelligence
- integrity / fraud checks
- FastAPI
- .NET integration
- databases
- audit pipelines

Design principles
-----------------
1. Bank-independent
2. Stable output contract
3. No fabricated values
4. Preserve field-level confidence
5. Preserve transaction-level confidence
6. Preserve reconciliation evidence
7. JSON serializable
8. Keep raw extraction internals out of the default public response
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    """
    Convert numeric values to JSON-friendly float.

    Returns None when conversion is not possible.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field_value(field_data: Any) -> Any:
    """
    Extract the actual value from metadata field structures.

    Supports:
        {"value": "...", "confidence": ..., "source": "..."}
        {"start_date": "...", "end_date": "..."}
        plain scalar values
    """
    if field_data is None:
        return None

    if isinstance(field_data, dict):
        if "value" in field_data:
            return field_data.get("value")

    return field_data


def _field_confidence(field_data: Any) -> float:
    if isinstance(field_data, dict):
        try:
            return float(field_data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def _field_source(field_data: Any) -> Optional[str]:
    if isinstance(field_data, dict):
        source = field_data.get("source")
        if source is not None:
            return str(source)

    return None


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------

@dataclass
class StandardizedField:
    value: Any = None
    confidence: float = 0.0
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatementPeriod:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    confidence: float = 0.0
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StandardizedMetadata:
    statement_period: StatementPeriod = field(default_factory=StatementPeriod)

    account_number: StandardizedField = field(default_factory=StandardizedField)
    account_type: StandardizedField = field(default_factory=StandardizedField)
    customer_name: StandardizedField = field(default_factory=StandardizedField)

    ifsc: StandardizedField = field(default_factory=StandardizedField)
    micr: StandardizedField = field(default_factory=StandardizedField)
    branch: StandardizedField = field(default_factory=StandardizedField)
    currency: StandardizedField = field(default_factory=StandardizedField)
    customer_id: StandardizedField = field(default_factory=StandardizedField)

    fields_found: int = 0
    metadata_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_period": self.statement_period.to_dict(),
            "account_number": self.account_number.to_dict(),
            "account_type": self.account_type.to_dict(),
            "customer_name": self.customer_name.to_dict(),
            "ifsc": self.ifsc.to_dict(),
            "micr": self.micr.to_dict(),
            "branch": self.branch.to_dict(),
            "currency": self.currency.to_dict(),
            "customer_id": self.customer_id.to_dict(),
            "fields_found": self.fields_found,
            "metadata_confidence": self.metadata_confidence,
        }


# ---------------------------------------------------------------------------
# Transaction schema
# ---------------------------------------------------------------------------

@dataclass
class StandardizedTransaction:
    sequence: Optional[int]
    date: Optional[str]
    description: str

    reference: Optional[str]

    debit: Optional[float]
    credit: Optional[float]
    balance: Optional[float]

    direction_source: Optional[str]

    balance_reconciled: Optional[bool]

    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Extraction diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ExtractionDiagnostics:
    extraction_method: Optional[str] = None

    page_count: int = 0
    text_char_count: int = 0
    line_count: int = 0

    ocr_used: bool = False

    transaction_header_detected: bool = False
    transaction_region_detected: bool = False

    rejected_transaction_blocks: int = 0
    unresolved_direction_count: int = 0
    reconciled_transaction_count: int = 0

    transaction_parser_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main standardized result
# ---------------------------------------------------------------------------

@dataclass
class StandardizedBankStatement:
    schema_version: str

    filename: str

    document_type: str

    metadata: StandardizedMetadata

    opening_balance: Optional[float]

    transactions: List[StandardizedTransaction]

    transaction_count: int

    diagnostics: ExtractionDiagnostics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "filename": self.filename,
            "document_type": self.document_type,
            "metadata": self.metadata.to_dict(),
            "opening_balance": self.opening_balance,
            "transaction_count": self.transaction_count,
            "transactions": [
                transaction.to_dict()
                for transaction in self.transactions
            ],
            "diagnostics": self.diagnostics.to_dict(),
        }


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

class BankStatementSchemaBuilder:
    """
    Converts the internal extraction result into the stable Phase-2
    bank-statement schema.

    The builder intentionally uses attribute access with safe defaults
    so that the public schema remains isolated from internal parser
    implementation details.
    """

    SCHEMA_VERSION = "2.0"

    DOCUMENT_TYPE = "bank_statement"

    METADATA_FIELDS = (
        "account_number",
        "account_type",
        "customer_name",
        "ifsc",
        "micr",
        "branch",
        "currency",
        "customer_id",
    )

    # ------------------------------------------------------------------

    @staticmethod
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)

        return getattr(obj, name, default)

    # ------------------------------------------------------------------

    def _build_field(self, metadata: Dict[str, Any], name: str) -> StandardizedField:
        field_data = metadata.get(name)

        return StandardizedField(
            value=_field_value(field_data),
            confidence=_field_confidence(field_data),
            source=_field_source(field_data),
        )

    # ------------------------------------------------------------------

    def _build_statement_period(
        self,
        metadata: Dict[str, Any],
    ) -> StatementPeriod:

        period = metadata.get("statement_period") or {}

        if not isinstance(period, dict):
            return StatementPeriod()

        return StatementPeriod(
            start_date=period.get("start_date"),
            end_date=period.get("end_date"),
            confidence=_field_confidence(period),
            source=_field_source(period),
        )

    # ------------------------------------------------------------------

    def _build_metadata(self, result: Any) -> StandardizedMetadata:
        metadata = self._get(result, "metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        fields = {
            name: self._build_field(metadata, name)
            for name in self.METADATA_FIELDS
        }

        try:
            fields_found = int(metadata.get("fields_found") or 0)
        except (TypeError, ValueError):
            fields_found = 0

        try:
            metadata_confidence = float(
                metadata.get("metadata_confidence") or 0.0
            )
        except (TypeError, ValueError):
            metadata_confidence = 0.0

        return StandardizedMetadata(
            statement_period=self._build_statement_period(metadata),

            account_number=fields["account_number"],
            account_type=fields["account_type"],
            customer_name=fields["customer_name"],

            ifsc=fields["ifsc"],
            micr=fields["micr"],
            branch=fields["branch"],
            currency=fields["currency"],
            customer_id=fields["customer_id"],

            fields_found=fields_found,
            metadata_confidence=metadata_confidence,
        )

    # ------------------------------------------------------------------

    def _build_transaction(
        self,
        transaction: Any,
    ) -> StandardizedTransaction:

        sequence = self._get(transaction, "sequence")

        try:
            sequence = int(sequence) if sequence is not None else None
        except (TypeError, ValueError):
            sequence = None

        description = self._get(transaction, "description", "")

        if description is None:
            description = ""

        confidence = self._get(transaction, "confidence", 0.0)

        try:
            confidence = float(confidence or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        reconciled = self._get(
            transaction,
            "balance_reconciled",
            None,
        )

        if reconciled not in (True, False):
            reconciled = None

        return StandardizedTransaction(
            sequence=sequence,
            date=self._get(transaction, "date"),
            description=str(description),

            reference=self._get(transaction, "reference"),

            debit=_to_float(
                self._get(transaction, "debit")
            ),
            credit=_to_float(
                self._get(transaction, "credit")
            ),
            balance=_to_float(
                self._get(transaction, "balance")
            ),

            direction_source=self._get(
                transaction,
                "direction_source",
            ),

            balance_reconciled=reconciled,

            confidence=confidence,
        )

    # ------------------------------------------------------------------

    def _build_transactions(
        self,
        result: Any,
    ) -> List[StandardizedTransaction]:

        transactions = self._get(result, "transactions", []) or []

        return [
            self._build_transaction(transaction)
            for transaction in transactions
        ]

    # ------------------------------------------------------------------

    def _build_diagnostics(
        self,
        result: Any,
    ) -> ExtractionDiagnostics:

        def safe_int(name: str) -> int:
            try:
                return int(self._get(result, name, 0) or 0)
            except (TypeError, ValueError):
                return 0

        def safe_float(name: str) -> float:
            try:
                return float(self._get(result, name, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return ExtractionDiagnostics(
            extraction_method=self._get(
                result,
                "extraction_method",
            ),

            page_count=safe_int("page_count"),
            text_char_count=safe_int("text_char_count"),
            line_count=safe_int("line_count"),

            ocr_used=bool(
                self._get(result, "ocr_used", False)
            ),

            transaction_header_detected=bool(
                self._get(
                    result,
                    "transaction_header_detected",
                    False,
                )
            ),

            transaction_region_detected=bool(
                self._get(
                    result,
                    "transaction_region_detected",
                    False,
                )
            ),

            rejected_transaction_blocks=safe_int(
                "rejected_transaction_blocks"
            ),

            unresolved_direction_count=safe_int(
                "unresolved_direction_count"
            ),

            reconciled_transaction_count=safe_int(
                "reconciled_transaction_count"
            ),

            transaction_parser_confidence=safe_float(
                "transaction_parser_confidence"
            ),
        )

    # ------------------------------------------------------------------

    def build(
        self,
        result: Any,
    ) -> StandardizedBankStatement:

        if result is None:
            raise ValueError(
                "Extraction result cannot be None."
            )

        filename = self._get(result, "filename", "")

        transactions = self._build_transactions(result)

        return StandardizedBankStatement(
            schema_version=self.SCHEMA_VERSION,

            filename=str(filename or ""),

            document_type=self.DOCUMENT_TYPE,

            metadata=self._build_metadata(result),

            opening_balance=_to_float(
                self._get(result, "opening_balance")
            ),

            transactions=transactions,

            # Do not blindly trust an upstream count.
            # The public contract reflects what is actually returned.
            transaction_count=len(transactions),

            diagnostics=self._build_diagnostics(result),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

bank_statement_schema_builder = BankStatementSchemaBuilder()