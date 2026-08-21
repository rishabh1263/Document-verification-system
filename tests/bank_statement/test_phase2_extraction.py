"""
Phase 2 - Bank Statement Extraction Integration Tests

These tests validate the complete extraction pipeline:

Document
    -> Extraction Router
    -> Native PDF / OCR routing
    -> Text Normalization
    -> Structure Parsing
    -> Metadata Extraction
    -> Transaction Parsing
    -> Bank Statement Extractor
    -> Standardized Schema

The tests intentionally validate business invariants rather than
bank-specific implementation details.
"""

from pathlib import Path

import pytest

from src.documents.bank_statement.extraction.bank_statement_extractor import (
    bank_statement_extractor,
)
from src.documents.bank_statement.extraction.extraction_schema import (
    bank_statement_schema_builder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_standardized(filename: str):
    path = SAMPLES_DIR / filename

    assert path.exists(), f"Sample file not found: {path}"

    raw = bank_statement_extractor.extract(
        path.read_bytes(),
        path.name,
    )

    return bank_statement_schema_builder.build(raw)


# ---------------------------------------------------------------------------
# Kotak
# ---------------------------------------------------------------------------

def test_kotak_full_extraction():
    result = extract_standardized(
        "KOTAK BANK STATEMENT.pdf"
    )

    assert result.schema_version == "2.0"
    assert result.document_type == "bank_statement"

    assert result.filename == "KOTAK BANK STATEMENT.pdf"

    assert result.diagnostics.extraction_method == "native_pdf"
    assert result.diagnostics.ocr_used is False

    assert result.diagnostics.page_count == 39

    assert result.diagnostics.transaction_header_detected is True
    assert result.diagnostics.transaction_region_detected is True

    assert result.opening_balance == pytest.approx(
        1317.44,
        abs=0.01,
    )

    assert result.transaction_count == 1339

    assert result.diagnostics.rejected_transaction_blocks == 0
    assert result.diagnostics.unresolved_direction_count == 0

    assert (
        result.diagnostics.reconciled_transaction_count
        == result.transaction_count
    )

    assert (
        result.diagnostics.transaction_parser_confidence
        >= 0.98
    )


def test_kotak_metadata():
    result = extract_standardized(
        "KOTAK BANK STATEMENT.pdf"
    )

    metadata = result.metadata

    assert metadata.statement_period.start_date == "2025-05-01"
    assert metadata.statement_period.end_date == "2026-05-01"

    assert metadata.account_number.value == "6347393980"

    assert metadata.account_type.value == "Savings"

    assert metadata.ifsc.value == "KKBK0004624"

    assert metadata.micr.value == "110485127"

    assert metadata.currency.value == "INR"

    assert metadata.metadata_confidence >= 0.90


def test_kotak_transaction_boundaries():
    result = extract_standardized(
        "KOTAK BANK STATEMENT.pdf"
    )

    first = result.transactions[0]
    last = result.transactions[-1]

    assert first.sequence == 1
    assert first.date == "2025-05-02"

    assert first.debit == pytest.approx(
        20.00,
        abs=0.01,
    )

    assert first.balance == pytest.approx(
        1297.44,
        abs=0.01,
    )

    assert first.balance_reconciled is True

    assert last.sequence == 1339
    assert last.date == "2026-05-01"

    assert last.credit == pytest.approx(
        20.00,
        abs=0.01,
    )

    assert last.balance == pytest.approx(
        1382.87,
        abs=0.01,
    )

    assert last.balance_reconciled is True


# ---------------------------------------------------------------------------
# Canara
# ---------------------------------------------------------------------------

def test_canara_full_extraction():
    result = extract_standardized(
        "Canara Bank Statement.pdf"
    )

    assert result.schema_version == "2.0"
    assert result.document_type == "bank_statement"

    assert result.filename == "Canara Bank Statement.pdf"

    assert result.diagnostics.extraction_method == "native_pdf"
    assert result.diagnostics.ocr_used is False

    assert result.diagnostics.page_count == 84

    assert result.diagnostics.transaction_header_detected is True
    assert result.diagnostics.transaction_region_detected is True

    assert result.opening_balance == pytest.approx(
        4824.70,
        abs=0.01,
    )

    assert result.transaction_count == 524

    assert result.diagnostics.rejected_transaction_blocks == 0
    assert result.diagnostics.unresolved_direction_count == 0

    assert (
        result.diagnostics.reconciled_transaction_count
        == result.transaction_count
    )

    assert (
        result.diagnostics.transaction_parser_confidence
        >= 0.99
    )


def test_canara_metadata():
    result = extract_standardized(
        "Canara Bank Statement.pdf"
    )

    metadata = result.metadata

    assert metadata.statement_period.start_date == "2026-01-23"
    assert metadata.statement_period.end_date == "2026-07-22"

    assert metadata.account_number.value == "XXXXXXXXXX5119"

    assert metadata.customer_name.value == "GUDDI DEVI"

    assert metadata.ifsc.value == "CNRB0002312"

    assert metadata.branch.value == "SITAMARHI"

    assert metadata.metadata_confidence >= 0.90


def test_canara_transaction_boundaries():
    result = extract_standardized(
        "Canara Bank Statement.pdf"
    )

    first = result.transactions[0]
    last = result.transactions[-1]

    assert first.sequence == 1
    assert first.date == "2026-01-22"

    assert first.credit == pytest.approx(
        400.00,
        abs=0.01,
    )

    assert first.balance == pytest.approx(
        5224.70,
        abs=0.01,
    )

    assert first.balance_reconciled is True

    assert last.sequence == 524
    assert last.date == "2026-07-21"

    assert last.debit == pytest.approx(
        500.00,
        abs=0.01,
    )

    assert last.balance == pytest.approx(
        229.70,
        abs=0.01,
    )

    assert last.balance_reconciled is True


# ---------------------------------------------------------------------------
# Generic contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "KOTAK BANK STATEMENT.pdf",
        "Canara Bank Statement.pdf",
    ],
)
def test_standardized_schema_contract(filename):
    result = extract_standardized(filename)

    payload = result.to_dict()

    assert payload["schema_version"] == "2.0"
    assert payload["document_type"] == "bank_statement"

    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["transactions"], list)
    assert isinstance(payload["diagnostics"], dict)

    assert payload["transaction_count"] == len(
        payload["transactions"]
    )

    # Public standardized transactions must not leak
    # parser-internal raw extraction blocks.
    for transaction in payload["transactions"]:
        assert "raw_text" not in transaction

        assert "sequence" in transaction
        assert "date" in transaction
        assert "description" in transaction
        assert "reference" in transaction
        assert "debit" in transaction
        assert "credit" in transaction
        assert "balance" in transaction
        assert "direction_source" in transaction
        assert "balance_reconciled" in transaction
        assert "confidence" in transaction


@pytest.mark.parametrize(
    "filename",
    [
        "KOTAK BANK STATEMENT.pdf",
        "Canara Bank Statement.pdf",
    ],
)
def test_all_transactions_reconcile(filename):
    result = extract_standardized(filename)

    assert result.transaction_count > 0

    non_reconciled = [
        transaction
        for transaction in result.transactions
        if transaction.balance_reconciled is not True
    ]

    assert non_reconciled == []