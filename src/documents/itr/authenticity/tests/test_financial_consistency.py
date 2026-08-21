"""
Tests for ITR financial consistency analysis.

The tests intentionally separate:

1. Financial consistency
2. Missing information
3. Suspicious financial relationships

Financial inconsistency is treated as evidence, not automatic proof
that an ITR is fake.
"""

from __future__ import annotations

from src.documents.itr.authenticity.financial_consistency import (
    FinancialConsistencyEngine,
    FinancialSeverity,
    FinancialSnapshot,
    FinancialStatus,
    analyze_financial_consistency,
    normalize_amount,
)


# ==========================================================
# NORMALIZATION TESTS
# ==========================================================


def test_normalize_integer() -> None:
    assert normalize_amount(498000) == 498000.0


def test_normalize_string_number() -> None:
    assert normalize_amount("498000") == 498000.0


def test_normalize_indian_comma_format() -> None:
    assert (
        normalize_amount("4,98,000")
        == 498000.0
    )


def test_normalize_rupee_format() -> None:
    assert (
        normalize_amount("₹4,98,000")
        == 498000.0
    )


def test_normalize_rs_format() -> None:
    assert (
        normalize_amount("Rs. 4,98,000")
        == 498000.0
    )


def test_normalize_parentheses_as_negative() -> None:
    assert (
        normalize_amount("(1,000)")
        == -1000.0
    )


def test_normalize_invalid_value_returns_none() -> None:
    assert (
        normalize_amount("not-a-number")
        is None
    )


def test_normalize_none_returns_none() -> None:
    assert (
        normalize_amount(None)
        is None
    )


# ==========================================================
# REAL VEDANT SCENARIO
# ==========================================================


def test_vedant_itr_financial_values_are_consistent() -> None:
    """
    Real extracted values from the Vedant dummy ITR:

        Business Income = 498000
        Total Income    = 498000

    These values are internally consistent.
    """

    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        tax_on_total_income=9900,

        rebate=9900,
    )

    result = (
        FinancialConsistencyEngine().analyze(
            snapshot
        )
    )

    assert (
        result.status
        == FinancialStatus.CLEAN
    )

    assert (
        result.risk_level
        == FinancialSeverity.INFO
    )

    assert (
        result.findings
        == ()
    )


# ==========================================================
# BUSINESS INCOME / TOTAL INCOME
# ==========================================================


def test_business_income_equal_total_income_is_clean() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,
        business_income=498000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert (
        result.status
        == FinancialStatus.CLEAN
    )

    assert not result.findings


def test_business_income_less_than_total_income_is_not_false_positive() -> None:
    """
    Other income may cause total income to exceed business income.

    Therefore this must NOT be treated as suspicious.
    """

    snapshot = FinancialSnapshot(
        total_income=600000,

        business_income=498000,

        other_income=102000,

        deductions=0,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "BUSINESS_INCOME_EXCEEDS_TOTAL_INCOME"
        for finding in result.findings
    )


def test_business_income_exceeds_total_income_without_deduction_is_suspicious() -> None:
    snapshot = FinancialSnapshot(
        total_income=400000,

        business_income=498000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert (
        result.status
        == FinancialStatus.SUSPICIOUS
    )

    assert (
        result.risk_level
        == FinancialSeverity.HIGH
    )

    assert any(
        finding.rule_id
        == "BUSINESS_INCOME_EXCEEDS_TOTAL_INCOME"
        for finding in result.findings
    )


def test_business_income_exceeds_total_income_with_deduction_is_not_automatically_suspicious() -> None:
    """
    Business income can exceed total income if a deduction explains
    the difference.
    """

    snapshot = FinancialSnapshot(
        total_income=450000,

        business_income=498000,

        deductions=48000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "BUSINESS_INCOME_EXCEEDS_TOTAL_INCOME"
        for finding in result.findings
    )


# ==========================================================
# COMPONENT RECONCILIATION
# ==========================================================


def test_total_income_components_reconcile() -> None:
    snapshot = FinancialSnapshot(
        business_income=498000,

        other_income=50000,

        deductions=48000,

        total_income=500000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "TOTAL_INCOME_COMPONENT_MISMATCH"
        for finding in result.findings
    )


def test_total_income_component_mismatch_is_detected() -> None:
    snapshot = FinancialSnapshot(
        business_income=498000,

        other_income=50000,

        deductions=48000,

        total_income=650000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert any(
        finding.rule_id
        == "TOTAL_INCOME_COMPONENT_MISMATCH"
        for finding in result.findings
    )

    assert (
        result.risk_level
        == FinancialSeverity.HIGH
    )


# ==========================================================
# TAX / REBATE
# ==========================================================


def test_rebate_equal_to_tax_is_valid() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        tax_on_total_income=9900,

        rebate=9900,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "REBATE_EXCEEDS_TAX"
        for finding in result.findings
    )


def test_rebate_less_than_tax_is_valid() -> None:
    snapshot = FinancialSnapshot(
        total_income=800000,

        business_income=800000,

        tax_on_total_income=30000,

        rebate=10000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "REBATE_EXCEEDS_TAX"
        for finding in result.findings
    )


def test_rebate_exceeding_tax_is_suspicious() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        tax_on_total_income=9900,

        rebate=15000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert any(
        finding.rule_id
        == "REBATE_EXCEEDS_TAX"
        for finding in result.findings
    )

    assert (
        result.risk_level
        == FinancialSeverity.HIGH
    )


# ==========================================================
# PAYABLE / REFUND SANITY
# ==========================================================


def test_normal_payable_has_no_sign_anomaly() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        tax_on_total_income=9900,

        rebate=9900,

        amount_payable=1000,

        net_tax_payable=1000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        in {
            "NEGATIVE_AMOUNT_PAYABLE",
            "NEGATIVE_NET_TAX_PAYABLE",
        }
        for finding in result.findings
    )


def test_negative_amount_payable_is_detected() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        amount_payable=-1000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert any(
        finding.rule_id
        == "NEGATIVE_AMOUNT_PAYABLE"
        for finding in result.findings
    )


def test_negative_net_tax_payable_is_detected() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        net_tax_payable=-1000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert any(
        finding.rule_id
        == "NEGATIVE_NET_TAX_PAYABLE"
        for finding in result.findings
    )


def test_negative_refund_is_detected() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        refund=-1000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert any(
        finding.rule_id
        == "NEGATIVE_REFUND"
        for finding in result.findings
    )


# ==========================================================
# MISSING DATA
# ==========================================================


def test_empty_snapshot_is_incomplete() -> None:
    snapshot = FinancialSnapshot()

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert (
        result.status
        == FinancialStatus.INCOMPLETE
    )

    assert not result.findings


def test_missing_business_income_does_not_create_false_positive() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "BUSINESS_INCOME_EXCEEDS_TOTAL_INCOME"
        for finding in result.findings
    )


def test_missing_total_income_does_not_create_false_positive() -> None:
    snapshot = FinancialSnapshot(
        business_income=498000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert not any(
        finding.rule_id
        == "BUSINESS_INCOME_EXCEEDS_TOTAL_INCOME"
        for finding in result.findings
    )


# ==========================================================
# SERIALIZATION
# ==========================================================


def test_result_serialization_contains_reason() -> None:
    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    payload = (
        result.to_dict()
    )

    assert (
        "status"
        in payload
    )

    assert (
        "risk_level"
        in payload
    )

    assert (
        "risk_score"
        in payload
    )

    assert (
        "confidence"
        in payload
    )

    assert (
        "snapshot"
        in payload
    )

    assert (
        "findings"
        in payload
    )

    assert (
        "reason"
        in payload
    )

    assert (
        "summary"
        in payload
    )


# ==========================================================
# IMPORTANT AUTHENTICITY PRINCIPLE
# ==========================================================


def test_financial_consistency_does_not_claim_authenticity() -> None:
    """
    A financially consistent document is NOT automatically genuine.

    This is critical for the overall architecture because a dummy
    PDF can contain perfectly consistent numbers.
    """

    snapshot = FinancialSnapshot(
        total_income=498000,

        business_income=498000,

        tax_on_total_income=9900,

        rebate=9900,
    )

    result = (
        analyze_financial_consistency(
            snapshot
        )
    )

    assert (
        result.status
        == FinancialStatus.CLEAN
    )

    # The engine reports consistency only.
    assert (
        "authentic"
        not in result.reason.lower()
    )

    assert (
        "genuine"
        not in result.reason.lower()
    )

    assert (
        "fake"
        not in result.reason.lower()
    )