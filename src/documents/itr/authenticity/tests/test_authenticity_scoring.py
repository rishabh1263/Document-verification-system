"""
Tests for the ITR authenticity scoring engine.
"""

from __future__ import annotations

from src.documents.itr.authenticity.authenticity_scoring import (
    AuthenticityDecision,
    AuthenticityEvidenceInput,
    AuthenticityRiskLevel,
    AuthenticityScoringEngine,
    EvidenceSeverity,
    EvidenceSignal,
    score_authenticity,
)


# ==========================================================
# HELPERS
# ==========================================================


def _signal(
    rule_id: str,
    severity: EvidenceSeverity,
    score: float,
    category: str = "test",
) -> EvidenceSignal:

    return EvidenceSignal(
        rule_id=rule_id,

        category=category,

        severity=severity,

        message=f"Test signal: {rule_id}",

        reason=f"Reason for {rule_id}",

        score=score,

        evidence={
            "test": True,
        },
    )


# ==========================================================
# EMPTY / UNVERIFIED
# ==========================================================


def test_no_evidence_is_unverified() -> None:
    result = (
        AuthenticityScoringEngine().analyze(
            AuthenticityEvidenceInput()
        )
    )

    assert (
        result.decision
        == AuthenticityDecision.UNVERIFIED
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.LOW
    )

    assert result.verified is False

    assert result.evidence_count == 0

    assert "unverified" in (
        result.reason.lower()
    )


# ==========================================================
# LOW RISK
# ==========================================================


def test_low_signal_remains_unverified() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "PDF_METADATA_MODIFIED",
                EvidenceSeverity.LOW,
                10.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.LOW
    )

    assert (
        result.decision
        == AuthenticityDecision.UNVERIFIED
    )

    assert result.verified is False


# ==========================================================
# MEDIUM
# ==========================================================


def test_medium_signal_requires_manual_review() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "PDF_EDITOR_METADATA",
                EvidenceSeverity.MEDIUM,
                30.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.MEDIUM
    )

    assert (
        result.decision
        == AuthenticityDecision.MANUAL_REVIEW
    )

    assert result.verified is False


# ==========================================================
# HIGH
# ==========================================================


def test_high_signal_creates_high_risk() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "IDENTITY_CONFLICT",
                EvidenceSeverity.HIGH,
                65.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.HIGH
    )

    assert (
        result.decision
        == AuthenticityDecision.HIGH_RISK
    )

    assert result.verified is False


# ==========================================================
# CRITICAL
# ==========================================================


def test_critical_signal_causes_reject() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "CROSS_DOCUMENT_IDENTITY_REUSE",
                EvidenceSeverity.CRITICAL,
                100.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.CRITICAL
    )

    assert (
        result.decision
        == AuthenticityDecision.REJECT
    )

    assert result.verified is False

    assert (
        result.risk_score
        == 1.0
    )


# ==========================================================
# DUMMY VEDANT -> SHASHIKANT SCENARIO
# ==========================================================


def test_dummy_identity_reuse_is_critical() -> None:
    """
    This represents the scenario discussed in the project:

        Vedant:
            PAN = MCVPS7350E

        Shashikant:
            PAN = AGRPV4014C

    But both documents reuse the same submission identity.
    """

    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "CROSS_DOCUMENT_IDENTITY_REUSE",
                EvidenceSeverity.CRITICAL,
                100.0,
                category="cross_document",
            ),

            _signal(
                "IDENTITY_CHANGED",
                EvidenceSeverity.HIGH,
                70.0,
                category="identity",
            ),
        ),

        cross_document_critical=True,
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.decision
        == AuthenticityDecision.REJECT
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.CRITICAL
    )

    assert result.critical_count == 1

    assert result.high_count == 1

    assert result.verified is False


# ==========================================================
# AUTHORITATIVE VERIFICATION
# ==========================================================


def test_authoritative_verification_overrides_risk_signals() -> None:
    """
    An authoritative verification result is stronger than
    heuristic PDF signals.
    """

    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "PDF_METADATA_MODIFIED",
                EvidenceSeverity.LOW,
                10.0,
            ),

            _signal(
                "PDF_EDITOR_METADATA",
                EvidenceSeverity.MEDIUM,
                30.0,
            ),
        ),

        authoritative_verified=True,
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.decision
        == AuthenticityDecision.VERIFIED
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.NONE
    )

    assert result.verified is True

    assert (
        result.confidence
        == 0.99
    )


def test_authoritative_failure_is_reject() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "EXTERNAL_MISMATCH",
                EvidenceSeverity.HIGH,
                80.0,
            ),
        ),

        authoritative_failed=True,
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.decision
        == AuthenticityDecision.REJECT
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.CRITICAL
    )

    assert result.verified is False

    assert (
        result.risk_score
        == 1.0
    )


# ==========================================================
# MULTIPLE SIGNALS
# ==========================================================


def test_multiple_independent_signals_raise_risk() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "PDF_EDITOR_METADATA",
                EvidenceSeverity.MEDIUM,
                30.0,
            ),

            _signal(
                "FINANCIAL_MISMATCH",
                EvidenceSeverity.HIGH,
                60.0,
            ),

            _signal(
                "MIXED_PAGE_STRUCTURE",
                EvidenceSeverity.LOW,
                10.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.risk_level
        == AuthenticityRiskLevel.HIGH
    )

    assert (
        result.decision
        == AuthenticityDecision.HIGH_RISK
    )

    assert (
        result.evidence_count
        == 3
    )


# ==========================================================
# REASON
# ==========================================================


def test_primary_reason_is_exposed() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            EvidenceSignal(
                rule_id="TEST_REASON",

                category="test",

                severity=(
                    EvidenceSeverity.HIGH
                ),

                message="Suspicious event",

                reason=(
                    "This is the important explanation."
                ),

                score=70.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert (
        result.reason
        ==
        "This is the important explanation."
    )


# ==========================================================
# SERIALIZATION
# ==========================================================


def test_result_serialization() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "TEST_SIGNAL",
                EvidenceSeverity.MEDIUM,
                30.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    payload = (
        result.to_dict()
    )

    assert (
        payload["status"]
        == "unverified"
    )

    assert (
        "decision"
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
        "verified"
        in payload
    )

    assert (
        "signals"
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

    assert isinstance(
        payload["signals"],
        list,
    )


# ==========================================================
# IMPORTANT SAFETY / ACCURACY RULE
# ==========================================================


def test_clean_document_is_not_claimed_genuine() -> None:
    """
    This is one of the most important tests.

    A clean document with no anomaly is still UNVERIFIED
    unless an authoritative source confirms it.
    """

    evidence = AuthenticityEvidenceInput(
        signals=()
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert result.verified is False

    assert (
        result.decision
        == AuthenticityDecision.UNVERIFIED
    )

    assert (
        "authoritative"
        in result.reason.lower()
    )


# ==========================================================
# SIGNAL COUNTS
# ==========================================================


def test_signal_counts_are_correct() -> None:
    evidence = AuthenticityEvidenceInput(
        signals=(
            _signal(
                "CRITICAL",
                EvidenceSeverity.CRITICAL,
                100.0,
            ),

            _signal(
                "HIGH",
                EvidenceSeverity.HIGH,
                70.0,
            ),

            _signal(
                "MEDIUM",
                EvidenceSeverity.MEDIUM,
                30.0,
            ),

            _signal(
                "LOW",
                EvidenceSeverity.LOW,
                10.0,
            ),
        ),
    )

    result = (
        score_authenticity(
            evidence
        )
    )

    assert result.critical_count == 1

    assert result.high_count == 1

    assert result.medium_count == 1

    assert result.low_count == 1

    assert result.evidence_count == 4