"""
Tests for the ITR authenticity layer.

These tests verify:

1. Identity occurrence extraction.
2. DOB conflict detection.
3. Multiple PAN handling.
4. Submission identifier extraction.
5. Cross-document identity reuse.
6. Critical risk classification.
7. Explainable reasons.
8. Empty-document handling.
9. Fingerprint generation.
"""

from __future__ import annotations

from src.documents.itr.authenticity.authenticity_engine import (
    ITRAuthenticityEngine,
)

from src.documents.itr.authenticity.fingerprint import (
    build_submission_fingerprint,
    fingerprints_match,
    shared_submission_fields,
)

from src.documents.itr.authenticity.models import (
    AuthenticityDecision,
    AuthenticityInput,
    RiskLevel,
    SubmissionSnapshot,
)

from src.documents.itr.authenticity.rules import (
    extract_identity_occurrences,
    extract_submission_snapshot,
    rule_identity_conflicts,
)


# ==========================================================
# IDENTITY EXTRACTION
# ==========================================================


def test_extract_multiple_dob_values() -> None:
    text = """
    Name: Vedant Ashish Sinagare
    PAN: MCVPS7350E
    Date of Birth: 04/01/2001

    Date of Birth
    04/01/2001
    """

    values = extract_identity_occurrences(
        text
    )

    assert "04/01/2001" in values["dob"]


def test_detect_conflicting_dob_values() -> None:
    text = """
    Name: Shashikant Sanjay Kandekar
    PAN: AGRPV4014C

    Date of Birth: 22/02/2022

    Date of Birth
    22/02/1972
    """

    values = extract_identity_occurrences(
        text
    )

    findings = rule_identity_conflicts(
        values
    )

    dob_findings = [
        finding
        for finding in findings
        if finding.rule_id
        == "IDENTITY_MULTI_DOB"
    ]

    assert dob_findings

    finding = dob_findings[0]

    assert finding.severity == RiskLevel.HIGH

    assert (
        "DOB"
        in finding.message.upper()
    )

    assert finding.reason


def test_same_dob_does_not_create_conflict() -> None:
    text = """
    Date of Birth: 04/01/2001
    Date of Birth
    04/01/2001
    """

    values = extract_identity_occurrences(
        text
    )

    findings = rule_identity_conflicts(
        values
    )

    dob_findings = [
        finding
        for finding in findings
        if finding.rule_id
        == "IDENTITY_MULTI_DOB"
    ]

    assert not dob_findings


# ==========================================================
# PAN HANDLING
# ==========================================================


def test_multiple_pan_values_are_extracted_without_false_conflict() -> None:
    text = """
    PAN: AGRPV4014C
    Verifier PAN: MCVPS7350E
    """

    values = extract_identity_occurrences(
        text
    )

    assert (
        "AGRPV4014C"
        in values["pan"]
    )

    assert (
        "MCVPS7350E"
        in values["pan"]
    )

    findings = rule_identity_conflicts(
        values
    )

    pan_findings = [
        finding
        for finding in findings
        if finding.rule_id
        == "IDENTITY_MULTI_PAN"
    ]

    assert not pan_findings


# ==========================================================
# SUBMISSION EXTRACTION
# ==========================================================


def test_extract_acknowledgement_number() -> None:
    text = """
    Acknowledgement Number: 917312470270325
    """

    submission = (
        extract_submission_snapshot(
            text
        )
    )

    assert (
        submission.acknowledgement_number
        == "917312470270325"
    )


def test_extract_eid() -> None:
    text = """
    EID: ABC123456789
    """

    submission = (
        extract_submission_snapshot(
            text
        )
    )

    assert (
        submission.eid
        == "ABC123456789"
    )


def test_extract_ip_address() -> None:
    text = """
    IP Address: 192.168.10.20
    """

    submission = (
        extract_submission_snapshot(
            text
        )
    )

    assert (
        submission.ip_address
        == "192.168.10.20"
    )


# ==========================================================
# FINGERPRINT
# ==========================================================


def test_fingerprint_requires_two_fields() -> None:
    submission = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        )
    )

    fingerprint = (
        build_submission_fingerprint(
            submission
        )
    )

    assert fingerprint is None


def test_fingerprint_is_generated() -> None:
    submission = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27-Mar-2025"
        ),
    )

    fingerprint = (
        build_submission_fingerprint(
            submission
        )
    )

    assert fingerprint is not None

    assert len(
        fingerprint
    ) == 64


def test_same_fingerprint_matches() -> None:
    left = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27-Mar-2025"
        ),
    )

    right = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27-Mar-2025"
        ),
    )

    assert fingerprints_match(
        left,
        right,
    )


def test_different_fingerprint_does_not_match() -> None:
    left = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27-Mar-2025"
        ),
    )

    right = SubmissionSnapshot(
        acknowledgement_number=(
            "999999999999999"
        ),

        filing_date=(
            "27-Mar-2025"
        ),
    )

    assert not fingerprints_match(
        left,
        right,
    )


def test_shared_submission_fields() -> None:
    left = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27-Mar-2025"
        ),

        ip_address=(
            "192.168.10.20"
        ),
    )

    right = SubmissionSnapshot(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27-Mar-2025"
        ),

        ip_address=(
            "10.10.10.10"
        ),
    )

    shared = shared_submission_fields(
        left,
        right,
    )

    assert (
        "acknowledgement_number"
        in shared
    )

    assert (
        "filing_date"
        in shared
    )

    assert (
        "ip_address"
        not in shared
    )


# ==========================================================
# ENGINE — EMPTY INPUT
# ==========================================================


def test_empty_document_is_high_risk() -> None:
    engine = (
        ITRAuthenticityEngine()
    )

    result = engine.analyze(
        AuthenticityInput(
            text=""
        )
    )

    assert (
        result.decision
        == AuthenticityDecision.HIGH_RISK
    )

    assert (
        result.verified
        is False
    )

    assert result.reason


# ==========================================================
# ENGINE — NO REFERENCE
# ==========================================================


def test_clean_document_is_unverified_without_reference() -> None:
    text = """
    Name: Vedant Ashish Sinagare
    PAN: MCVPS7350E
    Date of Birth: 04/01/2001
    """

    engine = (
        ITRAuthenticityEngine()
    )

    result = engine.analyze(
        AuthenticityInput(
            text=text
        )
    )

    assert (
        result.decision
        == AuthenticityDecision.UNVERIFIED
    )

    assert (
        result.verified
        is False
    )

    assert result.reason


# ==========================================================
# ENGINE — DOB CONFLICT
# ==========================================================


def test_dob_conflict_becomes_suspicious_or_high_risk() -> None:
    text = """
    Name: Shashikant Sanjay Kandekar
    PAN: AGRPV4014C

    Date of Birth: 22/02/2022

    Date of Birth
    22/02/1972
    """

    engine = (
        ITRAuthenticityEngine()
    )

    result = engine.analyze(
        AuthenticityInput(
            text=text
        )
    )

    assert result.decision in {
        AuthenticityDecision.SUSPICIOUS,
        AuthenticityDecision.HIGH_RISK,
    }

    assert result.findings

    assert result.reason


# ==========================================================
# ENGINE — CROSS DOCUMENT DUMMY DETECTION
# ==========================================================


def test_cross_document_identity_reuse_is_critical() -> None:
    """
    Simulates the real dummy-document scenario.

    Reference document:

        Vedant
        PAN = MCVPS7350E

    Submitted document:

        Shashikant
        PAN = AGRPV4014C

    Both reuse the same acknowledgement number and filing
    metadata.
    """

    reference_text = """
    Acknowledgement Number: 917312470270325
    Date of filing: 27-Mar-2025
    IP Address: 192.168.10.20

    Name: Vedant Ashish Sinagare
    PAN: MCVPS7350E
    Date of Birth: 04/01/2001
    """

    submitted_text = """
    Acknowledgement Number: 917312470270325
    Date of filing: 27-Mar-2025
    IP Address: 192.168.10.20

    Name: Shashikant Sanjay Kandekar
    PAN: AGRPV4014C
    Date of Birth: 22/02/1972
    """

    reference = AuthenticityInput(
        text=reference_text,

        document_id=(
            "vedant-reference"
        ),
    )

    submitted = AuthenticityInput(
        text=submitted_text,

        document_id=(
            "shashikant-submitted"
        ),

        reference_documents=(
            reference,
        ),
    )

    engine = (
        ITRAuthenticityEngine()
    )

    result = engine.analyze(
        submitted
    )

    assert (
        result.decision
        == AuthenticityDecision.HIGH_RISK
    )

    assert (
        result.risk_level
        == RiskLevel.CRITICAL
    )

    assert (
        result.verified
        is False
    )

    critical_findings = [
        finding
        for finding in result.findings
        if finding.severity
        == RiskLevel.CRITICAL
    ]

    assert critical_findings

    finding = (
        critical_findings[0]
    )

    assert (
        finding.rule_id
        == "CROSS_DOCUMENT_IDENTITY_REUSE"
    )

    assert finding.reason

    assert (
        "different"
        in finding.reason.lower()
        or
        "identity"
        in finding.reason.lower()
    )

    assert result.reason


# ==========================================================
# ENGINE — API SERIALIZATION
# ==========================================================


def test_result_serialization_contains_reason() -> None:
    text = """
    Name: Vedant Ashish Sinagare
    PAN: MCVPS7350E
    Date of Birth: 04/01/2001
    """

    engine = (
        ITRAuthenticityEngine()
    )

    result = engine.analyze(
        AuthenticityInput(
            text=text
        )
    )

    payload = (
        result.to_dict()
    )

    assert "decision" in payload

    assert "risk_level" in payload

    assert "reason" in payload

    assert "findings" in payload

    assert isinstance(
        payload["findings"],
        list,
    )