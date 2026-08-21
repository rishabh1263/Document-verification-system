"""
Tests for ITR cross-document reference comparison.

These tests specifically cover the dummy-document scenario:

    Vedant ITR
        vs
    Shashikant ITR

where taxpayer identity is changed while submission-level
metadata remains reused.
"""

from __future__ import annotations

from src.documents.itr.authenticity.reference.reference_comparison import (
    ITRReferenceComparisonEngine,
    ITRReferenceIdentity,
    ITRSubmissionMetadata,
    ITRReferenceSnapshot,
)


# ============================================================
# FIXTURES
# ============================================================


VEDANT_TEXT = """
Acknowledgement Number: 917312470270325
Date of filing : 27-Mar-2025

Assessment Year
2024-25

PAN
MCVPS7350E

Name
Vedant Ashish Sinagare

Date of Birth
04/01/2001

Updated Income Tax Return submitted electronically
on 27-Mar-2025 17:30:18 from IP address
150.129.88.225 and verified by
KRISHNARAJANAGAR LAKSHMI PATHAIAH VISHNUVARDHANA
having PAN AGRPV4014C on 27-Mar-2025 using
EIDKJ56GDI generated through Aadhaar
OTP mode
"""


SHASHIKANT_TEXT = """
Acknowledgement Number: 917312470270325
Date of filing : 27-Mar-2025

Assessment Year
2025-26

PAN
AGRPV4014C

Name
Shashikant Sanjay Kandekar

Date of Birth
22/02/1972

Updated Income Tax Return submitted electronically
on 27-Mar-2025 17:30:18 from IP address
150.129.88.225 and verified by
KRISHNARAJANAGAR LAKSHMI PATHAIAH VISHNUVARDHANA
having PAN AGRPV4014C on 27-Mar-2025 using
EIDKJ56GDI generated through Aadhaar
OTP mode
"""


# ============================================================
# SNAPSHOT EXTRACTION
# ============================================================


def test_build_snapshot_extracts_identity() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    snapshot = (
        engine.build_snapshot(
            VEDANT_TEXT,
            document_id="vedant",
        )
    )

    assert (
        snapshot.identity.name
        == "Vedant Ashish Sinagare"
    )

    assert (
        snapshot.identity.pan
        == "MCVPS7350E"
    )

    assert (
        snapshot.identity.dob
        == "04/01/2001"
    )

    assert (
        snapshot.identity.assessment_year
        == "2024-25"
    )


# ============================================================
# SUBMISSION METADATA
# ============================================================


def test_build_snapshot_extracts_submission_metadata() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    snapshot = (
        engine.build_snapshot(
            VEDANT_TEXT
        )
    )

    submission = (
        snapshot.submission
    )

    assert (
        submission.acknowledgement_number
        == "917312470270325"
    )

    assert (
        submission.filing_date
        == "27/03/2025"
    )

    assert (
        submission.submission_timestamp
        == "27/03/2025 17:30:18"
    )

    assert (
        submission.ip_address
        == "150.129.88.225"
    )

    assert (
        submission.eid
        == "EIDKJ56GDI"
    )

    assert (
        submission.verifier_pan
        == "AGRPV4014C"
    )


# ============================================================
# FINGERPRINT
# ============================================================


def test_fingerprint_requires_two_submission_fields() -> None:

    submission = ITRSubmissionMetadata(
        acknowledgement_number=(
            "917312470270325"
        ),
    )

    assert (
        ITRReferenceComparisonEngine
        .generate_submission_fingerprint(
            submission
        )
        is None
    )


def test_fingerprint_is_deterministic() -> None:

    submission = ITRSubmissionMetadata(
        acknowledgement_number=(
            "917312470270325"
        ),

        filing_date=(
            "27/03/2025"
        ),

        ip_address=(
            "150.129.88.225"
        ),
    )

    fingerprint_one = (
        ITRReferenceComparisonEngine
        .generate_submission_fingerprint(
            submission
        )
    )

    fingerprint_two = (
        ITRReferenceComparisonEngine
        .generate_submission_fingerprint(
            submission
        )
    )

    assert fingerprint_one is not None

    assert (
        fingerprint_one
        == fingerprint_two
    )


# ============================================================
# SAME SUBMISSION FINGERPRINT
# ============================================================


def test_dummy_documents_have_same_submission_fingerprint() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    vedant = (
        engine.build_snapshot(
            VEDANT_TEXT
        )
    )

    shashikant = (
        engine.build_snapshot(
            SHASHIKANT_TEXT
        )
    )

    assert (
        vedant.fingerprint
        is not None
    )

    assert (
        shashikant.fingerprint
        is not None
    )

    assert (
        vedant.fingerprint
        == shashikant.fingerprint
    )


# ============================================================
# IDENTITY CONFLICT
# ============================================================


def test_dummy_documents_have_identity_conflict() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    vedant = (
        engine.build_snapshot(
            VEDANT_TEXT
        )

    )

    shashikant = (
        engine.build_snapshot(
            SHASHIKANT_TEXT
        )
    )

    result = engine.compare(
        submitted=shashikant,
        reference=vedant,
    )

    assert (
        result.identity_conflict
        is True
    )

    assert (
        "name"
        in result.conflicting_identity_fields
    )

    assert (
        "pan"
        in result.conflicting_identity_fields
    )

    assert (
        "dob"
        in result.conflicting_identity_fields
    )


# ============================================================
# SHARED SUBMISSION FIELDS
# ============================================================


def test_dummy_documents_share_submission_fields() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    vedant = (
        engine.build_snapshot(
            VEDANT_TEXT
        )
    )

    shashikant = (
        engine.build_snapshot(
            SHASHIKANT_TEXT
        )
    )

    result = engine.compare(
        submitted=shashikant,
        reference=vedant,
    )

    assert (
        "acknowledgement_number"
        in result.shared_submission_fields
    )

    assert (
        "filing_date"
        in result.shared_submission_fields
    )

    assert (
        "submission_timestamp"
        in result.shared_submission_fields
    )

    assert (
        "ip_address"
        in result.shared_submission_fields
    )

    assert (
        "eid"
        in result.shared_submission_fields
    )

    assert (
        "verifier_pan"
        in result.shared_submission_fields
    )


# ============================================================
# CRITICAL DUMMY DETECTION
# ============================================================


def test_dummy_identity_reuse_is_critical() -> None:

    result = (
        ITRReferenceComparisonEngine()
        .compare(
            submitted=(
                ITRReferenceComparisonEngine()
                .build_snapshot(
                    SHASHIKANT_TEXT,
                    document_id="shashikant",
                )
            ),

            reference=(
                ITRReferenceComparisonEngine()
                .build_snapshot(
                    VEDANT_TEXT,
                    document_id="vedant",
                )
            ),
        )
    )

    assert (
        result.critical_reuse
        is True
    )

    assert (
        result.suspicious_reuse
        is True
    )

    assert (
        result.fingerprint_match
        is True
    )

    assert any(
        signal.rule_id
        == "REFERENCE_FINGERPRINT_IDENTITY_REUSE"
        for signal in result.signals
    )


# ============================================================
# SAME IDENTITY SHOULD NOT BE FALSE POSITIVE
# ============================================================


def test_same_identity_is_not_critical() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    first = (
        engine.build_snapshot(
            VEDANT_TEXT,
            document_id="first",
        )
    )

    second = (
        engine.build_snapshot(
            VEDANT_TEXT,
            document_id="second",
        )
    )

    result = engine.compare(
        submitted=second,
        reference=first,
    )

    assert (
        result.identity_conflict
        is False
    )

    assert (
        result.critical_reuse
        is False
    )


# ============================================================
# ONE SHARED FIELD
# ============================================================


def test_one_shared_field_is_not_critical() -> None:

    submitted = ITRReferenceSnapshot(
        identity=ITRReferenceIdentity(
            name="Shashikant Sanjay Kandekar",
            pan="AGRPV4014C",
            dob="22/02/1972",
            assessment_year="2025-26",
        ),

        submission=ITRSubmissionMetadata(
            acknowledgement_number=(
                "917312470270325"
            ),
        ),

        fingerprint=None,
    )

    reference = ITRReferenceSnapshot(
        identity=ITRReferenceIdentity(
            name="Vedant Ashish Sinagare",
            pan="MCVPS7350E",
            dob="04/01/2001",
            assessment_year="2024-25",
        ),

        submission=ITRSubmissionMetadata(
            acknowledgement_number=(
                "917312470270325"
            ),

            filing_date=(
                "27/03/2025"
            ),
        ),

        fingerprint=None,
    )

    result = (
        ITRReferenceComparisonEngine()
        .compare(
            submitted,
            reference,
        )
    )

    assert (
        result.identity_conflict
        is True
    )

    assert (
        result.critical_reuse
        is False
    )

    assert (
        result.suspicious_reuse
        is True
    )

    assert (
        result.signals[0].severity
        == "medium"
    )


# ============================================================
# MISSING VALUES
# ============================================================


def test_missing_reference_fields_do_not_create_conflict() -> None:

    submitted = ITRReferenceSnapshot(
        identity=ITRReferenceIdentity(
            name="Vedant Ashish Sinagare",
            pan="MCVPS7350E",
        ),

        submission=ITRSubmissionMetadata(
            acknowledgement_number=(
                "917312470270325"
            ),
        ),
    )

    reference = ITRReferenceSnapshot(
        identity=ITRReferenceIdentity(
            name=None,
            pan=None,
        ),

        submission=ITRSubmissionMetadata(
            acknowledgement_number=None,
        ),
    )

    result = (
        ITRReferenceComparisonEngine()
        .compare(
            submitted,
            reference,
        )
    )

    assert (
        result.identity_conflict
        is False
    )

    assert (
        result.shared_submission_fields
        == ()
    )


# ============================================================
# SERIALIZATION
# ============================================================


def test_result_serialization_contains_reason() -> None:

    engine = (
        ITRReferenceComparisonEngine()
    )

    result = engine.compare(
        submitted=(
            engine.build_snapshot(
                SHASHIKANT_TEXT
            )
        ),

        reference=(
            engine.build_snapshot(
                VEDANT_TEXT
            )
        ),
    )

    payload = result.to_dict()

    assert (
        "reason"
        in payload
    )

    assert (
        payload["reason"]
    )

    assert (
        "signals"
        in payload
    )

    assert isinstance(
        payload["signals"],
        list,
    )