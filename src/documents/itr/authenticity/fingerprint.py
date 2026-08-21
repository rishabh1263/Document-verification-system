"""
Stable fingerprints for ITR authenticity comparison.

The fingerprint intentionally excludes mutable taxpayer identity
fields.

It is designed to help detect reuse of stable submission-level
information across documents.

It is NOT a cryptographic proof that a document is genuine.
"""

from __future__ import annotations

import hashlib
import re

from .models import SubmissionSnapshot


# ==============================================================
# NORMALIZATION
# ==============================================================


def normalize_identifier(
    value: str | None,
) -> str | None:
    """
    Normalize an identifier for deterministic comparison.

    Examples:

        " ABCD 123 "
            ->
        "ABCD123"
    """

    if not value:
        return None

    normalized = re.sub(
        r"\s+",
        "",
        value,
    ).strip().upper()

    return (
        normalized
        or None
    )


# ==============================================================
# FINGERPRINT
# ==============================================================


def build_submission_fingerprint(
    submission: SubmissionSnapshot,
) -> str | None:
    """
    Build a deterministic SHA-256 fingerprint from stable
    submission-level information.

    Fields currently considered:

        acknowledgement number
        EID
        filing date
        submission timestamp
        IP address
        verifier PAN

    At least TWO fields must be available.

    This prevents a single OCR-extracted value from becoming
    an overly strong fingerprint.
    """

    fields = {
        "acknowledgement_number": (
            normalize_identifier(
                submission.acknowledgement_number
            )
        ),

        "eid": (
            normalize_identifier(
                submission.eid
            )
        ),

        "filing_date": (
            normalize_identifier(
                submission.filing_date
            )
        ),

        "submission_timestamp": (
            normalize_identifier(
                submission.submission_timestamp
            )
        ),

        "ip_address": (
            normalize_identifier(
                submission.ip_address
            )
        ),

        "verifier_pan": (
            normalize_identifier(
                submission.verifier_pan
            )
        ),
    }

    present_fields = [
        f"{key}={value}"
        for key, value
        in sorted(
            fields.items()
        )
        if value
    ]

    if len(
        present_fields
    ) < 2:
        return None

    payload = "|".join(
        present_fields
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


# ==============================================================
# SHARED FIELD DETECTION
# ==============================================================


def shared_submission_fields(
    left: SubmissionSnapshot,

    right: SubmissionSnapshot,
) -> tuple[str, ...]:
    """
    Return stable fields that are identical in both snapshots.

    This is useful for explainability.

    Example:

        (
            "acknowledgement_number",
            "filing_date",
            "ip_address",
        )
    """

    pairs = (
        (
            "acknowledgement_number",

            left.acknowledgement_number,

            right.acknowledgement_number,
        ),

        (
            "eid",

            left.eid,

            right.eid,
        ),

        (
            "filing_date",

            left.filing_date,

            right.filing_date,
        ),

        (
            "submission_timestamp",

            left.submission_timestamp,

            right.submission_timestamp,
        ),

        (
            "ip_address",

            left.ip_address,

            right.ip_address,
        ),

        (
            "verifier_pan",

            left.verifier_pan,

            right.verifier_pan,
        ),
    )

    shared: list[str] = []

    for (
        name,
        left_value,
        right_value,
    ) in pairs:

        left_normalized = (
            normalize_identifier(
                left_value
            )
        )

        right_normalized = (
            normalize_identifier(
                right_value
            )
        )

        if (
            left_normalized
            and
            right_normalized
            and
            left_normalized
            == right_normalized
        ):

            shared.append(
                name
            )

    return tuple(
        shared
    )


# ==============================================================
# FINGERPRINT COMPARISON
# ==============================================================


def fingerprints_match(
    left: SubmissionSnapshot,

    right: SubmissionSnapshot,
) -> bool:
    """
    Return True when both submissions produce the same stable
    fingerprint.

    Returns False when either document does not contain enough
    stable fields to generate a fingerprint.
    """

    left_fingerprint = (
        build_submission_fingerprint(
            left
        )
    )

    right_fingerprint = (
        build_submission_fingerprint(
            right
        )
    )

    if not left_fingerprint:
        return False

    if not right_fingerprint:
        return False

    return (
        left_fingerprint
        ==
        right_fingerprint
    )


# ==============================================================
# FINGERPRINT DETAILS
# ==============================================================


def fingerprint_details(
    submission: SubmissionSnapshot,
) -> dict[str, object]:
    """
    Return explainable fingerprint information.

    The raw values are included because this function is intended
    for controlled internal/audit use. API exposure should be
    considered separately because some fields may be sensitive.
    """

    fingerprint = (
        build_submission_fingerprint(
            submission
        )
    )

    available_fields = []

    field_values = {
        "acknowledgement_number": (
            submission.acknowledgement_number
        ),

        "eid": (
            submission.eid
        ),

        "filing_date": (
            submission.filing_date
        ),

        "submission_timestamp": (
            submission.submission_timestamp
        ),

        "ip_address": (
            submission.ip_address
        ),

        "verifier_pan": (
            submission.verifier_pan
        ),
    }

    for (
        field,
        value,
    ) in field_values.items():

        if value:
            available_fields.append(
                field
            )

    return {
        "fingerprint": fingerprint,

        "available_fields": (
            available_fields
        ),

        "field_count": len(
            available_fields
        ),

        "fingerprint_available": (
            fingerprint is not None
        ),
    }