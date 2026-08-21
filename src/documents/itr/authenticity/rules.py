"""
Authenticity rules for ITR documents.

Rules are deterministic and explainable.

Important:
    A single weak signal must NOT automatically classify a document
    as fake.

The authenticity layer should produce evidence and risk, not invent
certainty.
"""

from __future__ import annotations

import re
from typing import Iterable

from .models import (
    AuthenticityFinding,
    IdentitySnapshot,
    RiskLevel,
    SubmissionSnapshot,
)


# ==============================================================
# REGEX PATTERNS
# ==============================================================

_DATE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(\d{1,2})"
    r"[\-/]"
    r"(\d{1,2})"
    r"[\-/]"
    r"(\d{4})"
    r"(?!\d)"
)


_DOB_TEXT_PATTERN = re.compile(
    r"\b"
    r"(?:date\s+of\s+birth|dob)"
    r"\b"
    r"\s*[:\-]?\s*"
    r"("
    r"(?:"
    r"\d{1,2}"
    r"[\-/]"
    r"\d{1,2}"
    r"[\-/]"
    r"\d{4}"
    r")"
    r"|"
    r"(?:"
    r"\d{1,2}"
    r"[-/]\s*"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[-/]\s*"
    r"\d{4}"
    r")"
    r")",
    re.IGNORECASE,
)


_PAN_PATTERN = re.compile(
    r"\b"
    r"[A-Z]{5}"
    r"\d{4}"
    r"[A-Z]"
    r"\b",
    re.IGNORECASE,
)


_ACK_PATTERN = re.compile(
    r"(?:"
    r"acknowledg(?:ement|e)"
    r"|ack\.?\s*no\.?"
    r"|acknowledgement\s+number"
    r")"
    r"\s*[:#\-]?\s*"
    r"(\d{8,20})",
    re.IGNORECASE,
)


_EID_PATTERN = re.compile(
    r"\b"
    r"(?:"
    r"eid"
    r"|e[- ]?filing\s+id"
    r")"
    r"\s*[:#\-]?\s*"
    r"([A-Z0-9\-/]{6,40})",
    re.IGNORECASE,
)


_IP_PATTERN = re.compile(
    r"\b"
    r"(?:ip\s+address|ip)"
    r"\s*[:#\-]?\s*"
    r"("
    r"\d{1,3}"
    r"(?:\.\d{1,3}){3}"
    r")",
    re.IGNORECASE,
)


_FILING_DATE_PATTERN = re.compile(
    r"\b"
    r"date\s+of\s+filing"
    r"\s*[:#\-]?\s*"
    r"("
    r"\d{1,2}[-/]\w{3}[-/]\d{4}"
    r"|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r")",
    re.IGNORECASE,
)


_TIMESTAMP_PATTERN = re.compile(
    r"(?:"
    r"submitted"
    r"|submission"
    r"|filed"
    r"|uploaded"
    r")"
    r".{0,80}?"
    r"("
    r"\d{1,2}[-/]\w{3}[-/]\d{4}"
    r"\s+"
    r"\d{1,2}:\d{2}:\d{2}"
    r")",
    re.IGNORECASE,
)


_NAME_PATTERN = re.compile(
    r"\b"
    r"name"
    r"(?:\s+of\s+(?:the\s+)?assessee)?"
    r"\b"
    r"\s*[:\-]\s*"
    r"("
    r"[A-Za-z]"
    r"[A-Za-z .&'\-]{2,100}"
    r")",
    re.IGNORECASE,
)


# ==============================================================
# NORMALIZATION
# ==============================================================


def _unique(
    values: Iterable[str],
) -> list[str]:
    """
    Return normalized unique values while preserving order.
    """

    result: list[str] = []

    seen: set[str] = set()

    for value in values:

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        key = value.casefold()

        if value and key not in seen:

            result.append(value)

            seen.add(key)

    return result


def _normalize_date(
    value: str,
) -> str | None:
    """
    Normalize numeric dates to DD/MM/YYYY.

    Non-numeric month formats are returned unchanged.
    """

    value = value.strip()

    match = _DATE_PATTERN.fullmatch(
        value
    )

    if not match:
        return value or None

    day, month, year = match.groups()

    day_i = int(day)

    month_i = int(month)

    year_i = int(year)

    if not 1 <= month_i <= 12:
        return None

    if not 1 <= day_i <= 31:
        return None

    return (
        f"{day_i:02d}/"
        f"{month_i:02d}/"
        f"{year_i:04d}"
    )


# ==============================================================
# IDENTITY EXTRACTION
# ==============================================================


def extract_identity_occurrences(
    text: str,
) -> dict[str, list[str]]:
    """
    Extract repeated identity values for authenticity analysis.

    Unlike the canonical ITR extractor, this function intentionally
    collects ALL occurrences.

    This is important because authenticity analysis needs to detect
    conflicts such as:

        DOB: 22-02-2022
        DOB: 22-Feb-1972

    rather than selecting only the first value.
    """

    if not text:
        return {
            "name": [],
            "pan": [],
            "dob": [],
        }

    # ----------------------------------------------------------
    # NAME
    # ----------------------------------------------------------

    names = _unique(
        match.group(1)
        for match in _NAME_PATTERN.finditer(
            text
        )
    )

    # ----------------------------------------------------------
    # PAN
    # ----------------------------------------------------------

    pans = _unique(
        match.group(0).upper()
        for match in _PAN_PATTERN.finditer(
            text
        )
    )

    # ----------------------------------------------------------
    # DOB
    #
    # The direct DOB pattern catches:
    #
    # Date of Birth: 04/01/2001
    #
    # But ITR PDFs can also produce:
    #
    # Date of Birth
    # 22-02-2022
    # : 22-Feb-1972
    #
    # Therefore we additionally inspect the nearby DOB block.
    # ----------------------------------------------------------

    dobs: list[str] = []

    for match in _DOB_TEXT_PATTERN.finditer(
        text
    ):

        raw = re.sub(
            r"\s+",
            "",
            match.group(1),
        )

        normalized = _normalize_date(
            raw
        )

        if normalized:
            dobs.append(
                normalized
            )
        else:
            dobs.append(
                raw
            )

    # ----------------------------------------------------------
    # TABLE-STYLE DOB EXTRACTION
    # ----------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(
        lines
    ):

        if not re.fullmatch(
            r"date\s+of\s+birth",
            line,
            re.IGNORECASE,
        ):
            continue

        # Search the following few lines for date values.
        nearby_lines = lines[
            index + 1:index + 8
        ]

        for candidate in nearby_lines:

            candidate = candidate.strip()

            # Remove leading colon.
            candidate = re.sub(
                r"^:\s*",
                "",
                candidate,
            )

            date_match = re.search(
                r"(?<!\d)"
                r"\d{1,2}"
                r"[-/]"
                r"(?:"
                r"\d{1,2}"
                r"|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|"
                r"Jul|Aug|Sep|Oct|Nov|Dec)"
                r")"
                r"[-/]"
                r"\d{4}"
                r"(?!\d)",
                candidate,
                re.IGNORECASE,
            )

            if not date_match:
                continue

            raw = re.sub(
                r"\s+",
                "",
                date_match.group(0),
            )

            normalized = _normalize_date(
                raw
            )

            dobs.append(
                normalized or raw
            )

    return {
        "name": _unique(names),

        "pan": _unique(pans),

        "dob": _unique(dobs),
    }


# ==============================================================
# SUBMISSION INFORMATION
# ==============================================================


def extract_submission_snapshot(
    text: str,
) -> SubmissionSnapshot:
    """
    Extract submission-level identifiers from native/OCR text.
    """

    if not text:
        return SubmissionSnapshot()

    def first(
        pattern: re.Pattern[str],
    ) -> str | None:

        match = pattern.search(
            text
        )

        if not match:
            return None

        return re.sub(
            r"\s+",
            " ",
            match.group(1),
        ).strip()

    return SubmissionSnapshot(
        acknowledgement_number=first(
            _ACK_PATTERN
        ),

        eid=first(
            _EID_PATTERN
        ),

        filing_date=first(
            _FILING_DATE_PATTERN
        ),

        submission_timestamp=first(
            _TIMESTAMP_PATTERN
        ),

        ip_address=first(
            _IP_PATTERN
        ),
    )


# ==============================================================
# IDENTITY SNAPSHOT
# ==============================================================


def identity_snapshot_from_occurrences(
    occurrences: dict[str, list[str]],
) -> IdentitySnapshot:
    """
    Convert occurrence lists into a best-effort identity snapshot.
    """

    names = occurrences.get(
        "name",
        [],
    )

    pans = occurrences.get(
        "pan",
        [],
    )

    dobs = occurrences.get(
        "dob",
        [],
    )

    return IdentitySnapshot(
        name=(
            names[0]
            if names
            else None
        ),

        pan=(
            pans[0]
            if pans
            else None
        ),

        dob=(
            dobs[0]
            if dobs
            else None
        ),
    )


# ==============================================================
# IN-DOCUMENT IDENTITY CONFLICTS
# ==============================================================


def rule_identity_conflicts(
    occurrences: dict[str, list[str]],
) -> list[AuthenticityFinding]:
    """
    Detect multiple conflicting taxpayer identity values.

    Important:
        PAN is intentionally excluded from generic conflict detection.

    An ITR can legitimately contain:

        taxpayer PAN
        verifier PAN
        representative PAN

    Therefore, seeing two PANs is not sufficient evidence of tampering.
    """

    findings: list[
        AuthenticityFinding
    ] = []

    checks = (
        (
            "name",
            RiskLevel.MEDIUM,
            25.0,
            (
                "Multiple different names were detected inside the "
                "same ITR. This can indicate that taxpayer identity "
                "information was altered, merged, or extracted from "
                "different document sections."
            ),
        ),

        (
            "dob",
            RiskLevel.HIGH,
            45.0,
            (
                "Multiple different dates of birth were detected "
                "inside the same ITR. DOB is a core taxpayer identity "
                "field, so conflicting DOB values are a strong "
                "authenticity anomaly."
            ),
        ),
    )

    for (
        field,
        severity,
        score,
        reason,
    ) in checks:

        values = occurrences.get(
            field,
            [],
        )

        if len(values) <= 1:
            continue

        findings.append(
            AuthenticityFinding(
                rule_id=(
                    f"IDENTITY_MULTI_"
                    f"{field.upper()}"
                ),

                category=(
                    "identity_consistency"
                ),

                severity=severity,

                message=(
                    f"Multiple {field.upper()} "
                    f"values were detected "
                    f"in the document."
                ),

                reason=reason,

                evidence={
                    "field": field,
                    "values": values,
                    "occurrence_count": len(
                        values
                    ),
                },

                score=score,
            )
        )

    return findings


# ==============================================================
# CROSS DOCUMENT SUBMISSION REUSE
# ==============================================================


def rule_submission_reuse(
    current_identity: IdentitySnapshot,

    current_submission: SubmissionSnapshot,

    references: Iterable[
        tuple[
            IdentitySnapshot,
            SubmissionSnapshot,
        ]
    ],
) -> list[AuthenticityFinding]:
    """
    Detect shared submission identifiers with conflicting identities.

    This is the most important rule for the dummy-document scenario.

    Example:

        Document A
            Name = Vedant
            PAN  = XXXXX
            ACK  = 123456

        Document B
            Name = Shashikant
            PAN  = YYYYY
            ACK  = 123456

    Reusing the same stable submission identifier while changing the
    taxpayer identity is a strong authenticity anomaly.
    """

    findings: list[
        AuthenticityFinding
    ] = []

    current_pan = (
        current_identity.pan.upper()
        if current_identity.pan
        else None
    )

    current_name = (
        current_identity.name.casefold()
        if current_identity.name
        else None
    )

    for (
        identity,
        submission,
    ) in references:

        shared: list[str] = []

        fields = (
            (
                "acknowledgement_number",
                current_submission.acknowledgement_number,
                submission.acknowledgement_number,
            ),

            (
                "eid",
                current_submission.eid,
                submission.eid,
            ),

            (
                "filing_date",
                current_submission.filing_date,
                submission.filing_date,
            ),

            (
                "submission_timestamp",
                current_submission.submission_timestamp,
                submission.submission_timestamp,
            ),

            (
                "ip_address",
                current_submission.ip_address,
                submission.ip_address,
            ),
        )

        for (
            field,
            left,
            right,
        ) in fields:

            if not left or not right:
                continue

            if (
                left.strip().casefold()
                ==
                right.strip().casefold()
            ):
                shared.append(
                    field
                )

        reference_pan = (
            identity.pan.upper()
            if identity.pan
            else None
        )

        reference_name = (
            identity.name.casefold()
            if identity.name
            else None
        )

        identity_conflict = (
            bool(
                current_pan
                and reference_pan
                and current_pan
                != reference_pan
            )
            or
            bool(
                current_name
                and reference_name
                and current_name
                != reference_name
            )
        )

        if not shared:
            continue

        if not identity_conflict:
            continue

        # ------------------------------------------------------
        # Score based on number of independently reused fields.
        #
        # ACK + IP + filing date is considerably stronger than
        # one weak matching field.
        # ------------------------------------------------------

        score = min(
            90.0,
            35.0
            + (
                len(shared)
                * 12.0
            ),
        )

        shared_text = ", ".join(
            shared
        )

        reason = (
            "The document shares "
            f"{shared_text} "
            "with another reference document, "
            "while the taxpayer identity differs. "
            "Stable submission identifiers should not normally "
            "belong to different taxpayer identities. "
            "This combination is therefore treated as a "
            "critical cross-document authenticity anomaly."
        )

        findings.append(
            AuthenticityFinding(
                rule_id=(
                    "CROSS_DOCUMENT_"
                    "IDENTITY_REUSE"
                ),

                category="cross_document",

                severity=RiskLevel.CRITICAL,

                message=(
                    "Submission-level identifiers "
                    "are shared with another document "
                    "while taxpayer identity differs."
                ),

                reason=reason,

                evidence={
                    "shared_fields": shared,

                    "shared_field_count": len(
                        shared
                    ),

                    "current_pan": (
                        current_identity.pan
                    ),

                    "reference_pan": (
                        identity.pan
                    ),

                    "current_name": (
                        current_identity.name
                    ),

                    "reference_name": (
                        identity.name
                    ),
                },

                score=score,
            )
        )

    return findings