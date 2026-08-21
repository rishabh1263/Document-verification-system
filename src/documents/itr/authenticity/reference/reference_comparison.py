"""
ITR Cross-Document Reference Comparison Engine.

Purpose
-------
Detect suspicious reuse of submission-level metadata across
different ITR documents.

This layer is specifically designed to detect scenarios such as:

    Reference:
        Name = Vedant Ashish Sinagare
        PAN  = MCVPS7350E
        DOB  = 04/01/2001

    Submitted:
        Name = Shashikant Sanjay Kandekar
        PAN  = AGRPV4014C
        DOB  = 22/02/1972

while submission-level information remains the same:

        Acknowledgement Number
        Filing Date
        Filing Timestamp
        IP Address
        EID
        Verifier PAN

Important
---------
Matching submission metadata does NOT automatically mean fraud.

This engine produces EVIDENCE.

The final authenticity scoring engine decides whether that
evidence is low, medium, high, or critical risk.

Design principles
-----------------
1. Identity fields and submission fields are treated separately.
2. One shared field is not enough to declare a document fake.
3. Two or more strong shared submission fields combined with
   conflicting taxpayer identity create a critical signal.
4. Missing fields are ignored rather than treated as mismatches.
5. Exact normalized values are used to avoid formatting noise.
6. The engine is deterministic and explainable.
7. No external network call is made by this module.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ============================================================
# CONSTANTS
# ============================================================


PAN_PATTERN = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b("
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|"
    r"\d{1,2}-"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"-\d{4}"
    r")\b",
    re.IGNORECASE,
)

IP_PATTERN = re.compile(
    r"\b"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\."
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"){3}"
    r"\b"
)

ACK_PATTERN = re.compile(
    r"(?:acknowledgement|acknowledgment)"
    r"\s+number\s*[:\-]?\s*"
    r"([0-9]{8,20})",
    re.IGNORECASE,
)

EFILING_ACK_PATTERN = re.compile(
    r"e[-\s]?filing\s+acknowledgement"
    r"\s+number\s*[:\-]?\s*"
    r"([0-9]{8,20})",
    re.IGNORECASE,
)

EID_PATTERN = re.compile(
    r"\bEID\s*[:\-]?\s*"
    r"([A-Z0-9]{5,40})\b",
    re.IGNORECASE,
)

# Handles:
#
#     Date of filing: 27-Mar-2025
#     Date of filing : 27-Mar-2025
#
# and:
#
#     Updated Income Tax Return submitted electronically
#     on 27-Mar-2025 17:30:18
#
FILING_DATE_PATTERN = re.compile(
    r"(?:date\s+of\s+filing|filed\s+on)"
    r"\s*[:\-]?\s*"
    r"("
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|"
    r"\d{1,2}-"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"-\d{4}"
    r")",
    re.IGNORECASE,
)

SUBMISSION_TIMESTAMP_PATTERN = re.compile(
    r"submitted\s+electronically"
    r".{0,100}?"
    r"on\s+"
    r"("
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|"
    r"\d{1,2}-"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"-\d{4}"
    r")"
    r"\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?)",
    re.IGNORECASE | re.DOTALL,
)

VERIFIER_PAN_PATTERN = re.compile(
    r"having\s+"
    r"(?:PAN|P\s*A\s*N)"
    r"\s*"
    r"([A-Z]{5}[0-9]{4}[A-Z])",
    re.IGNORECASE,
)

VERIFIED_BY_PATTERN = re.compile(
    r"verified\s+by\s+"
    r"([A-Za-z][A-Za-z .'\-&]{2,120}?)"
    r"\s+having\s+"
    r"(?:PAN|P\s*A\s*N)",
    re.IGNORECASE,
)

NAME_PATTERN = re.compile(
    r"\bname\s*[:\-]?\s*"
    r"([A-Za-z][A-Za-z .'\-&]{2,120})",
    re.IGNORECASE,
)

DOB_PATTERN = re.compile(
    r"(?:date\s+of\s+birth|dob)"
    r"\s*[:\-]?\s*"
    r"("
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|"
    r"\d{1,2}-"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"-\d{4}"
    r")",
    re.IGNORECASE,
)

ASSESSMENT_YEAR_PATTERN = re.compile(
    r"(?:assessment\s+year|asst\.?\s+year|a\.?\s*y\.?)"
    r"\s*[:\-]?\s*"
    r"((?:19|20)\d{2}\s*[-/]\s*(?:\d{2}|\d{4}))",
    re.IGNORECASE,
)


# ============================================================
# DATA MODELS
# ============================================================


@dataclass(frozen=True)
class ITRReferenceIdentity:
    """
    Taxpayer identity fields.
    """

    name: str | None = None

    pan: str | None = None

    dob: str | None = None

    assessment_year: str | None = None


@dataclass(frozen=True)
class ITRSubmissionMetadata:
    """
    Submission-level metadata.

    These fields are intentionally separated from taxpayer
    identity because a dummy document can preserve these
    values while changing the taxpayer identity.
    """

    acknowledgement_number: str | None = None

    filing_date: str | None = None

    submission_timestamp: str | None = None

    ip_address: str | None = None

    eid: str | None = None

    verifier_pan: str | None = None

    verifier_name: str | None = None


@dataclass(frozen=True)
class ITRReferenceSnapshot:
    """
    Complete reference snapshot.
    """

    identity: ITRReferenceIdentity

    submission: ITRSubmissionMetadata

    fingerprint: str | None = None

    document_id: str | None = None


@dataclass(frozen=True)
class ReferenceComparisonSignal:
    """
    One explainable cross-document signal.
    """

    rule_id: str

    severity: str

    score: float

    message: str

    reason: str

    matched_fields: tuple[str, ...] = ()

    conflicting_identity_fields: tuple[str, ...] = ()

    evidence: dict[str, Any] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize signal.
        """

        return {
            "rule_id": self.rule_id,

            "severity": self.severity,

            "score": self.score,

            "message": self.message,

            "reason": self.reason,

            "matched_fields": list(
                self.matched_fields
            ),

            "conflicting_identity_fields": list(
                self.conflicting_identity_fields
            ),

            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ReferenceComparisonResult:
    """
    Result of comparing one submitted ITR against one
    reference document.
    """

    matched_fields: tuple[str, ...] = ()

    conflicting_identity_fields: tuple[str, ...] = ()

    shared_submission_fields: tuple[str, ...] = ()

    fingerprint_match: bool = False

    identity_conflict: bool = False

    suspicious_reuse: bool = False

    critical_reuse: bool = False

    signals: tuple[
        ReferenceComparisonSignal,
        ...
    ] = ()

    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize comparison result.
        """

        return {
            "matched_fields": list(
                self.matched_fields
            ),

            "conflicting_identity_fields": list(
                self.conflicting_identity_fields
            ),

            "shared_submission_fields": list(
                self.shared_submission_fields
            ),

            "fingerprint_match": (
                self.fingerprint_match
            ),

            "identity_conflict": (
                self.identity_conflict
            ),

            "suspicious_reuse": (
                self.suspicious_reuse
            ),

            "critical_reuse": (
                self.critical_reuse
            ),

            "signals": [
                signal.to_dict()
                for signal in self.signals
            ],

            "reason": self.reason,
        }


# ============================================================
# ENGINE
# ============================================================


class ITRReferenceComparisonEngine:
    """
    Compare an ITR against a trusted/reference document.

    This engine does NOT decide whether a document is genuine.

    It identifies reusable submission metadata and identity
    conflicts that can be passed to the authenticity scoring
    engine.
    """

    # Strong submission-level fields.
    STRONG_SUBMISSION_FIELDS = (
        "acknowledgement_number",
        "filing_date",
        "submission_timestamp",
        "ip_address",
        "eid",
        "verifier_pan",
    )

    # Identity fields.
    IDENTITY_FIELDS = (
        "name",
        "pan",
        "dob",
        "assessment_year",
    )

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def compare(
        self,
        submitted: ITRReferenceSnapshot,
        reference: ITRReferenceSnapshot,
    ) -> ReferenceComparisonResult:
        """
        Compare submitted document against reference.

        Rules
        -----
        - Same identity is not suspicious merely because
          submission metadata matches.
        - Different identity + 2 strong shared submission
          fields is suspicious.
        - Different identity + 3+ strong shared fields is
          critical.
        - Exact fingerprint match with conflicting identity
          is critical.
        """

        matched_fields = self._compare_submission_fields(
            submitted.submission,
            reference.submission,
        )

        conflicting_identity_fields = (
            self._compare_identity_fields(
                submitted.identity,
                reference.identity,
            )
        )

        identity_conflict = bool(
            conflicting_identity_fields
        )

        fingerprint_match = (
            submitted.fingerprint is not None
            and reference.fingerprint is not None
            and submitted.fingerprint
            == reference.fingerprint
        )

        shared_count = len(
            matched_fields
        )

        signals: list[
            ReferenceComparisonSignal
        ] = []

        # ----------------------------------------------------
        # EXACT FINGERPRINT + IDENTITY CHANGE
        # ----------------------------------------------------

        if (
            fingerprint_match
            and identity_conflict
        ):
            signals.append(
                ReferenceComparisonSignal(
                    rule_id=(
                        "REFERENCE_FINGERPRINT_IDENTITY_REUSE"
                    ),

                    severity="critical",

                    score=100.0,

                    message=(
                        "The submitted document has the same "
                        "submission fingerprint as a reference "
                        "document while taxpayer identity fields "
                        "differ."
                    ),

                    reason=(
                        "This combination is highly suspicious "
                        "because the document appears to reuse "
                        "the same submission-level structure "
                        "for a different taxpayer identity."
                    ),

                    matched_fields=matched_fields,

                    conflicting_identity_fields=(
                        conflicting_identity_fields
                    ),

                    evidence={
                        "fingerprint_match": True,

                        "submitted_document_id": (
                            submitted.document_id
                        ),

                        "reference_document_id": (
                            reference.document_id
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # MULTIPLE SHARED SUBMISSION FIELDS
        # ----------------------------------------------------

        if (
            identity_conflict
            and shared_count >= 3
        ):
            signals.append(
                ReferenceComparisonSignal(
                    rule_id=(
                        "MULTIPLE_SUBMISSION_FIELDS_REUSED"
                    ),

                    severity="critical",

                    score=100.0,

                    message=(
                        "Multiple strong submission-level "
                        "fields are reused across documents "
                        "with conflicting taxpayer identity."
                    ),

                    reason=(
                        "The submitted ITR shares multiple "
                        "independent filing metadata values with "
                        "another ITR while taxpayer identity "
                        "fields differ."
                    ),

                    matched_fields=matched_fields,

                    conflicting_identity_fields=(
                        conflicting_identity_fields
                    ),

                    evidence={
                        "shared_field_count": (
                            shared_count
                        ),

                        "shared_fields": (
                            list(matched_fields)
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # TWO SHARED FIELDS
        # ----------------------------------------------------

        elif (
            identity_conflict
            and shared_count == 2
        ):
            signals.append(
                ReferenceComparisonSignal(
                    rule_id=(
                        "SUBMISSION_METADATA_REUSE"
                    ),

                    severity="high",

                    score=85.0,

                    message=(
                        "Two strong submission-level fields "
                        "are shared with another document "
                        "while taxpayer identity differs."
                    ),

                    reason=(
                        "The document contains reusable filing "
                        "metadata that conflicts with its "
                        "taxpayer identity."
                    ),

                    matched_fields=matched_fields,

                    conflicting_identity_fields=(
                        conflicting_identity_fields
                    ),

                    evidence={
                        "shared_field_count": (
                            shared_count
                        ),

                        "shared_fields": (
                            list(matched_fields)
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # ONE SHARED FIELD
        # ----------------------------------------------------

        elif (
            identity_conflict
            and shared_count == 1
        ):
            signals.append(
                ReferenceComparisonSignal(
                    rule_id=(
                        "SINGLE_SUBMISSION_FIELD_REUSE"
                    ),

                    severity="medium",

                    score=50.0,

                    message=(
                        "One submission-level field is shared "
                        "with another document while taxpayer "
                        "identity differs."
                    ),

                    reason=(
                        "A single shared filing attribute is "
                        "not enough to establish document "
                        "fabrication, but it should remain "
                        "available as supporting evidence."
                    ),

                    matched_fields=matched_fields,

                    conflicting_identity_fields=(
                        conflicting_identity_fields
                    ),

                    evidence={
                        "shared_field": (
                            matched_fields[0]
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # SAME IDENTITY
        # ----------------------------------------------------

        if (
            not identity_conflict
            and shared_count > 0
            and not fingerprint_match
        ):
            signals.append(
                ReferenceComparisonSignal(
                    rule_id=(
                        "SAME_IDENTITY_SHARED_SUBMISSION"
                    ),

                    severity="info",

                    score=0.0,

                    message=(
                        "Submission metadata matches a reference "
                        "document with the same taxpayer identity."
                    ),

                    reason=(
                        "Matching filing metadata with matching "
                        "taxpayer identity does not by itself "
                        "indicate document reuse or fraud."
                    ),

                    matched_fields=matched_fields,

                    conflicting_identity_fields=(),

                    evidence={
                        "shared_field_count": (
                            shared_count
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # BUILD FINAL FLAGS
        # ----------------------------------------------------

        suspicious_reuse = any(
            signal.severity
            in {"medium", "high", "critical"}
            for signal in signals
        )

        critical_reuse = any(
            signal.severity
            == "critical"
            for signal in signals
        )

        # ----------------------------------------------------
        # REASON
        # ----------------------------------------------------

        reason = self._build_reason(
            signals
        )

        return ReferenceComparisonResult(
            matched_fields=matched_fields,

            conflicting_identity_fields=(
                conflicting_identity_fields
            ),

            shared_submission_fields=(
                matched_fields
            ),

            fingerprint_match=(
                fingerprint_match
            ),

            identity_conflict=(
                identity_conflict
            ),

            suspicious_reuse=(
                suspicious_reuse
            ),

            critical_reuse=(
                critical_reuse
            ),

            signals=tuple(
                signals
            ),

            reason=reason,
        )

    # ========================================================
    # TEXT -> SNAPSHOT
    # ========================================================

    def build_snapshot(
        self,
        text: str,
        document_id: str | None = None,
    ) -> ITRReferenceSnapshot:
        """
        Extract a reference snapshot from ITR text.

        This parser is deliberately independent from the main
        ITR extractor so the authenticity layer remains isolated.
        """

        normalized = self._normalize_text(
            text
        )

        identity = ITRReferenceIdentity(
            name=self._extract_name(
                normalized
            ),

            pan=self._extract_taxpayer_pan(
                normalized
            ),

            dob=self._extract_dob(
                normalized
            ),

            assessment_year=(
                self._extract_assessment_year(
                    normalized
                )
            ),
        )

        submission = ITRSubmissionMetadata(
            acknowledgement_number=(
                self._extract_acknowledgement(
                    normalized
                )
            ),

            filing_date=(
                self._extract_filing_date(
                    normalized
                )
            ),

            submission_timestamp=(
                self._extract_submission_timestamp(
                    normalized
                )
            ),

            ip_address=(
                self._extract_ip(
                    normalized
                )
            ),

            eid=(
                self._extract_eid(
                    normalized
                )
            ),

            verifier_pan=(
                self._extract_verifier_pan(
                    normalized
                )
            ),

            verifier_name=(
                self._extract_verifier_name(
                    normalized
                )
            ),
        )

        fingerprint = (
            self.generate_submission_fingerprint(
                submission
            )
        )

        return ITRReferenceSnapshot(
            identity=identity,

            submission=submission,

            fingerprint=fingerprint,

            document_id=document_id,
        )

    # ========================================================
    # FINGERPRINT
    # ========================================================

    @staticmethod
    def generate_submission_fingerprint(
        submission: ITRSubmissionMetadata,
    ) -> str | None:
        """
        Generate a deterministic fingerprint from submission
        metadata.

        Taxpayer identity is intentionally excluded.

        This is important.

        If a dummy document changes:

            Name
            PAN
            DOB

        but preserves:

            Acknowledgement
            Filing date
            Timestamp
            IP
            EID

        the fingerprint can remain the same.
        """

        values = {
            "acknowledgement_number": (
                submission.acknowledgement_number
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

            "eid": (
                submission.eid
            ),

            "verifier_pan": (
                submission.verifier_pan
            ),
        }

        usable = [
            f"{key}={value}"
            for key, value in values.items()
            if value
        ]

        # At least two independent fields are required.
        if len(usable) < 2:
            return None

        payload = "|".join(
            sorted(usable)
        )

        return hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

    # ========================================================
    # SUBMISSION COMPARISON
    # ========================================================

    @classmethod
    def _compare_submission_fields(
        cls,
        submitted: ITRSubmissionMetadata,
        reference: ITRSubmissionMetadata,
    ) -> tuple[str, ...]:
        """
        Return submission fields that match exactly.

        Missing values are ignored.
        """

        matched: list[str] = []

        for field_name in (
            cls.STRONG_SUBMISSION_FIELDS
        ):
            submitted_value = getattr(
                submitted,
                field_name,
            )

            reference_value = getattr(
                reference,
                field_name,
            )

            if (
                submitted_value
                and reference_value
                and cls._normalize_value(
                    submitted_value
                )
                == cls._normalize_value(
                    reference_value
                )
            ):
                matched.append(
                    field_name
                )

        return tuple(
            matched
        )

    # ========================================================
    # IDENTITY COMPARISON
    # ========================================================

    @classmethod
    def _compare_identity_fields(
        cls,
        submitted: ITRReferenceIdentity,
        reference: ITRReferenceIdentity,
    ) -> tuple[str, ...]:
        """
        Return identity fields that explicitly conflict.

        Missing values do NOT create conflicts.
        """

        conflicts: list[str] = []

        for field_name in (
            cls.IDENTITY_FIELDS
        ):
            submitted_value = getattr(
                submitted,
                field_name,
            )

            reference_value = getattr(
                reference,
                field_name,
            )

            if not submitted_value:
                continue

            if not reference_value:
                continue

            if (
                cls._normalize_value(
                    submitted_value
                )
                != cls._normalize_value(
                    reference_value
                )
            ):
                conflicts.append(
                    field_name
                )

        return tuple(
            conflicts
        )

    # ========================================================
    # NAME
    # ========================================================

    @staticmethod
    def _extract_name(
        text: str,
    ) -> str | None:
        """
        Extract taxpayer name.

        Prefer the labelled taxpayer field and avoid the
        verifier name when possible.
        """

        patterns = (
            re.compile(
                r"\bname\s+of\s+(?:the\s+)?assessee"
                r"\s*[:\-]?\s*"
                r"([A-Za-z][A-Za-z .'\-&]{2,120})",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bname\s*[:\-]\s*"
                r"([A-Za-z][A-Za-z .'\-&]{2,120})",
                re.IGNORECASE,
            ),
        )

        for pattern in patterns:

            match = pattern.search(
                text
            )

            if not match:
                continue

            value = (
                match.group(1)
                .strip()
            )

            value = re.sub(
                r"\s+",
                " ",
                value,
            )

            # Stop accidental capture of adjacent labels.
            value = re.split(
                r"\b(?:address|status|form\s+number|pan)\b",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

            if len(value) >= 3:
                return value

        return None

    # ========================================================
    # PAN
    # ========================================================

    @staticmethod
    def _extract_taxpayer_pan(
        text: str,
    ) -> str | None:
        """
        Extract taxpayer PAN.

        The verifier PAN is deliberately excluded.
        """

        lines = text.splitlines()

        for index, line in enumerate(
            lines
        ):

            clean = line.strip()

            if re.fullmatch(
                r"pan",
                clean,
                re.IGNORECASE,
            ):
                block = "\n".join(
                    lines[
                        index:index + 4
                    ]
                )

                # Remove verifier area from consideration.
                block = re.split(
                    r"verified\s+by",
                    block,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]

                match = PAN_PATTERN.search(
                    block
                )

                if match:
                    return (
                        match.group(0)
                        .upper()
                    )

            match = re.search(
                r"\bpan\b\s*[:\-]\s*"
                r"([A-Z]{5}[0-9]{4}[A-Z])\b",
                clean,
                re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1)
                    .upper()
                )

        return None

    # ========================================================
    # DOB
    # ========================================================

    @staticmethod
    def _extract_dob(
        text: str,
    ) -> str | None:
        """
        Extract labelled DOB.
        """

        match = DOB_PATTERN.search(
            text
        )

        if not match:
            return None

        return (
            ITRReferenceComparisonEngine
            ._normalize_date(
                match.group(1)
            )
        )

    # ========================================================
    # ASSESSMENT YEAR
    # ========================================================

    @staticmethod
    def _extract_assessment_year(
        text: str,
    ) -> str | None:

        match = (
            ASSESSMENT_YEAR_PATTERN.search(
                text
            )
        )

        if not match:
            return None

        value = re.sub(
            r"\s+",
            "",
            match.group(1),
        )

        value = value.replace(
            "/",
            "-",
        )

        parts = value.split(
            "-"
        )

        if len(parts) != 2:
            return None

        first = parts[0]

        second = parts[1]

        if len(second) == 4:
            second = second[-2:]

        return (
            f"{first}-{second}"
        )

    # ========================================================
    # ACKNOWLEDGEMENT
    # ========================================================

    @staticmethod
    def _extract_acknowledgement(
        text: str,
    ) -> str | None:

        for pattern in (
            ACK_PATTERN,
            EFILING_ACK_PATTERN,
        ):

            match = pattern.search(
                text
            )

            if match:
                return match.group(
                    1
                )

        return None

    # ========================================================
    # FILING DATE
    # ========================================================

    @staticmethod
    def _extract_filing_date(
        text: str,
    ) -> str | None:

        match = FILING_DATE_PATTERN.search(
            text
        )

        if not match:
            return None

        return (
            ITRReferenceComparisonEngine
            ._normalize_date(
                match.group(1)
            )
        )

    # ========================================================
    # SUBMISSION TIMESTAMP
    # ========================================================

    @staticmethod
    def _extract_submission_timestamp(
        text: str,
    ) -> str | None:

        match = (
            SUBMISSION_TIMESTAMP_PATTERN.search(
                text
            )
        )

        if not match:
            return None

        date_value = (
            ITRReferenceComparisonEngine
            ._normalize_date(
                match.group(1)
            )
        )

        time_value = match.group(
            2
        )

        if not date_value:
            return None

        return (
            f"{date_value} {time_value}"
        )

    # ========================================================
    # IP
    # ========================================================

    @staticmethod
    def _extract_ip(
        text: str,
    ) -> str | None:

        # Prefer IP following "IP address".
        labelled = re.search(
            r"IP\s+address"
            r"\s*[:\-]?\s*"
            r"("
            r"(?:\d{1,3}\.){3}\d{1,3}"
            r")",
            text,
            re.IGNORECASE,
        )

        if labelled:
            value = labelled.group(
                1
            )

            if (
                ITRReferenceComparisonEngine
                ._valid_ip(value)
            ):
                return value

        # Fallback: first valid IP.
        for match in IP_PATTERN.finditer(
            text
        ):
            value = match.group(
                0
            )

            if (
                ITRReferenceComparisonEngine
                ._valid_ip(value)
            ):
                return value

        return None

    # ========================================================
    # EID
    # ========================================================

    @staticmethod
    def _extract_eid(
        text: str,
    ) -> str | None:

        match = EID_PATTERN.search(
            text
        )

        if match:
            return (
                match.group(1)
                .upper()
            )

        # Some ITR text contains:
        #
        # using EIDKJ56GDI generated through Aadhaar
        #
        fallback = re.search(
            r"\busing\s+"
            r"([A-Z0-9]{6,40})"
            r"\s+generated\s+through",
            text,
            re.IGNORECASE,
        )

        if fallback:
            return (
                fallback.group(1)
                .upper()
            )

        return None

    # ========================================================
    # VERIFIER PAN
    # ========================================================

    @staticmethod
    def _extract_verifier_pan(
        text: str,
    ) -> str | None:

        match = (
            VERIFIER_PAN_PATTERN.search(
                text
            )
        )

        if not match:
            return None

        return (
            match.group(1)
            .upper()
        )

    # ========================================================
    # VERIFIER NAME
    # ========================================================

    @staticmethod
    def _extract_verifier_name(
        text: str,
    ) -> str | None:

        match = (
            VERIFIED_BY_PATTERN.search(
                text
            )
        )

        if not match:
            return None

        value = (
            match.group(1)
            .strip()
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value or None

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        if not text:
            return ""

        return (
            text
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

    @staticmethod
    def _normalize_value(
        value: str,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]",
            "",
            str(value).casefold(),
        )

    # ========================================================
    # DATE NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_date(
        value: str,
    ) -> str | None:

        if not value:
            return None

        value = value.strip()

        formats = (
            "%d-%b-%Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
        )

        for fmt in formats:

            try:

                parsed = datetime.strptime(
                    value,
                    fmt,
                )

                return parsed.strftime(
                    "%d/%m/%Y"
                )

            except ValueError:
                continue

        return None

    # ========================================================
    # IP VALIDATION
    # ========================================================

    @staticmethod
    def _valid_ip(
        value: str,
    ) -> bool:

        try:

            parts = value.split(
                "."
            )

            return (
                len(parts) == 4
                and all(
                    0 <= int(part) <= 255
                    for part in parts
                )
            )

        except (
            ValueError,
            AttributeError,
        ):
            return False

    # ========================================================
    # REASON
    # ========================================================

    @staticmethod
    def _build_reason(
        signals: list[
            ReferenceComparisonSignal
        ],
    ) -> str:

        if not signals:
            return (
                "No suspicious cross-document reuse "
                "evidence was detected."
            )

        critical = [
            signal
            for signal in signals
            if signal.severity
            == "critical"
        ]

        if critical:
            return (
                "CRITICAL: "
                + critical[0].reason
            )

        high = [
            signal
            for signal in signals
            if signal.severity
            == "high"
        ]

        if high:
            return (
                "HIGH RISK: "
                + high[0].reason
            )

        medium = [
            signal
            for signal in signals
            if signal.severity
            == "medium"
        ]

        if medium:
            return (
                "SUSPICIOUS: "
                + medium[0].reason
            )

        return signals[0].reason


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def compare_itr_documents(
    submitted_text: str,
    reference_text: str,
    submitted_document_id: str | None = None,
    reference_document_id: str | None = None,
) -> ReferenceComparisonResult:
    """
    Convenience wrapper for text-based comparison.
    """

    engine = (
        ITRReferenceComparisonEngine()
    )

    submitted = (
        engine.build_snapshot(
            submitted_text,
            document_id=(
                submitted_document_id
            ),
        )
    )

    reference = (
        engine.build_snapshot(
            reference_text,
            document_id=(
                reference_document_id
            ),
        )
    )

    return engine.compare(
        submitted,
        reference,
    )