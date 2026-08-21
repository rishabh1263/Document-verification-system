"""
Production data models for ITR authenticity analysis.

This module contains the canonical domain models used by the
authenticity layer.

Architecture:

    Extraction
        ↓
    AuthenticityInput
        ↓
    Authenticity rules
        ↓
    AuthenticityResult
        ↓
    API adapter

Important:

    validation != authenticity

A technically valid ITR is not automatically genuine.

A document is marked VERIFIED only when authoritative evidence
supports authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


# ============================================================
# RISK LEVEL
# ============================================================


class RiskLevel(str, Enum):
    """
    Overall authenticity risk level.
    """

    INFO = "info"

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"


# ============================================================
# AUTHENTICITY DECISION
# ============================================================


class AuthenticityDecision(str, Enum):
    """
    Final authenticity decision.

    VERIFIED
        An authoritative source confirmed the document.

    UNVERIFIED
        No sufficient evidence exists to prove authenticity.

    SUSPICIOUS
        Meaningful authenticity anomalies were detected.

    HIGH_RISK
        Strong authenticity anomalies were detected.

    REJECT
        Critical evidence indicates that automatic acceptance
        is not permitted.

    MANUAL_REVIEW
        Evidence requires human investigation.
    """

    VERIFIED = "verified"

    UNVERIFIED = "unverified"

    SUSPICIOUS = "suspicious"

    HIGH_RISK = "high_risk"

    REJECT = "reject"

    MANUAL_REVIEW = "manual_review"


# ============================================================
# FINDING
# ============================================================


@dataclass(frozen=True)
class AuthenticityFinding:
    """
    One explainable authenticity finding.

    rule_id
        Stable identifier for the rule.

    category
        Logical category such as:

            pdf_integrity
            financial_consistency
            identity
            cross_document
            metadata

    severity
        Severity assigned to the finding.

    message
        Human-readable description of what was detected.

    reason
        Explanation of why the finding matters.

    evidence
        Machine-readable supporting information.

    score
        Rule contribution from 0 to 100.
    """

    rule_id: str

    category: str

    severity: RiskLevel

    message: str

    reason: str = ""

    evidence: Mapping[str, Any] = field(
        default_factory=dict
    )

    score: float = 0.0


# ============================================================
# IDENTITY SNAPSHOT
# ============================================================


@dataclass(frozen=True)
class IdentitySnapshot:
    """
    Identity fields extracted from one ITR.
    """

    name: str | None = None

    pan: str | None = None

    dob: str | None = None

    assessment_year: str | None = None


# ============================================================
# SUBMISSION SNAPSHOT
# ============================================================


@dataclass(frozen=True)
class SubmissionSnapshot:
    """
    Submission-level metadata.

    These values are evidence, not proof.

    Their importance increases when they are compared against
    trusted/reference documents.
    """

    acknowledgement_number: str | None = None

    eid: str | None = None

    filing_date: str | None = None

    submission_timestamp: str | None = None

    ip_address: str | None = None

    verifier_name: str | None = None

    verifier_pan: str | None = None


# ============================================================
# REFERENCE COMPARISON
# ============================================================


@dataclass(frozen=True)
class ReferenceComparison:
    """
    Cross-document comparison result.

    This model deliberately keeps comparison evidence separate
    from the final authenticity decision.
    """

    matched_fields: tuple[
        str,
        ...
    ] = ()

    conflicting_identity_fields: tuple[
        str,
        ...
    ] = ()

    fingerprint_match: bool = False

    identity_conflict: bool = False

    suspicious_reuse: bool = False

    critical_reuse: bool = False

    reference_document_id: str | None = None

    submitted_document_id: str | None = None

    reason: str = ""

    evidence: Mapping[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# AUTHENTICITY INPUT
# ============================================================


@dataclass(frozen=True)
class AuthenticityInput:
    """
    Normalized input supplied to the authenticity layer.

    text
        Native/OCR extracted document text.

    document_id
        Internal identifier for traceability.

    metadata
        Additional document-level metadata.

    reference_documents
        Trusted/reference documents used for comparison.
    """

    text: str

    document_id: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    reference_documents: tuple[
        "AuthenticityInput",
        ...
    ] = ()

    # Optional normalized snapshots.
    identity: IdentitySnapshot = field(
        default_factory=IdentitySnapshot
    )

    submission: SubmissionSnapshot = field(
        default_factory=SubmissionSnapshot
    )


# ============================================================
# AUTHENTICITY RESULT
# ============================================================


@dataclass(frozen=True)
class AuthenticityResult:
    """
    Complete explainable authenticity result.
    """

    decision: AuthenticityDecision

    risk_level: RiskLevel

    confidence: float

    risk_score: float

    verified: bool

    findings: tuple[
        AuthenticityFinding,
        ...
    ] = ()

    identity: IdentitySnapshot = field(
        default_factory=IdentitySnapshot
    )

    submission: SubmissionSnapshot = field(
        default_factory=SubmissionSnapshot
    )

    reference_match: bool = False

    reference_comparison: ReferenceComparison | None = None

    summary: str = ""

    reason: str = ""

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Serialize the authenticity result.

        The output is deliberately explainable.
        """

        comparison = (
            self.reference_comparison
        )

        return {
            "decision": (
                self.decision.value
            ),

            "risk_level": (
                self.risk_level.value
            ),

            "confidence": round(
                self.confidence,
                4,
            ),

            "risk_score": round(
                self.risk_score,
                4,
            ),

            "verified": self.verified,

            "reference_match": (
                self.reference_match
            ),

            "summary": self.summary,

            "reason": self.reason,

            "identity": {
                "name": self.identity.name,

                "pan": self.identity.pan,

                "dob": self.identity.dob,

                "assessment_year": (
                    self.identity.assessment_year
                ),
            },

            "submission": {
                "acknowledgement_number": (
                    self.submission
                    .acknowledgement_number
                ),

                "eid": (
                    self.submission.eid
                ),

                "filing_date": (
                    self.submission.filing_date
                ),

                "submission_timestamp": (
                    self.submission
                    .submission_timestamp
                ),

                "ip_address": (
                    self.submission.ip_address
                ),

                "verifier_name": (
                    self.submission.verifier_name
                ),

                "verifier_pan": (
                    self.submission.verifier_pan
                ),
            },

            "reference_comparison": (
                comparison.to_dict()
                if comparison
                else None
            ),

            "findings": [
                {
                    "rule_id": item.rule_id,

                    "category": item.category,

                    "severity": (
                        item.severity.value
                    ),

                    "message": item.message,

                    "reason": item.reason,

                    "evidence": dict(
                        item.evidence
                    ),

                    "score": round(
                        item.score,
                        4,
                    ),
                }

                for item in self.findings
            ],
        }