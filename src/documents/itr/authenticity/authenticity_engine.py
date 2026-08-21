"""
Production-oriented ITR authenticity engine.

Pipeline:

    Detection
        ↓
    Extraction
        ↓
    Internal Validation
        ↓
    Authenticity
        ↓
    Final Decision

Important:

    VALID != GENUINE

A document can be structurally valid and internally consistent while
still being suspicious or unverified from an authenticity perspective.

The engine therefore produces explainable evidence rather than making
unsupported claims of government authenticity.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .models import (
    AuthenticityDecision,
    AuthenticityFinding,
    AuthenticityInput,
    AuthenticityResult,
    IdentitySnapshot,
    RiskLevel,
    SubmissionSnapshot,
)

from .rules import (
    extract_identity_occurrences,
    extract_submission_snapshot,
    rule_identity_conflicts,
    rule_submission_reuse,
)


logger = logging.getLogger(__name__)


class ITRAuthenticityEngine:
    """
    Analyze an ITR for authenticity anomalies.

    The engine does NOT claim that a document is genuine merely because
    no anomaly was detected.

    Positive verification requires trusted reference evidence.
    """

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def analyze(
        self,
        document: AuthenticityInput,
    ) -> AuthenticityResult:
        """
        Run complete authenticity analysis.
        """

        text = (
            document.text
            or ""
        ).strip()

        # ------------------------------------------------------
        # EMPTY INPUT
        # ------------------------------------------------------

        if not text:

            finding = AuthenticityFinding(
                rule_id="INPUT_EMPTY",

                category="document",

                severity=RiskLevel.HIGH,

                message=(
                    "Authenticity analysis received empty "
                    "document text."
                ),

                reason=(
                    "The authenticity engine cannot establish "
                    "identity, submission metadata, or document "
                    "consistency without readable document content."
                ),

                evidence={},

                score=100.0,
            )

            return AuthenticityResult(
                decision=(
                    AuthenticityDecision.HIGH_RISK
                ),

                risk_level=RiskLevel.HIGH,

                confidence=1.0,

                risk_score=1.0,

                verified=False,

                findings=(finding,),

                summary=(
                    "Authenticity analysis could not be performed."
                ),

                reason=(
                    "The submitted document contains no readable "
                    "text for authenticity analysis."
                ),
            )

        # ======================================================
        # 1. EXTRACT ALL IDENTITY OCCURRENCES
        # ======================================================

        occurrences = (
            extract_identity_occurrences(
                text
            )
        )

        identity = (
            self._build_identity_snapshot(
                occurrences
            )
        )

        # ======================================================
        # 2. EXTRACT SUBMISSION IDENTIFIERS
        # ======================================================

        submission = (
            extract_submission_snapshot(
                text
            )
        )

        findings: list[
            AuthenticityFinding
        ] = []

        # ======================================================
        # 3. IN-DOCUMENT IDENTITY CHECKS
        # ======================================================

        findings.extend(
            rule_identity_conflicts(
                occurrences
            )
        )

        # ======================================================
        # 4. LOAD REFERENCE DOCUMENTS
        # ======================================================

        reference_pairs: list[
            tuple[
                IdentitySnapshot,
                SubmissionSnapshot,
            ]
        ] = []

        for reference in (
            document.reference_documents
        ):

            reference_text = (
                reference.text
                or ""
            ).strip()

            if not reference_text:
                logger.warning(
                    "Skipping empty reference document: %s",
                    reference.document_id,
                )
                continue

            reference_occurrences = (
                extract_identity_occurrences(
                    reference_text
                )
            )

            reference_identity = (
                self._build_identity_snapshot(
                    reference_occurrences
                )
            )

            reference_submission = (
                extract_submission_snapshot(
                    reference_text
                )
            )

            reference_pairs.append(
                (
                    reference_identity,
                    reference_submission,
                )
            )

        # ======================================================
        # 5. CROSS-DOCUMENT AUTHENTICITY CHECK
        # ======================================================

        findings.extend(
            rule_submission_reuse(
                identity,
                submission,
                reference_pairs,
            )
        )

        # ======================================================
        # 6. PDF METADATA CHECKS
        # ======================================================

        findings.extend(
            self._analyze_metadata(
                document.metadata
            )
        )

        # ======================================================
        # 7. CALCULATE NUMERICAL RISK
        # ======================================================

        risk_score = (
            self._calculate_risk(
                findings
            )
        )

        # ======================================================
        # 8. CALCULATE CATEGORICAL RISK
        #
        # IMPORTANT:
        #
        # A CRITICAL finding must remain CRITICAL even when the
        # numerical score is below the generic threshold.
        # ======================================================

        risk_level = (
            self._risk_level(
                findings=findings,
                risk_score=risk_score,
            )
        )

        # ======================================================
        # 9. HIGH-SEVERITY FLAGS
        # ======================================================

        has_critical = any(
            finding.severity
            == RiskLevel.CRITICAL
            for finding in findings
        )

        has_high = any(
            finding.severity
            == RiskLevel.HIGH
            for finding in findings
        )

        # ======================================================
        # 10. POSITIVE REFERENCE MATCH
        # ======================================================

        reference_match = (
            self._positive_reference_match(
                identity=identity,
                submission=submission,
                references=reference_pairs,
                findings=findings,
            )
        )

        # ======================================================
        # 11. FINAL DECISION
        # ======================================================

        decision = (
            self._decision(
                risk_score=risk_score,
                has_critical=has_critical,
                has_high=has_high,
                reference_match=reference_match,
            )
        )

        verified = (
            decision
            == AuthenticityDecision.VERIFIED
        )

        # ======================================================
        # 12. HUMAN-READABLE REASON
        # ======================================================

        reason = (
            self._build_reason(
                decision=decision,
                findings=findings,
                reference_match=reference_match,
            )
        )

        summary = (
            self._build_summary(
                decision=decision,
                findings=findings,
                reference_match=reference_match,
            )
        )

        # ======================================================
        # 13. DECISION CONFIDENCE
        # ======================================================

        confidence = (
            self._confidence(
                decision=decision,
                risk_score=risk_score,
                finding_count=len(
                    findings
                ),
                reference_match=reference_match,
            )
        )

        return AuthenticityResult(
            decision=decision,

            risk_level=risk_level,

            confidence=confidence,

            risk_score=risk_score,

            verified=verified,

            findings=tuple(
                findings
            ),

            identity=identity,

            submission=submission,

            reference_match=reference_match,

            summary=summary,

            reason=reason,
        )

    # ==========================================================
    # IDENTITY SNAPSHOT
    # ==========================================================

    @staticmethod
    def _build_identity_snapshot(
        occurrences: dict[
            str,
            list[str],
        ],
    ) -> IdentitySnapshot:
        """
        Build canonical identity from the first occurrence.

        Conflict detection separately uses all occurrences.
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

    # ==========================================================
    # NUMERICAL RISK
    # ==========================================================

    @staticmethod
    def _calculate_risk(
        findings: Iterable[
            AuthenticityFinding
        ],
    ) -> float:
        """
        Convert rule scores into a bounded numerical risk score.

        The strongest finding receives full weight.

        Additional findings receive reduced weight.

        This score is useful for ranking and analytics, but it does
        NOT override explicit severity levels.
        """

        scores = sorted(
            (
                max(
                    0.0,
                    min(
                        100.0,
                        finding.score,
                    ),
                )

                for finding in findings
            ),
            reverse=True,
        )

        if not scores:
            return 0.0

        weighted_score = (
            scores[0]
            +
            sum(
                score * 0.45
                for score in scores[1:]
            )
        )

        return min(
            1.0,
            weighted_score / 100.0,
        )

    # ==========================================================
    # CATEGORICAL RISK
    # ==========================================================

    @staticmethod
    def _risk_level(
        findings: Iterable[
            AuthenticityFinding
        ],

        risk_score: float,
    ) -> RiskLevel:
        """
        Convert findings into the final categorical risk.

        Severity has priority over numerical score.

        This is critical for production behavior.

        Example:

            finding severity = CRITICAL
            numerical score  = 0.59

        Result:

            CRITICAL

        NOT:

            HIGH
        """

        severities = [
            finding.severity
            for finding in findings
        ]

        # ------------------------------------------------------
        # CRITICAL ALWAYS WINS
        # ------------------------------------------------------

        if RiskLevel.CRITICAL in severities:
            return RiskLevel.CRITICAL

        # ------------------------------------------------------
        # HIGH FINDING
        # ------------------------------------------------------

        if RiskLevel.HIGH in severities:
            return RiskLevel.HIGH

        # ------------------------------------------------------
        # MEDIUM FINDING
        # ------------------------------------------------------

        if RiskLevel.MEDIUM in severities:
            return RiskLevel.MEDIUM

        # ------------------------------------------------------
        # LOW FINDING
        # ------------------------------------------------------

        if RiskLevel.LOW in severities:
            return RiskLevel.LOW

        # ------------------------------------------------------
        # NO EXPLICIT FINDING
        #
        # Numerical score is now used only as a fallback.
        # ------------------------------------------------------

        if risk_score >= 0.75:
            return RiskLevel.CRITICAL

        if risk_score >= 0.50:
            return RiskLevel.HIGH

        if risk_score >= 0.25:
            return RiskLevel.MEDIUM

        if risk_score > 0.0:
            return RiskLevel.LOW

        return RiskLevel.INFO

    # ==========================================================
    # FINAL DECISION
    # ==========================================================

    @staticmethod
    def _decision(
        risk_score: float,

        has_critical: bool,

        has_high: bool,

        reference_match: bool,
    ) -> AuthenticityDecision:
        """
        Produce final authenticity decision.

        Critical/high authenticity evidence takes precedence over
        positive reference evidence.
        """

        if has_critical:
            return (
                AuthenticityDecision.HIGH_RISK
            )

        if risk_score >= 0.70:
            return (
                AuthenticityDecision.HIGH_RISK
            )

        if has_high:
            return (
                AuthenticityDecision.HIGH_RISK
            )

        if reference_match:
            return (
                AuthenticityDecision.VERIFIED
            )

        if risk_score >= 0.30:
            return (
                AuthenticityDecision.SUSPICIOUS
            )

        return (
            AuthenticityDecision.UNVERIFIED
        )

    # ==========================================================
    # POSITIVE REFERENCE MATCH
    # ==========================================================

    @staticmethod
    def _positive_reference_match(
        identity: IdentitySnapshot,

        submission: SubmissionSnapshot,

        references: list[
            tuple[
                IdentitySnapshot,
                SubmissionSnapshot,
            ]
        ],

        findings: list[
            AuthenticityFinding
        ],
    ) -> bool:
        """
        Determine whether positive trusted reference evidence exists.

        Current V1 requirement:

            taxpayer PAN matches
            AND
            acknowledgement number matches
        """

        if not references:
            return False

        for (
            reference_identity,
            reference_submission,
        ) in references:

            identity_matches = (
                bool(
                    identity.pan
                )
                and
                bool(
                    reference_identity.pan
                )
                and
                identity.pan.upper()
                ==
                reference_identity.pan.upper()
            )

            acknowledgement_matches = (
                bool(
                    submission.acknowledgement_number
                )
                and
                bool(
                    reference_submission
                    .acknowledgement_number
                )
                and
                submission.acknowledgement_number
                ==
                reference_submission
                .acknowledgement_number
            )

            if (
                identity_matches
                and
                acknowledgement_matches
            ):
                return True

        return False

    # ==========================================================
    # TOP-LEVEL REASON
    # ==========================================================

    @staticmethod
    def _build_reason(
        decision: AuthenticityDecision,

        findings: list[
            AuthenticityFinding
        ],

        reference_match: bool,
    ) -> str:
        """
        Build an explainable top-level API reason.
        """

        if (
            decision
            == AuthenticityDecision.HIGH_RISK
        ):

            severe_findings = [
                finding
                for finding in findings
                if finding.severity
                in {
                    RiskLevel.CRITICAL,
                    RiskLevel.HIGH,
                }
            ]

            if severe_findings:

                primary = max(
                    severe_findings,
                    key=lambda item: (
                        item.severity
                        == RiskLevel.CRITICAL,
                        item.score,
                    ),
                )

                return (
                    primary.reason
                    or primary.message
                )

            return (
                "Strong authenticity anomalies were "
                "detected in the submitted ITR."
            )

        if (
            decision
            == AuthenticityDecision.SUSPICIOUS
        ):

            if findings:

                primary = max(
                    findings,
                    key=lambda item: item.score,
                )

                return (
                    primary.reason
                    or primary.message
                )

            return (
                "Authenticity anomalies were detected "
                "and further verification is required."
            )

        if (
            decision
            == AuthenticityDecision.VERIFIED
        ):

            return (
                "The taxpayer identity and stable "
                "submission identifiers matched the "
                "available reference evidence, with no "
                "high-risk authenticity conflicts detected."
            )

        return (
            "No strong authenticity anomaly was detected, "
            "but no trusted reference evidence was available "
            "to prove that the document is genuine."
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    @staticmethod
    def _build_summary(
        decision: AuthenticityDecision,

        findings: list[
            AuthenticityFinding
        ],

        reference_match: bool,
    ) -> str:
        """
        Build short operational summary.
        """

        if (
            decision
            == AuthenticityDecision.HIGH_RISK
        ):

            return (
                "High-risk authenticity anomaly detected."
            )

        if (
            decision
            == AuthenticityDecision.SUSPICIOUS
        ):

            return (
                "Authenticity anomaly detected; "
                "manual or external verification recommended."
            )

        if (
            decision
            == AuthenticityDecision.VERIFIED
        ):

            return (
                "Authenticity supported by reference evidence."
            )

        return (
            "Authenticity remains unverified."
        )

    # ==========================================================
    # CONFIDENCE
    # ==========================================================

    @staticmethod
    def _confidence(
        decision: AuthenticityDecision,

        risk_score: float,

        finding_count: int,

        reference_match: bool,
    ) -> float:
        """
        Estimate confidence in the rule-based classification.

        This is NOT a probability that the document is fraudulent.
        """

        if (
            decision
            == AuthenticityDecision.VERIFIED
        ):

            if reference_match:
                return 0.95

            return 0.75

        if (
            decision
            == AuthenticityDecision.HIGH_RISK
        ):

            base = (
                0.85
                +
                min(
                    0.10,
                    finding_count * 0.02,
                )
            )

            return min(
                0.99,
                base,
            )

        if (
            decision
            == AuthenticityDecision.SUSPICIOUS
        ):

            return min(
                0.90,
                0.60
                +
                (
                    risk_score
                    * 0.25
                ),
            )

        return 0.55

    # ==========================================================
    # PDF METADATA
    # ==========================================================

    @staticmethod
    def _analyze_metadata(
        metadata: object,
    ) -> list[
        AuthenticityFinding
    ]:
        """
        Inspect supplied PDF metadata.

        Editor metadata is only a signal, never standalone proof
        of forgery.
        """

        if not isinstance(
            metadata,
            dict,
        ):
            return []

        findings: list[
            AuthenticityFinding
        ] = []

        suspicious_creator_terms = (
            "photoshop",
            "gimp",
            "canva",
            "illustrator",
            "paint",
        )

        creator = str(
            metadata.get(
                "creator",
                "",
            )
        ).casefold()

        producer = str(
            metadata.get(
                "producer",
                "",
            )
        ).casefold()

        matched = [
            term
            for term
            in suspicious_creator_terms
            if (
                term in creator
                or
                term in producer
            )
        ]

        if matched:

            findings.append(
                AuthenticityFinding(
                    rule_id=(
                        "PDF_EDITOR_METADATA"
                    ),

                    category="pdf_metadata",

                    severity=RiskLevel.MEDIUM,

                    message=(
                        "PDF metadata identifies "
                        "an image/design editor."
                    ),

                    reason=(
                        "The PDF creator or producer metadata "
                        "contains an application commonly used "
                        "to edit documents. This is a tampering "
                        "signal, but it is not standalone proof "
                        "that the ITR was forged."
                    ),

                    evidence={
                        "matched_editors": matched,

                        "creator": metadata.get(
                            "creator"
                        ),

                        "producer": metadata.get(
                            "producer"
                        ),
                    },

                    score=30.0,
                )
            )

        return findings


# ==============================================================
# CONVENIENCE FUNCTION
# ==============================================================


def analyze_itr_authenticity(
    text: str,

    *,
    document_id: str | None = None,

    metadata: dict | None = None,

    reference_documents: tuple[
        AuthenticityInput,
        ...
    ] = (),
) -> AuthenticityResult:
    """
    Convenience wrapper for API integration.
    """

    return ITRAuthenticityEngine().analyze(
        AuthenticityInput(
            text=text,

            document_id=document_id,

            metadata=metadata or {},

            reference_documents=(
                reference_documents
            ),
        )
    )