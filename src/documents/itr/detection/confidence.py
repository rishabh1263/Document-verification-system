"""
==============================================================
ITR Detection Confidence Engine
==============================================================

Purpose
-------
Combine independent ITR detection evidence:

    Keyword
    Layout
    Structure

Metadata is treated as document-quality evidence and is
NOT directly used as ITR identity evidence.

Final identity score:

    Keyword    = 45%
    Layout     = 30%
    Structure  = 25%

Detection threshold is read from the ITR constants module.

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import logging

from ..constants import MIN_DETECTION_CONFIDENCE
from ..models import (
    DetectionEvidence,
    DetectionResult,
    DetectionStatus,
    DocumentType,
)


logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """
    Calculate final ITR detection confidence.
    """

    # ==========================================================
    # IDENTITY WEIGHTS
    # ==========================================================

    KEYWORD_WEIGHT = 0.45

    LAYOUT_WEIGHT = 0.30

    STRUCTURE_WEIGHT = 0.25

    # ==========================================================
    # THRESHOLD
    # ==========================================================

    DETECTION_THRESHOLD = (
        MIN_DETECTION_CONFIDENCE
    )

    # ==========================================================
    # PUBLIC
    # ==========================================================

    def calculate(
        self,
        evidence: DetectionEvidence,
    ) -> DetectionResult:
        """
        Calculate final detection result from all evidence.
        """

        # ------------------------------------------------------
        # Individual evidence scores
        # ------------------------------------------------------

        keyword_score = self._clamp(
            evidence.keyword.score
        )

        layout_score = self._clamp(
            evidence.layout.score
        )

        structure_score = self._clamp(
            evidence.structure.score
        )

        metadata_score = self._clamp(
            evidence.metadata.score
        )

        # ------------------------------------------------------
        # Contradictory evidence
        # ------------------------------------------------------

        contradictory = (
            self._has_strong_contradictory_evidence(
                evidence
            )
        )

        # ------------------------------------------------------
        # Final identity score
        # ------------------------------------------------------

        identity_score = (
            self._calculate_identity_score(
                keyword_score=keyword_score,
                layout_score=layout_score,
                structure_score=structure_score,
            )
        )

        # ------------------------------------------------------
        # Detection decision
        # ------------------------------------------------------

        detected = (
            identity_score
            >= self.DETECTION_THRESHOLD
            and not contradictory
        )

        # ------------------------------------------------------
        # Status
        # ------------------------------------------------------

        status = (
            DetectionStatus.SUCCESS
            if detected
            else DetectionStatus.FAILED
        )

        # ------------------------------------------------------
        # Document type
        # ------------------------------------------------------

        document_type = (
            DocumentType.ITR
            if detected
            else DocumentType.UNKNOWN
        )

        # ------------------------------------------------------
        # Final result
        # ------------------------------------------------------

        result = DetectionResult(
            status=status,
            detected=detected,
            document_type=document_type,
            mode=evidence.metadata.mode,
            confidence=round(
                identity_score,
                3,
            ),
            page_count=(
                evidence.metadata.page_count
            ),
            evidence=evidence,
        )

        # ------------------------------------------------------
        # Reasons
        # ------------------------------------------------------

        self._add_reasons(
            result=result,
            metadata_score=metadata_score,
            keyword_score=keyword_score,
            layout_score=layout_score,
            structure_score=structure_score,
            identity_score=identity_score,
            contradictory=contradictory,
        )

        return result

    # ==========================================================
    # IDENTITY SCORE
    # ==========================================================

    @classmethod
    def _calculate_identity_score(
        cls,
        keyword_score: float,
        layout_score: float,
        structure_score: float,
    ) -> float:
        """
        Calculate weighted ITR identity score.

        Formula:

            keyword    * 0.45
          + layout     * 0.30
          + structure  * 0.25
        """

        score = (
            keyword_score
            * cls.KEYWORD_WEIGHT
        )

        score += (
            layout_score
            * cls.LAYOUT_WEIGHT
        )

        score += (
            structure_score
            * cls.STRUCTURE_WEIGHT
        )

        return cls._clamp(
            score
        )

    # ==========================================================
    # CONTRADICTORY EVIDENCE
    # ==========================================================

    @staticmethod
    def _has_strong_contradictory_evidence(
        evidence: DetectionEvidence,
    ) -> bool:
        """
        Determine whether strong negative evidence exists.

        Current rule:

        If no positive ITR keyword evidence exists but
        contradictory keywords are present, reject the document.
        """

        keyword = evidence.keyword

        if (
            keyword.score <= 0.0
            and
            keyword.total_negative_score > 0
        ):
            return True

        return False

    # ==========================================================
    # REASONS
    # ==========================================================

    def _add_reasons(
        self,
        result: DetectionResult,
        metadata_score: float,
        keyword_score: float,
        layout_score: float,
        structure_score: float,
        identity_score: float,
        contradictory: bool,
    ) -> None:
        """
        Add human-readable scoring reasons.
        """

        reasons = (
            result.evidence.keyword.reasons
        )

        # ------------------------------------------------------
        # Metadata
        # ------------------------------------------------------

        reasons.append(
            f"Metadata quality score: "
            f"{metadata_score:.3f}"
        )

        # ------------------------------------------------------
        # Keyword
        # ------------------------------------------------------

        reasons.append(
            f"Keyword identity score: "
            f"{keyword_score:.3f}"
        )

        # ------------------------------------------------------
        # Layout
        # ------------------------------------------------------

        reasons.append(
            f"Layout identity score: "
            f"{layout_score:.3f}"
        )

        # ------------------------------------------------------
        # Structure
        # ------------------------------------------------------

        reasons.append(
            f"Structure identity score: "
            f"{structure_score:.3f}"
        )

        # ------------------------------------------------------
        # Final
        # ------------------------------------------------------

        reasons.append(
            f"ITR identity score: "
            f"{identity_score:.3f}"
        )

        # ------------------------------------------------------
        # Contradiction
        # ------------------------------------------------------

        if contradictory:

            reasons.append(
                "Strong contradictory evidence detected"
            )

        # ------------------------------------------------------
        # Detection
        # ------------------------------------------------------

        elif result.detected:

            reasons.append(
                "ITR detection threshold satisfied"
            )

        else:

            reasons.append(
                "ITR detection threshold not satisfied"
            )

    # ==========================================================
    # CLAMP
    # ==========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """
        Clamp score to [0, 1].
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )