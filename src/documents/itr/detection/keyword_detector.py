"""
==============================================================
ITR Keyword Detector
==============================================================

Purpose
-------
Detect ITR-related evidence from extracted document text.

Detection only:
    - No PAN extraction
    - No name extraction
    - No DOB extraction
    - No field validation
    - No tampering validation

Features
--------
- Primary keyword matching
- Secondary keyword matching
- Negative evidence
- Exact matching
- OCR-tolerant fuzzy matching
- Weighted evidence
- Position tracking
- Duplicate protection
- Explainable scoring

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import logging
from typing import Tuple

from rapidfuzz import fuzz

from ..config import CONFIG
from ..models import (
    KeywordMatch,
    KeywordResult,
    KeywordType,
    NegativeKeywordMatch,
)
from .utils import normalize_text


logger = logging.getLogger(__name__)


class KeywordDetector:
    """
    Detect ITR using weighted textual evidence.
    """

    # ==========================================================
    # FUZZY MATCHING
    # ==========================================================

    FUZZY_THRESHOLD = 90

    # ==========================================================
    # EVIDENCE WEIGHTS
    # ==========================================================

    PRIMARY_SCORE_CAP = 0.85

    SECONDARY_SCORE_CAP = 0.15

    NEGATIVE_PENALTY_CAP = 0.15

    # ==========================================================
    # PUBLIC METHOD
    # ==========================================================

    def analyze(self, text: str) -> KeywordResult:
        """
        Analyze extracted document text.

        Returns
        -------
        KeywordResult
            Keyword evidence and confidence score.
        """

        result = KeywordResult()

        # ------------------------------------------------------
        # Empty input
        # ------------------------------------------------------

        if not text or not text.strip():

            result.reasons.append(
                "No text available for keyword detection"
            )

            return result

        # ------------------------------------------------------
        # Normalize text
        # ------------------------------------------------------

        normalized_text = normalize_text(text)

        if not normalized_text:

            result.reasons.append(
                "Text became empty after normalization"
            )

            return result

        # ------------------------------------------------------
        # Tracking
        # ------------------------------------------------------

        matched_keywords: set[str] = set()

        positive_score = 0

        negative_score = 0

        # ======================================================
        # PRIMARY KEYWORDS
        # ======================================================

        for keyword, weight in CONFIG.primary_keywords.items():

            found, position = self._match_keyword(
                keyword=keyword,
                text=normalized_text,
            )

            if not found:
                continue

            normalized_keyword = (
                keyword.lower().strip()
            )

            if normalized_keyword in matched_keywords:
                continue

            matched_keywords.add(normalized_keyword)

            match = KeywordMatch(
                keyword=keyword,
                weight=int(weight),
                position=position,
                type=KeywordType.PRIMARY,
            )

            result.matched_keywords.append(match)

            positive_score += int(weight)

        # ======================================================
        # SECONDARY KEYWORDS
        # ======================================================

        for keyword, weight in CONFIG.secondary_keywords.items():

            found, position = self._match_keyword(
                keyword=keyword,
                text=normalized_text,
            )

            if not found:
                continue

            normalized_keyword = (
                keyword.lower().strip()
            )

            if normalized_keyword in matched_keywords:
                continue

            matched_keywords.add(normalized_keyword)

            match = KeywordMatch(
                keyword=keyword,
                weight=int(weight),
                position=position,
                type=KeywordType.SECONDARY,
            )

            result.matched_keywords.append(match)

            positive_score += int(weight)

        # ======================================================
        # NEGATIVE KEYWORDS
        # ======================================================

        for keyword, penalty in CONFIG.negative_keywords.items():

            found, position = self._match_keyword(
                keyword=keyword,
                text=normalized_text,
            )

            if not found:
                continue

            penalty_value = abs(int(penalty))

            negative_match = NegativeKeywordMatch(
                keyword=keyword,
                penalty=penalty_value,
                position=position,
            )

            result.negative_keywords.append(
                negative_match
            )

            negative_score += penalty_value

        # ======================================================
        # SCORE BREAKDOWN
        # ======================================================

        result.total_positive_score = positive_score

        result.total_negative_score = negative_score

        result.total_keywords_found = len(
            result.matched_keywords
        )

        # ======================================================
        # CALCULATE EVIDENCE SCORES
        # ======================================================

        primary_score = (
            self._calculate_primary_score(result)
        )

        secondary_score = (
            self._calculate_secondary_score(result)
        )

        negative_penalty = (
            self._calculate_negative_penalty(result)
        )

        # ======================================================
        # FINAL KEYWORD SCORE
        # ======================================================

        final_score = (
            primary_score
            + secondary_score
            - negative_penalty
        )

        final_score = max(
            0.0,
            min(1.0, final_score)
        )

        result.score = round(
            final_score,
            3,
        )

        # ======================================================
        # REASONS
        # ======================================================

        self._build_reasons(
            result=result,
            primary_score=primary_score,
            secondary_score=secondary_score,
            negative_penalty=negative_penalty,
        )

        return result

    # ==========================================================
    # PRIMARY SCORE
    # ==========================================================

    def _calculate_primary_score(
        self,
        result: KeywordResult,
    ) -> float:
        """
        Calculate primary evidence.

        Primary keywords represent the strongest evidence.

        The score is based on matched weighted evidence,
        but uses saturation so that a document does not need
        every possible keyword to achieve strong confidence.
        """

        primary_matches = [
            item
            for item in result.matched_keywords
            if item.type == KeywordType.PRIMARY
        ]

        if not primary_matches:
            return 0.0

        matched_weight = sum(
            item.weight
            for item in primary_matches
        )

        maximum_weight = sum(
            int(weight)
            for weight in CONFIG.primary_keywords.values()
        )

        if maximum_weight <= 0:
            return 0.0

        ratio = (
            matched_weight
            / maximum_weight
        )

        ratio = min(
            ratio,
            1.0,
        )

        # Evidence saturation.
        #
        # 4 strong primary matches should already represent
        # very strong evidence even if the configuration later
        # grows with more optional keywords.

        if len(primary_matches) >= 4:

            ratio = max(
                ratio,
                0.90,
            )

        elif len(primary_matches) >= 3:

            ratio = max(
                ratio,
                0.75,
            )

        elif len(primary_matches) >= 2:

            ratio = max(
                ratio,
                0.50,
            )

        elif len(primary_matches) >= 1:

            ratio = max(
                ratio,
                0.20,
            )

        return min(
            ratio * self.PRIMARY_SCORE_CAP,
            self.PRIMARY_SCORE_CAP,
        )

    # ==========================================================
    # SECONDARY SCORE
    # ==========================================================

    def _calculate_secondary_score(
        self,
        result: KeywordResult,
    ) -> float:
        """
        Calculate supporting secondary evidence.

        Secondary keywords support primary evidence but should
        never dominate the detection decision.
        """

        secondary_matches = [
            item
            for item in result.matched_keywords
            if item.type == KeywordType.SECONDARY
        ]

        if not secondary_matches:
            return 0.0

        matched_weight = sum(
            item.weight
            for item in secondary_matches
        )

        maximum_weight = sum(
            int(weight)
            for weight in CONFIG.secondary_keywords.values()
        )

        if maximum_weight <= 0:
            return 0.0

        ratio = (
            matched_weight
            / maximum_weight
        )

        ratio = min(
            ratio,
            1.0,
        )

        return min(
            ratio * self.SECONDARY_SCORE_CAP,
            self.SECONDARY_SCORE_CAP,
        )

    # ==========================================================
    # NEGATIVE PENALTY
    # ==========================================================

    def _calculate_negative_penalty(
        self,
        result: KeywordResult,
    ) -> float:
        """
        Calculate contradictory evidence penalty.

        Negative keywords should represent evidence against ITR,
        not merely information that can legitimately appear in
        an ITR document.

        Penalty is capped so one keyword cannot destroy strong
        positive evidence.
        """

        if not result.negative_keywords:
            return 0.0

        total_penalty = sum(
            item.penalty
            for item in result.negative_keywords
        )

        maximum_positive_weight = (
            sum(
                int(weight)
                for weight in CONFIG.primary_keywords.values()
            )
            +
            sum(
                int(weight)
                for weight in CONFIG.secondary_keywords.values()
            )
        )

        if maximum_positive_weight <= 0:
            return 0.0

        penalty_ratio = (
            total_penalty
            / maximum_positive_weight
        )

        penalty = (
            penalty_ratio
            * self.NEGATIVE_PENALTY_CAP
        )

        return min(
            penalty,
            self.NEGATIVE_PENALTY_CAP,
        )

    # ==========================================================
    # REASONS
    # ==========================================================

    @staticmethod
    def _build_reasons(
        result: KeywordResult,
        primary_score: float,
        secondary_score: float,
        negative_penalty: float,
    ) -> None:
        """
        Generate explainable detection reasons.
        """

        primary_count = sum(
            1
            for item in result.matched_keywords
            if item.type == KeywordType.PRIMARY
        )

        secondary_count = sum(
            1
            for item in result.matched_keywords
            if item.type == KeywordType.SECONDARY
        )

        # ------------------------------------------------------
        # Positive evidence
        # ------------------------------------------------------

        if result.matched_keywords:

            result.reasons.append(
                f"{result.total_keywords_found} keywords matched"
            )

            result.reasons.append(
                f"{primary_count} primary keywords matched"
            )

            result.reasons.append(
                f"{secondary_count} secondary keywords matched"
            )

            result.reasons.append(
                f"Primary evidence score: "
                f"{primary_score:.3f}"
            )

            result.reasons.append(
                f"Secondary evidence score: "
                f"{secondary_score:.3f}"
            )

        else:

            result.reasons.append(
                "No ITR keywords matched"
            )

        # ------------------------------------------------------
        # Negative evidence
        # ------------------------------------------------------

        if result.negative_keywords:

            result.reasons.append(
                f"{len(result.negative_keywords)} "
                f"negative keywords matched"
            )

            result.reasons.append(
                f"Negative evidence penalty: "
                f"{negative_penalty:.3f}"
            )

        else:

            result.reasons.append(
                "No contradictory keywords matched"
            )

        # ------------------------------------------------------
        # Evidence classification
        # ------------------------------------------------------

        if result.score >= 0.85:

            result.reasons.append(
                "Strong ITR keyword evidence"
            )

        elif result.score >= 0.65:

            result.reasons.append(
                "Good ITR keyword evidence"
            )

        elif result.score >= 0.45:

            result.reasons.append(
                "Moderate ITR keyword evidence"
            )

        elif result.score > 0:

            result.reasons.append(
                "Weak ITR keyword evidence"
            )

        else:

            result.reasons.append(
                "No meaningful ITR keyword evidence"
            )

    # ==========================================================
    # KEYWORD MATCHING
    # ==========================================================

    def _match_keyword(
        self,
        keyword: str,
        text: str,
    ) -> Tuple[bool, int]:
        """
        Match keyword using:

        1. Exact phrase matching
        2. Fuzzy phrase matching

        Returns
        -------
        Tuple[bool, int]
            found, character position
        """

        if not keyword or not text:

            return False, -1

        keyword_normalized = normalize_text(
            keyword
        )

        if not keyword_normalized:

            return False, -1

        # ======================================================
        # EXACT MATCH
        # ======================================================

        exact_position = text.find(
            keyword_normalized
        )

        if exact_position != -1:

            return True, exact_position

        # ======================================================
        # FUZZY MATCH
        # ======================================================

        keyword_words = (
            keyword_normalized.split()
        )

        text_words = text.split()

        keyword_length = len(
            keyword_words
        )

        if not keyword_length:

            return False, -1

        if len(text_words) < keyword_length:

            return False, -1

        for index in range(
            len(text_words)
            - keyword_length
            + 1
        ):

            candidate = " ".join(
                text_words[
                    index:index + keyword_length
                ]
            )

            similarity = fuzz.ratio(
                keyword_normalized,
                candidate,
            )

            if similarity >= self.FUZZY_THRESHOLD:

                position = text.find(
                    candidate
                )

                return True, position

        return False, -1