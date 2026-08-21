"""
==============================================================
ITR Layout Detector
==============================================================

Purpose
-------
Detect ITR-like structural/layout evidence from PDF text.

This detector does NOT perform:
    - PAN extraction
    - Name extraction
    - DOB extraction
    - Validation
    - Tampering detection

It looks for relationships between ITR-related sections rather
than relying only on individual keywords.

Detection evidence:
    1. Taxpayer / identity section
    2. Assessment Year
    3. Income section
    4. Deduction section
    5. Tax computation
    6. Refund / tax payable
    7. Verification / acknowledgement

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..models import LayoutResult


class LayoutDetector:
    """
    Detect structural evidence indicating an ITR document.
    """

    # ==========================================================
    # SECTION PATTERNS
    # ==========================================================

    SECTION_PATTERNS: Dict[str, Tuple[str, ...]] = {

        "assessment": (
            r"\bassessment\s+year\b",
            r"\ba\.?y\.?\s*[:\-]?\s*20\d{2}\s*[-/]\s*20\d{2}\b",
        ),

        "taxpayer": (
            r"\bpan\b",
            r"\bpermanent\s+account\s+number\b",
            r"\bname\s+of\s+(?:the\s+)?assessee\b",
            r"\bassessee\b",
        ),

        "income": (
            r"\bgross\s+total\s+income\b",
            r"\btotal\s+income\b",
            r"\bincome\s+from\s+salary\b",
            r"\bincome\s+from\s+house\s+property\b",
            r"\bincome\s+from\s+business\b",
            r"\bincome\s+from\s+other\s+sources\b",
        ),

        "deductions": (
            r"\bdeductions?\b",
            r"\bchapter\s+vi[-\s]?a\b",
            r"\b80c\b",
            r"\b80d\b",
            r"\b80tta\b",
        ),

        "tax_computation": (
            r"\btax\s+payable\b",
            r"\btotal\s+tax\b",
            r"\btax\s+paid\b",
            r"\bnet\s+tax\s+liability\b",
            r"\btax\s+computation\b",
        ),

        "refund": (
            r"\brefund\b",
            r"\brefund\s+due\b",
            r"\btax\s+refund\b",
        ),

        "verification": (
            r"\bverification\b",
            r"\bverified\b",
            r"\bverification\s+under\b",
            r"\btrue\s+and\s+correct\b",
        ),

        "acknowledgement": (
            r"\backnowledg(?:e?ment|ement)\b",
            r"\backnowledgement\s+number\b",
            r"\backnowledgment\s+number\b",
        ),

        "itr_identity": (
            r"\bincome\s+tax\s+return\b",
            r"\bincome\s+tax\s+department\b",
            r"\bdepartment\s+of\s+income\s+tax\b",
            r"\bform\s+itr[-\s]?[1-7]\b",
        ),
    }

    # ==========================================================
    # SECTION WEIGHTS
    # ==========================================================

    SECTION_WEIGHTS: Dict[str, float] = {

        "assessment": 0.20,

        "taxpayer": 0.15,

        "income": 0.15,

        "deductions": 0.10,

        "tax_computation": 0.15,

        "refund": 0.05,

        "verification": 0.05,

        "acknowledgement": 0.05,

        "itr_identity": 0.10,
    }

    # ==========================================================
    # SECTION ORDER
    # ==========================================================

    EXPECTED_ORDER = (
        "assessment",
        "taxpayer",
        "income",
        "deductions",
        "tax_computation",
        "refund",
        "verification",
    )

    # ==========================================================
    # PUBLIC
    # ==========================================================

    def analyze(
        self,
        text: str,
    ) -> LayoutResult:
        """
        Analyze text for ITR structural evidence.

        Parameters
        ----------
        text:
            Extracted document text.

        Returns
        -------
        LayoutResult
        """

        if not text or not text.strip():

            return LayoutResult(
                score=0.0,
                detected_sections=[],
                reasons=[
                    "No text available for layout detection"
                ],
            )

        normalized_text = self._normalize_text(
            text
        )

        # ------------------------------------------------------
        # Detect sections
        # ------------------------------------------------------

        section_positions = self._detect_sections(
            normalized_text
        )

        detected_sections = list(
            section_positions.keys()
        )

        # ------------------------------------------------------
        # No sections
        # ------------------------------------------------------

        if not detected_sections:

            return LayoutResult(
                score=0.0,
                detected_sections=[],
                reasons=[
                    "No ITR structural sections detected"
                ],
            )

        # ------------------------------------------------------
        # Evidence score
        # ------------------------------------------------------

        base_score = self._calculate_section_score(
            detected_sections
        )

        # ------------------------------------------------------
        # Relationship evidence
        # ------------------------------------------------------

        relationship_score = (
            self._calculate_relationship_score(
                section_positions
            )
        )

        # ------------------------------------------------------
        # Order evidence
        # ------------------------------------------------------

        order_score = (
            self._calculate_order_score(
                section_positions
            )
        )

        # ------------------------------------------------------
        # Final layout score
        # ------------------------------------------------------

        score = (
            (base_score * 0.60)
            +
            (relationship_score * 0.25)
            +
            (order_score * 0.15)
        )

        score = self._clamp(
            score
        )

        # ------------------------------------------------------
        # Reasons
        # ------------------------------------------------------

        reasons: List[str] = []

        reasons.append(
            f"{len(detected_sections)} ITR structural sections detected"
        )

        if relationship_score > 0:

            reasons.append(
                f"Section relationship score: "
                f"{relationship_score:.3f}"
            )

        if order_score > 0:

            reasons.append(
                f"Section order score: "
                f"{order_score:.3f}"
            )

        if score >= 0.70:

            reasons.append(
                "Strong ITR layout evidence"
            )

        elif score >= 0.50:

            reasons.append(
                "Moderate ITR layout evidence"
            )

        elif score > 0:

            reasons.append(
                "Weak ITR layout evidence"
            )

        return LayoutResult(
            score=round(
                score,
                3,
            ),
            detected_sections=detected_sections,
            reasons=reasons,
        )

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize PDF text for robust pattern matching.
        """

        text = text.lower()

        # ------------------------------------------------------
        # Normalize common PDF whitespace
        # ------------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n+",
            "\n",
            text,
        )

        return text

    # ==========================================================
    # SECTION DETECTION
    # ==========================================================

    def _detect_sections(
        self,
        text: str,
    ) -> Dict[str, int]:
        """
        Detect each structural section and store its first
        position in the document.
        """

        detected: Dict[str, int] = {}

        for section, patterns in self.SECTION_PATTERNS.items():

            positions: List[int] = []

            for pattern in patterns:

                match = re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )

                if match:

                    positions.append(
                        match.start()
                    )

            if positions:

                detected[section] = min(
                    positions
                )

        return detected

    # ==========================================================
    # SECTION SCORE
    # ==========================================================

    def _calculate_section_score(
        self,
        sections: List[str],
    ) -> float:
        """
        Calculate weighted section coverage.
        """

        if not sections:

            return 0.0

        total_weight = sum(
            self.SECTION_WEIGHTS.values()
        )

        matched_weight = sum(
            self.SECTION_WEIGHTS.get(
                section,
                0.0,
            )
            for section in sections
        )

        if total_weight <= 0:

            return 0.0

        return self._clamp(
            matched_weight
            / total_weight
        )

    # ==========================================================
    # RELATIONSHIP SCORE
    # ==========================================================

    @staticmethod
    def _calculate_relationship_score(
        positions: Dict[str, int],
    ) -> float:
        """
        Measure whether meaningful ITR sections coexist.

        Strong relationships:
            assessment + taxpayer
            taxpayer + income
            income + tax computation
            tax computation + verification
        """

        relationships = (

            (
                "assessment",
                "taxpayer",
            ),

            (
                "taxpayer",
                "income",
            ),

            (
                "income",
                "tax_computation",
            ),

            (
                "tax_computation",
                "verification",
            ),

            (
                "assessment",
                "itr_identity",
            ),
        )

        if not positions:

            return 0.0

        matched = 0

        for first, second in relationships:

            if (
                first in positions
                and second in positions
            ):

                matched += 1

        return min(
            1.0,
            matched
            / len(relationships),
        )

    # ==========================================================
    # ORDER SCORE
    # ==========================================================

    def _calculate_order_score(
        self,
        positions: Dict[str, int],
    ) -> float:
        """
        Check whether detected sections broadly follow
        expected ITR ordering.

        This is intentionally tolerant because different ITR
        forms can have different layouts.
        """

        available = [
            section
            for section in self.EXPECTED_ORDER
            if section in positions
        ]

        if len(available) < 2:

            return 0.0

        correct_pairs = 0

        total_pairs = 0

        for index in range(
            len(available) - 1
        ):

            first = available[index]

            second = available[index + 1]

            total_pairs += 1

            if (
                positions[first]
                <= positions[second]
            ):

                correct_pairs += 1

        if total_pairs == 0:

            return 0.0

        return (
            correct_pairs
            / total_pairs
        )

    # ==========================================================
    # CLAMP
    # ==========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """
        Clamp score between 0 and 1.
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )