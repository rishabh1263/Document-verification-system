"""
==============================================================
ITR Structure Detector - V2
==============================================================

Purpose
-------
Detect internal structural relationships in an ITR document.

Detection layers:

    KeywordDetector
        -> individual terminology

    LayoutDetector
        -> broad section arrangement

    StructureDetector
        -> internal component relationships and ordering

This detector performs DETECTION only.

It does NOT:
    - validate PAN
    - validate name
    - validate DOB
    - perform field matching
    - detect tampering
    - perform authenticity validation

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..models import StructureResult


class StructureDetector:
    """
    Detect internal ITR document structure.
    """

    # ==========================================================
    # COMPONENT PATTERNS
    # ==========================================================

    COMPONENT_PATTERNS: Dict[str, Tuple[str, ...]] = {

        "assessment_year": (
            r"\bassessment\s+year\b",
            r"\ba\.?\s*y\.?\s*[:\-]?\s*20\d{2}\s*[-/]\s*20\d{2}\b",
        ),

        "pan": (
            r"\bpan\b",
            r"\bpermanent\s+account\s+number\b",
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        ),

        "assessee": (
            r"\bname\s+of\s+(?:the\s+)?assessee\b",
            r"\bassessee\s+name\b",
            r"\bassessee\b",
            r"\btaxpayer\b",
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
            r"\btax\s+computation\b",
            r"\btotal\s+tax\b",
            r"\btax\s+payable\b",
            r"\bnet\s+tax\s+liability\b",
        ),

        "tax_paid": (
            r"\btax\s+paid\b",
            r"\badvance\s+tax\b",
            r"\bself[-\s]?assessment\s+tax\b",
            r"\btds\b",
            r"\btds\s+claimed\b",
        ),

        "refund": (
            r"\brefund\b",
            r"\brefund\s+due\b",
            r"\btax\s+refund\b",
        ),

        "verification": (
            r"\bverification\b",
            r"\bverified\b",
            r"\btrue\s+and\s+correct\b",
            r"\bverify\s+and\s+submit\b",
        ),

        "itr_identity": (
            r"\bincome\s+tax\s+return\b",
            r"\bincome\s+tax\s+department\b",
            r"\bdepartment\s+of\s+income\s+tax\b",
            r"\bform\s+itr[-\s]?[1-7]\b",
        ),

        "acknowledgement": (
            r"\backnowledg(?:e?ment|ement)\b",
            r"\backnowledgment\b",
            r"\backnowledgement\s+number\b",
            r"\backnowledgment\s+number\b",
        ),
    }

    # ==========================================================
    # COMPONENT WEIGHTS
    # ==========================================================

    COMPONENT_WEIGHTS: Dict[str, float] = {

        "assessment_year": 0.15,

        "pan": 0.15,

        "assessee": 0.10,

        "income": 0.15,

        "deductions": 0.08,

        "tax_computation": 0.12,

        "tax_paid": 0.08,

        "refund": 0.05,

        "verification": 0.05,

        "itr_identity": 0.04,

        "acknowledgement": 0.03,
    }

    # ==========================================================
    # EXPECTED RELATIONSHIPS
    # ==========================================================
    #
    # Direction matters.
    #
    # We do NOT accept:
    #
    #     tax_paid -> tax_computation
    #
    # when expected is:
    #
    #     tax_computation -> tax_paid
    #
    # ==========================================================

    REQUIRED_RELATIONSHIPS = (

        (
            "assessment_year",
            "pan",
        ),

        (
            "pan",
            "income",
        ),

        (
            "income",
            "tax_computation",
        ),

        (
            "tax_computation",
            "tax_paid",
        ),

        (
            "tax_computation",
            "refund",
        ),

        (
            "tax_computation",
            "verification",
        ),

        (
            "itr_identity",
            "assessment_year",
        ),
    )

    # ==========================================================
    # PUBLIC
    # ==========================================================

    def analyze(
        self,
        text: str,
    ) -> StructureResult:
        """
        Analyze document text for internal ITR structure.
        """

        if not text or not text.strip():

            return StructureResult(
                score=0.0,
                detected_components=[],
                relationships_found=[],
                reasons=[
                    "No text available for structure detection"
                ],
            )

        normalized_text = self._normalize_text(
            text
        )

        # ------------------------------------------------------
        # Detect components
        # ------------------------------------------------------

        component_positions = (
            self._detect_components(
                normalized_text
            )
        )

        detected_components = list(
            component_positions.keys()
        )

        if not detected_components:

            return StructureResult(
                score=0.0,
                detected_components=[],
                relationships_found=[],
                reasons=[
                    "No ITR structural components detected"
                ],
            )

        # ------------------------------------------------------
        # Component coverage
        # ------------------------------------------------------

        component_score = (
            self._component_score(
                detected_components
            )
        )

        # ------------------------------------------------------
        # Relationships
        # ------------------------------------------------------

        relationships_found = (
            self._find_relationships(
                component_positions
            )
        )

        relationship_score = (
            self._relationship_score(
                relationships_found
            )
        )

        # ------------------------------------------------------
        # Ordering
        # ------------------------------------------------------

        order_score = (
            self._order_score(
                component_positions
            )
        )

        # ------------------------------------------------------
        # Core structural identity
        # ------------------------------------------------------

        identity_score = (
            self._identity_score(
                detected_components
            )
        )

        # ------------------------------------------------------
        # Final score
        # ------------------------------------------------------

        score = (
            component_score * 0.35
            +
            relationship_score * 0.40
            +
            order_score * 0.25
        )

        # ------------------------------------------------------
        # Strong core bonus
        # ------------------------------------------------------

        if identity_score >= 0.80:

            score += 0.05

        score = self._clamp(
            score
        )

        # ------------------------------------------------------
        # Reasons
        # ------------------------------------------------------

        reasons: List[str] = []

        reasons.append(
            f"{len(detected_components)} structural "
            f"components detected"
        )

        reasons.append(
            f"{len(relationships_found)} structural "
            f"relationships detected"
        )

        reasons.append(
            f"Component coverage score: "
            f"{component_score:.3f}"
        )

        reasons.append(
            f"Relationship score: "
            f"{relationship_score:.3f}"
        )

        reasons.append(
            f"Structure order score: "
            f"{order_score:.3f}"
        )

        reasons.append(
            f"Structural identity score: "
            f"{identity_score:.3f}"
        )

        if score >= 0.70:

            reasons.append(
                "Strong ITR structural evidence"
            )

        elif score >= 0.50:

            reasons.append(
                "Moderate ITR structural evidence"
            )

        elif score > 0:

            reasons.append(
                "Weak ITR structural evidence"
            )

        return StructureResult(
            score=round(
                score,
                3,
            ),
            detected_components=detected_components,
            relationships_found=relationships_found,
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
        Normalize PDF text.
        """

        text = text.lower()

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
    # COMPONENT DETECTION
    # ==========================================================

    def _detect_components(
        self,
        text: str,
    ) -> Dict[str, int]:
        """
        Detect components and record their first position.
        """

        detected: Dict[str, int] = {}

        for component, patterns in (
            self.COMPONENT_PATTERNS.items()
        ):

            positions: List[int] = []

            for pattern in patterns:

                matches = re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )

                for match in matches:

                    positions.append(
                        match.start()
                    )

            if positions:

                detected[component] = min(
                    positions
                )

        return detected

    # ==========================================================
    # COMPONENT SCORE
    # ==========================================================

    def _component_score(
        self,
        components: List[str],
    ) -> float:
        """
        Calculate weighted component coverage.
        """

        if not components:

            return 0.0

        total_weight = sum(
            self.COMPONENT_WEIGHTS.values()
        )

        matched_weight = sum(
            self.COMPONENT_WEIGHTS.get(
                component,
                0.0,
            )
            for component in components
        )

        if total_weight <= 0:

            return 0.0

        return self._clamp(
            matched_weight
            / total_weight
        )

    # ==========================================================
    # RELATIONSHIPS
    # ==========================================================

    def _find_relationships(
        self,
        positions: Dict[str, int],
    ) -> List[str]:
        """
        Find correctly ordered structural relationships.

        Direction matters.
        """

        found: List[str] = []

        for first, second in (
            self.REQUIRED_RELATIONSHIPS
        ):

            if (
                first not in positions
                or second not in positions
            ):

                continue

            # --------------------------------------------------
            # Correct directional relationship
            # --------------------------------------------------

            if (
                positions[first]
                <= positions[second]
            ):

                found.append(
                    f"{first} -> {second}"
                )

        return found

    # ==========================================================
    # RELATIONSHIP SCORE
    # ==========================================================

    def _relationship_score(
        self,
        relationships: List[str],
    ) -> float:
        """
        Calculate correctly ordered relationship coverage.
        """

        if not relationships:

            return 0.0

        total = len(
            self.REQUIRED_RELATIONSHIPS
        )

        return self._clamp(
            len(relationships)
            / total
        )

    # ==========================================================
    # ORDER SCORE
    # ==========================================================

    def _order_score(
        self,
        positions: Dict[str, int],
    ) -> float:
        """
        Evaluate broad ITR component ordering.

        Expected broad flow:

            ITR identity
                ↓
            Assessment Year
                ↓
            PAN
                ↓
            Assessee
                ↓
            Income
                ↓
            Tax computation
                ↓
            Tax paid / refund
                ↓
            Verification
        """

        ordered_components = (
            "itr_identity",
            "assessment_year",
            "pan",
            "assessee",
            "income",
            "tax_computation",
            "tax_paid",
            "refund",
            "verification",
        )

        available = [
            component
            for component in ordered_components
            if component in positions
        ]

        if len(available) < 2:

            return 0.0

        correct = 0

        total = 0

        for index in range(
            len(available) - 1
        ):

            current = available[index]

            following = available[
                index + 1
            ]

            total += 1

            if (
                positions[current]
                <= positions[following]
            ):

                correct += 1

        if total == 0:

            return 0.0

        return self._clamp(
            correct / total
        )

    # ==========================================================
    # STRUCTURAL IDENTITY
    # ==========================================================

    @staticmethod
    def _identity_score(
        components: List[str],
    ) -> float:
        """
        Calculate core ITR structural identity.

        Core:

            ITR identity
            Assessment Year
            PAN
            Income
            Tax computation
        """

        required = (
            "itr_identity",
            "assessment_year",
            "pan",
            "income",
            "tax_computation",
        )

        matched = sum(
            1
            for item in required
            if item in components
        )

        return StructureDetector._clamp(
            matched
            / len(required)
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