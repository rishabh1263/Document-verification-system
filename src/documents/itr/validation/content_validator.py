"""
==============================================================
ITR Content Validator
==============================================================

Purpose
-------
Validate whether the detected ITR contains the expected
mandatory content.

Checks:
    1. Assessment Year
    2. PAN
    3. Taxpayer information
    4. Income information
    5. Tax computation
    6. Verification
    7. Acknowledgement

IMPORTANT
---------
This validator performs CONTENT validation only.

It does NOT perform:
    - ITR detection
    - OCR
    - PAN external verification
    - Name matching
    - DOB matching
    - Authenticity verification
    - Tax calculation verification

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations

import logging
import re
from time import perf_counter

from .constants import (
    CONTENT_WEIGHTS,
    REASON_ACKNOWLEDGEMENT_FOUND,
    REASON_ACKNOWLEDGEMENT_MISSING,
    REASON_ASSESSMENT_YEAR_FOUND,
    REASON_ASSESSMENT_YEAR_MISSING,
    REASON_INCOME_FOUND,
    REASON_INCOME_MISSING,
    REASON_PAN_FOUND,
    REASON_PAN_MISSING,
    REASON_TAX_COMPUTATION_FOUND,
    REASON_TAX_COMPUTATION_MISSING,
    REASON_TAXPAYER_FOUND,
    REASON_TAXPAYER_MISSING,
    REASON_VERIFICATION_FOUND,
    REASON_VERIFICATION_MISSING,
    REQUIRED_CONTENT,
)
from .models import ContentResult


logger = logging.getLogger(__name__)


class ContentValidator:
    """
    Validate mandatory ITR content.
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self) -> None:
        """
        Initialize the content validator.
        """

        logger.debug(
            "ContentValidator initialized"
        )

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def validate(
        self,
        text: str,
    ) -> ContentResult:
        """
        Validate required ITR content.

        Parameters
        ----------
        text:
            Extracted document text.

        Returns
        -------
        ContentResult
            Structured content validation result.
        """

        start_time = perf_counter()

        result = ContentResult()

        try:

            # --------------------------------------------------
            # NORMALIZE TEXT
            # --------------------------------------------------

            normalized_text = self._normalize_text(
                text
            )

            if not normalized_text:

                result.reasons.append(
                    "No document text available"
                )

                return self._finalize(
                    result,
                    start_time,
                )

            # --------------------------------------------------
            # DETECT REQUIRED CONTENT
            # --------------------------------------------------

            result.assessment_year_present = (
                self._detect_assessment_year(
                    normalized_text
                )
            )

            result.pan_present = (
                self._detect_pan(
                    normalized_text
                )
            )

            result.taxpayer_information_present = (
                self._detect_taxpayer_information(
                    normalized_text
                )
            )

            result.income_information_present = (
                self._detect_income_information(
                    normalized_text
                )
            )

            result.tax_computation_present = (
                self._detect_tax_computation(
                    normalized_text
                )
            )

            result.verification_present = (
                self._detect_verification(
                    normalized_text
                )
            )

            result.acknowledgement_present = (
                self._detect_acknowledgement(
                    normalized_text
                )
            )

            # --------------------------------------------------
            # REASONS
            # --------------------------------------------------

            self._add_content_reasons(
                result
            )

            # --------------------------------------------------
            # MISSING ITEMS
            # --------------------------------------------------

            self._collect_missing_items(
                result
            )

            # --------------------------------------------------
            # REQUIRED CONTENT
            # --------------------------------------------------

            result.required_content_present = (
                len(
                    result.missing_items
                ) == 0
            )

            # --------------------------------------------------
            # FINALIZE
            # --------------------------------------------------

            return self._finalize(
                result,
                start_time,
            )

        except Exception as exc:

            logger.exception(
                "ITR content validation failed"
            )

            result.reasons.append(
                f"Content validation error: {exc}"
            )

            return self._finalize(
                result,
                start_time,
            )

    # ==========================================================
    # TEXT NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize extracted document text.

        Normalization:
            - Convert to lowercase
            - Normalize whitespace
            - Preserve useful alphanumeric characters
        """

        if not text:

            return ""

        normalized = text.lower()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    # ==========================================================
    # ASSESSMENT YEAR
    # ==========================================================

    @staticmethod
    def _detect_assessment_year(
        text: str,
    ) -> bool:
        """
        Detect an Indian Income Tax Assessment Year.

        Examples:
            Assessment Year 2025-26
            AY 2025-26
            A.Y. 2025-26
        """

        patterns = [
            r"\bassessment\s+year\s*[:\-]?\s*20\d{2}\s*[-/]\s*\d{2}\b",
            r"\ba\.?\s*y\.?\s*[:\-]?\s*20\d{2}\s*[-/]\s*\d{2}\b",
            r"\bay\s*[:\-]?\s*20\d{2}\s*[-/]\s*\d{2}\b",
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    # ==========================================================
    # PAN
    # ==========================================================

    @staticmethod
    def _detect_pan(
        text: str,
    ) -> bool:
        """
        Detect a PAN number.

        PAN format:
            ABCDE1234F
        """

        pan_pattern = (
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
        )

        return bool(
            re.search(
                pan_pattern,
                text,
                re.IGNORECASE,
            )
        )

    # ==========================================================
    # TAXPAYER INFORMATION
    # ==========================================================

    @staticmethod
    def _detect_taxpayer_information(
        text: str,
    ) -> bool:
        """
        Detect taxpayer-related information.

        Uses multiple signals instead of relying on one
        exact label.
        """

        taxpayer_patterns = [
            r"\btaxpayer\b",
            r"\bassessee\b",
            r"\bname of assessee\b",
            r"\bname\b.*\bpan\b",
            r"\bfather'?s?\s+name\b",
            r"\baddress\b",
            r"\bdate of birth\b",
        ]

        matches = sum(
            1
            for pattern in taxpayer_patterns
            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
        )

        # Require at least two taxpayer signals.
        return matches >= 2

    # ==========================================================
    # INCOME INFORMATION
    # ==========================================================

    @staticmethod
    def _detect_income_information(
        text: str,
    ) -> bool:
        """
        Detect income-related information.
        """

        income_patterns = [
            r"\bincome\b",
            r"\bgross\s+total\s+income\b",
            r"\btotal\s+income\b",
            r"\bincome\s+from\s+salary\b",
            r"\bincome\s+from\s+house\s+property\b",
            r"\bincome\s+from\s+other\s+sources\b",
            r"\bcapital\s+gains?\b",
            r"\bprofits?\s+and\s+gains?\b",
        ]

        matches = sum(
            1
            for pattern in income_patterns
            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
        )

        return matches >= 1

    # ==========================================================
    # TAX COMPUTATION
    # ==========================================================

    @staticmethod
    def _detect_tax_computation(
        text: str,
    ) -> bool:
        """
        Detect tax computation information.
        """

        tax_patterns = [
            r"\btax\s+computation\b",
            r"\btax\s+payable\b",
            r"\btax\s+payable\s+on\s+total\s+income\b",
            r"\btotal\s+tax\b",
            r"\btax\s+paid\b",
            r"\bself\s+assessment\s+tax\b",
            r"\badvance\s+tax\b",
            r"\btds\b",
            r"\btds\s+deducted\b",
            r"\brefund\b",
        ]

        matches = sum(
            1
            for pattern in tax_patterns
            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
        )

        return matches >= 2

    # ==========================================================
    # VERIFICATION
    # ==========================================================

    @staticmethod
    def _detect_verification(
        text: str,
    ) -> bool:
        """
        Detect ITR verification section.
        """

        verification_patterns = [
            r"\bverification\b",
            r"\bverify\b",
            r"\bverified\b",
            r"\btrue\s+and\s+correct\b",
            r"\bdeclare\b",
            r"\bdeclaration\b",
        ]

        matches = sum(
            1
            for pattern in verification_patterns
            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
        )

        return matches >= 1

    # ==========================================================
    # ACKNOWLEDGEMENT
    # ==========================================================

    @staticmethod
    def _detect_acknowledgement(
        text: str,
    ) -> bool:
        """
        Detect ITR acknowledgement information.
        """

        acknowledgement_patterns = [
            r"\backnowledgement\b",
            r"\backnowledgment\b",
            r"\backnowledgement\s+number\b",
            r"\backnowledgment\s+number\b",
            r"\backnowledgement\s+no\b",
            r"\backnowledgment\s+no\b",
            r"\backnowledgement\s+number\b",
            r"\bitr-v\b",
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in acknowledgement_patterns
        )

    # ==========================================================
    # REASONS
    # ==========================================================

    @staticmethod
    def _add_content_reasons(
        result: ContentResult,
    ) -> None:
        """
        Add human-readable reasons for every content check.
        """

        if result.assessment_year_present:

            result.reasons.append(
                REASON_ASSESSMENT_YEAR_FOUND
            )

        else:

            result.reasons.append(
                REASON_ASSESSMENT_YEAR_MISSING
            )

        if result.pan_present:

            result.reasons.append(
                REASON_PAN_FOUND
            )

        else:

            result.reasons.append(
                REASON_PAN_MISSING
            )

        if result.taxpayer_information_present:

            result.reasons.append(
                REASON_TAXPAYER_FOUND
            )

        else:

            result.reasons.append(
                REASON_TAXPAYER_MISSING
            )

        if result.income_information_present:

            result.reasons.append(
                REASON_INCOME_FOUND
            )

        else:

            result.reasons.append(
                REASON_INCOME_MISSING
            )

        if result.tax_computation_present:

            result.reasons.append(
                REASON_TAX_COMPUTATION_FOUND
            )

        else:

            result.reasons.append(
                REASON_TAX_COMPUTATION_MISSING
            )

        if result.verification_present:

            result.reasons.append(
                REASON_VERIFICATION_FOUND
            )

        else:

            result.reasons.append(
                REASON_VERIFICATION_MISSING
            )

        if result.acknowledgement_present:

            result.reasons.append(
                REASON_ACKNOWLEDGEMENT_FOUND
            )

        else:

            result.reasons.append(
                REASON_ACKNOWLEDGEMENT_MISSING
            )

    # ==========================================================
    # MISSING ITEMS
    # ==========================================================

    @staticmethod
    def _collect_missing_items(
        result: ContentResult,
    ) -> None:
        """
        Collect mandatory content that was not detected.
        """

        if not result.assessment_year_present:

            result.missing_items.append(
                "assessment_year"
            )

        if not result.pan_present:

            result.missing_items.append(
                "pan"
            )

        if not result.taxpayer_information_present:

            result.missing_items.append(
                "taxpayer_information"
            )

        if not result.income_information_present:

            result.missing_items.append(
                "income_information"
            )

        if not result.tax_computation_present:

            result.missing_items.append(
                "tax_computation"
            )

        if not result.verification_present:

            result.missing_items.append(
                "verification"
            )

        if not result.acknowledgement_present:

            result.missing_items.append(
                "acknowledgement"
            )

    # ==========================================================
    # SCORE
    # ==========================================================

    @staticmethod
    def _calculate_score(
        result: ContentResult,
    ) -> float:
        """
        Calculate weighted content validation score.

        Each required content component has a centralized
        weight in constants.py.
        """

        checks = {
            "assessment_year": (
                result.assessment_year_present
            ),
            "pan": (
                result.pan_present
            ),
            "taxpayer_information": (
                result.taxpayer_information_present
            ),
            "income_information": (
                result.income_information_present
            ),
            "tax_computation": (
                result.tax_computation_present
            ),
            "verification": (
                result.verification_present
            ),
            "acknowledgement": (
                result.acknowledgement_present
            ),
        }

        score = 0.0

        for name, passed in checks.items():

            if passed:

                score += CONTENT_WEIGHTS.get(
                    name,
                    0.0,
                )

        return round(
            min(
                score,
                1.0,
            ),
            3,
        )

    # ==========================================================
    # FINALIZE
    # ==========================================================

    def _finalize(
        self,
        result: ContentResult,
        start_time: float,
    ) -> ContentResult:
        """
        Calculate final score and processing time.
        """

        result.score = (
            self._calculate_score(
                result
            )
        )

        result.processing_time_ms = round(
            (
                perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        return result


# ==========================================================
# CONVENIENCE FUNCTION
# ==========================================================


def validate_content(
    text: str,
) -> ContentResult:
    """
    Convenience function for content validation.
    """

    validator = ContentValidator()

    return validator.validate(
        text
    )


# ==========================================================
# MODULE TEST
# ==========================================================


if __name__ == "__main__":

    sample_text = """
    Income Tax Return

    Assessment Year 2025-26

    PAN: ABCDE1234F

    Name of Assessee:
    Vedant Sinagare

    Gross Total Income
    Total Income

    Tax Computation
    Tax Paid
    Tax Payable

    Verification

    Acknowledgement Number
    """

    validator = ContentValidator()

    result = validator.validate(
        sample_text
    )

    print(
        "=" * 80
    )

    print(
        "ITR CONTENT VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        result.model_dump()
    )

    print(
        "=" * 80
    )