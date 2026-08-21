"""
Passport LOS Decision Engine - Phase 1.

PHASE 1:
    - Document detection
    - Structural validation
    - Image quality validation
    - Basic LOS routing

NOT PERFORMED:
    - OCR
    - MRZ extraction
    - MRZ checksum validation
    - Field extraction
    - Identity verification
    - Final authenticity verification

Important:
    Phase 1 must NOT reject a passport merely because OCR/MRZ
    has not been executed.
"""

from __future__ import annotations

from typing import Any

from src.documents.passport.core.constants import (
    DOCUMENT_REJECTED,
    DOCUMENT_REVIEW,
    DOCUMENT_VERIFIED,
)


class DecisionEngine:
    """
    Passport Phase-1 decision engine.

    The purpose of this engine is to determine whether the uploaded
    document is structurally eligible to continue through the LOS
    document-verification flow.

    It does NOT make the final passport authenticity/identity decision.
    """

    # ================================================================
    # SCORE WEIGHTS
    # ================================================================

    DOCUMENT_WEIGHT = 40.0
    STRUCTURE_WEIGHT = 30.0
    QUALITY_WEIGHT = 30.0

    @staticmethod
    def _safe_bool(
        value: Any,
    ) -> bool:
        """
        Safely convert a value to boolean.
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if value is None:
            return False

        if isinstance(
            value,
            str,
        ):

            return value.strip().lower() in {
                "true",
                "1",
                "yes",
                "pass",
                "passed",
            }

        return bool(value)

    @classmethod
    def _structural_validation(
        cls,
        document: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform fast structural validation.

        No OCR is required.
        """

        checks = dict(
            document.get(
                "checks",
                {},
            )
            or {}
        )

        file_exists = cls._safe_bool(
            checks.get(
                "file_exists",
                True,
            )
        )

        supported_extension = cls._safe_bool(
            checks.get(
                "supported_extension",
                True,
            )
        )

        pdf_readable = cls._safe_bool(
            checks.get(
                "pdf_readable",
                True,
            )
        )

        has_pages = cls._safe_bool(
            checks.get(
                "has_pages",
                True,
            )
        )

        within_page_limit = cls._safe_bool(
            checks.get(
                "within_page_limit",
                True,
            )
        )

        page_count = int(
            preprocessing.get(
                "page_count",
                document.get(
                    "metadata",
                    {},
                ).get(
                    "page_count",
                    0,
                ),
            )
            or 0
        )

        page_check = (
            page_count > 0
            and
            page_count <= 10
        )

        structural_checks = {

            "file_exists":
                file_exists,

            "supported_extension":
                supported_extension,

            "pdf_readable":
                pdf_readable,

            "has_pages":
                has_pages,

            "within_page_limit":
                within_page_limit,

            "page_count_valid":
                page_check,

        }

        passed = all(
            structural_checks.values()
        )

        return {

            "passed":
                passed,

            "checks":
                structural_checks,

            "page_count":
                page_count,

        }

    @classmethod
    def _quality_validation(
        cls,
        quality: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize the passport quality result.
        """

        available = cls._safe_bool(
            quality.get(
                "available",
                False,
            )
        )

        passed = cls._safe_bool(
            quality.get(
                "passed",
                False,
            )
        )

        try:

            score = float(
                quality.get(
                    "score",
                    0.0,
                )
                or 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        if not available:

            status = "NOT_CHECKED"

        elif passed:

            status = "GOOD"

        else:

            status = "POOR"

        return {

            "available":
                available,

            "passed":
                passed,

            "score":
                round(
                    score,
                    2,
                ),

            "status":
                status,

            "reason":
                quality.get(
                    "reason",
                    "",
                ),

        }

    @classmethod
    def evaluate(
        cls,
        document: dict[str, Any],
        quality: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> dict[str, Any]:

        # ============================================================
        # 1. DOCUMENT DETECTION
        # ============================================================

        document_detected = cls._safe_bool(
            document.get(
                "eligible",
                False,
            )
        )

        # ============================================================
        # 2. STRUCTURAL VALIDATION
        # ============================================================

        structural = (
            cls._structural_validation(
                document=document,
                preprocessing=preprocessing,
            )
        )

        # ============================================================
        # 3. QUALITY
        # ============================================================

        quality_result = (
            cls._quality_validation(
                quality
            )
        )

        # ============================================================
        # 4. SCORE
        # ============================================================

        document_score = (
            cls.DOCUMENT_WEIGHT
            if document_detected
            else 0.0
        )

        structure_score = (
            cls.STRUCTURE_WEIGHT
            if structural["passed"]
            else 0.0
        )

        quality_score = (
            (
                quality_result["score"]
                /
                100.0
            )
            *
            cls.QUALITY_WEIGHT
        )

        score = round(
            document_score
            +
            structure_score
            +
            quality_score,
            2,
        )

        # ============================================================
        # 5. DECISION
        # ============================================================

        #
        # Hard rejection:
        #   Document itself is not detected OR structural validation
        #   failed.
        #

        if not document_detected:

            decision = (
                DOCUMENT_REJECTED
            )

            status = (
                DOCUMENT_REJECTED
            )

        elif not structural["passed"]:

            decision = (
                DOCUMENT_REJECTED
            )

            status = (
                DOCUMENT_REJECTED
            )

        #
        # Quality unavailable:
        #   Do NOT reject automatically.
        #   Send to review / next verification stage.
        #

        elif not quality_result["available"]:

            decision = (
                DOCUMENT_REVIEW
            )

            status = (
                DOCUMENT_REVIEW
            )

        #
        # Quality failed:
        #   Do not claim validation success.
        #

        elif not quality_result["passed"]:

            decision = (
                DOCUMENT_REVIEW
            )

            status = (
                DOCUMENT_REVIEW
            )

        #
        # Structural + quality passed:
        #   Phase-1 document validation passed.
        #
        # IMPORTANT:
        # This does NOT mean passport authenticity/identity is verified.
        # OCR/MRZ remains mandatory for Phase 2.
        #

        else:

            decision = (
                DOCUMENT_VERIFIED
            )

            status = (
                DOCUMENT_VERIFIED
            )

        # ============================================================
        # 6. LOS ROUTING
        # ============================================================

        requires_ocr_phase = True

        requires_rcu_review = (
            decision
            ==
            DOCUMENT_REVIEW
        )

        # Credit does not make a decision here.
        credit_decision = None

        # ============================================================
        # 7. ERRORS
        # ============================================================

        errors: list[str] = []

        if not document_detected:

            errors.append(
                "Passport document could not be detected."
            )

        if not structural["passed"]:

            failed_checks = [
                key
                for key, value
                in structural["checks"].items()
                if not value
            ]

            if failed_checks:

                errors.append(
                    "Structural validation failed: "
                    +
                    ", ".join(
                        failed_checks
                    )
                )

        # ============================================================
        # 8. WARNINGS
        # ============================================================

        warnings = [

            (
                "Phase 1 performs non-OCR passport "
                "validation only."
            ),

            (
                "OCR/MRZ identity verification is "
                "deferred to Phase 2."
            ),

        ]

        if not quality_result["available"]:

            warnings.append(
                "Image quality could not be evaluated."
            )

        elif not quality_result["passed"]:

            warnings.append(
                "Image quality is below the configured "
                "passport verification threshold."
            )

        # ============================================================
        # 9. FINAL RESULT
        # ============================================================

        return {

            "status":
                status,

            "decision":
                decision,

            "score":
                score,

            "confidence":
                round(
                    score / 100.0,
                    3,
                ),

            "validation_mode":
                "PHASE_1_NO_OCR",

            "common_validation":
                None,

            "errors":
                errors,

            "warnings":
                warnings,

            "validation": {

                "document_detected":
                    document_detected,

                "image_quality":
                    quality_result["status"],

                "structural_validation":
                    (
                        "PASS"
                        if structural["passed"]
                        else "FAIL"
                    ),

                "structural_checks":
                    structural["checks"],

                "page_count":
                    structural["page_count"],

                "quality_score":
                    quality_result["score"],

            },

            "los": {

                "document_validation":
                    decision,

                "requires_ocr_phase":
                    requires_ocr_phase,

                "requires_rcu_review":
                    requires_rcu_review,

                "credit_decision":
                    credit_decision,

            },

        }