"""
Passport Verification Pipeline - Phase 1

PHASE 1 SCOPE
-------------
Validation only.

OCR / MRZ extraction is intentionally NOT executed.

Flow:

    Upload
        ↓
    Common file security
        ↓
    Preprocessing
        ↓
    Common image quality
        ↓
    Non-OCR document classification
        ↓
    Common document validation
        ↓
    LOS document decision


PHASE 2
-------
OCR
MRZ detection
MRZ correction
MRZ parsing
MRZ checksum validation
Field extraction
Visible-field ↔ MRZ consistency
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.documents.passport.core.constants import (
    DOCUMENT_REJECTED,
    DOCUMENT_REVIEW,
)

from src.documents.passport.preprocessing.pipeline.preprocessing_pipeline import (
    PreprocessingPipeline,
)

from src.documents.passport.verification.classifier.document_classifier import (
    DocumentClassifier,
)

from src.documents.passport.verification.decision.decision_engine import (
    DecisionEngine,
)


class VerificationPipeline:
    """
    Orchestrates Passport Phase-1 validation.

    IMPORTANT
    ---------
    This pipeline intentionally DOES NOT import or execute:

        - AuthenticityEngine
        - ForgeryEngine
        - OCREngine
        - MRZCandidate
        - MRZCorrector
        - MRZParser
        - MRZValidator
        - RiskEngine
        - ConfidenceEngine

    Those components are reserved for later phases.

    Phase 1 is focused on fast, non-OCR document validation for LOS.
    """

    @classmethod
    def verify(
        cls,
        file_path: str,
        request_id: str = "",
    ) -> dict[str, Any]:
        """
        Run the complete Phase-1 passport validation flow.

        Parameters
        ----------
        file_path:
            Path of the uploaded passport file.

        request_id:
            LOS/request correlation identifier.

        Returns
        -------
        dict
            Standardized passport validation result.
        """

        # ===============================================================
        # 1. INPUT VALIDATION
        # ===============================================================

        path = Path(
            file_path
        )

        if not path.exists():

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Uploaded document does not exist."
                ),
            )

        if not path.is_file():

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Uploaded path is not a file."
                ),
            )

        # ===============================================================
        # 2. PREPROCESSING
        # ===============================================================

        try:

            preprocessing = (
                PreprocessingPipeline.process(
                    str(path)
                )
            )

        except Exception as exc:

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Passport preprocessing failed: "
                    f"{exc}"
                ),
            )

        if not isinstance(
            preprocessing,
            dict,
        ):

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Invalid preprocessing result."
                ),
            )

        # ===============================================================
        # 3. QUALITY
        # ===============================================================

        quality_results = (
            preprocessing.get(
                "quality",
                [],
            )
            or []
        )

        if quality_results:

            # Phase 1 only evaluates the first image/page.
            quality = quality_results[0]

        else:

            quality = {
                "available": False,
                "passed": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason": (
                    "No image quality result available."
                ),
                "checks": {},
            }

        if not isinstance(
            quality,
            dict,
        ):

            quality = {
                "available": False,
                "passed": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason": (
                    "Invalid image quality result."
                ),
                "checks": {},
            }

        # ===============================================================
        # 4. NON-OCR DOCUMENT CLASSIFICATION
        # ===============================================================

        try:

            document = (
                DocumentClassifier.classify(
                    file_path=str(path),
                    preprocessing_result=preprocessing,
                )
            )

        except Exception as exc:

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Passport document classification "
                    f"failed: {exc}"
                ),
            )

        if not isinstance(
            document,
            dict,
        ):

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Invalid document classification result."
                ),
            )

        # ===============================================================
        # 5. COMMON DOCUMENT VALIDATION
        # ===============================================================

        try:

            decision = (
                DecisionEngine.evaluate(
                    document=document,
                    quality=quality,
                    preprocessing=preprocessing,
                )
            )

        except Exception as exc:

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Passport decision engine failed: "
                    f"{exc}"
                ),
            )

        if not isinstance(
            decision,
            dict,
        ):

            return cls._failed(
                request_id=request_id,
                reason=(
                    "Invalid decision engine result."
                ),
            )

        # ===============================================================
        # 6. EXTRACT DECISION
        # ===============================================================

        status = (
            decision.get(
                "status",
                DOCUMENT_REJECTED,
            )
        )

        final_decision = (
            decision.get(
                "decision",
                DOCUMENT_REJECTED,
            )
        )

        score = (
            decision.get(
                "score",
                0.0,
            )
        )

        confidence = (
            decision.get(
                "confidence",
                0.0,
            )
        )

        # ===============================================================
        # 7. LOS ROUTING
        # ===============================================================

        los = decision.get(
            "los",
            {},
        )

        if not isinstance(
            los,
            dict,
        ):

            los = {}

        los.setdefault(
            "document_validation",
            final_decision,
        )

        # OCR is intentionally disabled in Phase 1.
        los.setdefault(
            "requires_ocr_phase",
            True,
        )

        # RCU should only become relevant when the document needs review.
        los.setdefault(
            "requires_rcu_review",
            final_decision
            ==
            DOCUMENT_REVIEW,
        )

        # Credit decision belongs to CRDT/credit policy,
        # not the document validation service.
        los.setdefault(
            "credit_decision",
            None,
        )

        # ===============================================================
        # 8. WARNINGS / ERRORS
        # ===============================================================

        errors = list(
            decision.get(
                "errors",
                [],
            )
            or []
        )

        warnings = list(
            decision.get(
                "warnings",
                [],
            )
            or []
        )

        # Explicitly communicate the scope of Phase 1.
        phase_warning = (
            "Phase 1 performs non-OCR passport validation only. "
            "Passport identity, MRZ fields and MRZ checksum "
            "validation are deferred to Phase 2."
        )

        if phase_warning not in warnings:

            warnings.append(
                phase_warning
            )

        # ===============================================================
        # 9. FINAL RESULT
        # ===============================================================

        return {

            # -----------------------------------------------------------
            # REQUEST
            # -----------------------------------------------------------

            "request_id":
                request_id,

            # -----------------------------------------------------------
            # DOCUMENT
            # -----------------------------------------------------------

            "document_type":
                document.get(
                    "document_type",
                    "UNKNOWN",
                ),

            # -----------------------------------------------------------
            # DECISION
            # -----------------------------------------------------------

            "status":
                status,

            "decision":
                final_decision,

            "score":
                cls._safe_float(
                    score
                ),

            "confidence":
                cls._safe_float(
                    confidence
                ),

            # -----------------------------------------------------------
            # VALIDATION MODE
            # -----------------------------------------------------------

            "validation_mode":
                "PHASE_1_NO_OCR",

            # -----------------------------------------------------------
            # VALIDATION EVIDENCE
            # -----------------------------------------------------------

            "validation": {

                "document":
                    document,

                "quality":
                    quality,

                "preprocessing": {

                    "is_pdf":
                        bool(
                            preprocessing.get(
                                "is_pdf",
                                False,
                            )
                        ),

                    "page_count":
                        preprocessing.get(
                            "page_count",
                            1,
                        ),

                    "image_count":
                        preprocessing.get(
                            "image_count",
                            0,
                        ),

                    "quality_passed":
                        bool(
                            preprocessing.get(
                                "quality_passed",
                                False,
                            )
                        ),

                },

                "common":
                    decision.get(
                        "common_validation",
                        {},
                    ),

            },

            # -----------------------------------------------------------
            # LOS
            # -----------------------------------------------------------

            "los":
                los,

            # -----------------------------------------------------------
            # ERRORS
            # -----------------------------------------------------------

            "errors":
                errors,

            # -----------------------------------------------------------
            # WARNINGS
            # -----------------------------------------------------------

            "warnings":
                warnings,

        }


    # ===================================================================
    # HELPERS
    # ===================================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        """
        Safely normalize numeric values.

        Prevents malformed confidence/score values from breaking the
        response.
        """

        try:

            return round(
                float(
                    value
                ),
                3,
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0


    @staticmethod
    def _failed(
        request_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """
        Standard pipeline failure response.

        Failure here means the validation pipeline itself could not
        complete. It is not a credit decision.
        """

        return {

            "request_id":
                request_id,

            "document_type":
                "UNKNOWN",

            "status":
                DOCUMENT_REJECTED,

            "decision":
                DOCUMENT_REJECTED,

            "score":
                0.0,

            "confidence":
                0.0,

            "validation_mode":
                "PHASE_1_NO_OCR",

            "validation":
                {},

            "los": {

                "document_validation":
                    DOCUMENT_REJECTED,

                "requires_ocr_phase":
                    False,

                "requires_rcu_review":
                    False,

                "credit_decision":
                    None,

            },

            "errors": [
                reason
            ],

            "warnings": [],

        }