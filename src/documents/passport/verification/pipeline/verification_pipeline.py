"""
Passport Verification Pipeline - Phase 1.

Fast LOS-oriented validation.

Phase 1 checks:
    - File / structural validation
    - Passport document classification
    - Image quality
    - Human face detection
    - Tampering risk

Not executed:
    - OCR
    - MRZ extraction
    - MRZ checksum validation
    - Field extraction
    - Identity matching
    - Credit decision
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.common.authenticity.tamper import (
    analyze_tampering,
)

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
    Passport Phase-1 verification.

    Face detection and tamper analysis are supporting evidence.
    They do not independently prove that a passport is genuine.
    """

    @classmethod
    def verify(
        cls,
        file_path: str,
        request_id: str = "",
    ) -> dict[str, Any]:

        # ============================================================
        # 1. INPUT
        # ============================================================

        path = Path(file_path)

        if not path.exists():

            return cls._failed(
                request_id,
                "Uploaded document does not exist.",
            )

        if not path.is_file():

            return cls._failed(
                request_id,
                "Uploaded path is not a file.",
            )

        # ============================================================
        # 2. PREPROCESSING
        # ============================================================

        try:

            preprocessing = (
                PreprocessingPipeline.process(
                    str(path)
                )
            )

        except Exception as exc:

            return cls._failed(
                request_id,
                f"Passport preprocessing failed: {exc}",
            )

        if not isinstance(
            preprocessing,
            dict,
        ):

            return cls._failed(
                request_id,
                "Invalid preprocessing result.",
            )

        # ============================================================
        # 3. QUALITY
        # ============================================================

        quality_results = (
            preprocessing.get(
                "quality",
                [],
            )
            or []
        )

        if quality_results:

            quality = quality_results[0]

        else:

            quality = {
                "available": False,
                "passed": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    "No quality result available.",
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
                "reason":
                    "Invalid quality result.",
                "checks": {},
            }

        # ============================================================
        # 4. DOCUMENT CLASSIFICATION
        # ============================================================

        try:

            document = (
                DocumentClassifier.classify(
                    file_path=str(path),
                    preprocessing_result=preprocessing,
                )
            )

        except Exception as exc:

            return cls._failed(
                request_id,
                f"Passport document classification failed: {exc}",
            )

        if not isinstance(
            document,
            dict,
        ):

            return cls._failed(
                request_id,
                "Invalid document classification result.",
            )

        # ============================================================
        # 5. FIRST PROCESSED IMAGE
        # ============================================================

        processed_images = (
            preprocessing.get(
                "processed_images",
                [],
            )
            or []
        )

        first_image = (
            processed_images[0]
            if processed_images
            else None
        )

        # ============================================================
        # 6. LOAD IMAGE ONCE
        # ============================================================

        image = None

        if first_image:

            try:

                image = cv2.imread(
                    str(first_image),
                    cv2.IMREAD_COLOR,
                )

            except Exception:

                image = None

        # ============================================================
        # 7. HUMAN FACE DETECTION
        # ============================================================

        face_result = (
            cls._detect_human_face(
                image
            )
        )

        # ============================================================
        # 8. TAMPER ANALYSIS
        # ============================================================

        tamper_result = (
            cls._analyze_tampering(
                image
            )
        )

        # ============================================================
        # 9. STRUCTURAL VALIDATION
        # ============================================================

        document_checks = (
            document.get(
                "checks",
                {},
            )
            or {}
        )

        structural_passed = all(
            bool(
                document_checks.get(
                    check,
                    True,
                )
            )
            for check in (
                "file_exists",
                "supported_extension",
                "pdf_readable",
                "has_pages",
                "within_page_limit",
            )
        )

        # ============================================================
        # 10. DECISION ENGINE
        # ============================================================

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
                request_id,
                f"Passport decision engine failed: {exc}",
            )

        if not isinstance(
            decision,
            dict,
        ):

            return cls._failed(
                request_id,
                "Invalid decision engine result.",
            )

        final_decision = decision.get(
            "decision",
            DOCUMENT_REJECTED,
        )

        score = cls._safe_float(
            decision.get(
                "score",
                0,
            )
        )

        confidence = cls._safe_float(
            decision.get(
                "confidence",
                0,
            )
        )

        # ============================================================
        # 11. LOS
        # ============================================================

        los = (
            decision.get(
                "los",
                {},
            )
            or {}
        )

        if not isinstance(
            los,
            dict,
        ):

            los = {}

        # High tamper risk should route to review.
        # It should not automatically be treated as proof of fraud.

        if (
            tamper_result["risk"]
            == "HIGH"
            and
            final_decision
            ==
            "DOCUMENT_VERIFIED"
        ):

            final_decision = DOCUMENT_REVIEW

            los[
                "document_validation"
            ] = DOCUMENT_REVIEW

            los[
                "requires_rcu_review"
            ] = True

        else:

            los.setdefault(
                "document_validation",
                final_decision,
            )

            los.setdefault(
                "requires_rcu_review",
                final_decision
                ==
                DOCUMENT_REVIEW,
            )

        los.setdefault(
            "requires_ocr_phase",
            True,
        )

        los.setdefault(
            "credit_decision",
            None,
        )

        # ============================================================
        # 12. IMAGE QUALITY LABEL
        # ============================================================

        quality_available = bool(
            quality.get(
                "available",
                False,
            )
        )

        quality_passed = bool(
            quality.get(
                "passed",
                False,
            )
        )

        if not quality_available:

            image_quality = "NOT_CHECKED"

        elif quality_passed:

            image_quality = "GOOD"

        else:

            image_quality = "POOR"

        # ============================================================
        # 13. DOCUMENT DETECTED
        # ============================================================

        document_detected = bool(
            document.get(
                "eligible",
                False,
            )
        )

        # ============================================================
        # 14. FINAL RESULT
        # ============================================================

        return {

            "request_id":
                request_id,

            "document_type":
                document.get(
                    "document_type",
                    "UNKNOWN",
                ),

            "status":
                final_decision,

            "decision":
                final_decision,

            "score":
                score,

            "confidence":
                confidence,

            "validation_mode":
                "PHASE_1_NO_OCR",

            "validation": {

                "document_detected":
                    document_detected,

                "image_quality":
                    image_quality,

                "human_photo_detected":
                    face_result[
                        "detected"
                    ],

                "tampering_risk":
                    tamper_result[
                        "risk"
                    ],

                "structural_validation":
                    (
                        "PASS"
                        if structural_passed
                        else
                        "FAIL"
                    ),

            },

            "los":
                los,

            "errors":
                list(
                    decision.get(
                        "errors",
                        [],
                    )
                    or []
                ),

            "warnings":
                list(
                    decision.get(
                        "warnings",
                        [],
                    )
                    or []
                ),

        }

    # ================================================================
    # HUMAN FACE DETECTION
    # ================================================================

    @staticmethod
    def _detect_human_face(
        image: Any,
    ) -> dict[str, Any]:

        if image is None:

            return {
                "detected":
                    "NOT_CHECKED",
                "available":
                    False,
            }

        try:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            cascade_path = (
                cv2.data.haarcascades
                +
                "haarcascade_frontalface_default.xml"
            )

            detector = cv2.CascadeClassifier(
                cascade_path
            )

            if detector.empty():

                return {
                    "detected":
                        "NOT_CHECKED",
                    "available":
                        False,
                }

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )

            return {

                "detected":
                    len(faces) > 0,

                "available":
                    True,

                "face_count":
                    len(faces),

            }

        except Exception:

            return {
                "detected":
                    "NOT_CHECKED",
                "available":
                    False,
            }

    # ================================================================
    # TAMPER ANALYSIS
    # ================================================================

    @staticmethod
    def _analyze_tampering(
        image: Any,
    ) -> dict[str, Any]:

        if image is None:

            return {
                "risk":
                    "NOT_CHECKED",
                "available":
                    False,
            }

        try:

            result = analyze_tampering(
                image
            )

        except Exception:

            return {
                "risk":
                    "NOT_CHECKED",
                "available":
                    False,
            }

        if not isinstance(
            result,
            dict,
        ):

            return {
                "risk":
                    "NOT_CHECKED",
                "available":
                    False,
            }

        risk = str(
            result.get(
                "risk",
                "UNKNOWN",
            )
        ).upper()

        if risk not in {
            "LOW",
            "MEDIUM",
            "HIGH",
        }:

            risk = "NOT_CHECKED"

        return {

            "risk":
                risk,

            "available":
                True,

            "tamper_score":
                result.get(
                    "tamper_score",
                    0,
                ),

        }

    # ================================================================
    # SAFE FLOAT
    # ================================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:

        try:

            return round(
                float(value),
                2,
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    # ================================================================
    # FAILED RESPONSE
    # ================================================================

    @staticmethod
    def _failed(
        request_id: str,
        reason: str,
    ) -> dict[str, Any]:

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
                0,

            "confidence":
                0,

            "validation_mode":
                "PHASE_1_NO_OCR",

            "validation": {

                "document_detected":
                    False,

                "image_quality":
                    "NOT_CHECKED",

                "human_photo_detected":
                    "NOT_CHECKED",

                "tampering_risk":
                    "NOT_CHECKED",

                "structural_validation":
                    "FAIL",

            },

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