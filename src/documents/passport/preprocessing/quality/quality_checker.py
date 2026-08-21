"""
Passport quality adapter.

Uses the common image-quality implementation while keeping
Passport-specific thresholds and LOS evidence separate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.common.verification.image_quality import (
    analyze_image_quality,
    calculate_quality_score,
)

from src.documents.passport.core.config import settings

from src.documents.passport.core.constants import (
    MIN_PASSPORT_IMAGE_HEIGHT,
    MIN_PASSPORT_IMAGE_WIDTH,
)


class QualityChecker:
    """
    Passport-specific quality adapter.

    The common quality module expects a NumPy image.
    This adapter converts the supplied image path into an image.
    """

    @classmethod
    def check(
        cls,
        image_path: str,
    ) -> dict[str, Any]:

        # ============================================================
        # 1. PATH
        # ============================================================

        path = Path(
            image_path
        )

        if not path.exists():

            return {
                "passed": False,
                "available": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    "Image file does not exist.",
                "checks": {},
            }

        if not path.is_file():

            return {
                "passed": False,
                "available": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    "Image path is not a file.",
                "checks": {},
            }

        # ============================================================
        # 2. LOAD IMAGE
        # ============================================================

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            return {
                "passed": False,
                "available": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    "Image could not be decoded.",
                "checks": {},
            }

        # ============================================================
        # 3. COMMON ANALYSIS
        # ============================================================

        try:

            result = (
                analyze_image_quality(
                    image
                )
            )

        except Exception as exc:

            return {
                "passed": False,
                "available": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    f"Image quality analysis failed: {exc}",
                "checks": {},
            }

        if not isinstance(
            result,
            dict,
        ):

            return {
                "passed": False,
                "available": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    "Invalid common quality result.",
                "checks": {},
            }

        # ============================================================
        # 4. SCORE
        # ============================================================

        try:

            quality_score, score_details = (
                calculate_quality_score(
                    result
                )
            )

            quality_score = float(
                quality_score
            )

        except Exception as exc:

            return {
                "passed": False,
                "available": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason":
                    f"Quality score calculation failed: {exc}",
                "checks": {},
            }

        quality_score = max(
            0.0,
            min(
                100.0,
                quality_score,
            ),
        )

        # ============================================================
        # 5. RESOLUTION
        # ============================================================

        try:

            width = int(
                result.get(
                    "width",
                    0,
                )
            )

            height = int(
                result.get(
                    "height",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            width = 0
            height = 0

        resolution_passed = (
            width >= MIN_PASSPORT_IMAGE_WIDTH
            and
            height >= MIN_PASSPORT_IMAGE_HEIGHT
        )

        # ============================================================
        # 6. CHECKS
        # ============================================================

        checks = {

            "resolution":
                resolution_passed,

            "passport_resolution":
                resolution_passed,

            "quality_score":
                quality_score
                >=
                settings.MIN_QUALITY_SCORE,

            "aspect_ratio":
                bool(
                    result.get(
                        "aspect_ratio_ok",
                        False,
                    )
                ),

        }

        # ============================================================
        # 7. FINAL PASS
        # ============================================================

        passed = (
            bool(
                result.get(
                    "available",
                    False,
                )
            )
            and
            resolution_passed
            and
            quality_score
            >=
            settings.MIN_QUALITY_SCORE
        )

        # ============================================================
        # 8. REASON
        # ============================================================

        if passed:

            reason = (
                "Passport image quality "
                "is sufficient."
            )

        elif not result.get(
            "available",
            False,
        ):

            reason = (
                "Passport image quality "
                "could not be evaluated."
            )

        elif not resolution_passed:

            reason = (
                "Passport image resolution "
                "is below the required threshold."
            )

        else:

            reason = (
                "Passport image quality score "
                "is below the configured threshold."
            )

        # ============================================================
        # 9. RESULT
        # ============================================================

        return {

            "available":
                True,

            "passed":
                passed,

            "score":
                round(
                    quality_score,
                    2,
                ),

            "confidence":
                round(
                    quality_score / 100.0,
                    3,
                ),

            "reason":
                reason,

            "checks":
                checks,

            "metrics":
                result,

            "score_details":
                score_details,

        }