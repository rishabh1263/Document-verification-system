"""
Passport OCR engine.

Architecture:
    - Reuse PaddleOCR already used by the common/PAN implementation.
    - Keep ONE OCR inference pass for the normal path.
    - Preserve OCR confidence and bounding boxes.
    - Do not perform expensive preprocessing here.
    - MRZ-specific OCR correction happens after OCR.

LOS requirement:
    OCR should provide evidence quickly.
    It must NOT itself decide whether the passport is genuine.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR

from src.documents.passport.core.config import settings


class PassportOCRError(Exception):
    """Raised when passport OCR cannot be completed."""


class OCREngine:
    """
    Fast PaddleOCR wrapper for passport verification.

    The model is initialized lazily and cached so that it is not loaded
    for every request.
    """

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_ocr() -> PaddleOCR:
        """
        Initialize PaddleOCR once per process.

        Keep the configuration lightweight for the LOS synchronous path.
        """

        try:

            return PaddleOCR(

                lang="en",

                # Fast mobile detector.
                text_detection_model_name=(
                    "PP-OCRv5_mobile_det"
                ),

                # Fast mobile recognizer.
                text_recognition_model_name=(
                    "en_PP-OCRv5_mobile_rec"
                ),

                # Keep detection image size controlled.
                text_det_limit_side_len=(
                    settings.OCR_DET_LIMIT_SIDE_LEN
                ),

                # Avoid unnecessary orientation processing.
                use_doc_orientation_classify=False,

                # Passport identity pages are normally not warped.
                use_doc_unwarping=False,

                # Text-line orientation is not required for the normal
                # passport fast path.
                use_textline_orientation=False,

                # Disable verbose logging.
                show_log=False,
            )

        except Exception as exc:

            raise PassportOCRError(
                f"Could not initialize PaddleOCR: {exc}"
            ) from exc


    @staticmethod
    def _prepare_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize only when the image is unnecessarily large.

        Do NOT:
            - denoise
            - CLAHE
            - sharpen
            - threshold
            - create multiple variants

        Those operations increase latency and should not be part of the
        normal LOS path.
        """

        if image is None:

            raise PassportOCRError(
                "Invalid image."
            )

        if image.size == 0:

            raise PassportOCRError(
                "Empty image."
            )

        height, width = (
            image.shape[:2]
        )

        max_width = (
            settings.OCR_MAX_WIDTH
        )

        if width > max_width:

            scale = (
                max_width
                /
                float(width)
            )

            new_height = max(
                1,
                int(
                    height * scale
                ),
            )

            image = cv2.resize(
                image,
                (
                    max_width,
                    new_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        return image


    @classmethod
    def _run_ocr(
        cls,
        image: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Perform exactly ONE PaddleOCR inference pass.
        """

        ocr = cls._get_ocr()

        try:

            result = ocr.predict(
                image
            )

        except Exception as exc:

            raise PassportOCRError(
                f"PaddleOCR inference failed: {exc}"
            ) from exc

        lines: list[
            dict[str, Any]
        ] = []

        for page_result in result:

            data = getattr(
                page_result,
                "json",
                None,
            )

            if callable(data):

                data = data()

            if data is None:

                data = getattr(
                    page_result,
                    "res",
                    None,
                )

            if data is None:

                continue

            if isinstance(
                data,
                list,
            ):

                objects = data

            elif isinstance(
                data,
                dict,
            ):

                objects = [
                    data
                ]

            else:

                continue

            for obj in objects:

                if not isinstance(
                    obj,
                    dict,
                ):

                    continue

                texts = obj.get(
                    "rec_texts",
                    [],
                )

                scores = obj.get(
                    "rec_scores",
                    [],
                )

                boxes = obj.get(
                    "rec_boxes",
                    obj.get(
                        "rec_polys",
                        [],
                    ),
                )

                if isinstance(
                    texts,
                    str,
                ):

                    texts = [
                        texts
                    ]

                for index, text in enumerate(
                    texts
                ):

                    text = str(
                        text
                    ).strip()

                    if not text:

                        continue

                    confidence = 0.0

                    if (
                        index
                        <
                        len(scores)
                    ):

                        try:

                            confidence = float(
                                scores[
                                    index
                                ]
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):

                            confidence = 0.0

                    bbox = None

                    if (
                        index
                        <
                        len(boxes)
                    ):

                        bbox = boxes[
                            index
                        ]

                    lines.append(
                        {
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox,
                        }
                    )

        return lines


    @staticmethod
    def _normal(
        text: str,
    ) -> str:
        """
        Normalize OCR text for matching.

        Preserve '<' because it is meaningful in MRZ.
        """

        return re.sub(
            r"\s+",
            " ",
            str(text)
            .upper()
            .strip(),
        )


    @staticmethod
    def _plain_lines(
        results: list[
            dict[str, Any]
        ],
    ) -> list[str]:
        """
        Return OCR text only.
        """

        return [
            str(
                item.get(
                    "text",
                    "",
                )
            ).strip()
            for item in results
            if item.get(
                "text"
            )
        ]


    @classmethod
    def extract(
        cls,
        image_path: str,
    ) -> dict[str, Any]:
        """
        Main passport OCR entry point.
        """

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise PassportOCRError(
                f"Unable to read image: "
                f"{image_path}"
            )

        image = cls._prepare_image(
            image
        )

        results = cls._run_ocr(
            image
        )

        lines = cls._plain_lines(
            results
        )

        confidence_values = [
            float(
                item.get(
                    "confidence",
                    0.0,
                )
            )
            for item in results
            if item.get(
                "confidence"
            ) is not None
        ]

        average_confidence = (
            sum(
                confidence_values
            )
            /
            len(
                confidence_values
            )
            if confidence_values
            else 0.0
        )

        mrz_candidates = [
            line
            for line in lines
            if cls._looks_like_mrz(
                line
            )
        ]

        return {

            "lines": lines,

            "results": results,

            "text": "\n".join(
                lines
            ),

            "line_count": len(
                lines
            ),

            "average_confidence": round(
                average_confidence,
                4,
            ),

            "mrz_candidates": (
                mrz_candidates
            ),

            "ocr_success": bool(
                lines
            ),

        }


    @staticmethod
    def _looks_like_mrz(
        text: str,
    ) -> bool:
        """
        Cheap MRZ signal.

        This is NOT MRZ validation.
        It only identifies candidate lines.

        Real MRZ validation happens in the MRZ layer.
        """

        compact = re.sub(
            r"\s+",
            "",
            str(text)
            .upper(),
        )

        if len(compact) < 30:

            return False

        # Passport MRZ starts with P<.
        if compact.startswith(
            "P<"
        ):

            return True

        # OCR may occasionally corrupt P<.
        # A long line containing mostly MRZ characters is still a useful
        # candidate for the MRZ correction layer.
        allowed_count = sum(
            1
            for char in compact
            if (
                char.isalpha()
                or
                char.isdigit()
                or
                char == "<"
            )
        )

        return (
            allowed_count
            /
            max(
                len(compact),
                1,
            )
            >= 0.95
        )