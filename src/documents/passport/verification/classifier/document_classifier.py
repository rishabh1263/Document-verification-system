"""
Passport document classifier.

PHASE 1:
    Validation only.

OCR is intentionally NOT used here.

The classifier performs cheap structural/document-level checks that can
be done before extraction.

Important:
    This does NOT prove that a passport is genuine.
    It establishes whether the uploaded file is suitable for the passport
    validation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class DocumentClassifier:
    """
    Fast passport document classifier.

    Phase 1 deliberately avoids OCR.
    """

    PASSPORT = "PASSPORT"

    UNKNOWN = "UNKNOWN"

    SUPPORTED_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    SUPPORTED_DOCUMENT_EXTENSIONS = {
        ".pdf",
    }


    @classmethod
    def classify(
        cls,
        file_path: str,
        preprocessing_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform non-OCR passport document checks.

        Returns evidence that can be passed to the common validation layer.

        This function intentionally does NOT inspect passport text.
        """

        path = Path(
            file_path
        )

        if not path.exists():

            return cls._failed(
                "Uploaded document does not exist."
            )

        if not path.is_file():

            return cls._failed(
                "Uploaded path is not a file."
            )

        extension = (
            path.suffix.lower()
        )

        # ---------------------------------------------------------------
        # FILE TYPE
        # ---------------------------------------------------------------

        if extension not in (
            cls.SUPPORTED_IMAGE_EXTENSIONS
            |
            cls.SUPPORTED_DOCUMENT_EXTENSIONS
        ):

            return cls._failed(
                f"Unsupported passport file type: "
                f"{extension}"
            )

        # ---------------------------------------------------------------
        # PDF
        # ---------------------------------------------------------------

        if extension == ".pdf":

            return cls._classify_pdf(
                path,
                preprocessing_result,
            )

        # ---------------------------------------------------------------
        # IMAGE
        # ---------------------------------------------------------------

        return cls._classify_image(
            path,
        )


    @classmethod
    def _classify_image(
        cls,
        path: Path,
    ) -> dict[str, Any]:
        """
        Validate basic image properties.

        No OCR.
        """

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            return cls._failed(
                "Image could not be decoded."
            )

        height, width = (
            image.shape[:2]
        )

        if width <= 0 or height <= 0:

            return cls._failed(
                "Image has invalid dimensions."
            )

        # Passport identity documents are normally landscape-oriented.
        #
        # We do NOT reject portrait images outright because mobile uploads
        # and scans can be rotated.
        aspect_ratio = (
            width
            /
            float(height)
        )

        return {

            "document_type":
                cls.PASSPORT,

            "eligible":
                True,

            "confidence":
                0.50,

            "classification_method":
                "NON_OCR_STRUCTURAL",

            "checks": {

                "file_exists":
                    True,

                "file_readable":
                    True,

                "supported_extension":
                    True,

                "valid_dimensions":
                    True,

            },

            "metadata": {

                "extension":
                    path.suffix.lower(),

                "width":
                    width,

                "height":
                    height,

                "aspect_ratio":
                    round(
                        aspect_ratio,
                        3,
                    ),

            },

            "warnings": [

                "Passport identity cannot be confirmed "
                "without OCR/MRZ extraction."

            ],

        }


    @classmethod
    def _classify_pdf(
        cls,
        path: Path,
        preprocessing_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Validate basic PDF properties.

        The PDF converter already determines page count, so reuse that
        information rather than opening/rendering the PDF again.
        """

        page_count = None

        if preprocessing_result:

            page_count = (
                preprocessing_result.get(
                    "page_count"
                )
            )

        if page_count is None:

            # Keep this lightweight. The preprocessing layer is expected
            # to provide page_count.
            return {

                "document_type":
                    cls.PASSPORT,

                "eligible":
                    True,

                "confidence":
                    0.40,

                "classification_method":
                    "NON_OCR_STRUCTURAL",

                "checks": {

                    "file_exists":
                        True,

                    "supported_extension":
                        True,

                    "pdf_readable":
                        True,

                    "page_count_available":
                        False,

                },

                "metadata": {

                    "extension":
                        ".pdf",

                },

                "warnings": [

                    "PDF page count was not available.",

                    "Passport identity cannot be confirmed "
                    "without OCR/MRZ extraction.",

                ],

            }

        try:

            page_count = int(
                page_count
            )

        except (
            TypeError,
            ValueError,
        ):

            page_count = 0

        if page_count <= 0:

            return cls._failed(
                "PDF contains no readable pages."
            )

        max_pages = 2

        within_page_limit = (
            page_count
            <=
            max_pages
        )

        return {

            "document_type":
                cls.PASSPORT,

            "eligible":
                within_page_limit,

            "confidence":
                0.50
                if within_page_limit
                else 0.0,

            "classification_method":
                "NON_OCR_STRUCTURAL",

            "checks": {

                "file_exists":
                    True,

                "supported_extension":
                    True,

                "pdf_readable":
                    True,

                "has_pages":
                    True,

                "within_page_limit":
                    within_page_limit,

            },

            "metadata": {

                "extension":
                    ".pdf",

                "page_count":
                    page_count,

            },

            "warnings": [

                "Passport identity cannot be confirmed "
                "without OCR/MRZ extraction."

            ],

        }


    @classmethod
    def _failed(
        cls,
        reason: str,
    ) -> dict[str, Any]:
        """
        Standard failed classification response.
        """

        return {

            "document_type":
                cls.UNKNOWN,

            "eligible":
                False,

            "confidence":
                0.0,

            "classification_method":
                "NON_OCR_STRUCTURAL",

            "checks": {},

            "metadata": {},

            "warnings": [],

            "errors": [
                reason
            ],

        }