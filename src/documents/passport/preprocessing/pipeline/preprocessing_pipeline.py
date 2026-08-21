"""
Passport preprocessing pipeline.

PHASE 1:
    Fast, non-OCR validation preparation.

Flow:
    PDF   -> render page 1 -> quality check
    Image -> quality check

OCR/MRZ is intentionally not called here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.documents.passport.core.config import settings

from src.documents.passport.preprocessing.converter.pdf_converter import (
    PDFConverter,
)

from src.documents.passport.preprocessing.quality.quality_checker import (
    QualityChecker,
)


class PreprocessingPipeline:
    """
    Lightweight passport preprocessing pipeline.
    """

    @staticmethod
    def _processed_directory() -> Path:

        directory = (
            Path(settings.STORAGE_DIR)
            / "processed"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @staticmethod
    def _validate_image_path(
        file_path: str,
    ) -> None:

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Uploaded image is not a file: {path}"
            )

    @classmethod
    def _process_pdf(
        cls,
        file_path: str,
    ) -> dict[str, Any]:

        page_count = (
            PDFConverter.get_page_count(
                file_path
            )
        )

        if page_count <= 0:

            raise ValueError(
                "PDF contains no pages."
            )

        processed_dir = (
            cls._processed_directory()
        )

        first_page = (
            PDFConverter.render_first_page(
                pdf_path=file_path,
                output_folder=str(
                    processed_dir
                ),
            )
        )

        if not first_page:

            raise ValueError(
                "PDF first page could not be rendered."
            )

        # QualityChecker expects the IMAGE PATH.
        quality = (
            QualityChecker.check(
                first_page
            )
        )

        return {

            "source_file":
                file_path,

            "is_pdf":
                True,

            "page_count":
                page_count,

            "processed_images": [
                first_page
            ],

            "quality": [
                quality
            ],

            "next_page":
                (
                    2
                    if page_count >= 2
                    else None
                ),

        }

    @classmethod
    def _process_image(
        cls,
        file_path: str,
    ) -> dict[str, Any]:

        cls._validate_image_path(
            file_path
        )

        # QualityChecker expects the IMAGE PATH.
        quality = (
            QualityChecker.check(
                file_path
            )
        )

        return {

            "source_file":
                file_path,

            "is_pdf":
                False,

            "page_count":
                1,

            "processed_images": [
                file_path
            ],

            "quality": [
                quality
            ],

            "next_page":
                None,

        }

    @classmethod
    def process(
        cls,
        file_path: str,
    ) -> dict[str, Any]:

        path = Path(
            file_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Input path is not a file: {path}"
            )

        extension = (
            path.suffix.lower()
        )

        if extension == ".pdf":

            result = (
                cls._process_pdf(
                    file_path
                )
            )

        else:

            result = (
                cls._process_image(
                    file_path
                )
            )

        result[
            "file_extension"
        ] = extension

        result[
            "image_count"
        ] = len(
            result.get(
                "processed_images",
                [],
            )
        )

        quality_results = (
            result.get(
                "quality",
                [],
            )
            or []
        )

        result[
            "quality_passed"
        ] = bool(
            quality_results
        ) and all(
            item.get(
                "passed",
                False,
            )
            for item in quality_results
        )

        return result