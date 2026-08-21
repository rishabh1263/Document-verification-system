"""
Passport PDF conversion.

LOS requirement:
    Keep PDF processing lightweight.

Strategy:
    1. Render only the requested page.
    2. Use approximately 150 DPI.
    3. Avoid expensive image enhancement.
    4. Return the rendered image path.

The verification layer decides whether another page is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz


class PDFConverter:
    """
    Fast PDF renderer for passport verification.
    """

    # 150 DPI is enough for normal passport OCR and is considerably
    # cheaper than rendering at 300 DPI.
    DEFAULT_DPI = 150


    @classmethod
    def get_page_count(
        cls,
        pdf_path: str,
    ) -> int:
        """
        Return PDF page count without rendering pages.
        """

        path = Path(
            pdf_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        document = None

        try:

            document = fitz.open(
                str(path)
            )

            return int(
                document.page_count
            )

        except Exception as exc:

            raise ValueError(
                f"Could not open PDF: {exc}"
            ) from exc

        finally:

            if document is not None:

                document.close()


    @classmethod
    def render_page(
        cls,
        pdf_path: str,
        output_folder: str,
        page_number: int = 1,
        dpi: Optional[int] = None,
    ) -> str:
        """
        Render exactly one PDF page.

        page_number is 1-based.

        Example:
            page_number=1
            means the first PDF page.
        """

        path = Path(
            pdf_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        document = None

        try:

            document = fitz.open(
                str(path)
            )

            page_count = (
                document.page_count
            )

            if page_count == 0:

                raise ValueError(
                    "PDF contains no pages."
                )

            if (
                page_number < 1
                or
                page_number > page_count
            ):

                raise ValueError(
                    f"Invalid PDF page: "
                    f"{page_number}. "
                    f"PDF has {page_count} page(s)."
                )

            # PyMuPDF uses zero-based page indexes.
            page = document.load_page(
                page_number - 1
            )

            render_dpi = (
                dpi
                if dpi is not None
                else cls.DEFAULT_DPI
            )

            scale = (
                render_dpi / 72.0
            )

            matrix = fitz.Matrix(
                scale,
                scale,
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            output_dir = Path(
                output_folder
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_dir
                /
                (
                    f"{path.stem}"
                    f"_page_{page_number}.jpg"
                )
            )

            pixmap.save(
                str(output_path)
            )

            return str(
                output_path
            )

        except ValueError:

            raise

        except Exception as exc:

            raise ValueError(
                f"Could not render PDF page "
                f"{page_number}: {exc}"
            ) from exc

        finally:

            if document is not None:

                document.close()


    @classmethod
    def render_first_page(
        cls,
        pdf_path: str,
        output_folder: str,
    ) -> str:
        """
        Fast-path helper.

        The LOS pipeline initially renders only page 1.
        """

        return cls.render_page(
            pdf_path=pdf_path,
            output_folder=output_folder,
            page_number=1,
        )