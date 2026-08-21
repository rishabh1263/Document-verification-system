"""
Common PDF renderer.

Converts PDF pages into OpenCV BGR images.

Used only when a PDF needs to go through OCR,
for example a scanned PDF.
"""

from __future__ import annotations

import pymupdf
import numpy as np


def render_pdf_pages(
    file_bytes: bytes,
    dpi: int = 150,
) -> list[np.ndarray]:
    """
    Render every PDF page as an OpenCV BGR image.

    Args:
        file_bytes:
            PDF file bytes.

        dpi:
            Rendering resolution.

    Returns:
        List of BGR numpy images.
    """

    if not file_bytes:
        raise ValueError(
            "PDF file is empty."
        )

    if dpi <= 0:
        raise ValueError(
            "DPI must be greater than 0."
        )

    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )

    except Exception as exc:
        raise ValueError(
            f"Unable to open PDF: {exc}"
        ) from exc

    try:

        if len(document) == 0:
            raise ValueError(
                "PDF contains no pages."
            )

        scale = dpi / 72.0

        matrix = pymupdf.Matrix(
            scale,
            scale,
        )

        images: list[np.ndarray] = []

        for page in document:

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            image = np.frombuffer(
                pixmap.samples,
                dtype=np.uint8,
            )

            image = image.reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )

            # PyMuPDF gives RGB.
            # OpenCV expects BGR.
            if pixmap.n >= 3:

                image = image[
                    :, :, :3
                ]

                image = image[
                    :, :, ::-1
                ].copy()

            else:

                image = image.copy()

            images.append(image)

        return images

    finally:

        document.close()