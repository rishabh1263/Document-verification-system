"""
Optimized Document Loader
"""

from pathlib import Path
from typing import List

import cv2
import fitz
import numpy as np

from src.documents.sale_deed.config.config import PDF_DPI


class DocumentLoader:

    def __init__(self, file_path: str):

        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

    # =====================================================
    # Document Type
    # =====================================================

    def get_extension(self):

        return self.file_path.suffix.lower()

    def is_pdf(self):

        return self.get_extension() == ".pdf"

    # =====================================================
    # Embedded Text Detection
    # =====================================================

    def has_embedded_text(self):

        if not self.is_pdf():
            return False

        with fitz.open(self.file_path) as pdf:

            for page in pdf:

                if len(page.get_text().strip()) > 50:
                    return True

        return False

    # =====================================================
    # Extract Embedded Text
    # =====================================================

    def extract_text(self):

        if not self.is_pdf():
            return ""

        pages = []

        with fitz.open(self.file_path) as pdf:

            for page in pdf:

                text = page.get_text().strip()

                if text:
                    pages.append(text)

        return "\n\n".join(pages)

    # =====================================================
    # Convert PDF to Images
    # =====================================================

    def extract_images(self):

        if not self.is_pdf():
            return self._load_image()

        pages = []

        zoom = PDF_DPI / 72

        matrix = fitz.Matrix(
            zoom,
            zoom
        )

        with fitz.open(self.file_path) as pdf:

            for page in pdf:

                pix = page.get_pixmap(

                    matrix=matrix,

                    alpha=False

                )

                image = np.frombuffer(

                    pix.samples,

                    dtype=np.uint8

                ).reshape(

                    pix.height,

                    pix.width,

                    3

                )

                image = cv2.cvtColor(

                    image,

                    cv2.COLOR_RGB2BGR

                )

                pages.append(image.copy())

                del pix

        return pages

    # =====================================================
    # Load Image
    # =====================================================

    def _load_image(self):

        image = cv2.imread(str(self.file_path))

        if image is None:

            raise FileNotFoundError(

                f"Unable to load image: {self.file_path}"

            )

        return [image]
