"""
Common document processing result models.

This file defines the standard structure returned by the
common document processing engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageProcessingResult:
    """
    Processing result for one document page.
    """

    page_number: int

    width: int

    height: int

    quality: dict[str, Any] = field(
        default_factory=dict
    )

    ocr: list[dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def text(self) -> str:
        """
        Combine OCR text from this page.
        """

        return "\n".join(
            item.get("text", "")
            for item in self.ocr
            if item.get("text")
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert page result to a dictionary.
        """

        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "quality": self.quality,
            "ocr": self.ocr,
            "text": self.text,
        }


@dataclass
class DocumentProcessingResult:
    """
    Complete result returned by the common document engine.
    """

    filename: str

    file_type: str

    page_count: int

    pages: list[PageProcessingResult] = field(
        default_factory=list
    )

    @property
    def full_text(self) -> str:
        """
        Combine OCR text from all pages.
        """

        return "\n".join(
            page.text
            for page in self.pages
            if page.text
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert complete document result to a dictionary.
        """

        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "page_count": self.page_count,
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
            "full_text": self.full_text,
        }