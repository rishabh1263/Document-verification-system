"""
Base class for all document field extractors.

Every extractor takes raw OCR text and returns a dict of normalized fields.
Keeping this interface identical across doc types is what makes the
pipeline "generic" â€” adding a new document type is just:
  1. add keywords to config.DOC_TYPE_KEYWORDS
  2. subclass BaseExtractor
  3. register it in extractors/__init__.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExtractor(ABC):
    doc_type: str = "generic"

    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dict of extracted fields from raw OCR/text-layer text."""
        raise NotImplementedError

    @staticmethod
    def _clean(value: str) -> str:
        return " ".join(value.split()).strip(" :\t-") if value else value

