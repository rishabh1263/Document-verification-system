import re
from typing import Any, Dict

from .base_extractor import BaseExtractor


class GenericExtractor(BaseExtractor):
    """
    Fallback extractor used when the document type can't be confidently
    classified. Pulls out any label:value pairs it can find so the
    pipeline never returns an empty result, and stays robust/generic
    for document types not explicitly modeled yet.
    """
    doc_type = "generic"

    LABEL_VALUE_PATTERN = re.compile(
        r"([A-Za-z][A-Za-z .]{2,30})\s*[:\-]\s*([A-Za-z0-9 ,./\-]{2,60})"
    )

    def extract(self, text: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for m in self.LABEL_VALUE_PATTERN.finditer(text):
            key = self._clean(m.group(1)).lower().replace(" ", "_")
            value = self._clean(m.group(2))
            if key and value and key not in fields:
                fields[key] = value
        return fields

