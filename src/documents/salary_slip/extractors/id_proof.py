import re
from typing import Any, Dict

from .base_extractor import BaseExtractor


class IDProofExtractor(BaseExtractor):
    doc_type = "id_proof"

    PATTERNS = {
        "name": [
            r"(?:name)\s*[:\-]\s*([A-Za-z .]+)",
        ],
        "date_of_birth": [
            r"(?:date\s*of\s*birth|dob)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        ],
        "pan_number": [
            r"\b([A-Z]{5}[0-9]{4}[A-Z])\b",
        ],
        "aadhaar_number": [
            r"\b(\d{4}\s?\d{4}\s?\d{4})\b",
        ],
        "father_name": [
            r"(?:father(?:'s)?\s*name)\s*[:\-]\s*([A-Za-z .]+)",
        ],
    }

    def extract(self, text: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for field_name, patterns in self.PATTERNS.items():
            fields[field_name] = self._first_match(text, patterns)
        return fields

    def _first_match(self, text: str, patterns):
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return self._clean(m.group(1))
        return None

