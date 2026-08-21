import re
from typing import Any, Dict

from .base_extractor import BaseExtractor


class BankStatementExtractor(BaseExtractor):
    doc_type = "bank_statement"

    PATTERNS = {
        "account_holder_name": [
            r"(?:account\s*holder(?:'s)?\s*name|name)\s*[:\-]\s*([A-Za-z .]+)",
        ],
        "account_number": [
            r"(?:account\s*(?:no\.?|number))\s*[:\-]\s*([0-9Xx\*]{6,20})",
        ],
        "ifsc_code": [
            r"\bIFSC\s*(?:code)?\s*[:\-]?\s*([A-Z]{4}0[A-Z0-9]{6})",
        ],
        "branch": [
            r"(?:branch)\s*[:\-]\s*([A-Za-z0-9 .,\-]+)",
        ],
        "statement_period": [
            r"(?:statement\s*period|period)\s*[:\-]\s*([0-9A-Za-z\-\/ ]+to[0-9A-Za-z\-\/ ]+)",
        ],
        "opening_balance": [
            r"(?:opening\s*balance)\s*[:\-]?\s*(?:rs\.?|inr|â‚¹)?\s*([\d,]+\.?\d*)",
        ],
        "closing_balance": [
            r"(?:closing\s*balance)\s*[:\-]?\s*(?:rs\.?|inr|â‚¹)?\s*([\d,]+\.?\d*)",
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

