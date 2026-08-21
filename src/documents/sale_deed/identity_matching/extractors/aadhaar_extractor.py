import re

from .base_extractor import BaseExtractor


class AadhaarExtractor(BaseExtractor):

    def extract(self, text: str):

        data = {
            "document_type": "Aadhaar",
            "name": None,
            "aadhaar_number": None,
            "address": None
        }

        aadhaar = re.search(
            r"\d{4}\s\d{4}\s\d{4}",
            text
        )

        if aadhaar:
            data["aadhaar_number"] = aadhaar.group()

        return data
