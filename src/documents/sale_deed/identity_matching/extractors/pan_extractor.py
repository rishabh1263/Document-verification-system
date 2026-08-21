import re

from .base_extractor import BaseExtractor


class PANExtractor(BaseExtractor):

    def extract(self, text: str):

        data = {
            "document_type": "PAN",
            "name": None,
            "pan_number": None
        }

        pan = re.search(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            text
        )

        if pan:
            data["pan_number"] = pan.group()

        return data
