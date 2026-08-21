"""
Data Normalizer
Standardizes extracted values before comparison.
No matching logic should exist here.
"""
import re
import unicodedata
from datetime import datetime


class DataNormalizer:

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Basic cleanup for OCR output.
        """
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Lowercase
        text = text.lower()
        # Remove leading/trailing spaces
        text = text.strip()
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize personal names.
        """
        name = DataNormalizer.normalize_text(name)
        # Remove common titles
        titles = [
            "mr",
            "mrs",
            "ms",
            "dr",
            "shri",
            "smt",
            "kumari"
        ]
        words = []
        for word in name.split():
            cleaned = word.replace(".", "")
            if cleaned not in titles:
                words.append(cleaned)
        name = " ".join(words)
        # Remove punctuation
        name = re.sub(r"[^a-z0-9\u0900-\u097F ]", "", name)
        return name

    @staticmethod
    def normalize_address(address: str) -> str:
        address = DataNormalizer.normalize_text(address)
        address = address.replace(",", " ")
        address = address.replace(".", " ")
        address = re.sub(r"\s+", " ", address)
        return address

    @staticmethod
    def normalize_gender(gender: str) -> str:
        gender = DataNormalizer.normalize_text(gender)
        mapping = {
            "m": "male",
            "male": "male",
            "f": "female",
            "female": "female",
            "other": "other"
        }
        return mapping.get(gender, gender)

    @staticmethod
    def normalize_date(date_text: str) -> str:
        if not date_text:
            return ""
        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d.%m.%Y"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(
                    date_text.strip(),
                    fmt
                ).strftime("%Y-%m-%d")
            except Exception:
                pass
        return date_text.strip()

    @staticmethod
    def normalize_document_number(number: str) -> str:
        if not number:
            return ""
        number = number.upper()
        number = re.sub(r"[^A-Z0-9]", "", number)
        return number
