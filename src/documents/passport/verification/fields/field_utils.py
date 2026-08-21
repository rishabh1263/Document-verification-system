import re
from datetime import datetime


class FieldUtils:
    """
    Utility functions for extracting and comparing
    passport fields.
    """

    @staticmethod
    def normalize_spaces(text: str) -> str:
        """Remove extra spaces."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Convert OCR text into a consistent format.
        """
        if not text:
            return ""

        text = text.upper()
        text = FieldUtils.normalize_spaces(text)

        return text

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Convert names into a comparable format.

        Example:
            NIDHI AJIT
            NIDHI   AJIT
            NIDHI-AJIT

        ->
            NIDHI AJIT
        """

        if not name:
            return ""

        name = name.upper()

        name = re.sub(r"[^A-Z ]", " ", name)

        name = FieldUtils.normalize_spaces(name)

        return name

    @staticmethod
    def normalize_date(date_string: str):
        """
        Convert

        12/09/1999
        12-09-1999
        1999-09-12

        into

        990912
        """

        if not date_string:
            return None

        formats = [

            "%d/%m/%Y",

            "%d-%m-%Y",

            "%Y-%m-%d"

        ]

        for fmt in formats:

            try:

                date = datetime.strptime(
                    date_string,
                    fmt
                )

                return date.strftime("%y%m%d")

            except ValueError:
                pass

        return None

    @staticmethod
    def clean_passport_number(value: str):
        """
        Remove OCR noise from passport number.
        """

        if not value:
            return None

        value = value.upper()

        value = re.sub(
            r"[^A-Z0-9]",
            "",
            value
        )

        return value

    @staticmethod
    def clean_country(value: str):

        if not value:
            return None

        return value.upper().strip()

    @staticmethod
    def is_valid_passport_number(value: str):
        """
        Indian passport numbers generally follow:

        Letter + 7 digits

        Example:
            U7208925
        """

        if not value:
            return False

        return bool(
            re.fullmatch(
                r"[A-Z][0-9]{7}",
                value
            )
        )
