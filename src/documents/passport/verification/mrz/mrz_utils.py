from datetime import datetime


class MRZUtils:
    """
    ICAO 9303 utility functions for passport MRZ.
    """

    WEIGHTS = [7, 3, 1]

    @staticmethod
    def character_value(char: str) -> int:
        """
        ICAO character values.

        0-9 -> 0-9
        A-Z -> 10-35
        <   -> 0
        """

        if char.isdigit():
            return int(char)

        if "A" <= char <= "Z":
            return ord(char) - ord("A") + 10

        if char == "<":
            return 0

        return 0

    @classmethod
    def calculate_check_digit(cls, value: str) -> str:
        """
        Calculate ICAO check digit.
        """

        total = 0

        for index, char in enumerate(value):

            weight = cls.WEIGHTS[index % 3]

            total += cls.character_value(char) * weight

        return str(total % 10)

    @classmethod
    def validate_check_digit(
        cls,
        value: str,
        expected: str
    ) -> bool:

        if expected is None:
            return False

        return cls.calculate_check_digit(value) == expected

    @staticmethod
    def validate_date(value: str) -> bool:
        """
        Validate YYMMDD date.

        ICAO stores dates as YYMMDD.
        """

        if len(value) != 6:
            return False

        try:

            yy = int(value[:2])

            mm = int(value[2:4])

            dd = int(value[4:6])

            # Guess century
            current = datetime.now().year % 100

            if yy <= current:
                year = 2000 + yy
            else:
                year = 1900 + yy

            datetime(year, mm, dd)

            return True

        except Exception:

            return False

    @staticmethod
    def normalize_ocr(text: str) -> str:
        """
        Correct common OCR mistakes before validation.
        """

        if not text:
            return ""

        replacements = {

            "O": "0",
            "Q": "0",
            "D": "0",

            "I": "1",
            "L": "1",

            "S": "5",
            "B": "8",

            "G": "6",
            "Z": "2"

        }

        result = ""

        for ch in text:

            result += replacements.get(ch, ch)

        return result

    @staticmethod
    def clean_field(value: str) -> str:
        """
        Remove filler characters.
        """

        if value is None:
            return ""

        return value.replace("<", "").strip()


# -------------------------------------------------------------------
# Backward-compatible wrapper functions
# -------------------------------------------------------------------

def calculate_check_digit(value: str) -> str:
    return MRZUtils.calculate_check_digit(value)


def validate_check_digit(value: str, expected: str) -> bool:
    return MRZUtils.validate_check_digit(value, expected)


def validate_date(value: str) -> bool:
    return MRZUtils.validate_date(value)


def normalize_ocr(text: str) -> str:
    return MRZUtils.normalize_ocr(text)


def clean_field(value: str) -> str:
    return MRZUtils.clean_field(value)
