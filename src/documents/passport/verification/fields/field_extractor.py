import re

from src.documents.passport.verification.fields.field_utils import FieldUtils


class FieldExtractor:

    @classmethod
    def extract(cls, ocr):

        text = FieldUtils.normalize_text(
            ocr["text"]
        )

        fields = {

            "passport_number": None,

            "date_of_birth": None,

            "date_of_issue": None,

            "date_of_expiry": None,

            "nationality": None,

            "surname": None,

            "given_names": None

        }

        passport = re.search(
            r"[A-Z][0-9]{7}",
            text
        )

        if passport:

            number = passport.group()

            if FieldUtils.is_valid_passport_number(number):

                fields["passport_number"] = number

        dates = re.findall(
            r"\d{2}/\d{2}/\d{4}",
            text
        )

        if len(dates) >= 1:
            fields["date_of_birth"] = dates[0]

        if len(dates) >= 2:
            fields["date_of_issue"] = dates[1]

        if len(dates) >= 3:
            fields["date_of_expiry"] = dates[2]

        if "INDIAN" in text:

            fields["nationality"] = "IND"

        surname = re.search(
            r"SURNAME[^A-Z]*([A-Z]+)",
            text
        )

        if surname:

            fields["surname"] = surname.group(1)

        given = re.search(
            r"GIVEN NAME\S*[^A-Z]*([A-Z ]+?)DATE",
            text
        )

        if given:

            fields["given_names"] = FieldUtils.normalize_name(
                given.group(1)
            )

        return fields
