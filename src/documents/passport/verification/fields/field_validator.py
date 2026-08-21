from src.documents.passport.verification.fields.field_utils import FieldUtils


class FieldValidator:
    """
    Compare OCR extracted fields with MRZ fields.

    The validator is tolerant to common OCR errors.
    """

    NATIONALITY_MAP = {
        "INDIAN": "IND",
        "IND": "IND"
    }

    @classmethod
    def validate(cls, visible_fields, mrz):

        if not mrz.get("parsed", False):

            return {

                "passed": False,

                "score": 0,

                "confidence": 0.0,

                "matches": {},

                "reason": "MRZ parsing failed."

            }

        matches = {}

        score = 0

        # -----------------------------
        # Passport Number
        # -----------------------------

        visible_passport = (
            visible_fields.get("passport_number", "")
            .replace(" ", "")
            .upper()
        )

        mrz_passport = (
            mrz.get("passport_number", "")
            .replace("<", "")
            .upper()
        )

        matches["passport_number"] = (
            visible_passport == mrz_passport
        )

        if matches["passport_number"]:
            score += 25

        # -----------------------------
        # Date of Birth
        # -----------------------------

        visible_birth = FieldUtils.normalize_date(
            visible_fields.get(
                "date_of_birth",
                ""
            )
        )

        mrz_birth = mrz.get(
            "birth_date",
            ""
        )

        matches["birth_date"] = (
            visible_birth == mrz_birth
        )

        if matches["birth_date"]:
            score += 25

        # -----------------------------
        # Expiry Date
        # -----------------------------

        visible_expiry = FieldUtils.normalize_date(
            visible_fields.get(
                "date_of_expiry",
                ""
            )
        )

        mrz_expiry = mrz.get(
            "expiry_date",
            ""
        )

        matches["expiry_date"] = (
            visible_expiry == mrz_expiry
        )

        if matches["expiry_date"]:
            score += 25

        # -----------------------------
        # Nationality
        # -----------------------------

        visible_nat = (
            visible_fields.get(
                "nationality",
                ""
            )
            .strip()
            .upper()
        )

        visible_nat = cls.NATIONALITY_MAP.get(
            visible_nat,
            visible_nat
        )

        mrz_nat = (
            mrz.get(
                "nationality",
                ""
            )
            .strip()
            .upper()
        )

        matches["nationality"] = (
            visible_nat == mrz_nat
        )

        if matches["nationality"]:
            score += 25

        confidence = round(score / 100, 2)

        return {

            "passed": score >= 75,

            "score": score,

            "confidence": confidence,

            "matches": matches

        }
