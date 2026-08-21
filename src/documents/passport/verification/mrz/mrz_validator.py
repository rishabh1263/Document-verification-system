from src.documents.passport.verification.mrz.mrz_utils import (
    validate_check_digit,
    validate_date,
)


class MRZValidator:
    """
    ICAO 9303 TD3 Passport MRZ Validator.

    Returns:
        passed
        valid
        score
        confidence
        checks
        errors
    """

    @classmethod
    def validate(cls, mrz):

        if not mrz or not mrz.get("parsed", False):
            return {
                "passed": False,
                "valid": False,
                "score": 0,
                "confidence": 0.0,
                "checks": {},
                "errors": ["MRZ could not be parsed."]
            }

        checks = {}
        errors = []

        score = 0
        max_score = 0

        def add_check(name, passed, weight, message=None):
            nonlocal score, max_score

            checks[name] = passed
            max_score += weight

            if passed:
                score += weight
            elif message:
                errors.append(message)

        # -----------------------------------------
        # Document Type
        # -----------------------------------------

        document_type = (
            mrz.get("document_type", "")
            .replace("<", "")
            .strip()
            .upper()
        )

        add_check(
            "document_type",
            document_type.startswith("P"),
            10,
            "Invalid document type."
        )

        # -----------------------------------------
        # Passport Number
        # -----------------------------------------

        passport_number = mrz.get("passport_number", "")
        passport_check = mrz.get("passport_number_check", "")

        add_check(
            "passport_number",
            validate_check_digit(
                passport_number,
                passport_check
            ),
            20,
            "Passport number checksum failed."
        )

        # -----------------------------------------
        # Birth Date
        # -----------------------------------------

        birth_date = mrz.get("birth_date", "")
        birth_check = mrz.get("birth_date_check", "")

        birth_valid = (
            validate_date(birth_date)
            and
            validate_check_digit(
                birth_date,
                birth_check
            )
        )

        add_check(
            "birth_date",
            birth_valid,
            20,
            "Birth date validation failed."
        )

        # -----------------------------------------
        # Expiry Date
        # -----------------------------------------

        expiry_date = mrz.get("expiry_date", "")
        expiry_check = mrz.get("expiry_date_check", "")

        expiry_valid = (
            validate_date(expiry_date)
            and
            validate_check_digit(
                expiry_date,
                expiry_check
            )
        )

        add_check(
            "expiry_date",
            expiry_valid,
            20,
            "Expiry date validation failed."
        )

        # -----------------------------------------
        # Personal Number (optional)
        # -----------------------------------------

        personal_number = mrz.get(
            "personal_number",
            ""
        )

        personal_check = mrz.get(
            "personal_number_check",
            ""
        )

        if personal_number.replace("<", "") == "":
            personal_valid = True
        else:
            personal_valid = validate_check_digit(
                personal_number,
                personal_check
            )

        add_check(
            "personal_number",
            personal_valid,
            10,
            "Personal number checksum failed."
        )

        # -----------------------------------------
        # Final Composite Check
        # -----------------------------------------

        final_check = mrz.get("final_check", "")

        composite = (
            passport_number
            + passport_check
            + birth_date
            + birth_check
            + expiry_date
            + expiry_check
            + personal_number
            + personal_check
        )

        if final_check:

            composite_valid = validate_check_digit(
                composite,
                final_check
            )

            add_check(
                "final_check",
                composite_valid,
                20,
                "Composite checksum failed."
            )

        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        confidence = (
            round(score / max_score, 2)
            if max_score
            else 0.0
        )

        # -----------------------------------------
        # Critical checks
        # -----------------------------------------

        critical = all([
            checks.get("document_type", False),
            checks.get("passport_number", False),
            checks.get("birth_date", False),
            checks.get("expiry_date", False),
        ])

        return {

            "passed": critical,

            "valid": critical,

            "score": round(confidence * 100, 2),

            "confidence": confidence,

            "checks": checks,

            "errors": errors

        }
