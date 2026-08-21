"""
Aadhaar Validation Agent

Validates the extracted Aadhaar fields and returns
a validation report.
"""

import re


class AadhaarValidationAgent:

    REQUIRED_FIELDS = [
        "name",
        "aadhaar_number",
        "dob",
        "gender",
        "address"
    ]

    def run(self, extracted_data: dict) -> dict:
        """
        Validate extracted Aadhaar data.

        Args:
            extracted_data (dict): Output from AadhaarExtractionEngine

        Returns:
            dict: Validation report
        """

        if not extracted_data:

            return {
                "success": False,
                "validation_score": 0,
                "errors": ["No extracted data found."],
                "warnings": []
            }

        fields = extracted_data.get("fields", {})

        errors = []
        warnings = []

        # -------------------------------------------------------
        # Required Fields
        # -------------------------------------------------------

        for field in self.REQUIRED_FIELDS:

            value = fields.get(field, {}).get("value", "")

            if not value:

                errors.append(f"{field} is missing.")

        # -------------------------------------------------------
        # Aadhaar Number
        # -------------------------------------------------------

        aadhaar = fields.get("aadhaar_number", {}).get("value", "")

        if aadhaar:

            aadhaar = aadhaar.replace(" ", "")

            if not re.fullmatch(r"\d{12}", aadhaar):

                errors.append("Invalid Aadhaar number.")

        # -------------------------------------------------------
        # DOB
        # -------------------------------------------------------

        dob = fields.get("dob", {}).get("value", "")

        if dob:

            if not re.fullmatch(r"\d{2}[/-]\d{2}[/-]\d{4}", dob):

                errors.append("Invalid DOB format.")

        # -------------------------------------------------------
        # Gender
        # -------------------------------------------------------

        gender = fields.get("gender", {}).get("value", "")

        if gender:

            if gender.lower() not in ["male", "female", "other"]:

                errors.append("Invalid gender.")

        # -------------------------------------------------------
        # Name Length
        # -------------------------------------------------------

        name = fields.get("name", {}).get("value", "")

        if name:

            if len(name.strip()) < 3:

                warnings.append("Name looks too short.")

        # -------------------------------------------------------
        # Address Length
        # -------------------------------------------------------

        address = fields.get("address", {}).get("value", "")

        if address:

            if len(address.strip()) < 10:

                warnings.append("Address looks incomplete.")

        # -------------------------------------------------------
        # Validation Score
        # -------------------------------------------------------

        total_checks = 5
        failed_checks = len(errors)

        validation_score = max(
            0,
            round(((total_checks - failed_checks) / total_checks) * 100, 2)
        )

        return {

            "success": len(errors) == 0,

            "validation_score": validation_score,

            "errors": errors,

            "warnings": warnings

        }
