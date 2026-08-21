"""
Validation Agent

Validates merged extraction results.
"""

import re


class ValidationAgent:

    REQUIRED_FIELDS = [
        ("document_details", "deed_number"),
        ("document_details", "registration_date"),
        ("seller",),
        ("buyer",),
        ("financial", "sale_consideration"),
    ]

    def validate(self, data):

        missing_fields = []

        warnings = []

        confidence = 100

        # ----------------------------
        # Required Fields
        # ----------------------------

        for field in self.REQUIRED_FIELDS:

            if len(field) == 2:

                section, key = field

                value = data.get(section, {}).get(key, "")

                if value == "":
                    missing_fields.append(key)
                    confidence -= 5

            else:

                section = field[0]

                if not data.get(section):
                    missing_fields.append(section)
                    confidence -= 10

        # ----------------------------
        # Date Validation
        # ----------------------------

        date = data.get(
            "document_details",
            {}
        ).get(
            "registration_date",
            ""
        )

        if date:

            if not re.match(
                r"\d{2}/\d{2}/\d{4}",
                date
            ):
                warnings.append(
                    "Invalid registration date format"
                )
                confidence -= 5

        # ----------------------------
        # Financial Validation
        # ----------------------------

        financial = data.get(
            "financial",
            {}
        )

        for key in [
            "stamp_duty",
            "registration_fee",
            "sale_consideration"
        ]:

            value = financial.get(
                key,
                ""
            )

            if value == "":
                warnings.append(
                    f"{key} missing"
                )
                confidence -= 3

        # ----------------------------
        # Confidence Floor
        # ----------------------------

        confidence = max(
            confidence,
            0
        )

        return {

            "status": (
                "Passed"
                if len(missing_fields) == 0
                else "Needs Review"
            ),

            "confidence": confidence,

            "missing_fields": missing_fields,

            "warnings": warnings
        }
