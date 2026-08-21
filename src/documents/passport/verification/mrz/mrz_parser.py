"""
ICAO TD3 passport MRZ parser.

TD3 passport MRZ:

    Line 1 = 44 characters
    Line 2 = 44 characters

This module parses the MRZ into structured fields.

It does NOT decide whether the passport is genuine.

That decision belongs to:
    MRZValidator
    Common document validation
    LOS/RCU workflow
"""

from __future__ import annotations

from typing import Any

from src.documents.passport.core.constants import (
    MRZ_LINE_LENGTH,
)

from src.documents.passport.verification.mrz.mrz_candidate import (
    MRZCandidate,
)

from src.documents.passport.verification.mrz.mrz_corrector import (
    MRZCorrector,
)


class MRZParser:
    """
    ICAO TD3 passport MRZ parser.
    """


    # ------------------------------------------------------------------
    # BASIC FIELD HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_filler(
        value: str,
    ) -> str:
        """
        Convert MRZ filler characters to normal text.

        Example:
            JOHN<<KUMAR<<<<
            ->
            JOHN KUMAR
        """

        value = str(
            value or ""
        )

        value = value.replace(
            "<",
            " ",
        )

        return " ".join(
            value.split()
        ).strip()


    @staticmethod
    def _clean_code(
        value: str,
    ) -> str:
        """
        Clean an MRZ code while preserving the actual code characters.
        """

        return str(
            value or ""
        ).replace(
            "<",
            "",
        ).strip()


    @staticmethod
    def _safe_slice(
        value: str,
        start: int,
        end: int,
    ) -> str:
        """
        Safe substring helper.
        """

        if not value:

            return ""

        return value[
            start:end
        ]


    # ------------------------------------------------------------------
    # NAME PARSING
    # ------------------------------------------------------------------

    @classmethod
    def _parse_names(
        cls,
        name_field: str,
    ) -> tuple[str, str]:
        """
        TD3 line-1 name structure:

            SURNAME<<GIVEN<NAMES<<<<

        Returns:
            surname
            given_names
        """

        value = (
            name_field
            .strip(
                "<"
            )
        )

        parts = value.split(
            "<<",
            1,
        )

        surname = (
            cls._clean_filler(
                parts[0]
            )
        )

        given_names = ""

        if len(parts) > 1:

            given_names = (
                cls._clean_filler(
                    parts[1]
                )
            )

        return (
            surname,
            given_names,
        )


    # ------------------------------------------------------------------
    # LINE 1
    # ------------------------------------------------------------------

    @classmethod
    def _parse_line_1(
        cls,
        line: str,
    ) -> dict[str, Any]:
        """
        Parse TD3 MRZ line 1.

        Layout:

            0-1    document type
            2-4    issuing country
            5-43   name
        """

        line = (
            line
            .ljust(
                MRZ_LINE_LENGTH,
                "<",
            )
            [
                :MRZ_LINE_LENGTH
            ]
        )

        document_type = (
            cls._safe_slice(
                line,
                0,
                2,
            )
        )

        issuing_country = (
            cls._safe_slice(
                line,
                2,
                5,
            )
        )

        name_field = (
            cls._safe_slice(
                line,
                5,
                44,
            )
        )

        surname, given_names = (
            cls._parse_names(
                name_field
            )
        )

        return {

            "document_type":
                document_type,

            "issuing_country":
                issuing_country,

            "surname":
                surname,

            "given_names":
                given_names,

            "name_raw":
                name_field,

        }


    # ------------------------------------------------------------------
    # LINE 2
    # ------------------------------------------------------------------

    @classmethod
    def _parse_line_2(
        cls,
        line: str,
    ) -> dict[str, Any]:
        """
        Parse TD3 MRZ line 2.

        Exact TD3 positions:

            0-8    passport number
            9      passport number check digit
            10-12  nationality
            13-18  date of birth
            19     DOB check digit
            20     sex
            21-26  date of expiry
            27     expiry check digit
            28-42  personal number
            43     composite check digit
        """

        line = (
            line
            .ljust(
                MRZ_LINE_LENGTH,
                "<",
            )
            [
                :MRZ_LINE_LENGTH
            ]
        )

        passport_number_raw = (
            cls._safe_slice(
                line,
                0,
                9,
            )
        )

        passport_number = (
            passport_number_raw
            .replace(
                "<",
                "",
            )
        )

        passport_number_check = (
            cls._safe_slice(
                line,
                9,
                10,
            )
        )

        nationality = (
            cls._safe_slice(
                line,
                10,
                13,
            )
        )

        birth_date = (
            cls._safe_slice(
                line,
                13,
                19,
            )
        )

        birth_date_check = (
            cls._safe_slice(
                line,
                19,
                20,
            )
        )

        sex = (
            cls._safe_slice(
                line,
                20,
                21,
            )
        )

        expiry_date = (
            cls._safe_slice(
                line,
                21,
                27,
            )
        )

        expiry_date_check = (
            cls._safe_slice(
                line,
                27,
                28,
            )
        )

        personal_number_raw = (
            cls._safe_slice(
                line,
                28,
                43,
            )
        )

        personal_number = (
            personal_number_raw
            .replace(
                "<",
                "",
            )
        )

        personal_number_check = (
            cls._safe_slice(
                line,
                43,
                44,
            )
        )

        return {

            # ----------------------------------------------------------
            # Passport number
            # ----------------------------------------------------------

            "passport_number":
                passport_number,

            # CRITICAL:
            # Preserve the raw 9-character MRZ field for checksum
            # calculation.
            "passport_number_raw":
                passport_number_raw,

            "passport_number_check":
                passport_number_check,

            # ----------------------------------------------------------
            # Nationality
            # ----------------------------------------------------------

            "nationality":
                nationality,

            # ----------------------------------------------------------
            # Birth date
            # ----------------------------------------------------------

            "birth_date":
                birth_date,

            "birth_date_check":
                birth_date_check,

            # ----------------------------------------------------------
            # Sex
            # ----------------------------------------------------------

            "sex":
                sex,

            # ----------------------------------------------------------
            # Expiry
            # ----------------------------------------------------------

            "expiry_date":
                expiry_date,

            "expiry_date_check":
                expiry_date_check,

            # ----------------------------------------------------------
            # Personal number
            # ----------------------------------------------------------

            "personal_number_raw":
                personal_number_raw,

            "personal_number":
                personal_number,

            "personal_number_check":
                personal_number_check,

            # ----------------------------------------------------------
            # Composite checksum
            # ----------------------------------------------------------

            "final_check":
                cls._safe_slice(
                    line,
                    43,
                    44,
                ),

            # Preserve complete line for debugging/audit.
            "raw_line":
                line,

        }


    # ------------------------------------------------------------------
    # MAIN PARSER
    # ------------------------------------------------------------------

    @classmethod
    def parse(
        cls,
        ocr_result: dict[str, Any]
        | list[str]
        | str
        | None,
    ) -> dict[str, Any]:
        """
        Parse MRZ from OCR output.

        Supported input:

            OCR result dictionary
            list[str]
            raw OCR string
        """

        # ---------------------------------------------------------------
        # Find candidate
        # ---------------------------------------------------------------

        if isinstance(
            ocr_result,
            dict,
        ):

            candidate = (
                MRZCandidate.find_from_ocr(
                    ocr_result
                )
            )

        elif isinstance(
            ocr_result,
            list,
        ):

            candidate = (
                MRZCandidate.find(
                    ocr_lines=ocr_result
                )
            )

        elif isinstance(
            ocr_result,
            str,
        ):

            candidate = (
                MRZCandidate.find(
                    ocr_lines=ocr_result.splitlines()
                )
            )

        else:

            candidate = []

        if len(candidate) != 2:

            return {

                "parsed": False,

                "valid_structure": False,

                "line_1": None,

                "line_2": None,

                "fields": {},

                "errors": [
                    "Two MRZ lines could not be identified."
                ],

            }


        # ---------------------------------------------------------------
        # OCR correction
        # ---------------------------------------------------------------

        corrected = (
            MRZCorrector.correct(
                candidate
            )
        )

        if len(corrected) != 2:

            return {

                "parsed": False,

                "valid_structure": False,

                "line_1": candidate[0],

                "line_2": candidate[1],

                "fields": {},

                "errors": [
                    "MRZ OCR correction failed."
                ],

            }


        line_1 = corrected[
            0
        ]

        line_2 = corrected[
            1
        ]


        # ---------------------------------------------------------------
        # Structural validation
        # ---------------------------------------------------------------

        line_1_length = (
            len(line_1)
        )

        line_2_length = (
            len(line_2)
        )

        structure_errors = []

        if line_1_length != (
            MRZ_LINE_LENGTH
        ):

            structure_errors.append(
                "MRZ line 1 is not 44 characters."
            )

        if line_2_length != (
            MRZ_LINE_LENGTH
        ):

            structure_errors.append(
                "MRZ line 2 is not 44 characters."
            )

        # Passport TD3 must start with P<.
        if not line_1.startswith(
            "P<"
        ):

            structure_errors.append(
                "MRZ does not start with P<."
            )

        # ---------------------------------------------------------------
        # Parse fields
        # ---------------------------------------------------------------

        fields_line_1 = (
            cls._parse_line_1(
                line_1
            )
        )

        fields_line_2 = (
            cls._parse_line_2(
                line_2
            )
        )

        fields = {
            **fields_line_1,
            **fields_line_2,
        }

        # ---------------------------------------------------------------
        # Structure result
        # ---------------------------------------------------------------

        valid_structure = (
            not structure_errors
        )

        return {

            "parsed": True,

            "valid_structure":
                valid_structure,

            "line_1":
                line_1,

            "line_2":
                line_2,

            "fields":
                fields,

            # -----------------------------------------------------------
            # Flatten fields for easier use by existing validators.
            # -----------------------------------------------------------

            "document_type":
                fields.get(
                    "document_type",
                    "",
                ),

            "issuing_country":
                fields.get(
                    "issuing_country",
                    "",
                ),

            "surname":
                fields.get(
                    "surname",
                    "",
                ),

            "given_names":
                fields.get(
                    "given_names",
                    "",
                ),

            "passport_number":
                fields.get(
                    "passport_number",
                    "",
                ),

            "passport_number_raw":
                fields.get(
                    "passport_number_raw",
                    "",
                ),

            "passport_number_check":
                fields.get(
                    "passport_number_check",
                    "",
                ),

            "nationality":
                fields.get(
                    "nationality",
                    "",
                ),

            "birth_date":
                fields.get(
                    "birth_date",
                    "",
                ),

            "birth_date_check":
                fields.get(
                    "birth_date_check",
                    "",
                ),

            "sex":
                fields.get(
                    "sex",
                    "",
                ),

            "expiry_date":
                fields.get(
                    "expiry_date",
                    "",
                ),

            "expiry_date_check":
                fields.get(
                    "expiry_date_check",
                    "",
                ),

            "personal_number_raw":
                fields.get(
                    "personal_number_raw",
                    "",
                ),

            "personal_number":
                fields.get(
                    "personal_number",
                    "",
                ),

            "personal_number_check":
                fields.get(
                    "personal_number_check",
                    "",
                ),

            "final_check":
                fields.get(
                    "final_check",
                    "",
                ),

            "errors":
                structure_errors,

        }