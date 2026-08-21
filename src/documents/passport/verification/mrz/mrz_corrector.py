"""
Passport MRZ OCR correction.

Purpose:
    Correct common OCR substitutions in ICAO TD3 MRZ data.

IMPORTANT:
    This module only proposes/carries out conservative OCR corrections.

    It does NOT declare the passport valid.

Final validity must come from MRZ checksum validation.

LOS principle:
    Never convert an uncertain OCR result directly into VERIFIED.
"""

from __future__ import annotations

from typing import Iterable

from src.documents.passport.core.constants import (
    MRZ_LINE_LENGTH,
    MRZ_ALLOWED_CHARACTERS,
)


class MRZCorrector:
    """
    Conservative MRZ OCR corrector.

    Corrections are intentionally limited because blindly replacing
    characters can create false passport numbers or dates.
    """

    # ------------------------------------------------------------------
    # Common OCR substitutions.
    #
    # These are only used where the expected MRZ field permits the
    # corresponding character type.
    # ------------------------------------------------------------------

    DIGIT_TO_LETTER = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
    }

    LETTER_TO_DIGIT = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }

    # Characters frequently confused with MRZ filler.
    FILLER_ALIASES = {
        ">": "<",
        "|": "<",
    }


    # ------------------------------------------------------------------
    # BASIC CLEANING
    # ------------------------------------------------------------------

    @classmethod
    def clean(
        cls,
        line: str,
    ) -> str:
        """
        Normalize an MRZ line without changing semantic characters.
        """

        value = str(
            line or ""
        ).upper()

        value = "".join(
            value.split()
        )

        for source, target in (
            cls.FILLER_ALIASES.items()
        ):

            value = value.replace(
                source,
                target,
            )

        value = "".join(
            char
            for char in value
            if char in MRZ_ALLOWED_CHARACTERS
        )

        return value


    # ------------------------------------------------------------------
    # LINE 1
    # ------------------------------------------------------------------

    @classmethod
    def correct_line_1(
        cls,
        line: str,
    ) -> str:
        """
        Correct the passport MRZ first line.

        TD3 line 1:

            P<ISSUER<<<<SURNAME<<GIVENNAMES<<<<<<

        The first three characters are especially important:

            P<IND

        We only perform very conservative correction here.
        """

        value = cls.clean(
            line
        )

        if not value:

            return value

        # --------------------------------------------------------------
        # Correct the document marker.
        # --------------------------------------------------------------

        if len(value) >= 1:

            if value[0] in {
                "8",
                "B",
                "R",
                "F",
            }:

                value = (
                    "P"
                    +
                    value[1:]
                )

        if len(value) >= 2:

            if value[1] in {
                ">",
                "|",
            }:

                value = (
                    value[0]
                    +
                    "<"
                    +
                    value[2:]
                )

        # --------------------------------------------------------------
        # India issuer code.
        #
        # If the OCR output already strongly resembles IND, correct only
        # obvious substitutions.
        # --------------------------------------------------------------

        if len(value) >= 5:

            prefix = value[
                :5
            ]

            # P<IND
            if prefix.startswith(
                "P<"
            ):

                issuer = prefix[
                    2:5
                ]

                corrected = list(
                    issuer
                )

                expected = (
                    "IND"
                )

                for index in range(
                    3
                ):

                    if (
                        corrected[
                            index
                        ]
                        in
                        cls.DIGIT_TO_LETTER
                    ):

                        replacement = (
                            cls.DIGIT_TO_LETTER[
                                corrected[
                                    index
                                ]
                            ]
                        )

                        # Only apply if it matches expected issuer.
                        if (
                            replacement
                            ==
                            expected[
                                index
                            ]
                        ):

                            corrected[
                                index
                            ] = replacement

                value = (
                    value[:2]
                    +
                    "".join(
                        corrected
                    )
                    +
                    value[5:]
                )

        return value[
            :MRZ_LINE_LENGTH
        ]


    # ------------------------------------------------------------------
    # LINE 2
    # ------------------------------------------------------------------

    @classmethod
    def correct_line_2(
        cls,
        line: str,
    ) -> str:
        """
        Conservative correction for TD3 line 2.

        TD3 line 2 layout:

            0-8    passport number
            9      passport number check
            10-12  nationality
            13-18  DOB
            19     DOB check
            20     sex
            21-26  expiry
            27     expiry check
            28-42  personal number
            43     composite check

        We do NOT blindly change every character because the passport
        number itself can contain letters and digits.
        """

        value = cls.clean(
            line
        )

        if not value:

            return value

        # --------------------------------------------------------------
        # Passport number
        # --------------------------------------------------------------

        if len(value) >= 9:

            passport_number = (
                value[:9]
            )

            corrected = []

            for char in passport_number:

                # In an alphanumeric passport number, these substitutions
                # are possible, but we only correct obvious OCR cases.
                if char in {
                    "O",
                    "I",
                    "Z",
                    "S",
                    "B",
                    "G",
                }:

                    corrected.append(
                        char
                    )

                else:

                    corrected.append(
                        char
                    )

            value = (
                "".join(
                    corrected
                )
                +
                value[9:]
            )

        # --------------------------------------------------------------
        # Check digits
        #
        # These positions MUST be digits.
        # If OCR returned an obvious letter, convert it.
        # --------------------------------------------------------------

        check_digit_positions = {
            9,
            19,
            27,
            43,
        }

        chars = list(
            value
        )

        for position in (
            check_digit_positions
        ):

            if position >= len(
                chars
            ):

                continue

            char = chars[
                position
            ]

            if char in cls.LETTER_TO_DIGIT:

                chars[
                    position
                ] = cls.LETTER_TO_DIGIT[
                    char
                ]

        value = "".join(
            chars
        )

        # --------------------------------------------------------------
        # DOB / EXPIRY
        #
        # These positions MUST be digits.
        # --------------------------------------------------------------

        date_positions = (
            set(
                range(
                    13,
                    19,
                )
            )
            |
            set(
                range(
                    21,
                    27,
                )
            )
        )

        chars = list(
            value
        )

        for position in date_positions:

            if position >= len(
                chars
            ):

                continue

            char = chars[
                position
            ]

            if char in cls.LETTER_TO_DIGIT:

                chars[
                    position
                ] = cls.LETTER_TO_DIGIT[
                    char
                ]

        value = "".join(
            chars
        )

        return value[
            :MRZ_LINE_LENGTH
        ]


    # ------------------------------------------------------------------
    # FULL MRZ
    # ------------------------------------------------------------------

    @classmethod
    def correct(
        cls,
        lines: Iterable[str],
    ) -> list[str]:
        """
        Correct a two-line MRZ candidate.

        Returns the corrected candidate.

        It does NOT validate it.
        """

        values = list(
            lines
        )

        if len(values) < 2:

            return []

        line_1 = (
            cls.correct_line_1(
                values[0]
            )
        )

        line_2 = (
            cls.correct_line_2(
                values[1]
            )
        )

        return [
            line_1,
            line_2,
        ]