"""
Passport MRZ candidate detection.

Purpose:
    Find the two ICAO TD3 MRZ lines from OCR output.

Important:
    This file ONLY finds candidates.

    It does NOT decide whether the MRZ is valid.

Validation happens later using:
        - MRZ structure
        - ICAO check digits
        - passport number checksum
        - DOB checksum
        - expiry checksum
        - composite checksum
"""

from __future__ import annotations

import re
from typing import Any

from src.documents.passport.core.constants import (
    MRZ_LINE_LENGTH,
)


class MRZCandidate:
    """
    Fast ICAO TD3 MRZ candidate detector.
    """


    @staticmethod
    def clean_line(
        text: str,
    ) -> str:
        """
        Normalize OCR output into MRZ-compatible characters.

        OCR commonly produces:
            > instead of <
            | instead of <
            spaces inside MRZ

        We normalize those here.

        NOTE:
            We do NOT blindly replace letters/numbers here.
            Character-level corrections belong in mrz_corrector.py because
            they require field context.
        """

        value = (
            str(text or "")
            .upper()
            .strip()
        )

        # Remove spaces first.
        value = re.sub(
            r"\s+",
            "",
            value,
        )

        # Common OCR substitutions for the MRZ filler character.
        value = value.replace(
            "|",
            "<",
        )

        value = value.replace(
            ">",
            "<",
        )

        # Remove characters that cannot occur in TD3 MRZ.
        value = re.sub(
            r"[^A-Z0-9<]",
            "",
            value,
        )

        return value


    @classmethod
    def _looks_like_mrz_line(
        cls,
        line: str,
    ) -> bool:
        """
        Check whether a line has the basic shape of an MRZ line.

        This is intentionally permissive because OCR can produce a few
        incorrect characters.

        We do NOT perform check-digit validation here.
        """

        clean = cls.clean_line(
            line
        )

        if len(clean) < 30:

            return False

        # TD3 passport MRZ lines are 44 characters.
        # OCR can truncate a few characters, so allow a minimum candidate
        # length here.
        if len(clean) > MRZ_LINE_LENGTH:

            clean = clean[
                :MRZ_LINE_LENGTH
            ]

        allowed = sum(
            1
            for char in clean
            if (
                char.isalpha()
                or
                char.isdigit()
                or
                char == "<"
            )
        )

        ratio = (
            allowed
            /
            max(
                len(clean),
                1,
            )
        )

        return (
            ratio >= 0.95
        )


    @classmethod
    def _normalize_length(
        cls,
        line: str,
    ) -> str | None:
        """
        Convert a candidate into a 44-character TD3 line when possible.

        We do NOT pad arbitrary short OCR strings because doing that could
        create false MRZs.

        A line must contain at least 40 characters before we accept it.
        """

        clean = cls.clean_line(
            line
        )

        if len(clean) < 40:

            return None

        if len(clean) >= MRZ_LINE_LENGTH:

            return clean[
                :MRZ_LINE_LENGTH
            ]

        # Only four missing characters are tolerated.
        # Padding is deferred to the correction/validation layer.
        return clean


    @classmethod
    def find(
        cls,
        ocr_results: list[
            dict[str, Any]
        ]
        | None = None,
        ocr_lines: list[str]
        | None = None,
    ) -> list[str]:
        """
        Find the most likely TD3 MRZ pair.

        Preferred input:
            OCR result objects containing text/confidence/bbox.

        Fallback:
            plain OCR lines.

        Returns:
            []                       -> no MRZ candidate
            [line1, line2]           -> candidate MRZ
        """

        candidates: list[
            dict[str, Any]
        ] = []

        # ---------------------------------------------------------------
        # Build candidate objects.
        # ---------------------------------------------------------------

        if ocr_results:

            for item in ocr_results:

                if not isinstance(
                    item,
                    dict,
                ):

                    continue

                text = str(
                    item.get(
                        "text",
                        "",
                    )
                )

                if not text:

                    continue

                if cls._looks_like_mrz_line(
                    text
                ):

                    candidates.append(
                        {
                            "text": cls.clean_line(
                                text
                            ),
                            "confidence": float(
                                item.get(
                                    "confidence",
                                    0.0,
                                )
                                or 0.0
                            ),
                            "bbox": item.get(
                                "bbox"
                            ),
                        }
                    )

        elif ocr_lines:

            for line in ocr_lines:

                if cls._looks_like_mrz_line(
                    line
                ):

                    candidates.append(
                        {
                            "text": cls.clean_line(
                                line
                            ),
                            "confidence": 0.0,
                            "bbox": None,
                        }
                    )

        if not candidates:

            return []


        # ---------------------------------------------------------------
        # Strongest candidate first.
        #
        # Passport MRZ should be near the bottom of the identity page.
        # If bounding boxes are available, use vertical position as a
        # secondary ranking signal.
        # ---------------------------------------------------------------

        def sort_key(
            item: dict[str, Any]
        ):

            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
                or 0.0
            )

            bbox = item.get(
                "bbox"
            )

            y_position = 0.0

            try:

                if bbox:

                    # rec_boxes commonly look like:
                    # [x1, y1, x2, y2]
                    if (
                        isinstance(
                            bbox,
                            (list, tuple),
                        )
                        and
                        len(bbox) >= 4
                    ):

                        y_position = float(
                            bbox[3]
                        )

            except (
                TypeError,
                ValueError,
            ):

                y_position = 0.0

            return (
                confidence,
                y_position,
                len(
                    item[
                        "text"
                    ]
                ),
            )


        candidates.sort(
            key=sort_key,
            reverse=True,
        )


        # ---------------------------------------------------------------
        # Preferred passport marker:
        #
        # The first MRZ line normally starts with P<.
        # ---------------------------------------------------------------

        first_line_index = None

        for index, item in enumerate(
            candidates
        ):

            text = item[
                "text"
            ]

            if text.startswith(
                "P<"
            ):

                first_line_index = index

                break


        if first_line_index is not None:

            first = candidates[
                first_line_index
            ][
                "text"
            ]

            # Search for the second line.
            for index, item in enumerate(
                candidates
            ):

                if index == first_line_index:

                    continue

                second = item[
                    "text"
                ]

                if len(second) >= 40:

                    return [
                        first[
                            :MRZ_LINE_LENGTH
                        ],
                        second[
                            :MRZ_LINE_LENGTH
                        ],
                    ]


        # ---------------------------------------------------------------
        # Fallback:
        #
        # Search every candidate pair.
        # ---------------------------------------------------------------

        for first_index in range(
            len(candidates)
        ):

            first = candidates[
                first_index
            ][
                "text"
            ]

            if len(first) < 40:

                continue

            for second_index in range(
                first_index + 1,
                len(candidates),
            ):

                second = candidates[
                    second_index
                ][
                    "text"
                ]

                if len(second) < 40:

                    continue

                return [
                    first[
                        :MRZ_LINE_LENGTH
                    ],
                    second[
                        :MRZ_LINE_LENGTH
                    ],
                ]

        return []


    @classmethod
    def find_from_ocr(
        cls,
        ocr_result: dict[str, Any],
    ) -> list[str]:
        """
        Convenience method for the OCR engine output.
        """

        if not isinstance(
            ocr_result,
            dict,
        ):

            return []

        return cls.find(
            ocr_results=ocr_result.get(
                "results",
                [],
            ),
            ocr_lines=ocr_result.get(
                "lines",
                [],
            ),
        )