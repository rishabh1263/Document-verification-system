"""
ITR Field Extraction.

Current extraction scope:
    1. Name
    2. PAN
    3. Assessment Year
    4. Date of Birth
    5. Total Income
    6. Business / Profession Income

The extractor is independent of OCR and accepts text produced by
native PDF extraction or OCR.
"""

from __future__ import annotations

import re


class ITRExtractor:
    """Generic ITR field extractor."""

    _NAME_PATTERNS = (
        re.compile(
            r"\bname\s+of\s+(?:the\s+)?assessee"
            r"\s*[:\-]?\s*"
            r"([A-Za-z][A-Za-z .'\-&]{2,100})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bname\s*[:\-]\s*"
            r"([A-Za-z][A-Za-z .'\-&]{2,100})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bname\b\s*\n+\s*"
            r"([A-Za-z][A-Za-z .'\-&]{2,100})"
            r"(?=\s*\n)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bname\b\s{2,}"
            r"([A-Za-z][A-Za-z .'\-&]{2,100})"
            r"(?=\s+(?:address|status|form\s+number|pan)\b)",
            re.IGNORECASE,
        ),
    )

    _PAN_PATTERN = re.compile(
        r"\b([A-Z]{5}[0-9]{4}[A-Z])\b",
        re.IGNORECASE,
    )

    _SPACED_PAN_PATTERN = re.compile(
        r"\b([A-Z])\s*([A-Z])\s*([A-Z])\s*([A-Z])\s*([A-Z])\s*"
        r"([0-9])\s*([0-9])\s*([0-9])\s*([0-9])\s*([A-Z])\b",
        re.IGNORECASE,
    )

    _ASSESSMENT_YEAR_PATTERN = re.compile(
        r"\b(?:assessment\s+year|a\.?\s*y\.?)"
        r"\s*[:\-]?\s*"
        r"((?:19|20)\d{2}\s*[-/]\s*\d{2})\b",
        re.IGNORECASE,
    )

    _ASSESSMENT_YEAR_SEPARATE_PATTERN = re.compile(
        r"\b(?:assessment\s+year|a\.?\s*y\.?)\b"
        r"\s*[:\-]?\s*\n+\s*"
        r"((?:19|20)\d{2}\s*[-/]\s*\d{2})\b",
        re.IGNORECASE,
    )

    _DOB_VALUE_PATTERN = re.compile(
        r"(?<!\d)("
        r"\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{4}"
        r"|"
        r"\d{1,2}\s*[-/]\s*"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s*[-/]\s*\d{4}"
        r")(?!\d)",
        re.IGNORECASE,
    )

    _DOB_PATTERN = re.compile(
        r"\b(?:date\s+of\s+birth|dob)"
        r"\s*[:\-]?\s*("
        r"[0-9]{1,2}\s*[/\-]\s*[0-9]{1,2}\s*[/\-]\s*[0-9]{4}"
        r"|"
        r"[0-9]{1,2}\s*[-/]\s*"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s*[-/]\s*[0-9]{4}"
        r")\b",
        re.IGNORECASE,
    )

    _DOB_SEPARATE_PATTERN = re.compile(
        r"\b(?:date\s+of\s+birth|dob)\b"
        r"\s*[:\-]?\s*\n+\s*("
        r"[0-9]{1,2}\s*[/\-]\s*[0-9]{1,2}\s*[/\-]\s*[0-9]{4}"
        r"|"
        r"[0-9]{1,2}\s*[-/]\s*"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s*[-/]\s*[0-9]{4}"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def extract_name(cls, text: str) -> str | None:
        """Extract taxpayer / assessee name."""
        if not text or not text.strip():
            return None

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)

        for pattern in cls._NAME_PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue

            name = cls._clean_name(match.group(1))
            if cls._is_valid_name(name):
                return name

        return None

    @classmethod
    def extract_pan(cls, text: str) -> str | None:
        """Extract PAN from ITR text."""
        if not text or not text.strip():
            return None

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        match = cls._PAN_PATTERN.search(normalized.upper())
        if match:
            return match.group(1).upper()

        match = cls._SPACED_PAN_PATTERN.search(normalized.upper())
        if match:
            return "".join(match.groups()).upper()

        return None

    @classmethod
    def extract_assessment_year(cls, text: str) -> str | None:
        """Extract and normalize Assessment Year."""
        if not text or not text.strip():
            return None

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        match = cls._ASSESSMENT_YEAR_PATTERN.search(normalized)
        if match:
            return cls._normalize_assessment_year(match.group(1))

        match = cls._ASSESSMENT_YEAR_SEPARATE_PATTERN.search(normalized)
        if match:
            return cls._normalize_assessment_year(match.group(1))

        return None

    @staticmethod
    def _normalize_assessment_year(value: str | None) -> str | None:
        if not value:
            return None

        value = re.sub(r"\s+", "", value).replace("/", "-")

        match = re.fullmatch(r"((?:19|20)\d{2})-(\d{2})", value)
        if not match:
            return None

        return f"{match.group(1)}-{match.group(2)}"

    @classmethod
    def extract_dob(
        cls,
        text: str,
    ) -> str | None:
        """
        Extract taxpayer Date of Birth from ITR text.

        Supports explicit DOB formats and ITR table-style
        native PDF extraction where multiple date-like values
        may occur near the Date of Birth field.

        For the affected table layout, colon-prefixed values
        are preferred over an earlier unlabelled date.
        """

        if not text or not text.strip():
            return None

        normalized = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        # ======================================================
        # 1. SAME-LINE EXPLICIT DOB
        # ======================================================

        same_line_pattern = re.compile(
            r"\b(?:date\s+of\s+birth|dob)"
            r"\s*[:\-]\s*"
            r"("
            r"\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{4}"
            r"|"
            r"\d{1,2}\s*[-/]\s*"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s*[-/]\s*\d{4}"
            r")\b",
            re.IGNORECASE,
        )

        match = same_line_pattern.search(normalized)

        if match:
            return cls._normalize_dob(match.group(1))

        # ======================================================
        # 2. TABLE-STYLE EXTRACTION
        # ======================================================

        lines = normalized.splitlines()

        for index, raw_line in enumerate(lines):

            if not re.fullmatch(
                r"\s*date\s+of\s+birth\s*",
                raw_line,
                re.IGNORECASE,
            ):
                continue

            candidates: list[tuple[int, str, bool]] = []

            # Native PDF extraction can reorder table columns.
            # Inspect a bounded region instead of taking the
            # first date blindly.
            for candidate_index in range(
                index + 1,
                min(len(lines), index + 16),
            ):
                candidate = lines[candidate_index].strip()

                if not candidate:
                    continue

                # Stop at clearly unrelated sections.
                if re.fullmatch(
                    r"(?:statement\s+of\s+income|"
                    r"schedule\s+\d+|"
                    r"bank\s+accounts?)",
                    candidate,
                    re.IGNORECASE,
                ):
                    break

                date_match = cls._DOB_VALUE_PATTERN.search(
                    candidate
                )

                if not date_match:
                    continue

                value = cls._normalize_dob(
                    date_match.group(1)
                )

                if value is None:
                    continue

                # In the affected ITR table layout, a colon
                # prefix identifies the actual value-column date.
                is_colon_value = candidate.startswith(":")

                candidates.append(
                    (
                        candidate_index,
                        value,
                        is_colon_value,
                    )
                )

            if not candidates:
                continue

            # Prefer the colon-prefixed table value.
            colon_candidates = [
                item
                for item in candidates
                if item[2]
            ]

            if colon_candidates:
                return min(
                    colon_candidates,
                    key=lambda item: item[0],
                )[1]

            # Otherwise use the nearest valid candidate.
            return min(
                candidates,
                key=lambda item: item[0],
            )[1]

        return None

    @staticmethod
    def _normalize_dob(value: str | None) -> str | None:
        """Normalize DOB to DD/MM/YYYY and validate date ranges."""
        if not value:
            return None

        value = re.sub(r"\s+", "", value)

        numeric = re.fullmatch(
            r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})",
            value,
        )

        if numeric:
            day = int(numeric.group(1))
            month = int(numeric.group(2))
            year = int(numeric.group(3))
        else:
            named = re.fullmatch(
                r"(\d{1,2})[-/]"
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[-/](\d{4})",
                value,
                re.IGNORECASE,
            )

            if not named:
                return None

            months = {
                "jan": 1, "feb": 2, "mar": 3,
                "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9,
                "oct": 10, "nov": 11, "dec": 12,
            }

            day = int(named.group(1))
            month = months[named.group(2).casefold()]
            year = int(named.group(3))

        if not 1 <= day <= 31:
            return None

        if not 1 <= month <= 12:
            return None

        if not 1900 <= year <= 2100:
            return None

        return f"{day:02d}/{month:02d}/{year:04d}"

    # ======================================================
    # BUSINESS / PROFESSION INCOME EXTRACTION
    # ======================================================

    _BUSINESS_LABEL_PATTERN = re.compile(
        r"(?:"
        r"\bbusiness\s*:\s*"
        r"|"
        r"\bbusiness\s+and\s+profession\b"
        r"|"
        r"\bincome\s+chargeable\s+under\s+the\s+head"
        r"\s*[\"“]?\s*business\s+and\s+profession"
        r")",
        re.IGNORECASE,
    )

    @classmethod
    def extract_business_income(
        cls,
        text: str,
    ) -> int | None:
        """
        Extract income attributable to Business / Profession.

        Handles ITR table layouts such as:

            Business: Presumptive profits u/s 44AD
            1
            4,98,000

        and:

            Income chargeable under the head "Business and Profession"
            4,98,000

        Important:
            ITR tables may contain a serial/index number (for example
            "1") between the field label and the actual amount.
            The extractor therefore collects nearby numeric candidates
            and ranks actual income amounts above table indexes.
        """

        if not text or not text.strip():
            return None

        normalized = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        lines = [
            line.strip()
            for line in normalized.splitlines()
            if line.strip()
        ]

        # --------------------------------------------------
        # 1. Same-line format:
        # Business: ... 498000
        # --------------------------------------------------

        same_line = re.compile(
            r"\bbusiness\s*:\s*.*?"
            r"(?:₹\s*)?"
            r"([0-9][0-9,\s]*)\s*$",
            re.IGNORECASE,
        )

        for line in lines:
            match = same_line.search(line)

            if not match:
                continue

            value = cls._normalize_income(
                match.group(1)
            )

            if value is not None:
                return value

        # --------------------------------------------------
        # 2. Table-style format.
        # --------------------------------------------------

        for index, line in enumerate(lines):

            if not cls._BUSINESS_LABEL_PATTERN.search(
                line
            ):
                continue

            candidates: list[tuple[int, int, bool]] = []

            for offset, candidate in enumerate(
                lines[index + 1:index + 9],
                start=1,
            ):
                # Skip obvious non-value labels.
                if re.search(
                    r"\b(?:tax|rebate|interest|"
                    r"deduction|assessment\s+year|"
                    r"total\s+income\s+as\s+per)"
                    r"\b",
                    candidate,
                    re.IGNORECASE,
                ):
                    continue

                amount_match = re.fullmatch(
                    r"(?:₹\s*)?"
                    r"([0-9][0-9,\s]*)",
                    candidate,
                )

                if not amount_match:
                    continue

                raw_amount = amount_match.group(1)

                value = cls._normalize_income(
                    raw_amount
                )

                if value is None:
                    continue

                # A comma-formatted amount such as 4,98,000 is
                # much more likely to be the actual rupee value
                # than a table serial number such as "1".
                has_grouping = "," in raw_amount

                candidates.append(
                    (
                        value,
                        offset,
                        has_grouping,
                    )
                )

            if not candidates:
                continue

            # Priority:
            #   1. Properly grouped Indian monetary amount
            #      e.g. 4,98,000
            #   2. Larger ungrouped amount
            #      e.g. 498000
            #   3. Small serial/index values last.
            #
            # Within the same category, use the nearest value.
            candidates.sort(
                key=lambda item: (
                    0 if item[2] else 1,
                    0 if item[0] >= 1000 else 1,
                    item[1],
                )
            )

            return candidates[0][0]

        return None

    @classmethod
    def extract_total_income(cls, text: str) -> int | None:
        """
        Extract actual tax-computation Total Income.

        Excludes:
            Total Income as per Updated return
            Total Income as per earlier return
            Tax on total income
        """
        if not text or not text.strip():
            return None

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Label and value on the same line.
        same_line = re.compile(
            r"(?:■\s*)?total\s+income"
            r"\s*[:\-]\s*"
            r"(?:₹\s*)?"
            r"([0-9][0-9,\s]*)"
            r"\s*$",
            re.IGNORECASE,
        )

        for line in normalized.splitlines():
            match = same_line.search(line.strip())
            if match:
                value = cls._normalize_income(match.group(1))
                if value is not None:
                    return value

        # Table format:
        # ■ Total Income
        # 4,98,000
        lines = [
            line.strip()
            for line in normalized.splitlines()
            if line.strip()
        ]

        for index, line in enumerate(lines):
            if not re.fullmatch(
                r"(?:■\s*)?total\s+income",
                line,
                re.IGNORECASE,
            ):
                continue

            for candidate in lines[index + 1:index + 3]:
                value = cls._normalize_income(candidate)
                if value is not None:
                    return value

                if re.search(r"[A-Za-z]", candidate):
                    break

        return None

    @staticmethod
    def _normalize_income(value: str | None) -> int | None:
        if not value:
            return None

        value = value.replace("₹", "")
        value = re.sub(r"\s+", "", value)
        value = value.replace(",", "")

        if not re.fullmatch(r"\d+", value):
            return None

        return int(value)

    @staticmethod
    def _clean_name(value: str | None) -> str | None:
        if not value:
            return None

        value = re.sub(r"\s+", " ", value).strip(" :-\t\r\n")

        value = re.split(
            r"\b(?:address|status|form\s+number|pan|"
            r"assessment\s+year|father'?s\s+name)\b",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        value = value.strip(" :-\t\r\n")
        return value or None

    @staticmethod
    def _is_valid_name(value: str | None) -> bool:
        if not value:
            return False

        if len(value) < 3 or len(value) > 100:
            return False

        if not re.search(r"[A-Za-z]", value):
            return False

        blocked = {
            "name",
            "address",
            "status",
            "individual",
            "company",
            "firm",
            "pan",
            "form number",
            "assessment year",
            "income",
            "total income",
            "verification",
            "acknowledgement",
        }

        if value.casefold() in blocked:
            return False

        return not bool(re.search(r"\d", value))


# ==========================================================
# PUBLIC CONVENIENCE FUNCTIONS
# ==========================================================

def extract_name(text: str) -> str | None:
    return ITRExtractor.extract_name(text)


def extract_pan(text: str) -> str | None:
    return ITRExtractor.extract_pan(text)


def extract_assessment_year(text: str) -> str | None:
    return ITRExtractor.extract_assessment_year(text)


def extract_dob(text: str) -> str | None:
    return ITRExtractor.extract_dob(text)


def extract_total_income(text: str) -> int | None:
    return ITRExtractor.extract_total_income(text)


def extract_business_income(text: str) -> int | None:
    return ITRExtractor.extract_business_income(text)