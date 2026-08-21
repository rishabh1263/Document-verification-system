"""
ITR Consistency Validator

Validates internal consistency of an Income Tax Return.

Checks:
    - Assessment Year
    - Taxpayer PAN
    - Income information
    - Tax information
    - Acknowledgement information

Important:
    This validator is intentionally conservative.

    It does NOT compare unrelated financial fields such as:
        tax payable vs tax paid
        tax due vs tax payable
        business income vs total income

    Those are different accounting concepts.

Author : SBFC Document Intelligence
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from time import perf_counter
from typing import Iterable, List, Optional, Sequence

try:
    from .models import ConsistencyResult
except ImportError:
    from src.documents.itr.validation.models import ConsistencyResult


logger = logging.getLogger(__name__)


class ConsistencyValidator:
    """
    Internal consistency validator for ITR documents.
    """

    # ==========================================================
    # PUBLIC VALIDATION
    # ==========================================================

    def validate(self, text: str) -> ConsistencyResult:
        start_time = perf_counter()

        assessment_year_consistent = True
        pan_consistent = True
        income_consistent = True
        tax_consistent = True
        acknowledgement_consistent = True

        inconsistencies: List[str] = []
        reasons: List[str] = []

        try:
            normalized_text = self._normalize_text(text)

            if not normalized_text:
                return self._result(
                    consistent=False,
                    assessment_year_consistent=False,
                    pan_consistent=False,
                    income_consistent=False,
                    tax_consistent=False,
                    acknowledgement_consistent=False,
                    score=0.0,
                    inconsistencies=[
                        "Document text is empty"
                    ],
                    reasons=[
                        "Consistency validation requires document text"
                    ],
                    start_time=start_time,
                )

            # ==================================================
            # ASSESSMENT YEAR
            # ==================================================

            assessment_year_consistent = (
                self._check_assessment_year(
                    normalized_text
                )
            )

            if assessment_year_consistent:
                reasons.append(
                    "Assessment year is consistent"
                )
            else:
                inconsistencies.append(
                    "Assessment year is inconsistent"
                )
                reasons.append(
                    "Assessment year is inconsistent"
                )

            # ==================================================
            # PAN
            # ==================================================

            pan_consistent = self._check_pan(
                normalized_text
            )

            if pan_consistent:
                reasons.append(
                    "PAN is consistent"
                )
            else:
                inconsistencies.append(
                    "PAN is inconsistent"
                )
                reasons.append(
                    "PAN is inconsistent"
                )

            # ==================================================
            # INCOME
            # ==================================================

            income_consistent = self._check_income(
                normalized_text
            )

            if income_consistent:
                reasons.append(
                    "Income information is consistent"
                )
            else:
                inconsistencies.append(
                    "Income information could not be verified as consistent"
                )
                reasons.append(
                    "Income information is inconsistent"
                )

            # ==================================================
            # TAX
            # ==================================================

            tax_consistent = self._check_tax(
                normalized_text
            )

            if tax_consistent:
                reasons.append(
                    "Tax information is consistent"
                )
            else:
                inconsistencies.append(
                    "Tax information could not be verified as consistent"
                )
                reasons.append(
                    "Tax information is inconsistent"
                )

            # ==================================================
            # ACKNOWLEDGEMENT
            # ==================================================

            acknowledgement_consistent = (
                self._check_acknowledgement(
                    normalized_text
                )
            )

            if acknowledgement_consistent:
                reasons.append(
                    "Acknowledgement information is consistent"
                )
            else:
                inconsistencies.append(
                    "Acknowledgement information could not be verified as consistent"
                )
                reasons.append(
                    "Acknowledgement information is inconsistent"
                )

            # ==================================================
            # SCORE
            # ==================================================

            checks = [
                assessment_year_consistent,
                pan_consistent,
                income_consistent,
                tax_consistent,
                acknowledgement_consistent,
            ]

            score = (
                sum(
                    1.0
                    for check in checks
                    if check
                )
                / len(checks)
            )

            return self._result(
                consistent=all(checks),
                assessment_year_consistent=(
                    assessment_year_consistent
                ),
                pan_consistent=pan_consistent,
                income_consistent=income_consistent,
                tax_consistent=tax_consistent,
                acknowledgement_consistent=(
                    acknowledgement_consistent
                ),
                score=score,
                inconsistencies=inconsistencies,
                reasons=reasons,
                start_time=start_time,
            )

        except Exception as exc:

            logger.exception(
                "ITR consistency validation failed"
            )

            return self._result(
                consistent=False,
                assessment_year_consistent=(
                    assessment_year_consistent
                ),
                pan_consistent=pan_consistent,
                income_consistent=income_consistent,
                tax_consistent=tax_consistent,
                acknowledgement_consistent=(
                    acknowledgement_consistent
                ),
                score=0.0,
                inconsistencies=[
                    f"Consistency validation error: {exc}"
                ],
                reasons=[
                    f"Consistency validation error: {exc}"
                ],
                start_time=start_time,
            )

    # ==========================================================
    # ASSESSMENT YEAR
    # ==========================================================

    @classmethod
    def _check_assessment_year(
        cls,
        text: str,
    ) -> bool:

        patterns = [
            r"\bassessment\s+year\b",
            r"\basst\.?\s+year\b",
            r"\ba\.?\s*y\.?\b",
        ]

        values: List[str] = []

        lines = text.splitlines()

        for index, line in enumerate(lines):

            if not any(
                re.search(
                    pattern,
                    line,
                    re.IGNORECASE,
                )
                for pattern in patterns
            ):
                continue

            # Search current line + next few lines.
            block = "\n".join(
                lines[index:index + 4]
            )

            matches = re.findall(
                r"\b(20\d{2})\s*[-/]\s*(\d{2,4})\b",
                block,
                flags=re.IGNORECASE,
            )

            for first, second in matches:

                if len(second) == 2:
                    second = first[:2] + second

                values.append(
                    f"{first}-{second}"
                )

        values = cls._unique(values)

        if not values:
            return True

        return len(values) == 1

    # ==========================================================
    # PAN
    # ==========================================================

    @classmethod
    def _check_pan(
        cls,
        text: str,
    ) -> bool:
        """
        Extract ONLY taxpayer PAN occurrences.

        Critical:
            The ITR acknowledgement contains the verifier's PAN
            after:

                verified by <person>
                having PAN <PAN>

            That PAN must NOT be compared with the taxpayer PAN.

        Therefore we do NOT scan every PAN-looking string in the
        document.
        """

        taxpayer_pans: List[str] = []

        lines = text.splitlines()

        for index, line in enumerate(lines):

            clean = cls._normalize_space(line)

            # --------------------------------------------------
            # Explicit taxpayer PAN label.
            # --------------------------------------------------

            if re.fullmatch(
                r"pan",
                clean,
                flags=re.IGNORECASE,
            ):

                block = "\n".join(
                    lines[index:index + 4]
                )

                matches = re.findall(
                    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                    block,
                    flags=re.IGNORECASE,
                )

                for value in matches:
                    taxpayer_pans.append(
                        value.upper()
                    )

                continue

            # --------------------------------------------------
            # PAN: XXXXX9999X
            # --------------------------------------------------

            match = re.search(
                r"\bpan\b\s*[:\-]\s*"
                r"([A-Z]{5}[0-9]{4}[A-Z])\b",
                clean,
                flags=re.IGNORECASE,
            )

            if match:
                taxpayer_pans.append(
                    match.group(1).upper()
                )

        taxpayer_pans = cls._unique(
            taxpayer_pans
        )

        # ------------------------------------------------------
        # If taxpayer PAN is not extractable, consistency cannot
        # prove a contradiction.
        # Content validation handles presence.
        # ------------------------------------------------------

        if not taxpayer_pans:
            return True

        return len(taxpayer_pans) == 1

    # ==========================================================
    # INCOME
    # ==========================================================

    @classmethod
    def _check_income(
        cls,
        text: str,
    ) -> bool:
        """
        Validate repeated occurrences of the same income field.

        The extractor is line-aware so table serial numbers such
        as:

            1
            2
            3

        are never interpreted as the actual financial value.
        """

        # ------------------------------------------------------
        # TOTAL INCOME - UPDATED RETURN
        # ------------------------------------------------------

        updated_total_income = (
            cls._extract_labeled_amounts(
                text,
                [
                    r"total\s+income\s+as\s+per\s+updated\s+return",
                ],
            )
        )

        if not cls._values_are_consistent(
            updated_total_income
        ):
            return False

        # ------------------------------------------------------
        # TOTAL INCOME - EARLIER RETURN
        # ------------------------------------------------------

        earlier_total_income = (
            cls._extract_labeled_amounts(
                text,
                [
                    r"total\s+income\s+as\s+per\s+earlier\s+return",
                ],
            )
        )

        if not cls._values_are_consistent(
            earlier_total_income
        ):
            return False

        # ------------------------------------------------------
        # GROSS TOTAL INCOME
        # ------------------------------------------------------

        gross_total_income = (
            cls._extract_labeled_amounts(
                text,
                [
                    r"gross\s+total\s+income",
                ],
            )
        )

        if not cls._values_are_consistent(
            gross_total_income
        ):
            return False

        # ------------------------------------------------------
        # INCOME CHARGEABLE UNDER BUSINESS / PROFESSION
        # ------------------------------------------------------

        business_income = (
            cls._extract_labeled_amounts(
                text,
                [
                    r"income\s+chargeable\s+under\s+the\s+head\s+"
                    r"business\s+and\s+profession",
                ],
            )
        )

        if not cls._values_are_consistent(
            business_income
        ):
            return False

        # ------------------------------------------------------
        # PRESUMPTIVE BUSINESS PROFIT
        # ------------------------------------------------------

        presumptive_profit = (
            cls._extract_labeled_amounts(
                text,
                [
                    r"business\s*:\s*presumptive\s+profits\s+u/s\s+44ad",
                ],
            )
        )

        if not cls._values_are_consistent(
            presumptive_profit
        ):
            return False

        return True

    # ==========================================================
    # TAX
    # ==========================================================

    @classmethod
    def _check_tax(
        cls,
        text: str,
    ) -> bool:
        """
        Validate consistency of repeated occurrences of the
        same logical tax field.

        Different tax stages are intentionally not compared
        against each other because updated-return ITRs contain
        legitimate intermediate values.

        Additional Tax u/s 140B(3) is intentionally excluded
        from generic amount extraction because PDF table
        extraction can interpret schedule numbers as amounts.
        """

        fields = [
            (
                "tax on total income",
                [
                    r"tax\s+on\s+total\s+income",
                ],
            ),
            (
                "tax after rebate",
                [
                    r"tax\s+after\s+rebate",
                ],
            ),
            (
                "fee u/s 234F",
                [
                    r"fee\s+u/s\s+234f",
                ],
            ),
            (
                "tax paid u/s 140B",
                [
                    r"tax\s+paid\s+u/s\s+140b",
                ],
            ),
            (
                "tax due",
                [
                    r"tax\s+due",
                ],
            ),
            (
                "balance tax payable",
                [
                    r"balance\s+tax\s+payable",
                ],
            ),
            (
                "net amount payable",
                [
                    r"net\s+amount\s+payable",
                ],
            ),
            (
                "net tax payable",
                [
                    r"net\s+tax\s+payable",
                ],
            ),
        ]

        for field_name, labels in fields:

            values = cls._extract_labeled_amounts(
                text,
                labels,
            )

            if not values:
                continue

            if not cls._values_are_consistent(
                values
            ):
                logger.warning(
                    "Tax consistency failed for %s: %s",
                    field_name,
                    values,
                )
                return False

        return True

    @classmethod
    def _check_acknowledgement(
        cls,
        text: str,
    ) -> bool:

        acknowledgement_values: List[str] = []

        patterns = [
            r"acknowledgement\s+number\s*[:\-]?\s*"
            r"([0-9]{8,20})",

            r"e[-\s]?filing\s+acknowledgement\s+number\s*"
            r"[:\-]?\s*([0-9]{8,20})",
        ]

        for pattern in patterns:

            matches = re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            for match in matches:
                acknowledgement_values.append(
                    match.group(1)
                )

        acknowledgement_values = cls._unique(
            acknowledgement_values
        )

        if len(acknowledgement_values) > 1:
            return False

        # ------------------------------------------------------
        # Filing date
        # ------------------------------------------------------

        dates = cls._extract_dates_after_labels(
            text,
            [
                r"date\s+of\s+filing",
                r"filed\s+on",
            ],
        )

        if len(set(dates)) > 1:
            return False

        return True

    # ==========================================================
    # LABEL -> AMOUNT EXTRACTION
    # ==========================================================

    @classmethod
    def _extract_labeled_amounts(
        cls,
        text: str,
        labels: Sequence[str],
        lookahead_lines: int = 5,
    ) -> List[Decimal]:
        """
        Extract financial values belonging to a specific labelled field.

        Design:
            - One amount is extracted per occurrence of the requested label.
            - Table serial numbers are ignored.
            - Numbers inside formulas such as ``(8-9)`` are ignored.
            - Percentages such as ``25%`` are ignored.
            - Extraction stops immediately after the first valid amount
              belonging to the label.
            - A new logical field terminates the current value block.

        This is deliberately conservative. Consistency validation must
        compare the same logical field, not every number appearing nearby.
        """

        values: List[Decimal] = []
        lines = text.splitlines()

        if not labels:
            return values

        compiled_labels = [
            re.compile(label, flags=re.IGNORECASE)
            for label in labels
        ]

        for index, line in enumerate(lines):
            clean_line = cls._normalize_space(line)

            if not clean_line:
                continue

            label_match = None

            for pattern in compiled_labels:
                match = pattern.search(clean_line)

                if match:
                    label_match = match
                    break

            if label_match is None:
                continue

            tax_field = cls._is_tax_label(
                clean_line,
                labels,
            )

            # ------------------------------------------------------
            # SAME-LINE VALUE
            # ------------------------------------------------------

            suffix = clean_line[label_match.end():].strip()

            # Ignore arithmetic expressions attached to labels:
            #
            #     Tax due (8-9)
            #
            # The actual value is on the following row.
            suffix_for_amount = re.sub(
                r"\(\s*\d+\s*[-+]\s*\d+\s*\)",
                " ",
                suffix,
            )

            same_line_candidates = cls._financial_candidates(
                suffix_for_amount,
                prefer_non_negative=tax_field,
            )

            same_line_candidates = [
                item
                for item in same_line_candidates
                if not cls._is_probable_serial_number(
                    suffix_for_amount,
                    item[0],
                )
            ]

            if same_line_candidates:
                values.append(
                    cls._select_value_candidate(
                        same_line_candidates,
                        suffix_for_amount,
                    )
                )
                continue

            # ------------------------------------------------------
            # FOLLOWING-LINE VALUE BLOCK
            # ------------------------------------------------------

            for next_line in lines[
                index + 1:
                index + 1 + max(1, lookahead_lines)
            ]:

                clean_next = cls._normalize_space(next_line)

                if not clean_next:
                    continue

                # A new field means the current label has no value in
                # this block.
                if cls._is_new_field_line(clean_next):
                    break

                # Avoid crossing another requested label.
                if any(
                    pattern.search(clean_next)
                    for pattern in compiled_labels
                ):
                    break

                # Ignore formula-only rows such as:
                #     (8-9)
                if re.fullmatch(
                    r"\(?\s*\d+\s*[-+]\s*\d+\s*\)?",
                    clean_next,
                ):
                    continue

                candidates = cls._financial_candidates(
                    clean_next,
                    prefer_non_negative=tax_field,
                )

                candidates = [
                    item
                    for item in candidates
                    if not cls._is_probable_serial_number(
                        clean_next,
                        item[0],
                    )
                ]

                if not candidates:
                    continue

                # IMPORTANT:
                # Once the first valid amount belonging to this label
                # is found, STOP. Never scan further rows and accidentally
                # consume the amount belonging to the next field.
                values.append(
                    cls._select_value_candidate_from_pairs(
                        candidates,
                    )
                )
                break

        return values

    @staticmethod
    def _is_tax_label(
        line: str,
        labels: Sequence[str],
    ) -> bool:
        """
        Identify labels for which a small negative OCR artifact is
        especially suspicious.

        This does NOT globally reject negative financial values.
        Business/profession income, for example, may legitimately be
        negative.
        """
        tax_keywords = (
            "tax on total income",
            "tax after rebate",
            "fee u/s 234f",
            "tax paid u/s 140b",
            "tax due",
            "balance tax payable",
            "net amount payable",
            "net tax payable",
            "additional tax u/s 140b",
        )

        value = line.lower()

        return any(
            keyword in value
            for keyword in tax_keywords
        )

    @classmethod
    def _financial_candidates(
        cls,
        text: str,
        *,
        prefer_non_negative: bool = False,
    ) -> List[tuple[Decimal, str]]:
        """
        Extract plausible monetary values from one logical row.

        Explicitly excludes:
            - percentages (25%)
            - numbers embedded in identifiers (140B, 234F)
            - arithmetic expressions
        """

        pattern = re.compile(
            r"""
            (?<![A-Za-z0-9])
            (?:
                \(\s*
                [+-]?
            )?
            [+-]?
            \s*
            (?:
                \d{1,3}(?:,\d{2,3})+
                |
                \d+
            )
            (?:\.\d{1,2})?
            \s*
            (?:CR|DR)?
            (?![A-Za-z0-9%])
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        candidates: List[tuple[Decimal, str]] = []

        for match in pattern.finditer(text):
            raw = match.group(0).strip()

            # --------------------------------------------------
            # Percentage exclusion
            # --------------------------------------------------
            after = text[match.end():]

            if re.match(
                r"\s*%",
                after,
            ):
                continue

            # --------------------------------------------------
            # Arithmetic expression exclusion
            # --------------------------------------------------
            before = text[:match.start()]
            after_full = text[match.end():]

            if re.search(
                r"\(\s*$",
                before,
            ) and re.match(
                r"\s*[-+]\s*\d+\s*\)",
                after_full,
            ):
                continue

            amount = cls._parse_amount(raw)

            if amount is None:
                continue

            candidates.append(
                (amount, raw)
            )

        if (
            prefer_non_negative
            and any(
                amount >= 0
                for amount, _ in candidates
            )
        ):
            candidates = [
                item
                for item in candidates
                if item[0] >= 0
            ]

        return candidates

    @classmethod
    def _select_value_candidate(
        cls,
        candidates: Sequence[tuple[Decimal, str]],
        source_line: str,
    ) -> Decimal:
        """
        Select the most likely financial value from a single line.

        Preference:
            - discard a standalone table serial;
            - otherwise prefer the right-most financial value;
            - for tax labels, tiny negative OCR fragments have already
              been filtered by _financial_candidates().
        """
        filtered = [
            (amount, raw)
            for amount, raw in candidates
            if not cls._is_probable_serial_number(
                source_line,
                amount,
            )
        ]

        if filtered:
            return filtered[-1][0]

        return candidates[-1][0]

    @classmethod
    def _select_value_candidate_from_pairs(
        cls,
        candidates: Sequence[tuple[Decimal, str]],
    ) -> Decimal:
        """
        Select the right-most candidate from the immediate value block.

        The ITR tables commonly place the actual monetary value at the
        right edge of the row. Serial-number-only lines are removed
        before this method is called.
        """
        return candidates[-1][0]
    # ==========================================================
    # FIELD BOUNDARY
    # ==========================================================

    @staticmethod
    def _is_new_field_line(
        line: str,
    ) -> bool:
        """
        Detect the beginning of another logical ITR field/row.

        These boundaries are important because ITR PDF text extraction
        frequently places table values on separate lines.
        """

        field_patterns = [
            r"^assessment\s+year",
            r"^asst\.?\s+year",
            r"^pan\b",
            r"^name\b",
            r"^address\b",
            r"^status\b",
            r"^form\s+number",
            r"^filed\s+u/s",
            r"^taxable\s+income",
            r"^current\s+year",
            r"^total\s+income",
            r"^gross\s+total\s+income",
            r"^income\s+chargeable\s+under",
            r"^business\s*:\s*presumptive",
            r"^tax\s+on\s+total",
            r"^tax\s+after",
            r"^rebate\s+u/s",
            r"^fee\s+u/s",
            r"^additional\s+income",
            r"^additional\s+tax",
            r"^amount\s+payable",
            r"^amount\s+refundable",
            r"^net\s+amount",
            r"^net\s+tax",
            r"^tax\s+paid",
            r"^tax\s+due",
            r"^balance\s+tax\s+payable",
            r"^tax\s+computation",
            r"^schedule\s+\d",
            r"^statement\s+of\s+income",
            r"^bank\s+a/cs",
            r"^bank\s+accounts",
        ]

        return any(
            re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            )
            for pattern in field_patterns
        )

    # ==========================================================
    # SERIAL NUMBER DETECTION
    # ==========================================================

    @staticmethod
    def _is_probable_serial_number(
        line: str,
        amount: Decimal,
    ) -> bool:

        clean = line.strip()

        if not clean:
            return False

        if amount < 1 or amount > 10:
            return False

        if amount != amount.to_integral_value():
            return False

        number = str(
            int(amount)
        )

        return clean == number

    # ==========================================================
    # AMOUNT EXTRACTION
    # ==========================================================

    @classmethod
    def _extract_all_amounts(
        cls,
        text: str,
    ) -> List[Decimal]:

        pattern = re.compile(
            r"""
            (?:
                \(\s*[+-]?\s*
            )?
            [+-]?
            \s*
            (?:
                \d{1,3}(?:,\d{2,3})+
                |
                \d+
            )
            (?:\.\d{1,2})?
            """,
            flags=re.VERBOSE,
        )

        values: List[Decimal] = []

        for match in pattern.finditer(text):

            value = cls._parse_amount(
                match.group(0)
            )

            if value is not None:
                values.append(value)

        return values

    @classmethod
    def _find_last_amount(
        cls,
        text: str,
    ) -> Optional[Decimal]:

        values = cls._extract_all_amounts(
            text
        )

        if not values:
            return None

        return values[-1]

    # ==========================================================
    # DATE EXTRACTION
    # ==========================================================

    @classmethod
    def _extract_dates_after_labels(
        cls,
        text: str,
        labels: Sequence[str],
    ) -> List[str]:

        dates: List[str] = []

        lines = text.splitlines()

        for index, line in enumerate(lines):

            if not any(
                re.search(
                    label,
                    line,
                    flags=re.IGNORECASE,
                )
                for label in labels
            ):
                continue

            block = "\n".join(
                lines[index:index + 4]
            )

            matches = re.findall(
                r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
                block,
            )

            for value in matches:
                dates.append(
                    cls._normalize_date(
                        value
                    )
                )

        return cls._unique(dates)

    # ==========================================================
    # AMOUNT PARSER
    # ==========================================================

    @staticmethod
    def _parse_amount(
        raw: str,
    ) -> Optional[Decimal]:

        if not raw:
            return None

        value = raw.strip()

        negative = (
            "(" in value
            and ")" in value
        )

        value = value.replace(
            ",",
            "",
        )

        value = value.replace(
            "(",
            "",
        )

        value = value.replace(
            ")",
            "",
        )

        value = value.strip()

        # CR/DR are directional suffixes, not part of Decimal syntax.
        value = re.sub(
            r"\s*(?:CR|DR)\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        if value.startswith("+"):
            value = value[1:]

        try:
            result = Decimal(
                value
            )

        except (
            InvalidOperation,
            ValueError,
        ):
            return None

        if negative:
            result = -result

        return result

    # ==========================================================
    # VALUE CONSISTENCY
    # ==========================================================

    @classmethod
    def _values_are_consistent(
        cls,
        values: Sequence[Decimal],
    ) -> bool:

        if len(values) <= 1:
            return True

        normalized: List[Decimal] = []

        for value in values:
            if not any(
                cls._amounts_equal(value, existing)
                for existing in normalized
            ):
                normalized.append(value)

        if len(normalized) <= 1:
            return True

        first = normalized[0]

        return all(
            cls._amounts_equal(
                first,
                value,
            )
            for value in normalized[1:]
        )

    @staticmethod
    def _amounts_equal(
        first: Decimal,
        second: Decimal,
    ) -> bool:

        return abs(
            first - second
        ) <= Decimal("0.01")

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        if not text:
            return ""

        text = text.replace(
            "\x00",
            " ",
        )

        text = text.replace(
            "\xa0",
            " ",
        )

        text = (
            text.replace("â€“", "-")
            .replace("â€”", "-")
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _normalize_space(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

    # ==========================================================
    # UNIQUE
    # ==========================================================

    @staticmethod
    def _unique(
        values: Iterable[str],
    ) -> List[str]:

        result: List[str] = []
        seen = set()

        for value in values:

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result

    # ==========================================================
    # DATE NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_date(
        value: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            value.strip().lower(),
        )

    # ==========================================================
    # RESULT
    # ==========================================================

    @classmethod
    def _result(
        cls,
        *,
        consistent: bool,
        assessment_year_consistent: bool,
        pan_consistent: bool,
        income_consistent: bool,
        tax_consistent: bool,
        acknowledgement_consistent: bool,
        score: float,
        inconsistencies: List[str],
        reasons: List[str],
        start_time: float,
    ) -> ConsistencyResult:

        processing_time_ms = round(
            (
                perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        return ConsistencyResult(
            consistent=consistent,
            assessment_year_consistent=(
                assessment_year_consistent
            ),
            pan_consistent=pan_consistent,
            income_consistent=income_consistent,
            tax_consistent=tax_consistent,
            acknowledgement_consistent=(
                acknowledgement_consistent
            ),
            score=round(
                max(
                    0.0,
                    min(
                        1.0,
                        score,
                    ),
                ),
                3,
            ),
            inconsistencies=inconsistencies,
            reasons=reasons,
            processing_time_ms=processing_time_ms,
        )


# ============================================================
# MODULE INSTANCE
# ============================================================

consistency_validator = ConsistencyValidator()
