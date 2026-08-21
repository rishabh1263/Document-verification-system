"""
Generic Bank Statement Structure Parser.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- convert normalized extracted text into structural lines
- identify likely statement-header lines
- identify transaction-table header regions
- identify likely transaction-start lines
- identify transaction-region end / footer boundary
- remove generic repeated page boilerplate from transaction body
- separate header/body/footer-like content
- remain bank-independent
- provide structural signals for later metadata and
  transaction parsers

Important:
This module does NOT:
- identify a specific bank template
- extract final transaction objects
- determine debit vs credit from bank-specific rules
- detect fraud/tampering
- calculate risk scores

The goal is structural understanding, not final parsing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


# ============================================================
# Result models
# ============================================================


@dataclass(frozen=True)
class StructuredLine:
    line_number: int
    text: str

    is_date_line: bool
    has_amount: bool
    has_reference_signal: bool
    has_transaction_header_signal: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StructureParseResult:
    line_count: int

    header_lines: tuple[str, ...]
    body_lines: tuple[str, ...]

    transaction_header_line: int | None
    transaction_start_line: int | None

    transaction_header_detected: bool
    transaction_region_detected: bool

    structured_lines: tuple[StructuredLine, ...]

    # Added without changing existing fields.
    transaction_end_line: int | None = None
    footer_lines: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Structure parser
# ============================================================


class StructureParser:
    """
    Bank-independent statement structure analyzer.

    This parser uses generic financial-document signals rather
    than bank names or bank-specific templates.
    """

    # --------------------------------------------------------
    # Generic date patterns
    # --------------------------------------------------------

    DATE_PATTERNS = (
        # 02 May 2025
        re.compile(
            r"\b"
            r"\d{1,2}\s+"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[a-z]*\s+"
            r"\d{2,4}"
            r"\b",
            re.IGNORECASE,
        ),

        # 02/05/2025
        # 02-05-2025
        # 02.05.2025
        re.compile(
            r"\b"
            r"\d{1,2}"
            r"[/.\-]"
            r"\d{1,2}"
            r"[/.\-]"
            r"\d{2,4}"
            r"\b"
        ),

        # 2025-05-02
        re.compile(
            r"\b"
            r"\d{4}"
            r"[/.\-]"
            r"\d{1,2}"
            r"[/.\-]"
            r"\d{1,2}"
            r"\b"
        ),
    )

    # --------------------------------------------------------
    # Generic amount pattern
    # --------------------------------------------------------

    # PATCHED: Stricter amount pattern to prevent reference number mis-parsing
    # - Requires comma separators for values >= 1000 (Indian format)
    # - Requires exactly 2 decimal places
    # - Only allows 1-2 digit numbers without commas (small values like 25.00)
    # - Excludes 11-digit reference numbers and single-digit noise
    AMOUNT_PATTERN = re.compile(
        r"(?<!\w)"
        r"(?:₹|Rs\.?|INR)?\s*"
        r"-?"
        r"(?:"
        r"\d{1,3}(?:,\d{2,3})+"  # Indian format: 1,00,000 or 12,34,567
        r"|"
        r"\d{1,2}"              # Small values without commas: 25.00
        r")"
        r"\.\d{2}"              # REQUIRED: exactly 2 decimal places
        r"(?!\w)",
        re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Generic transaction header vocabulary
    # --------------------------------------------------------

    TRANSACTION_HEADER_TERMS = {
        "date",
        "description",
        "narration",
        "particulars",
        "transaction",
        "remarks",

        "withdrawal",
        "withdrawals",
        "debit",
        "dr",

        "deposit",
        "deposits",
        "credit",
        "cr",

        "balance",

        "reference",
        "ref",
        "chq",
        "cheque",
        "instrument",
    }

    # --------------------------------------------------------
    # Reference / payment signals
    # --------------------------------------------------------

    REFERENCE_SIGNALS = (
        "upi",
        "imps",
        "neft",
        "rtgs",
        "utr",
        "ref",
        "reference",
        "chq",
        "cheque",
        "txn",
        "transaction id",
        "transaction no",
    )

    # --------------------------------------------------------
    # Statement/account header signals
    # --------------------------------------------------------

    HEADER_SIGNALS = (
        "account statement",
        "statement period",
        "account no",
        "account number",
        "account type",
        "customer id",
        "customer no",
        "ifsc",
        "micr",
        "branch",
        "currency",
        "opening balance",
    )

    # --------------------------------------------------------
    # Strong generic transaction-end signals
    # --------------------------------------------------------
    #
    # These indicate that transaction data has ended.
    #
    # IMPORTANT:
    # A standalone "Closing Balance <amount>" line is a strong
    # semantic boundary. It belongs to the post-transaction summary,
    # not to the final transaction row.
    #
    # We intentionally match only a closing-balance label followed
    # by an optional currency marker and numeric amount. This keeps
    # the rule generic and avoids bank-name/template dependencies.
    # --------------------------------------------------------

    STRONG_END_PATTERNS = (
        re.compile(
            r"^\s*-*\s*end\s+of\s+statement\s*-*\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*end\s+of\s+account\s+statement\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*end\s+of\s+transaction(?:s)?\s*$",
            re.IGNORECASE,
        ),

        # Generic closing-balance boundary, for example:
        #   Closing Balance 229.70
        #   Closing Balance: INR 1,382.87
        #   Closing Balance ₹ 5,224.70 Cr
        re.compile(
            r"^\s*closing\s+balance\s*[:\-]?\s*"
            r"(?:(?:₹|rs\.?|inr)\s*)?"
            r"-?(?:\d{1,3}(?:,\d{2,3})+|\d+)"
            r"(?:\.\d{1,2})?"
            r"(?:\s*(?:cr|dr))?\s*$",
            re.IGNORECASE,
        ),
    )

    # --------------------------------------------------------
    # Generic post-transaction section headings
    # --------------------------------------------------------
    #
    # These are weaker than END OF STATEMENT.
    #
    # We only accept them as transaction boundaries when the
    # surrounding lines also look like footer/summary content.
    # --------------------------------------------------------

    POST_TRANSACTION_SECTION_PATTERNS = (
        re.compile(
            r"^\s*account\s+summary\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*important\s+information\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*important\s+notice\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*commonly\s+used\s+narrations\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*terms\s+(?:and|&)\s+conditions\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*disclaimer\s*$",
            re.IGNORECASE,
        ),
    )

    # --------------------------------------------------------
    # Generic repeated page boilerplate
    # --------------------------------------------------------
    #
    # These lines often appear between transaction rows when
    # text is extracted page-by-page.
    #
    # They are removed only when they match strongly generic
    # page/header patterns.
    # --------------------------------------------------------

    PAGE_NUMBER_PATTERN = re.compile(
        r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$",
        re.IGNORECASE,
    )

    PAGE_NUMBER_PARTIAL_PATTERN = re.compile(
        r"^\s*page\s+\d+\s+of\s*$",
        re.IGNORECASE,
    )

    PAGE_TOTAL_ONLY_PATTERN = re.compile(
        r"^\s*\d+\s*$"
    )

    STATEMENT_GENERATED_PATTERN = re.compile(
        r"^\s*statement\s+generated\b",
        re.IGNORECASE,
    )

    ACCOUNT_STATEMENT_PERIOD_PATTERN = re.compile(
        r"^\s*account\s+statement\b",
        re.IGNORECASE,
    )

    ACCOUNT_NUMBER_PATTERN = re.compile(
        r"^\s*account\s+(?:no\.?|number)\b",
        re.IGNORECASE,
    )

    TRANSACTION_SECTION_PATTERN = re.compile(
        r"^\s*(?:savings|current|salary)?\s*"
        r"account\s+transactions?\s*$",
        re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def parse(
        self,
        text: str,
    ) -> StructureParseResult:

        if text is None:
            raise ValueError(
                "Text cannot be None."
            )

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Text must be a string."
            )

        raw_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not raw_lines:
            return StructureParseResult(
                line_count=0,
                header_lines=(),
                body_lines=(),
                transaction_header_line=None,
                transaction_start_line=None,
                transaction_header_detected=False,
                transaction_region_detected=False,
                structured_lines=(),
                transaction_end_line=None,
                footer_lines=(),
            )

        structured_lines = tuple(
            self._analyze_line(
                line_number=index,
                text=line,
            )
            for index, line in enumerate(
                raw_lines,
                start=1,
            )
        )

        transaction_header_line = (
            self._find_transaction_header(
                structured_lines
            )
        )

        transaction_start_line = (
            self._find_transaction_start(
                structured_lines,
                transaction_header_line,
            )
        )

        transaction_end_line = (
            self._find_transaction_end(
                structured_lines,
                transaction_start_line,
            )
        )

        split_line = (
            transaction_start_line
            or transaction_header_line
        )

        # ----------------------------------------------------
        # Header / body / footer separation
        # ----------------------------------------------------

        if split_line is not None:

            header_lines = tuple(
                line.text
                for line in structured_lines
                if line.line_number < split_line
            )

            if transaction_end_line is not None:

                candidate_body = tuple(
                    line.text
                    for line in structured_lines
                    if (
                        line.line_number >= split_line
                        and
                        line.line_number < transaction_end_line
                    )
                )

                footer_lines = tuple(
                    line.text
                    for line in structured_lines
                    if line.line_number >= transaction_end_line
                )

            else:

                candidate_body = tuple(
                    line.text
                    for line in structured_lines
                    if line.line_number >= split_line
                )

                footer_lines = ()

            body_lines = (
                self._clean_transaction_body(
                    candidate_body
                )
            )

        else:

            header_lines = tuple(
                line.text
                for line in structured_lines
            )

            body_lines = ()
            footer_lines = ()

        return StructureParseResult(
            line_count=len(
                structured_lines
            ),
            header_lines=header_lines,
            body_lines=body_lines,
            transaction_header_line=(
                transaction_header_line
            ),
            transaction_start_line=(
                transaction_start_line
            ),
            transaction_header_detected=(
                transaction_header_line
                is not None
            ),
            transaction_region_detected=(
                transaction_start_line
                is not None
            ),
            structured_lines=(
                structured_lines
            ),
            transaction_end_line=(
                transaction_end_line
            ),
            footer_lines=(
                footer_lines
            ),
        )

    # ========================================================
    # Individual line analysis
    # ========================================================

    def _analyze_line(
        self,
        line_number: int,
        text: str,
    ) -> StructuredLine:

        normalized = text.lower()

        return StructuredLine(
            line_number=line_number,
            text=text,
            is_date_line=self._contains_date(
                text
            ),
            has_amount=self._contains_amount(
                text
            ),
            has_reference_signal=any(
                signal in normalized
                for signal in self.REFERENCE_SIGNALS
            ),
            has_transaction_header_signal=(
                self._has_transaction_header_signal(
                    text
                )
            ),
        )

    # ========================================================
    # Date detection
    # ========================================================

    @classmethod
    def _contains_date(
        cls,
        text: str,
    ) -> bool:

        return any(
            pattern.search(text)
            is not None
            for pattern in cls.DATE_PATTERNS
        )

    # ========================================================
    # Amount detection
    # ========================================================

    @classmethod
    def _contains_amount(
        cls,
        text: str,
    ) -> bool:

        return (
            cls.AMOUNT_PATTERN.search(
                text
            )
            is not None
        )

    # ========================================================
    # Transaction header detection
    # ========================================================

    @classmethod
    def _has_transaction_header_signal(
        cls,
        text: str,
    ) -> bool:

        words = set(
            re.findall(
                r"[a-z]+",
                text.lower(),
            )
        )

        matches = (
            words
            & cls.TRANSACTION_HEADER_TERMS
        )

        return len(matches) >= 2

    def _find_transaction_header(
        self,
        lines: tuple[
            StructuredLine,
            ...
        ],
    ) -> int | None:
        """
        Locate likely transaction-column headings.

        Handles both:

        Native PDF:
            Date Description Withdrawal Deposit Balance

        OCR:
            Date
            Description
            Withdrawal
            Deposit
            Balance
        """

        # First try single-line header detection.
        for line in lines:

            if (
                line.has_transaction_header_signal
            ):
                return line.line_number

        # Then inspect small windows for OCR output where every
        # column heading may appear on its own line.
        window_size = 8

        for index in range(
            len(lines)
        ):

            window = lines[
                index:
                index + window_size
            ]

            if not window:
                continue

            combined = " ".join(
                line.text.lower()
                for line in window
            )

            words = set(
                re.findall(
                    r"[a-z]+",
                    combined,
                )
            )

            matches = (
                words
                & self.TRANSACTION_HEADER_TERMS
            )

            has_date = (
                "date"
                in matches
            )

            has_balance = (
                "balance"
                in matches
            )

            has_description = bool(
                {
                    "description",
                    "narration",
                    "particulars",
                    "remarks",
                }
                & matches
            )

            has_money_column = bool(
                {
                    "withdrawal",
                    "withdrawals",
                    "debit",
                    "dr",
                    "deposit",
                    "deposits",
                    "credit",
                    "cr",
                }
                & matches
            )

            score = sum(
                (
                    has_date,
                    has_balance,
                    has_description,
                    has_money_column,
                )
            )

            if score >= 3:
                return window[
                    0
                ].line_number

        return None

    # ========================================================
    # Transaction-start detection
    # ========================================================

    def _find_transaction_start(
        self,
        lines: tuple[
            StructuredLine,
            ...
        ],
        transaction_header_line: int | None,
    ) -> int | None:
        """
        Locate the first likely transaction record.

        We intentionally avoid requiring one fixed row format.

        Native extraction may put an entire transaction on one
        line while OCR may place date, description and amounts
        on separate lines.
        """

        if not lines:
            return None

        start_index = 0

        if transaction_header_line is not None:

            for index, line in enumerate(
                lines
            ):
                if (
                    line.line_number
                    >= transaction_header_line
                ):
                    start_index = index
                    break

        search_lines = lines[
            start_index:
        ]

        for index, line in enumerate(
            search_lines
        ):

            if not line.is_date_line:
                continue

            # Strong case:
            # date and amount appear on same line.
            if line.has_amount:
                return line.line_number

            # OCR case:
            # date isolated; financial information follows.
            following = search_lines[
                index + 1:
                index + 7
            ]

            amount_found = any(
                candidate.has_amount
                for candidate in following
            )

            reference_found = any(
                candidate.has_reference_signal
                for candidate in following
            )

            if (
                amount_found
                or reference_found
            ):
                return line.line_number

        return None

    # ========================================================
    # Transaction-end detection
    # ========================================================

    def _find_transaction_end(
        self,
        lines: tuple[
            StructuredLine,
            ...
        ],
        transaction_start_line: int | None,
    ) -> int | None:
        """
        Find the first reliable boundary after transaction data.

        Strong markers such as "End of Statement" are accepted
        directly.

        Weaker section headings such as "Account Summary" are
        accepted only when the surrounding region looks like
        post-transaction/footer content.

        This remains bank-independent.
        """

        if (
            not lines
            or transaction_start_line is None
        ):
            return None

        start_index = None

        for index, line in enumerate(
            lines
        ):
            if (
                line.line_number
                >= transaction_start_line
            ):
                start_index = index
                break

        if start_index is None:
            return None

        # ----------------------------------------------------
        # Pass 1:
        # Strong explicit end markers.
        # ----------------------------------------------------

        for line in lines[
            start_index:
        ]:

            if self._is_strong_end_line(
                line.text
            ):
                return line.line_number

        # ----------------------------------------------------
        # Pass 2:
        # Generic post-transaction sections.
        #
        # Require evidence that transaction activity has stopped.
        # ----------------------------------------------------

        for index in range(
            start_index,
            len(lines),
        ):

            line = lines[index]

            if not self._is_post_transaction_section(
                line.text
            ):
                continue

            previous_window = lines[
                max(
                    start_index,
                    index - 12,
                ):
                index
            ]

            following_window = lines[
                index:
                index + 12
            ]

            previous_has_transaction = (
                self._window_has_transaction_activity(
                    previous_window
                )
            )

            following_has_transaction = (
                self._window_has_transaction_activity(
                    following_window
                )
            )

            if (
                previous_has_transaction
                and
                not following_has_transaction
            ):
                return line.line_number

        return None

    # ========================================================
    # Strong end markers
    # ========================================================

    @classmethod
    def _is_strong_end_line(
        cls,
        text: str,
    ) -> bool:

        normalized = (
            text
            .strip()
        )

        return any(
            pattern.search(
                normalized
            )
            is not None
            for pattern
            in cls.STRONG_END_PATTERNS
        )

    # ========================================================
    # Post-transaction section detection
    # ========================================================

    @classmethod
    def _is_post_transaction_section(
        cls,
        text: str,
    ) -> bool:

        normalized = (
            text
            .strip()
        )

        return any(
            pattern.search(
                normalized
            )
            is not None
            for pattern
            in cls.POST_TRANSACTION_SECTION_PATTERNS
        )

    # ========================================================
    # Transaction activity in a line window
    # ========================================================

    def _window_has_transaction_activity(
        self,
        lines: tuple[
            StructuredLine,
            ...
        ],
    ) -> bool:
        """
        Determine whether a small line window looks like it
        contains actual transaction activity.

        Avoids depending on one bank's row format.
        """

        if not lines:
            return False

        date_count = sum(
            1
            for line in lines
            if line.is_date_line
        )

        amount_count = sum(
            1
            for line in lines
            if line.has_amount
        )

        reference_count = sum(
            1
            for line in lines
            if line.has_reference_signal
        )

        # Strong transaction activity.
        if (
            date_count >= 1
            and amount_count >= 1
        ):
            return True

        if (
            date_count >= 1
            and reference_count >= 1
        ):
            return True

        return False

    # ========================================================
    # Transaction-body cleanup
    # ========================================================

    def _clean_transaction_body(
        self,
        lines: tuple[str, ...],
    ) -> tuple[str, ...]:
        """
        Remove generic repeated page boilerplate while preserving
        transaction descriptions and continuation lines.

        We deliberately use conservative rules here.

        Unknown lines are kept.
        """

        if not lines:
            return ()

        cleaned: list[str] = []

        index = 0

        while index < len(lines):

            line = lines[index]
            stripped = line.strip()

            if not stripped:
                index += 1
                continue

            # ------------------------------------------------
            # Remove:
            #
            # Statement Generated on ...
            # ------------------------------------------------

            if self.STATEMENT_GENERATED_PATTERN.search(
                stripped
            ):
                index += 1
                continue

            # ------------------------------------------------
            # Remove:
            #
            # Page 10
            # Page 10 of 39
            # ------------------------------------------------

            if self.PAGE_NUMBER_PATTERN.fullmatch(
                stripped
            ):
                index += 1
                continue

            # ------------------------------------------------
            # Handle split extraction:
            #
            # Page 36 of
            # 39
            # ------------------------------------------------

            if self.PAGE_NUMBER_PARTIAL_PATTERN.fullmatch(
                stripped
            ):

                if (
                    index + 1 < len(lines)
                    and
                    self.PAGE_TOTAL_ONLY_PATTERN.fullmatch(
                        lines[
                            index + 1
                        ].strip()
                    )
                ):
                    index += 2
                    continue

                index += 1
                continue

            # ------------------------------------------------
            # Remove repeated transaction column headers.
            #
            # Example:
            # Date Particulars Deposits Withdrawals Balance
            # ------------------------------------------------

            if self._has_transaction_header_signal(
                stripped
            ):
                index += 1
                continue

            # ------------------------------------------------
            # Remove repeated transaction section heading.
            #
            # Example:
            # Savings Account Transactions
            # ------------------------------------------------

            if self.TRANSACTION_SECTION_PATTERN.fullmatch(
                stripped
            ):
                index += 1
                continue

            # ------------------------------------------------
            # Repeated "Account Statement ..." page header.
            #
            # We only remove when it contains a date/period-like
            # signal or appears near other page boilerplate.
            # ------------------------------------------------

            if self.ACCOUNT_STATEMENT_PERIOD_PATTERN.search(
                stripped
            ):

                if self._contains_date(
                    stripped
                ):
                    index += 1
                    continue

            # ------------------------------------------------
            # Repeated account number header.
            #
            # Be conservative:
            # remove only if nearby lines clearly indicate
            # repeated statement/page boilerplate.
            # ------------------------------------------------

            if self.ACCOUNT_NUMBER_PATTERN.search(
                stripped
            ):

                nearby = lines[
                    max(
                        0,
                        index - 2,
                    ):
                    min(
                        len(lines),
                        index + 4,
                    )
                ]

                if self._looks_like_page_header_context(
                    nearby
                ):
                    index += 1
                    continue

            # ------------------------------------------------
            # Generic customer-name-only lines cannot safely be
            # removed because a transaction description itself
            # may consist of a person's name.
            #
            # Therefore unknown standalone text is preserved.
            # ------------------------------------------------

            cleaned.append(
                stripped
            )

            index += 1

        return tuple(
            cleaned
        )

    # ========================================================
    # Repeated page-header context
    # ========================================================

    def _looks_like_page_header_context(
        self,
        lines: tuple[str, ...],
    ) -> bool:
        """
        Determine whether nearby lines contain strong generic
        statement/page-header signals.

        Used only for conservative boilerplate removal.
        """

        if not lines:
            return False

        score = 0

        for line in lines:

            stripped = (
                line
                .strip()
            )

            if self.STATEMENT_GENERATED_PATTERN.search(
                stripped
            ):
                score += 2

            if self.PAGE_NUMBER_PATTERN.fullmatch(
                stripped
            ):
                score += 2

            if self.PAGE_NUMBER_PARTIAL_PATTERN.fullmatch(
                stripped
            ):
                score += 2

            if (
                self.ACCOUNT_STATEMENT_PERIOD_PATTERN.search(
                    stripped
                )
                and
                self._contains_date(
                    stripped
                )
            ):
                score += 1

            if self.TRANSACTION_SECTION_PATTERN.fullmatch(
                stripped
            ):
                score += 1

            if self._has_transaction_header_signal(
                stripped
            ):
                score += 1

        return score >= 2


structure_parser = StructureParser()