"""
Generic Bank Statement Transaction Parser.

Phase 2 - Document Intelligence / Extraction.

Responsibilities:
- parse transaction regions into structured records
- support single-line and multiline transaction layouts
- detect transaction boundaries using dates
- extract descriptions and transaction references
- extract transaction amounts and balances
- determine debit/credit direction using multiple generic signals
- use opening/previous balance reconciliation when table cells are lost
- remain bank-independent

Direction inference priority:
1. Previous-balance arithmetic reconciliation when available
2. Preserved debit / credit table columns
3. Explicit DR / CR / narration signal as fallback
4. Leave unresolved rather than inventing direction

Why arithmetic comes first:
- narration can contain misleading words such as merchant/company names
- balance movement is objective financial evidence
- this keeps the parser bank-independent without merchant-specific exceptions

Important:
This module does NOT:
- contain bank names
- select bank-specific templates
- detect tampering
- calculate fraud/risk scores
- perform loan eligibility logic
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation


# ============================================================
# Result models
# ============================================================


@dataclass(frozen=True)
class Transaction:
    sequence: int

    date: str | None

    description: str | None
    reference: str | None

    debit: Decimal | None
    credit: Decimal | None
    balance: Decimal | None

    direction_source: str | None

    balance_reconciled: bool | None

    confidence: float

    raw_text: str

    def to_dict(self) -> dict:
        data = asdict(self)

        for key in (
            "debit",
            "credit",
            "balance",
        ):
            value = data[key]

            if value is not None:
                data[key] = float(value)

        return data


@dataclass(frozen=True)
class TransactionParseResult:
    transaction_count: int

    transactions: tuple[
        Transaction,
        ...
    ]

    rejected_blocks: int

    unresolved_direction_count: int

    reconciled_count: int

    parser_confidence: float

    opening_balance: Decimal | None

    def to_dict(self) -> dict:
        return {
            "transaction_count":
                self.transaction_count,

            "transactions": [
                transaction.to_dict()
                for transaction
                in self.transactions
            ],

            "rejected_blocks":
                self.rejected_blocks,

            "unresolved_direction_count":
                self.unresolved_direction_count,

            "reconciled_count":
                self.reconciled_count,

            "parser_confidence":
                self.parser_confidence,

            "opening_balance": (
                float(self.opening_balance)
                if self.opening_balance
                is not None
                else None
            ),
        }


# ============================================================
# Internal transaction block
# ============================================================


@dataclass
class _TransactionBlock:
    lines: list[str]


# ============================================================
# Transaction parser
# ============================================================


class TransactionParser:
    """
    Generic transaction parser.

    The parser does not know which bank produced the document.

    It relies on financial-document semantics:

        date
        transaction description
        transaction reference
        monetary amount
        running balance
        DR / CR markers
        debit / credit column headings
        previous balance reconciliation
    """

    # Small tolerance for Decimal reconciliation.
    #
    # Bank statements normally use two decimal places, but this
    # prevents formatting/conversion noise from causing a false
    # mismatch.
    BALANCE_TOLERANCE = Decimal("0.01")

    # ========================================================
    # Date patterns
    # ========================================================

    DATE_PATTERNS = (
        # 02 May 2025
        # 02 May 25
        re.compile(
            r"\b"
            r"(\d{1,2}\s+"
            r"(?:"
            r"Jan|Feb|Mar|Apr|May|Jun|"
            r"Jul|Aug|Sep|Oct|Nov|Dec"
            r")"
            r"[a-z]*\s+"
            r"\d{2,4})"
            r"\b",
            re.IGNORECASE,
        ),

        # 22-01-2026
        # 22/01/2026
        # 22.01.2026
        re.compile(
            r"\b"
            r"(\d{1,2}"
            r"[/.\-]"
            r"\d{1,2}"
            r"[/.\-]"
            r"\d{2,4})"
            r"\b"
        ),

        # 23-Jan-2026
        # 23/Jan/2026
        re.compile(
            r"\b"
            r"(\d{1,2}"
            r"[\-/]"
            r"(?:"
            r"Jan|Feb|Mar|Apr|May|Jun|"
            r"Jul|Aug|Sep|Oct|Nov|Dec"
            r")"
            r"[a-z]*"
            r"[\-/]"
            r"\d{2,4})"
            r"\b",
            re.IGNORECASE,
        ),

        # 2026-01-22
        # 2026/01/22
        re.compile(
            r"\b"
            r"(\d{4}"
            r"[/.\-]"
            r"\d{1,2}"
            r"[/.\-]"
            r"\d{1,2})"
            r"\b"
        ),
    )

    # ========================================================
    # Amount recognition
    # ========================================================

    AMOUNT_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:₹|Rs\.?|INR)?\s*"
        r"("
        r"-?"
        r"(?:"
        r"\d{1,3}(?:,\d{2,3})+"
        r"|"
        r"\d+"
        r")"
        r"\.\d{2}"
        r")"
        r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    # ========================================================
    # Opening balance
    # ========================================================

    OPENING_BALANCE_PATTERN = re.compile(
        r"\bopening\s+balance\b"
        r".*?"
        r"(?:₹|Rs\.?|INR)?\s*"
        r"("
        r"-?"
        r"(?:"
        r"\d{1,3}(?:,\d{2,3})+"
        r"|"
        r"\d+"
        r")"
        r"\.\d{2}"
        r")",
        re.IGNORECASE,
    )

    # ========================================================
    # Explicit transaction direction
    # ========================================================

    CREDIT_PATTERNS = (
        # UPI/CR/...
        re.compile(
            r"(?:^|[/\s\-])"
            r"CR"
            r"(?:[/\s\-]|$)",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bCREDIT\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bCREDITED\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bDEPOSIT\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bDEPOSITED\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bRECEIVED\b",
            re.IGNORECASE,
        ),
    )

    DEBIT_PATTERNS = (
        # UPI/DR/...
        re.compile(
            r"(?:^|[/\s\-])"
            r"DR"
            r"(?:[/\s\-]|$)",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bDEBIT\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bDEBITED\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bWITHDRAWAL\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bWITHDRAWN\b",
            re.IGNORECASE,
        ),
    )

    # ========================================================
    # Reference extraction
    # ========================================================

    # Prefer structured transaction IDs before searching generic
    # description paths.

    STRUCTURED_REFERENCE_PATTERNS = (
        # UPI-512281161274
        # IMPS-512419091489
        # NEFT-ABC123
        # RTGS-ABC123
        # MB-998486503610
        re.compile(
            r"\b("
            r"(?:"
            r"UPI|IMPS|NEFT|RTGS|MB"
            r")"
            r"-"
            r"[A-Z0-9][A-Z0-9\-]{3,60}"
            r")\b",
            re.IGNORECASE,
        ),

        # Chq: 602261008536
        # Cheque: 602261008536
        re.compile(
            r"\b(?:Chq|Cheque)"
            r"\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9\-]{3,40})",
            re.IGNORECASE,
        ),

        # UTR: XXXXX
        # Ref: XXXXX
        # Reference: XXXXX
        re.compile(
            r"\b(?:UTR|REF|REFERENCE)"
            r"\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9\-/:]{3,60})",
            re.IGNORECASE,
        ),
    )

    # Generic fallback only after structured references fail.
    FALLBACK_REFERENCE_PATTERNS = (
        # UPI/Adesh/100477198091/NA
        # IMPS/512419359116/...
        re.compile(
            r"\b("
            r"(?:UPI|IMPS|NEFT|RTGS)"
            r"/"
            r"[A-Z0-9][A-Z0-9/._@*\-]{5,100}"
            r")",
            re.IGNORECASE,
        ),
    )

    # ========================================================
    # Noise / repeated page lines
    # ========================================================

    NOISE_PATTERNS = (
        re.compile(
            r"^page\s+\d+$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^date\s+.*balance$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^date$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^particulars$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^description$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^chq/?ref\.?\s*no\.?$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^deposits?$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^withdrawals?$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^debits?$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^credits?$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^balance$",
            re.IGNORECASE,
        ),
    )

    # ========================================================
    # Public API
    # ========================================================

    def parse(
        self,
        body_text: str,
        transaction_header: str | None = None,
        header_text: str | None = None,
        opening_balance: Decimal | float | str | None = None,
    ) -> TransactionParseResult:
        """
        Parse transaction body into structured transactions.

        Parameters
        ----------
        body_text:
            Transaction-region text.

        transaction_header:
            Optional transaction table header, for example:

                Date Description Withdrawal Deposit Balance

        header_text:
            Optional statement header text. Used only to extract
            opening balance when opening_balance was not supplied.

        opening_balance:
            Optional explicit opening balance. Highest-priority
            source for balance reconciliation.
        """

        if body_text is None:
            raise ValueError(
                "body_text cannot be None."
            )

        if not isinstance(
            body_text,
            str,
        ):
            raise TypeError(
                "body_text must be a string."
            )

        lines = [
            line.strip()
            for line in body_text.splitlines()
            if line.strip()
        ]

        if not lines:
            return TransactionParseResult(
                transaction_count=0,
                transactions=(),
                rejected_blocks=0,
                unresolved_direction_count=0,
                reconciled_count=0,
                parser_confidence=0.0,
                opening_balance=None,
            )

        resolved_opening_balance = (
            self._resolve_opening_balance(
                opening_balance=opening_balance,
                header_text=header_text,
            )
        )

        cleaned_lines = (
            self._remove_noise_lines(
                lines
            )
        )

        blocks = (
            self._segment_transactions(
                cleaned_lines
            )
        )

        header_semantics = (
            self._infer_header_semantics(
                transaction_header
            )
        )

        transactions: list[
            Transaction
        ] = []

        rejected_blocks = 0

        previous_balance = (
            resolved_opening_balance
        )

        for block in blocks:

            transaction = (
                self._parse_block(
                    block=block,
                    sequence=(
                        len(transactions)
                        + 1
                    ),
                    header_semantics=(
                        header_semantics
                    ),
                    previous_balance=(
                        previous_balance
                    ),
                )
            )

            if transaction is None:
                rejected_blocks += 1
                continue

            transactions.append(
                transaction
            )

            # Only a successfully parsed transaction can advance
            # the running balance.
            if transaction.balance is not None:
                previous_balance = (
                    transaction.balance
                )

        unresolved_direction_count = sum(
            1
            for transaction in transactions
            if (
                transaction.debit is None
                and transaction.credit is None
            )
        )

        reconciled_count = sum(
            1
            for transaction in transactions
            if transaction.balance_reconciled
            is True
        )

        confidence_values = [
            transaction.confidence
            for transaction
            in transactions
        ]

        parser_confidence = (
            sum(confidence_values)
            / len(confidence_values)
            if confidence_values
            else 0.0
        )

        return TransactionParseResult(
            transaction_count=len(
                transactions
            ),
            transactions=tuple(
                transactions
            ),
            rejected_blocks=(
                rejected_blocks
            ),
            unresolved_direction_count=(
                unresolved_direction_count
            ),
            reconciled_count=(
                reconciled_count
            ),
            parser_confidence=round(
                parser_confidence,
                4,
            ),
            opening_balance=(
                resolved_opening_balance
            ),
        )

    # ========================================================
    # Opening balance
    # ========================================================

    def _resolve_opening_balance(
        self,
        opening_balance:
            Decimal | float | str | None,
        header_text: str | None,
    ) -> Decimal | None:

        if opening_balance is not None:

            parsed = self._to_decimal(
                opening_balance
            )

            if parsed is not None:
                return parsed

        if not header_text:
            return None

        return self._extract_opening_balance(
            header_text
        )

    def _extract_opening_balance(
        self,
        header_text: str,
    ) -> Decimal | None:

        if not header_text:
            return None

        match = (
            self.OPENING_BALANCE_PATTERN.search(
                header_text
            )
        )

        if not match:
            return None

        return self._to_decimal(
            match.group(1)
        )

    # ========================================================
    # Transaction segmentation
    # ========================================================

    def _segment_transactions(
        self,
        lines: list[str],
    ) -> list[_TransactionBlock]:
        """
        Start a new transaction whenever a line starts with a
        recognizable transaction date.

        Handles both:

            1 02 May 2025 UPI/...
            22-01-2026
        """

        blocks: list[
            _TransactionBlock
        ] = []

        current_lines: list[str] = []

        for line in lines:

            if self._starts_with_date(
                line
            ):

                if current_lines:

                    blocks.append(
                        _TransactionBlock(
                            lines=current_lines
                        )
                    )

                current_lines = [
                    line
                ]

            elif current_lines:

                current_lines.append(
                    line
                )

        if current_lines:

            blocks.append(
                _TransactionBlock(
                    lines=current_lines
                )
            )

        return blocks

    # ========================================================
    # Parse transaction block
    # ========================================================

    def _parse_block(
        self,
        block: _TransactionBlock,
        sequence: int,
        header_semantics: dict,
        previous_balance: Decimal | None,
    ) -> Transaction | None:

        if not block.lines:
            return None

        raw_text = "\n".join(
            block.lines
        )

        date = self._extract_date(
            block.lines[0]
        )

        if date is None:
            return None

        amounts = self._extract_amounts(
            block.lines
        )

        # At minimum we need a monetary value.
        if not amounts:
            return None

        # The last monetary value in the extracted transaction
        # block is treated as the running balance.
        balance = amounts[-1]

        transaction_amounts = (
            amounts[:-1]
        )

        debit: Decimal | None = None
        credit: Decimal | None = None

        direction_source: str | None = None

        balance_reconciled: bool | None = None

        # ----------------------------------------------------
        # Step 1: previous-balance arithmetic reconciliation
        # ----------------------------------------------------
        #
        # This is the strongest generic signal whenever the previous
        # running balance is available. It prevents narration words
        # (for example a company/merchant name containing "CREDIT")
        # from overriding mathematically provable balance movement.
        if (
            transaction_amounts
            and previous_balance is not None
            and balance is not None
        ):
            (
                inferred_debit,
                inferred_credit,
                reconciled,
            ) = self._infer_from_balance_delta(
                previous_balance=previous_balance,
                current_balance=balance,
                transaction_amounts=transaction_amounts,
            )

            if reconciled:
                debit = inferred_debit
                credit = inferred_credit
                direction_source = "balance_delta"
                balance_reconciled = True

        # ----------------------------------------------------
        # Step 2: preserved debit/credit table columns
        # ----------------------------------------------------
        #
        # Use table-column semantics only when arithmetic could not
        # resolve direction. This is useful for the first transaction
        # when no opening balance is available, or for damaged text.
        if (
            debit is None
            and credit is None
            and transaction_amounts
        ):
            (
                column_debit,
                column_credit,
            ) = self._infer_from_amount_columns(
                transaction_amounts,
                header_semantics,
            )

            if (
                column_debit is not None
                or column_credit is not None
            ):
                debit = column_debit
                credit = column_credit
                direction_source = "column_semantics"

        # ----------------------------------------------------
        # Step 3: explicit DR / CR / narration semantic signal
        # ----------------------------------------------------
        #
        # Explicit text is deliberately a fallback rather than the
        # first authority. Words such as CREDIT or DEBIT can occur
        # inside legitimate narration/merchant names and therefore
        # must not defeat successful balance reconciliation.
        if (
            debit is None
            and credit is None
            and transaction_amounts
        ):
            explicit_signal = self._detect_direction_signal(
                raw_text
            )

            if explicit_signal in {"debit", "credit"}:
                amount = self._select_primary_amount(
                    transaction_amounts
                )

                if explicit_signal == "debit":
                    debit = amount
                else:
                    credit = amount

                direction_source = "explicit_signal"

        # ----------------------------------------------------
        # Step 4: validate fallback direction against balance
        # ----------------------------------------------------
        #
        # Arithmetic-derived transactions are already reconciled.
        # Column/explicit fallback results are checked here whenever
        # the previous and current balances are available.
        if (
            balance_reconciled is not True
            and previous_balance is not None
            and balance is not None
            and (
                debit is not None
                or credit is not None
            )
        ):
            balance_reconciled = self._reconcile_transaction(
                previous_balance=previous_balance,
                current_balance=balance,
                debit=debit,
                credit=credit,
            )

        reference = (
            self._extract_reference(
                raw_text
            )
        )

        description = (
            self._extract_description(
                block.lines
            )
        )

        confidence = (
            self._calculate_confidence(
                date=date,
                description=description,
                debit=debit,
                credit=credit,
                balance=balance,
                reference=reference,
                direction_source=(
                    direction_source
                ),
                balance_reconciled=(
                    balance_reconciled
                ),
            )
        )

        return Transaction(
            sequence=sequence,
            date=date,
            description=description,
            reference=reference,
            debit=debit,
            credit=credit,
            balance=balance,
            direction_source=(
                direction_source
            ),
            balance_reconciled=(
                balance_reconciled
            ),
            confidence=confidence,
            raw_text=raw_text,
        )

    # ========================================================
    # Date handling
    # ========================================================

    @classmethod
    def _starts_with_date(
        cls,
        line: str,
    ) -> bool:

        stripped = cls._remove_sequence_prefix(
            line.strip()
        )

        for pattern in cls.DATE_PATTERNS:

            match = pattern.search(
                stripped
            )

            if (
                match
                and match.start() == 0
            ):
                return True

        return False

    @classmethod
    def _extract_date(
        cls,
        line: str,
    ) -> str | None:

        candidate = (
            cls._remove_sequence_prefix(
                line.strip()
            )
        )

        for pattern in cls.DATE_PATTERNS:

            match = pattern.search(
                candidate
            )

            if not match:
                continue

            normalized = (
                cls._normalize_date(
                    match.group(1)
                )
            )

            if normalized:
                return normalized

        return None

    @staticmethod
    def _remove_sequence_prefix(
        line: str,
    ) -> str:
        """
        Remove optional transaction serial number.

        Example:
            12 08 May 2025 ...
                ↓
            08 May 2025 ...
        """

        return re.sub(
            r"^\d+\s+(?=\d{1,2}(?:\s|[-/.]))",
            "",
            line,
            count=1,
        )

    @staticmethod
    def _normalize_date(
        value: str,
    ) -> str | None:

        value = (
            value
            .strip()
            .replace(",", "")
        )

        formats = (
            "%d %B %Y",
            "%d %b %Y",
            "%d %B %y",
            "%d %b %y",

            "%d-%b-%Y",
            "%d-%B-%Y",
            "%d-%b-%y",
            "%d-%B-%y",

            "%d/%b/%Y",
            "%d/%B/%Y",
            "%d/%b/%y",
            "%d/%B/%y",

            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",

            "%d/%m/%y",
            "%d-%m-%y",
            "%d.%m.%y",

            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
        )

        for date_format in formats:

            try:

                parsed = datetime.strptime(
                    value,
                    date_format,
                )

                return parsed.strftime(
                    "%Y-%m-%d"
                )

            except ValueError:
                continue

        return None

    # ========================================================
    # Amount extraction
    # ========================================================

    @classmethod
    def _extract_amounts(
        cls,
        lines: list[str],
    ) -> list[Decimal]:

        values: list[
            Decimal
        ] = []

        for line in lines:

            for match in (
                cls.AMOUNT_PATTERN.finditer(
                    line
                )
            ):

                value = cls._to_decimal(
                    match.group(1)
                )

                if value is not None:
                    values.append(
                        value
                    )

        return values

    @staticmethod
    def _to_decimal(
        value:
            Decimal | float | int | str,
    ) -> Decimal | None:

        if isinstance(
            value,
            Decimal,
        ):
            return value

        raw = (
            str(value)
            .replace(",", "")
            .replace("₹", "")
            .strip()
        )

        raw = re.sub(
            r"^(?:Rs\.?|INR)\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        try:
            return Decimal(raw)

        except (
            InvalidOperation,
            ValueError,
        ):
            return None

    # ========================================================
    # Direction signals
    # ========================================================

    @classmethod
    def _detect_direction_signal(
        cls,
        text: str,
    ) -> str | None:

        credit = any(
            pattern.search(text)
            for pattern
            in cls.CREDIT_PATTERNS
        )

        debit = any(
            pattern.search(text)
            for pattern
            in cls.DEBIT_PATTERNS
        )

        if (
            credit
            and not debit
        ):
            return "credit"

        if (
            debit
            and not credit
        ):
            return "debit"

        # Conflicting signals should not be guessed.
        return None

    # ========================================================
    # Header semantics
    # ========================================================

    @staticmethod
    def _infer_header_semantics(
        header: str | None,
    ) -> dict:

        result = {
            "money_columns": [],
        }

        if not header:
            return result

        words = re.findall(
            r"[A-Za-z]+",
            header.lower(),
        )

        for word in words:

            if word in {
                "withdrawal",
                "withdrawals",
                "debit",
                "debits",
                "dr",
            }:

                if (
                    "debit"
                    not in result[
                        "money_columns"
                    ]
                ):
                    result[
                        "money_columns"
                    ].append(
                        "debit"
                    )

            elif word in {
                "deposit",
                "deposits",
                "credit",
                "credits",
                "cr",
            }:

                if (
                    "credit"
                    not in result[
                        "money_columns"
                    ]
                ):
                    result[
                        "money_columns"
                    ].append(
                        "credit"
                    )

        return result

    # ========================================================
    # Column-based inference
    # ========================================================

    @staticmethod
    def _infer_from_amount_columns(
        amounts: list[Decimal],
        header_semantics: dict,
    ) -> tuple[
        Decimal | None,
        Decimal | None,
    ]:

        if not amounts:
            return None, None

        columns = header_semantics.get(
            "money_columns",
            [],
        )

        # If only one amount survived before balance, its
        # original blank-cell position has been lost.
        #
        # Therefore column order cannot safely determine whether
        # it was debit or credit.
        if len(amounts) == 1:
            return None, None

        if (
            len(amounts) >= 2
            and len(columns) >= 2
        ):

            first = amounts[0]
            second = amounts[1]

            debit = None
            credit = None

            if columns[0] == "debit":
                debit = first

            elif columns[0] == "credit":
                credit = first

            if columns[1] == "debit":
                debit = second

            elif columns[1] == "credit":
                credit = second

            return debit, credit

        return None, None

    # ========================================================
    # Balance-delta inference
    # ========================================================

    def _infer_from_balance_delta(
        self,
        previous_balance: Decimal,
        current_balance: Decimal,
        transaction_amounts: list[Decimal],
    ) -> tuple[
        Decimal | None,
        Decimal | None,
        bool,
    ]:
        """
        Determine direction by reconciling:

            previous_balance - amount = current_balance

        or:

            previous_balance + amount = current_balance

        This is bank-independent and useful when PDF extraction
        removes blank debit/credit cells.
        """

        if not transaction_amounts:
            return None, None, False

        candidates: list[
            tuple[
                str,
                Decimal,
                Decimal,
            ]
        ] = []

        for amount in transaction_amounts:

            debit_expected = (
                previous_balance
                - amount
            )

            debit_error = abs(
                debit_expected
                - current_balance
            )

            if (
                debit_error
                <= self.BALANCE_TOLERANCE
            ):
                candidates.append(
                    (
                        "debit",
                        amount,
                        debit_error,
                    )
                )

            credit_expected = (
                previous_balance
                + amount
            )

            credit_error = abs(
                credit_expected
                - current_balance
            )

            if (
                credit_error
                <= self.BALANCE_TOLERANCE
            ):
                candidates.append(
                    (
                        "credit",
                        amount,
                        credit_error,
                    )
                )

        # No amount reconciles.
        if not candidates:
            return None, None, False

        # Sort by smallest reconciliation error.
        candidates.sort(
            key=lambda item: item[2]
        )

        best_direction = (
            candidates[0][0]
        )

        best_amount = (
            candidates[0][1]
        )

        # Defensive ambiguity check.
        #
        # If two different interpretations reconcile equally,
        # leave unresolved instead of inventing direction.
        if len(candidates) > 1:

            first_error = (
                candidates[0][2]
            )

            second_error = (
                candidates[1][2]
            )

            if (
                first_error
                == second_error
                and (
                    candidates[0][0]
                    != candidates[1][0]
                    or candidates[0][1]
                    != candidates[1][1]
                )
            ):
                return None, None, False

        if best_direction == "debit":

            return (
                best_amount,
                None,
                True,
            )

        return (
            None,
            best_amount,
            True,
        )

    # ========================================================
    # Balance reconciliation
    # ========================================================

    def _reconcile_transaction(
        self,
        previous_balance: Decimal,
        current_balance: Decimal,
        debit: Decimal | None,
        credit: Decimal | None,
    ) -> bool:

        expected = previous_balance

        if debit is not None:
            expected -= debit

        if credit is not None:
            expected += credit

        difference = abs(
            expected
            - current_balance
        )

        return (
            difference
            <= self.BALANCE_TOLERANCE
        )

    # ========================================================
    # Primary amount
    # ========================================================

    @staticmethod
    def _select_primary_amount(
        amounts: list[Decimal],
    ) -> Decimal:

        return amounts[0]

    # ========================================================
    # Reference extraction
    # ========================================================

    @classmethod
    def _extract_reference(
        cls,
        text: str,
    ) -> str | None:

        # First priority:
        #
        # explicit structured transaction references such as
        # UPI-..., IMPS-..., Chq: ..., UTR: ...
        for pattern in (
            cls.STRUCTURED_REFERENCE_PATTERNS
        ):

            match = pattern.search(
                text
            )

            if not match:
                continue

            value = (
                match.group(1)
                .strip()
            )

            if value:
                return value

        # Second priority:
        # generic transaction path/reference.
        for pattern in (
            cls.FALLBACK_REFERENCE_PATTERNS
        ):

            match = pattern.search(
                text
            )

            if not match:
                continue

            value = (
                match.group(1)
                .strip()
            )

            if value:
                return value

        return None

    # ========================================================
    # Description extraction
    # ========================================================

    @classmethod
    def _extract_description(
        cls,
        lines: list[str],
    ) -> str | None:

        if not lines:
            return None

        description_parts: list[
            str
        ] = []

        for index, line in enumerate(
            lines
        ):

            cleaned = line.strip()

            if not cleaned:
                continue

            # First transaction line can contain:
            #
            # 1 02 May 2025 UPI/...
            #
            # Remove optional sequence + transaction date while
            # preserving the description.
            if index == 0:

                cleaned = (
                    cls._remove_sequence_prefix(
                        cleaned
                    )
                )

                for pattern in (
                    cls.DATE_PATTERNS
                ):

                    match = pattern.search(
                        cleaned
                    )

                    if (
                        match
                        and match.start() == 0
                    ):

                        cleaned = (
                            cleaned[
                                match.end():
                            ]
                            .strip()
                        )

                        break

            if not cleaned:
                continue

            # Remove monetary values but preserve any surrounding
            # description text.
            if cls.AMOUNT_PATTERN.search(
                cleaned
            ):

                residual = (
                    cls.AMOUNT_PATTERN.sub(
                        "",
                        cleaned,
                    )
                    .strip(" ,-")
                )

                if not residual:
                    continue

                cleaned = residual

            # Reference-only cheque line.
            if re.fullmatch(
                r"(?:Chq|Cheque)"
                r"\s*[:#\-]?\s*"
                r"[A-Z0-9\-]+",
                cleaned,
                re.IGNORECASE,
            ):
                continue

            # Time-only line.
            if re.fullmatch(
                r"\d{1,2}:\d{2}"
                r"(?::\d{2})?",
                cleaned,
            ):
                continue

            if any(
                pattern.fullmatch(
                    cleaned
                )
                for pattern
                in cls.NOISE_PATTERNS
            ):
                continue

            description_parts.append(
                cleaned
            )

        if not description_parts:
            return None

        description = " ".join(
            description_parts
        )

        description = re.sub(
            r"\s+",
            " ",
            description,
        ).strip()

        return (
            description
            if description
            else None
        )

    # ========================================================
    # Noise removal
    # ========================================================

    @classmethod
    def _remove_noise_lines(
        cls,
        lines: list[str],
    ) -> list[str]:

        cleaned: list[str] = []

        for line in lines:

            stripped = line.strip()

            if any(
                pattern.fullmatch(
                    stripped
                )
                for pattern
                in cls.NOISE_PATTERNS
            ):
                continue

            cleaned.append(
                stripped
            )

        return cleaned

    # ========================================================
    # Confidence
    # ========================================================

    @staticmethod
    def _calculate_confidence(
        date: str | None,
        description: str | None,
        debit: Decimal | None,
        credit: Decimal | None,
        balance: Decimal | None,
        reference: str | None,
        direction_source: str | None,
        balance_reconciled: bool | None,
    ) -> float:

        score = 0.0

        if date:
            score += 0.20

        if description:
            score += 0.15

        if balance is not None:
            score += 0.20

        if (
            debit is not None
            or credit is not None
        ):
            score += 0.20

        if reference:
            score += 0.10

        if direction_source:
            score += 0.05

        if balance_reconciled is True:
            score += 0.10

        return round(
            min(score, 1.0),
            4,
        )


transaction_parser = TransactionParser()