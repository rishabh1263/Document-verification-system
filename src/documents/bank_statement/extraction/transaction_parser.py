"""
Generic Bank Statement Transaction Parser (PATCHED v2).

Fixes applied:
- Multi-line transaction merging: prevents orphan transactions when 
  continuation lines start with dates but have no amounts
- Reference number filtering: excludes 11-digit SBI reference numbers 
  from amount extraction
- Stricter amount regex: requires comma-separated format for large values
- Noise filtering: single digits and very small values without context 
  are treated as noise
- Better orphan handling: merges description-only blocks back into parent
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
        for key in ("debit", "credit", "balance"):
            value = data[key]
            if value is not None:
                data[key] = float(value)
        return data


@dataclass(frozen=True)
class TransactionParseResult:
    transaction_count: int
    transactions: tuple[Transaction, ...]
    rejected_blocks: int
    unresolved_direction_count: int
    reconciled_count: int
    parser_confidence: float
    opening_balance: Decimal | None

    def to_dict(self) -> dict:
        return {
            "transaction_count": self.transaction_count,
            "transactions": [t.to_dict() for t in self.transactions],
            "rejected_blocks": self.rejected_blocks,
            "unresolved_direction_count": self.unresolved_direction_count,
            "reconciled_count": self.reconciled_count,
            "parser_confidence": self.parser_confidence,
            "opening_balance": float(self.opening_balance) if self.opening_balance is not None else None,
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
    Generic transaction parser (PATCHED).

    Handles SBI-style multi-line transactions where:
    - Descriptions wrap across lines (some starting with dates)
    - Reference numbers appear before amounts
    - Amounts use Indian comma format (1,00,000.00)
    """

    BALANCE_TOLERANCE = Decimal("0.01")

    # ========================================================
    # Date patterns
    # ========================================================

    DATE_PATTERNS = (
        re.compile(
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})\b"),
        re.compile(
            r"\b(\d{1,2}[\-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\-/]\d{2,4})\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(\d{4}[/\.\-]\d{1,2}[/\.\-]\d{1,2})\b"),
    )

    # ========================================================
    # Amount recognition (PATCHED)
    # ========================================================
    #
    # Key changes:
    # 1. Requires comma separators for values >= 1000 (Indian format)
    # 2. Excludes 11-digit numbers (SBI reference numbers)
    # 3. Requires exactly 2 decimal places
    # 4. Negative sign support

    AMOUNT_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:₹|Rs\.?|INR)?\s*"
        r"("
        r"-?"
        r"(?:"
        r"\d{1,3}(?:,\d{2,3})+"  # Indian format: 1,00,000 or 12,34,567
        r"|"
        r"\d{1,2}"              # Small values: 25.00
        r")"
        r"\.\d{2}"
        r")"
        r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    # Also match amounts with CR/DR suffix like "1,43,278.86CR"
    AMOUNT_WITH_SUFFIX_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:₹|Rs\.?|INR)?\s*"
        r"("
        r"-?"
        r"(?:"
        r"\d{1,3}(?:,\d{2,3})+"
        r"|"
        r"\d{1,2}"
        r")"
        r"\.\d{2}"
        r")"
        r"\s*(?:CR|DR)?"
        r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    # ========================================================
    # Reference number patterns (PATCHED)
    # ========================================================
    # SBI puts 11-digit reference numbers before transaction descriptions
    # These must be excluded from amount extraction

    REFERENCE_NUMBER_PATTERNS = (
        # SBI transaction reference: 0097657105217
        re.compile(r"\b0\d{10}\b"),
        # Generic 11+ digit numbers that are likely references
        re.compile(r"\b\d{11,}\b"),
    )

    # ========================================================
    # Opening balance
    # ========================================================

    OPENING_BALANCE_PATTERN = re.compile(
        r"\bopening\s+balance\b.*?"
        r"(?:₹|Rs\.?|INR)?\s*"
        r"(\d{1,3}(?:,\d{2,3})*\.\d{2})",
        re.IGNORECASE,
    )

    BROUGHT_FORWARD_PATTERN = re.compile(
        r"\bbrought\s+forward\b.*?"
        r"(\d{1,3}(?:,\d{2,3})*\.\d{2})",
        re.IGNORECASE,
    )

    # ========================================================
    # Explicit transaction direction
    # ========================================================

    CREDIT_PATTERNS = (
        re.compile(r"(?:^|[/\s\-])CR(?:[/\s\-]|$)", re.IGNORECASE),
        re.compile(r"\bCREDIT\b", re.IGNORECASE),
        re.compile(r"\bCREDITED\b", re.IGNORECASE),
        re.compile(r"\bDEPOSIT\b", re.IGNORECASE),
        re.compile(r"\bDEPOSITED\b", re.IGNORECASE),
        re.compile(r"\bRECEIVED\b", re.IGNORECASE),
        re.compile(r"\bDEP\s+TFR\b", re.IGNORECASE),  # PATCH: DEP TFR = deposit/transfer (credit)
    )

    DEBIT_PATTERNS = (
        re.compile(r"(?:^|[/\s\-])DR(?:[/\s\-]|$)", re.IGNORECASE),
        re.compile(r"\bDEBIT\b", re.IGNORECASE),
        re.compile(r"\bDEBITED\b", re.IGNORECASE),
        re.compile(r"\bWITHDRAWAL\b", re.IGNORECASE),
        re.compile(r"\bWITHDRAWN\b", re.IGNORECASE),
        re.compile(r"\bWDL\b", re.IGNORECASE),  # PATCH: WDL = withdrawal
        re.compile(r"\bATM\s+WDL\b", re.IGNORECASE),
        re.compile(r"\bDIRECT\s+DR\b", re.IGNORECASE),
        re.compile(r"\bCASH\s+WITHDRAWAL\b", re.IGNORECASE),
    )

    # ========================================================
    # Reference extraction
    # ========================================================

    STRUCTURED_REFERENCE_PATTERNS = (
        re.compile(
            r"\b((?:UPI|IMPS|NEFT|RTGS|MB)-[A-Z0-9][A-Z0-9\-]{3,60})\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:Chq|Cheque)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-]{3,40})", re.IGNORECASE),
        re.compile(r"\b(?:UTR|REF|REFERENCE)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-/:]{3,60})", re.IGNORECASE),
    )

    FALLBACK_REFERENCE_PATTERNS = (
        re.compile(
            r"\b((?:UPI|IMPS|NEFT|RTGS)/[A-Z0-9][A-Z0-9/._@*\-]{5,100})",
            re.IGNORECASE,
        ),
    )

    # SBI-specific reference patterns
    SBI_REFERENCE_PATTERNS = (
        re.compile(r"\b(0\d{10})\b"),  # 11-digit SBI reference
        re.compile(r"\b(P\d{6}[A-Z0-9]{4,})\b"),  # HPCL subsidy refs like P012401D309B7
    )

    # ========================================================
    # Noise / repeated page lines
    # ========================================================

    NOISE_PATTERNS = (
        re.compile(r"^page\s+\d+$", re.IGNORECASE),
        re.compile(r"^date\s+.*balance$", re.IGNORECASE),
        re.compile(r"^date$", re.IGNORECASE),
        re.compile(r"^particulars$", re.IGNORECASE),
        re.compile(r"^description$", re.IGNORECASE),
        re.compile(r"^chq/?ref\.?\s*no\.?$", re.IGNORECASE),
        re.compile(r"^deposits?$", re.IGNORECASE),
        re.compile(r"^withdrawals?$", re.IGNORECASE),
        re.compile(r"^debits?$", re.IGNORECASE),
        re.compile(r"^credits?$", re.IGNORECASE),
        re.compile(r"^balance$", re.IGNORECASE),
        re.compile(r"^value\s+date$", re.IGNORECASE),
        re.compile(r"^post\s+date$", re.IGNORECASE),
        re.compile(r"^cheque\s+no/reference$", re.IGNORECASE),
        re.compile(r"^cheque\s+no$", re.IGNORECASE),
        re.compile(r"^no/reference$", re.IGNORECASE),
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

        if body_text is None:
            raise ValueError("body_text cannot be None.")
        if not isinstance(body_text, str):
            raise TypeError("body_text must be a string.")

        lines = [line.strip() for line in body_text.splitlines() if line.strip()]

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

        resolved_opening_balance = self._resolve_opening_balance(
            opening_balance=opening_balance,
            header_text=header_text,
        )

        cleaned_lines = self._remove_noise_lines(lines)

        # PATCH: Use smart segmentation that handles multi-line SBI transactions
        blocks = self._segment_transactions_smart(cleaned_lines)

        header_semantics = self._infer_header_semantics(transaction_header)

        transactions: list[Transaction] = []
        rejected_blocks = 0
        previous_balance = resolved_opening_balance

        for block in blocks:
            transaction = self._parse_block(
                block=block,
                sequence=len(transactions) + 1,
                header_semantics=header_semantics,
                previous_balance=previous_balance,
            )

            if transaction is None:
                rejected_blocks += 1
                continue

            transactions.append(transaction)

            if transaction.balance is not None:
                previous_balance = transaction.balance

        unresolved_direction_count = sum(
            1 for t in transactions if t.debit is None and t.credit is None
        )

        reconciled_count = sum(
            1 for t in transactions if t.balance_reconciled is True
        )

        confidence_values = [t.confidence for t in transactions]
        parser_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

        return TransactionParseResult(
            transaction_count=len(transactions),
            transactions=tuple(transactions),
            rejected_blocks=rejected_blocks,
            unresolved_direction_count=unresolved_direction_count,
            reconciled_count=reconciled_count,
            parser_confidence=round(parser_confidence, 4),
            opening_balance=resolved_opening_balance,
        )

    # ========================================================
    # PATCHED: reconcile_structured - missing method added
    # ========================================================

    def reconcile_structured(
        self,
        transactions: list[dict],
        opening_balance: Decimal | float | str | None = None,
        header_text: str | None = None,  # PATCH: Accept header_text to avoid TypeError
    ) -> list[dict]:
        """
        Reconcile transaction directions and balances after initial parsing.

        This method is called by the API routes to post-process parsed transactions
        and resolve any remaining direction ambiguities using balance delta logic.

        Args:
            transactions: List of transaction dicts from parse()
            opening_balance: Starting balance for reconciliation chain
            header_text: Header text (accepted for API compatibility, not used)

        Returns:
            List of reconciled transaction dicts
        """
        if not transactions:
            return transactions

        prev_balance = self._to_decimal(opening_balance) if opening_balance else None
        reconciled = []

        for txn in transactions:
            # Work with dict format (from to_dict())
            debit_val = txn.get('debit')
            credit_val = txn.get('credit')
            balance_val = txn.get('balance')

            d_debit = Decimal(str(debit_val)) if debit_val is not None else None
            d_credit = Decimal(str(credit_val)) if credit_val is not None else None
            d_balance = Decimal(str(balance_val)) if balance_val is not None else None

            # If direction is missing but we have balance delta, infer it
            if d_debit is None and d_credit is None and prev_balance is not None and d_balance is not None:
                delta = d_balance - prev_balance
                if abs(delta) > self.BALANCE_TOLERANCE:
                    if delta > 0:
                        txn['credit'] = float(abs(delta))
                        txn['direction_source'] = txn.get('direction_source', 'balance_delta_override')
                    else:
                        txn['debit'] = float(abs(delta))
                        txn['direction_source'] = txn.get('direction_source', 'balance_delta_override')

            # Verify balance reconciliation
            if prev_balance is not None and d_balance is not None:
                expected = prev_balance
                if txn.get('debit') is not None:
                    expected -= Decimal(str(txn['debit']))
                if txn.get('credit') is not None:
                    expected += Decimal(str(txn['credit']))
                txn['balance_reconciled'] = abs(expected - d_balance) < self.BALANCE_TOLERANCE
            else:
                txn['balance_reconciled'] = None

            if d_balance is not None:
                prev_balance = d_balance

            reconciled.append(txn)

        return reconciled

    # ========================================================
    # PATCHED: Smart transaction segmentation
    # ========================================================

    def _segment_transactions_smart(
        self,
        lines: list[str],
    ) -> list[_TransactionBlock]:
        """
        Smart segmentation that handles SBI multi-line transactions.

        A new transaction starts when:
        1. Line starts with a date AND
        2. Line contains at least one monetary amount OR
        3. Previous block already has a balance (is complete)

        Otherwise, append to current block (it's a continuation).
        """
        blocks: list[_TransactionBlock] = []
        current_lines: list[str] = []
        current_has_amounts = False
        current_has_balance = False

        def _finalize_block():
            nonlocal current_lines, current_has_amounts, current_has_balance
            if current_lines:
                blocks.append(_TransactionBlock(lines=current_lines))
            current_lines = []
            current_has_amounts = False
            current_has_balance = False

        for line in lines:
            if self._starts_with_date(line):
                # Check if this date line has amounts
                line_amounts = self._extract_amounts([line])
                line_has_amounts = len(line_amounts) > 0

                # Check if previous block is complete (has balance)
                prev_complete = current_has_balance

                # Start new transaction if:
                # - Previous block is complete, OR
                # - Current line has amounts (likely a real transaction start)
                if prev_complete or line_has_amounts:
                    _finalize_block()
                    current_lines = [line]
                    current_has_amounts = line_has_amounts
                    # Check if this line alone has a balance (2+ amounts)
                    current_has_balance = len(line_amounts) >= 2
                else:
                    # This is a continuation line that happens to start with a date
                    current_lines.append(line)
                    if line_has_amounts:
                        current_has_amounts = True
                        current_has_balance = len(line_amounts) >= 2
            else:
                # Not a date line - always append to current block
                if not current_lines:
                    # Skip lines before first date
                    continue
                current_lines.append(line)
                line_amounts = self._extract_amounts([line])
                if line_amounts:
                    current_has_amounts = True
                    # If we now have 2+ amounts total, we have a balance
                    total_amounts = self._extract_amounts(current_lines)
                    current_has_balance = len(total_amounts) >= 2

        _finalize_block()
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

        raw_text = "\n".join(block.lines)

        date = self._extract_date(block.lines[0])
        if date is None:
            return None

        # PATCH: Extract amounts with reference number filtering
        amounts = self._extract_amounts_safe(block.lines)

        if not amounts:
            # No amounts found - this might be a description-only orphan
            # Try to merge logic handled by smart segmentation
            return None

        # The last amount is the running balance
        balance = amounts[-1]
        transaction_amounts = amounts[:-1]

        debit: Decimal | None = None
        credit: Decimal | None = None
        direction_source: str | None = None
        balance_reconciled: bool | None = None

        # Step 1: previous-balance arithmetic reconciliation
        if transaction_amounts and previous_balance is not None and balance is not None:
            inferred_debit, inferred_credit, reconciled = self._infer_from_balance_delta(
                previous_balance=previous_balance,
                current_balance=balance,
                transaction_amounts=transaction_amounts,
            )
            if reconciled:
                debit = inferred_debit
                credit = inferred_credit
                direction_source = "balance_delta"
                balance_reconciled = True

        # Step 2: preserved debit/credit table columns
        if debit is None and credit is None and transaction_amounts:
            column_debit, column_credit = self._infer_from_amount_columns(
                transaction_amounts, header_semantics
            )
            if column_debit is not None or column_credit is not None:
                debit = column_debit
                credit = column_credit
                direction_source = "column_semantics"

        # Step 3: explicit DR / CR / narration semantic signal
        if debit is None and credit is None and transaction_amounts:
            explicit_signal = self._detect_direction_signal(raw_text)
            if explicit_signal in {"debit", "credit"}:
                amount = self._select_primary_amount(transaction_amounts)
                if explicit_signal == "debit":
                    debit = amount
                else:
                    credit = amount
                direction_source = "explicit_signal"

        # Step 4: validate fallback direction against balance
        if balance_reconciled is not True and previous_balance is not None and balance is not None and (debit is not None or credit is not None):
            balance_reconciled = self._reconcile_transaction(
                previous_balance=previous_balance,
                current_balance=balance,
                debit=debit,
                credit=credit,
            )

        reference = self._extract_reference(raw_text)
        description = self._extract_description(block.lines)

        confidence = self._calculate_confidence(
            date=date,
            description=description,
            debit=debit,
            credit=credit,
            balance=balance,
            reference=reference,
            direction_source=direction_source,
            balance_reconciled=balance_reconciled,
        )

        return Transaction(
            sequence=sequence,
            date=date,
            description=description,
            reference=reference,
            debit=debit,
            credit=credit,
            balance=balance,
            direction_source=direction_source,
            balance_reconciled=balance_reconciled,
            confidence=confidence,
            raw_text=raw_text,
        )

    # ========================================================
    # PATCHED: Safe amount extraction
    # ========================================================

    def _extract_amounts_safe(
        self,
        lines: list[str],
    ) -> list[Decimal]:
        """
        Extract amounts while filtering out reference numbers.
        """
        values: list[Decimal] = []

        for line in lines:
            # First, mask out reference numbers so they don't get parsed as amounts
            masked_line = line
            for pattern in self.REFERENCE_NUMBER_PATTERNS:
                masked_line = pattern.sub(" REF ", masked_line)

            # Also mask out single digits that are likely noise
            masked_line = re.sub(r"(?<![A-Za-z0-9])\d(?![A-Za-z0-9.,])", " ", masked_line)

            for match in self.AMOUNT_PATTERN.finditer(masked_line):
                value = self._to_decimal(match.group(1))
                if value is not None:
                    values.append(value)

        return values

    # ========================================================
    # Opening balance
    # ========================================================

    def _resolve_opening_balance(
        self,
        opening_balance: Decimal | float | str | None,
        header_text: str | None,
    ) -> Decimal | None:
        if opening_balance is not None:
            parsed = self._to_decimal(opening_balance)
            if parsed is not None:
                return parsed
        if not header_text:
            return None
        return self._extract_opening_balance(header_text)

    def _extract_opening_balance(
        self,
        header_text: str,
    ) -> Decimal | None:
        if not header_text:
            return None

        # Try "Opening Balance" first
        match = self.OPENING_BALANCE_PATTERN.search(header_text)
        if match:
            return self._to_decimal(match.group(1))

        # Try "Brought Forward"
        match = self.BROUGHT_FORWARD_PATTERN.search(header_text)
        if match:
            return self._to_decimal(match.group(1))

        return None

    # ========================================================
    # Date handling
    # ========================================================

    @classmethod
    def _starts_with_date(cls, line: str) -> bool:
        stripped = cls._remove_sequence_prefix(line.strip())
        for pattern in cls.DATE_PATTERNS:
            match = pattern.search(stripped)
            if match and match.start() == 0:
                return True
        return False

    @classmethod
    def _extract_date(cls, line: str) -> str | None:
        candidate = cls._remove_sequence_prefix(line.strip())
        for pattern in cls.DATE_PATTERNS:
            match = pattern.search(candidate)
            if not match:
                continue
            normalized = cls._normalize_date(match.group(1))
            if normalized:
                return normalized
        return None

    @staticmethod
    def _remove_sequence_prefix(line: str) -> str:
        return re.sub(r"^\d+\s+(?=\d{1,2}(?:\s|[-/.]))", "", line, count=1)

    @staticmethod
    def _normalize_date(value: str) -> str | None:
        value = value.strip().replace(",", "")
        formats = (
            "%d %B %Y", "%d %b %Y", "%d %B %y", "%d %b %y",
            "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y", "%d-%B-%y",
            "%d/%b/%Y", "%d/%B/%Y", "%d/%b/%y", "%d/%B/%y",
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
            "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        )
        for date_format in formats:
            try:
                parsed = datetime.strptime(value, date_format)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    # ========================================================
    # Amount extraction (legacy, used by other methods)
    # ========================================================

    @classmethod
    def _extract_amounts(cls, lines: list[str]) -> list[Decimal]:
        values: list[Decimal] = []
        for line in lines:
            for match in cls.AMOUNT_PATTERN.finditer(line):
                value = cls._to_decimal(match.group(1))
                if value is not None:
                    values.append(value)
        return values

    @staticmethod
    def _to_decimal(value: Decimal | float | int | str) -> Decimal | None:
        if isinstance(value, Decimal):
            return value
        raw = str(value).replace(",", "").replace("₹", "").strip()
        raw = re.sub(r"^(?:Rs\.?|INR)\s*", "", raw, flags=re.IGNORECASE)

        suffix_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(CR|DR)", raw, flags=re.IGNORECASE)
        if suffix_match:
            number_text = suffix_match.group(1)
            suffix = suffix_match.group(2).upper()
            try:
                amount = Decimal(number_text)
            except (InvalidOperation, ValueError):
                return None
            return -abs(amount) if suffix == "DR" else amount

        try:
            return Decimal(raw)
        except (InvalidOperation, ValueError):
            return None

    # ========================================================
    # Direction signals
    # ========================================================

    @classmethod
    def _detect_direction_signal(cls, text: str) -> str | None:
        credit = any(pattern.search(text) for pattern in cls.CREDIT_PATTERNS)
        debit = any(pattern.search(text) for pattern in cls.DEBIT_PATTERNS)
        if credit and not debit:
            return "credit"
        if debit and not credit:
            return "debit"
        return None

    # ========================================================
    # Header semantics
    # ========================================================

    @staticmethod
    def _infer_header_semantics(header: str | None) -> dict:
        result = {"money_columns": []}
        if not header:
            return result
        words = re.findall(r"[A-Za-z]+", header.lower())
        for word in words:
            if word in {"withdrawal", "withdrawals", "debit", "debits", "dr"}:
                if "debit" not in result["money_columns"]:
                    result["money_columns"].append("debit")
            elif word in {"deposit", "deposits", "credit", "credits", "cr"}:
                if "credit" not in result["money_columns"]:
                    result["money_columns"].append("credit")
        return result

    # ========================================================
    # Column-based inference
    # ========================================================

    @staticmethod
    def _infer_from_amount_columns(
        amounts: list[Decimal],
        header_semantics: dict,
    ) -> tuple[Decimal | None, Decimal | None]:
        if not amounts:
            return None, None
        if len(amounts) == 1:
            return None, None
        columns = header_semantics.get("money_columns", [])
        if len(amounts) >= 2 and len(columns) >= 2:
            first, second = amounts[0], amounts[1]
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
    ) -> tuple[Decimal | None, Decimal | None, bool]:
        if not transaction_amounts:
            return None, None, False

        candidates: list[tuple[str, Decimal, Decimal]] = []
        for amount in transaction_amounts:
            debit_expected = previous_balance - amount
            debit_error = abs(debit_expected - current_balance)
            if debit_error <= self.BALANCE_TOLERANCE:
                candidates.append(("debit", amount, debit_error))

            credit_expected = previous_balance + amount
            credit_error = abs(credit_expected - current_balance)
            if credit_error <= self.BALANCE_TOLERANCE:
                candidates.append(("credit", amount, credit_error))

        if not candidates:
            return None, None, False

        candidates.sort(key=lambda item: item[2])
        best_direction = candidates[0][0]
        best_amount = candidates[0][1]

        if len(candidates) > 1:
            first_error = candidates[0][2]
            second_error = candidates[1][2]
            if first_error == second_error and (candidates[0][0] != candidates[1][0] or candidates[0][1] != candidates[1][1]):
                return None, None, False

        if best_direction == "debit":
            return best_amount, None, True
        return None, best_amount, True

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
        difference = abs(expected - current_balance)
        return difference <= self.BALANCE_TOLERANCE

    # ========================================================
    # Primary amount
    # ========================================================

    @staticmethod
    def _select_primary_amount(amounts: list[Decimal]) -> Decimal:
        return amounts[0]

    # ========================================================
    # Reference extraction
    # ========================================================

    @classmethod
    def _extract_reference(cls, text: str) -> str | None:
        for pattern in cls.STRUCTURED_REFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        for pattern in cls.FALLBACK_REFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        # PATCH: SBI-specific references
        for pattern in cls.SBI_REFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    # ========================================================
    # Description extraction
    # ========================================================

    @classmethod
    def _extract_description(cls, lines: list[str]) -> str | None:
        if not lines:
            return None

        description_parts: list[str] = []
        for index, line in enumerate(lines):
            cleaned = line.strip()
            if not cleaned:
                continue

            if index == 0:
                cleaned = cls._remove_sequence_prefix(cleaned)
                for pattern in cls.DATE_PATTERNS:
                    match = pattern.search(cleaned)
                    if match and match.start() == 0:
                        cleaned = cleaned[match.end():].strip()
                        break

            if not cleaned:
                continue

            # Remove monetary values but preserve surrounding text
            if cls.AMOUNT_PATTERN.search(cleaned):
                residual = cls.AMOUNT_PATTERN.sub("", cleaned).strip(" ,-")
                if not residual:
                    continue
                cleaned = residual

            # Remove reference numbers from description
            for pattern in cls.REFERENCE_NUMBER_PATTERNS:
                cleaned = pattern.sub("", cleaned).strip()

            # Reference-only cheque line
            if re.fullmatch(r"(?:Chq|Cheque)\s*[:#\-]?\s*[A-Z0-9\-]+", cleaned, re.IGNORECASE):
                continue

            # Time-only line
            if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", cleaned):
                continue

            if any(pattern.fullmatch(cleaned) for pattern in cls.NOISE_PATTERNS):
                continue

            description_parts.append(cleaned)

        if not description_parts:
            return None

        description = " ".join(description_parts)
        description = re.sub(r"\s+", " ", description).strip()
        return description if description else None

    # ========================================================
    # Noise removal
    # ========================================================

    @classmethod
    def _remove_noise_lines(cls, lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if any(pattern.fullmatch(stripped) for pattern in cls.NOISE_PATTERNS):
                continue
            cleaned.append(stripped)
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
        if debit is not None or credit is not None:
            score += 0.20
        if reference:
            score += 0.10
        if direction_source:
            score += 0.05
        if balance_reconciled is True:
            score += 0.10
        return round(min(score, 1.0), 4)


transaction_parser = TransactionParser()