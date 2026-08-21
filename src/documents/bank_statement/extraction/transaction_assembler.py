"""
Generic Logical Transaction Assembler.

Phase 2 - Document Intelligence / Extraction

Purpose
-------
Convert geometry-aware physical OCR rows into logical bank transactions.

This module deliberately does NOT contain:
    - bank names
    - SBI/Kotak/Canara-specific rules
    - fixed transaction templates
    - OCR engine logic

It operates on semantic cells produced by the spatial table parser.

Pipeline
--------
OCR tokens
    -> spatial rows
    -> semantic cells
    -> logical transaction assembly
    -> transaction parser / reconciliation
    -> standardized extraction schema

Design principles
-----------------
1. A date alone does not prove a new transaction; monetary evidence makes the start strong.
2. Dated text-only rows are resolved with conservative forward lookahead and may be continuations.
3. Continuation text is attached according to semantic column.
4. Monetary values remain column-aware.
5. Header/footer/noise rows are ignored conservatively.
6. No bank-specific assumptions are made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# REGEX
# ============================================================

_DATE_PATTERNS = (
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(r"\b\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}\b"),
    re.compile(r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}\b"),
)


_AMOUNT_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        ₹\s*
    )?
    [+-]?
    (?:
        \d{1,3}(?:,\d{2,3})*
        |
        \d+
    )
    (?:\.\d{1,2})?
    \s*
    (?:CR|DR)?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


_HEADER_WORDS = {
    "date",
    "post date",
    "value date",
    "transaction date",
    "description",
    "particulars",
    "narration",
    "reference",
    "no/reference",
    "ref no",
    "debit",
    "credit",
    "withdrawal",
    "deposit",
    "balance",
}


_NOISE_PATTERNS = (
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*-+\s*$"),
    re.compile(r"^\s*end\s+of\s+statement\s*$", re.IGNORECASE),
)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class LogicalTransaction:
    """
    Geometry-derived logical transaction before financial reconciliation.
    """

    sequence: int
    page_number: Optional[int]

    date: Optional[str] = None
    value_date: Optional[str] = None

    description: Optional[str] = None
    reference: Optional[str] = None

    debit: Optional[str] = None
    credit: Optional[str] = None
    balance: Optional[str] = None

    raw_text: str = ""

    source_row_numbers: List[int] = field(default_factory=list)

    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "page_number": self.page_number,
            "date": self.date,
            "value_date": self.value_date,
            "description": self.description,
            "reference": self.reference,
            "debit": self.debit,
            "credit": self.credit,
            "balance": self.balance,
            "raw_text": self.raw_text,
            "source_row_numbers": list(self.source_row_numbers),
            "confidence": self.confidence,
        }


@dataclass
class TransactionAssemblyResult:
    transactions: List[LogicalTransaction] = field(default_factory=list)

    input_row_count: int = 0
    transaction_count: int = 0
    continuation_row_count: int = 0
    ignored_row_count: int = 0
    orphan_row_count: int = 0

    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transactions": [t.to_dict() for t in self.transactions],
            "input_row_count": self.input_row_count,
            "transaction_count": self.transaction_count,
            "continuation_row_count": self.continuation_row_count,
            "ignored_row_count": self.ignored_row_count,
            "orphan_row_count": self.orphan_row_count,
            "confidence": self.confidence,
        }


# ============================================================
# ASSEMBLER
# ============================================================

class TransactionAssembler:
    """
    Assemble spatial OCR rows into logical transaction blocks.

    Expected row shape
    ------------------
    The assembler intentionally uses duck typing.

    Each row may be:
        - dataclass/object
        - dictionary

    Expected attributes/keys:

        row_number
        page_number
        raw_text
        cells
        confidence

    Each cell may contain:

        semantic
        text
        confidence
    """

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def assemble(
        self,
        rows: Sequence[Any],
    ) -> TransactionAssemblyResult:

        result = TransactionAssemblyResult(
            input_row_count=len(rows),
        )

        transactions: List[LogicalTransaction] = []
        current: Optional[LogicalTransaction] = None

        continuation_count = 0
        ignored_count = 0
        orphan_count = 0

        # Analyse once so weak dated rows can be resolved using
        # nearby physical rows without introducing bank-specific rules.
        analysed_rows: List[Tuple[Any, Optional[Dict[str, Any]], bool]] = []

        for row in rows:
            row_text = self._clean_text(
                self._get(row, "raw_text", "") or ""
            )

            if not row_text:
                analysed_rows.append((row, None, True))
                continue

            if self._is_noise_row(row_text):
                analysed_rows.append((row, None, True))
                continue

            cells = self._extract_cells(row)

            if self._is_header_row(cells, row_text):
                analysed_rows.append((row, None, True))
                continue

            row_info = self._analyse_row(
                row=row,
                cells=cells,
                row_text=row_text,
            )

            analysed_rows.append((row, row_info, False))

        for index, (row, row_info, preignored) in enumerate(analysed_rows):

            if preignored or row_info is None:
                ignored_count += 1
                continue

            # =================================================
            # START DECISION
            # =================================================
            #
            # Strong start:
            #     date + monetary evidence
            #
            # Weak dated row:
            #     date + description/reference but no money
            #
            # A weak dated row starts a transaction only when
            # subsequent undated continuation rows provide money
            # before another dated row appears. Otherwise, when a
            # current transaction exists, it is treated as wrapped
            # continuation text. This handles OCR layouts where the
            # same transaction date is repeated on physical lines.
            # =================================================

            starts_transaction = row_info["starts_transaction"]

            if (
                row_info["has_date"]
                and not row_info["has_money"]
                and row_info["has_transaction_text"]
            ):
                gains_money = self._weak_dated_row_gains_money_before_next_dated(
                    analysed_rows=analysed_rows,
                    start_index=index,
                )

                if current is None:
                    # At the beginning of a stream, preserve a plausible
                    # dated transaction even if its monetary row is missing.
                    starts_transaction = True
                elif row_info["starts_transaction"]:
                    # Preserve a positive structural boundary decision.
                    # A dated transaction-like row does not need monetary
                    # evidence on the same physical row. OCR/scanned layouts
                    # may split fields vertically, and some transactions may
                    # have incomplete monetary extraction.
                    starts_transaction = True
                else:
                    # Only ambiguous weak dated rows require look-ahead
                    # monetary evidence before being promoted to a boundary.
                    starts_transaction = gains_money

            # =================================================
            # NEW TRANSACTION
            # =================================================

            if starts_transaction:

                if current is not None:
                    self._finalize_transaction(current)
                    transactions.append(current)

                current = self._create_transaction(
                    sequence=len(transactions) + 1,
                    row=row,
                    row_info=row_info,
                )

                continue

            # =================================================
            # CONTINUATION ROW
            # =================================================

            if current is not None and self._is_continuation(
                row_info=row_info,
                current=current,
            ):
                self._merge_continuation(
                    transaction=current,
                    row=row,
                    row_info=row_info,
                )

                continuation_count += 1
                continue

            # =================================================
            # NON-TRANSACTION / ORPHAN
            # =================================================

            if self._contains_useful_transaction_content(row_info):
                orphan_count += 1
            else:
                ignored_count += 1

        if current is not None:
            self._finalize_transaction(current)
            transactions.append(current)

        for index, transaction in enumerate(transactions, start=1):
            transaction.sequence = index

        result.transactions = transactions
        result.transaction_count = len(transactions)
        result.continuation_row_count = continuation_count
        result.ignored_row_count = ignored_count
        result.orphan_row_count = orphan_count

        result.confidence = self._calculate_result_confidence(
            transactions=transactions,
            orphan_count=orphan_count,
        )

        return result

    def _weak_dated_row_gains_money_before_next_dated(
        self,
        analysed_rows: Sequence[
            Tuple[Any, Optional[Dict[str, Any]], bool]
        ],
        start_index: int,
    ) -> bool:
        """
        Resolve a dated text-only physical row conservatively.

        Look only at following non-ignored rows on the same page.
        If monetary evidence appears before another dated row, the
        weak row is a genuine transaction start. If another dated row
        arrives first, the weak row is most likely a wrapped/repeated
        date continuation belonging to the current transaction.

        This is intentionally bank-independent.
        """

        start_row = analysed_rows[start_index][0]
        start_page = self._safe_int(
            self._get(start_row, "page_number", None)
        )

        for next_index in range(
            start_index + 1,
            len(analysed_rows),
        ):
            next_row, next_info, next_ignored = analysed_rows[next_index]

            if next_ignored or next_info is None:
                continue

            next_page = self._safe_int(
                self._get(next_row, "page_number", None)
            )

            if (
                start_page is not None
                and next_page is not None
                and next_page != start_page
            ):
                return False

            if next_info["has_date"]:
                return False

            if next_info["has_money"]:
                return True

        return False

    # ========================================================
    # ROW ANALYSIS
    # ========================================================

    def _analyse_row(
        self,
        row: Any,
        cells: Sequence[Any],
        row_text: str,
    ) -> Dict[str, Any]:

        semantic_values: Dict[str, List[str]] = {}

        confidence_values: List[float] = []

        for cell in cells:

            semantic = str(
                self._get(cell, "semantic", "") or ""
            ).strip().lower()

            text = self._clean_text(
                self._get(cell, "text", "") or ""
            )

            if not semantic or not text:
                continue

            semantic_values.setdefault(
                semantic,
                [],
            ).append(text)

            confidence = self._safe_float(
                self._get(cell, "confidence", None)
            )

            if confidence is not None:
                confidence_values.append(confidence)

        # -----------------------------------------------------
        # DATE
        # -----------------------------------------------------

        date_text = self._join_values(
            semantic_values.get("date", [])
        )

        value_date_text = self._join_values(
            semantic_values.get("value_date", [])
        )

        detected_date = self._extract_date(date_text)

        if detected_date is None:
            detected_date = self._extract_date(row_text)

        detected_value_date = self._extract_date(
            value_date_text
        )

        # -----------------------------------------------------
        # DESCRIPTION
        # -----------------------------------------------------

        description = self._join_values(
            semantic_values.get("description", [])
        )

        # Some geometry parsers may use alternative semantic
        # labels. Support generic aliases.
        if not description:
            description = self._join_values(
                semantic_values.get("particulars", [])
            )

        if not description:
            description = self._join_values(
                semantic_values.get("narration", [])
            )

        # -----------------------------------------------------
        # REFERENCE
        # -----------------------------------------------------

        reference = self._join_values(
            semantic_values.get("reference", [])
        )

        # -----------------------------------------------------
        # MONEY
        # -----------------------------------------------------

        debit = self._extract_amount_from_values(
            semantic_values.get("debit", [])
        )

        credit = self._extract_amount_from_values(
            semantic_values.get("credit", [])
        )

        balance = self._extract_amount_from_values(
            semantic_values.get("balance", [])
        )

        # -----------------------------------------------------
        # START DECISION
        # -----------------------------------------------------

        has_date = detected_date is not None

        has_money = any(
            value is not None
            for value in (
                debit,
                credit,
                balance,
            )
        )

        has_transaction_text = bool(
            description or reference
        )

        starts_transaction = bool(
            has_date
            and (
                has_money
                or has_transaction_text
            )
        )

        row_confidence = self._average(
            confidence_values
        )

        return {
            "date": detected_date,
            "value_date": detected_value_date,
            "description": description,
            "reference": reference,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "has_date": has_date,
            "has_money": has_money,
            "has_transaction_text": has_transaction_text,
            "starts_transaction": starts_transaction,
            "confidence": row_confidence,
            "raw_text": row_text,
        }

    # ========================================================
    # TRANSACTION CREATION
    # ========================================================

    def _create_transaction(
        self,
        sequence: int,
        row: Any,
        row_info: Dict[str, Any],
    ) -> LogicalTransaction:

        row_number = self._safe_int(
            self._get(row, "row_number", None)
        )

        page_number = self._safe_int(
            self._get(row, "page_number", None)
        )

        source_rows: List[int] = []

        if row_number is not None:
            source_rows.append(row_number)

        return LogicalTransaction(
            sequence=sequence,
            page_number=page_number,
            date=row_info["date"],
            value_date=row_info["value_date"],
            description=row_info["description"],
            reference=row_info["reference"],
            debit=row_info["debit"],
            credit=row_info["credit"],
            balance=row_info["balance"],
            raw_text=row_info["raw_text"],
            source_row_numbers=source_rows,
            confidence=row_info["confidence"],
        )

    # ========================================================
    # CONTINUATION LOGIC
    # ========================================================

    def _is_continuation(
        self,
        row_info: Dict[str, Any],
        current: LogicalTransaction,
    ) -> bool:

        # A strong dated row with monetary evidence should already
        # have started a new transaction. A dated text-only row may
        # still be a continuation when assemble() rejected it as a
        # weak start after lookahead.
        if row_info["has_date"] and row_info["has_money"]:
            return False

        # Description/reference-only rows are the strongest
        # continuation signal.
        if (
            row_info["description"]
            or row_info["reference"]
        ):
            return True

        # Monetary information without a date can be a wrapped
        # transaction line. Allow it only when it can fill
        # missing financial fields.
        if row_info["debit"] and not current.debit:
            return True

        if row_info["credit"] and not current.credit:
            return True

        if row_info["balance"] and not current.balance:
            return True

        return False

    def _merge_continuation(
        self,
        transaction: LogicalTransaction,
        row: Any,
        row_info: Dict[str, Any],
    ) -> None:

        # -----------------------------------------------------
        # DESCRIPTION
        # -----------------------------------------------------

        transaction.description = self._merge_text(
            transaction.description,
            row_info["description"],
        )

        # -----------------------------------------------------
        # REFERENCE
        # -----------------------------------------------------

        transaction.reference = self._merge_text(
            transaction.reference,
            row_info["reference"],
        )

        # -----------------------------------------------------
        # VALUE DATE
        # -----------------------------------------------------

        if (
            not transaction.value_date
            and row_info["value_date"]
        ):
            transaction.value_date = row_info[
                "value_date"
            ]

        # -----------------------------------------------------
        # MONEY
        # -----------------------------------------------------

        if (
            transaction.debit is None
            and row_info["debit"] is not None
        ):
            transaction.debit = row_info["debit"]

        if (
            transaction.credit is None
            and row_info["credit"] is not None
        ):
            transaction.credit = row_info["credit"]

        if (
            transaction.balance is None
            and row_info["balance"] is not None
        ):
            transaction.balance = row_info["balance"]

        # -----------------------------------------------------
        # RAW TEXT
        # -----------------------------------------------------

        if row_info["raw_text"]:
            if transaction.raw_text:
                transaction.raw_text += (
                    "\n" + row_info["raw_text"]
                )
            else:
                transaction.raw_text = row_info[
                    "raw_text"
                ]

        # -----------------------------------------------------
        # SOURCE ROW
        # -----------------------------------------------------

        row_number = self._safe_int(
            self._get(row, "row_number", None)
        )

        if (
            row_number is not None
            and row_number
            not in transaction.source_row_numbers
        ):
            transaction.source_row_numbers.append(
                row_number
            )

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        incoming_confidence = row_info[
            "confidence"
        ]

        if incoming_confidence > 0:

            if transaction.confidence > 0:
                transaction.confidence = (
                    transaction.confidence
                    + incoming_confidence
                ) / 2.0
            else:
                transaction.confidence = (
                    incoming_confidence
                )

    # ========================================================
    # FINALIZATION
    # ========================================================

    def _finalize_transaction(
        self,
        transaction: LogicalTransaction,
    ) -> None:

        transaction.description = self._clean_optional(
            transaction.description
        )

        transaction.reference = self._clean_optional(
            transaction.reference
        )

        transaction.raw_text = (
            transaction.raw_text.strip()
        )

        transaction.confidence = round(
            max(
                0.0,
                min(
                    1.0,
                    transaction.confidence,
                ),
            ),
            4,
        )

    # ========================================================
    # HEADER / NOISE
    # ========================================================

    def _is_header_row(
        self,
        cells: Sequence[Any],
        row_text: str,
    ) -> bool:

        normalized = self._normalize_for_match(
            row_text
        )

        matches = 0

        for header_word in _HEADER_WORDS:
            if header_word in normalized:
                matches += 1

        # Multiple table-header semantics appearing together
        # strongly indicate a header row.
        if matches >= 3:
            return True

        semantics = {
            str(
                self._get(cell, "semantic", "") or ""
            )
            .strip()
            .lower()
            for cell in cells
        }

        header_semantics = {
            "date",
            "description",
            "reference",
            "debit",
            "credit",
            "balance",
        }

        if (
            len(
                semantics.intersection(
                    header_semantics
                )
            )
            >= 4
            and matches >= 2
        ):
            return True

        return False

    def _is_noise_row(
        self,
        text: str,
    ) -> bool:

        for pattern in _NOISE_PATTERNS:
            if pattern.match(text):
                return True

        return False

    # ========================================================
    # CONTENT HELPERS
    # ========================================================

    def _contains_useful_transaction_content(
        self,
        row_info: Dict[str, Any],
    ) -> bool:

        return bool(
            row_info["description"]
            or row_info["reference"]
            or row_info["debit"]
            or row_info["credit"]
            or row_info["balance"]
        )

    def _extract_date(
        self,
        text: Optional[str],
    ) -> Optional[str]:

        if not text:
            return None

        for pattern in _DATE_PATTERNS:
            match = pattern.search(text)

            if match:
                return match.group(0)

        return None

    def _extract_amount_from_values(
        self,
        values: Iterable[str],
    ) -> Optional[str]:

        for value in values:

            cleaned = self._clean_text(value)

            if not cleaned:
                continue

            # First try the entire semantic cell.
            if _AMOUNT_PATTERN.match(cleaned):
                return cleaned

            # Then try individual whitespace tokens.
            parts = cleaned.split()

            for part in reversed(parts):
                if _AMOUNT_PATTERN.match(part):
                    return part

        return None

    # ========================================================
    # CELL EXTRACTION
    # ========================================================

    def _extract_cells(
        self,
        row: Any,
    ) -> Sequence[Any]:

        cells = self._get(
            row,
            "cells",
            (),
        )

        if cells is None:
            return ()

        return cells

    # ========================================================
    # RESULT CONFIDENCE
    # ========================================================

    def _calculate_result_confidence(
        self,
        transactions: Sequence[
            LogicalTransaction
        ],
        orphan_count: int,
    ) -> float:

        if not transactions:
            return 0.0

        transaction_confidences = [
            transaction.confidence
            for transaction in transactions
            if transaction.confidence > 0
        ]

        base = self._average(
            transaction_confidences
        )

        if base <= 0:
            base = 0.5

        total = (
            len(transactions)
            + orphan_count
        )

        if total > 0:
            orphan_ratio = (
                orphan_count / total
            )
        else:
            orphan_ratio = 0.0

        penalty = min(
            0.35,
            orphan_ratio * 0.5,
        )

        confidence = base - penalty

        return round(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            4,
        )

    # ========================================================
    # GENERIC UTILITIES
    # ========================================================

    @staticmethod
    def _get(
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:

        if isinstance(obj, dict):
            return obj.get(
                name,
                default,
            )

        return getattr(
            obj,
            name,
            default,
        )

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(value)

        text = text.replace(
            "\u00a0",
            " ",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text.strip()

    def _clean_optional(
        self,
        value: Optional[str],
    ) -> Optional[str]:

        if value is None:
            return None

        cleaned = self._clean_text(
            value
        )

        return cleaned or None

    def _join_values(
        self,
        values: Iterable[str],
    ) -> Optional[str]:

        cleaned = [
            self._clean_text(value)
            for value in values
            if self._clean_text(value)
        ]

        if not cleaned:
            return None

        return " ".join(cleaned)

    def _merge_text(
        self,
        existing: Optional[str],
        incoming: Optional[str],
    ) -> Optional[str]:

        existing_clean = self._clean_optional(
            existing
        )

        incoming_clean = self._clean_optional(
            incoming
        )

        if not incoming_clean:
            return existing_clean

        if not existing_clean:
            return incoming_clean

        # Prevent exact duplicate continuation fragments.
        if (
            incoming_clean.lower()
            == existing_clean.lower()
        ):
            return existing_clean

        return (
            existing_clean
            + " "
            + incoming_clean
        )

    @staticmethod
    def _normalize_for_match(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9/ ]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:

        try:
            if value is None:
                return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> Optional[int]:

        try:
            if value is None:
                return None

            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _average(
        values: Iterable[float],
    ) -> float:

        values = list(values)

        if not values:
            return 0.0

        return sum(values) / len(values)


# ============================================================
# SINGLETON
# ============================================================

transaction_assembler = TransactionAssembler()