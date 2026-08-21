"""
Generic OCR Layout Reconstructor.

Phase 2 - Document Intelligence / Extraction.

Purpose
-------
OCR text alone loses table structure.

For example, OCR may produce:

    05-02-2024
    INTEREST CREDIT
    25.00
    303.86CR

without telling downstream parsers which numeric value belongs to
debit, credit, or balance.

ocr_extractor.py preserves OCR token geometry.

This module converts those spatial OCR tokens into generic logical
rows and infers transaction-table column semantics.

Responsibilities
----------------
- group OCR tokens into physical/logical rows
- sort row tokens left-to-right
- detect transaction-table header rows
- infer generic semantic columns
- assign tokens to inferred columns
- reconstruct transaction-like rows
- preserve confidence and geometry
- remain bank-independent

This module does NOT:
- identify a bank
- contain bank-specific templates
- contain bank-specific coordinates
- parse metadata
- perform OCR
- reconcile balances
- determine fraud/tampering
- calculate risk scores

Semantic concepts supported:
    date
    description
    reference
    debit
    credit
    balance

The same concepts may appear under different labels such as:

    Date / Txn Date / Transaction Date
    Description / Particulars / Narration / Details
    Chq No / Ref No / Reference
    Debit / Withdrawal / Dr
    Credit / Deposit / Cr
    Balance / Closing Balance

Coordinates are normalized to [0, 1], so the reconstruction does not
depend on page DPI or physical page dimensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from statistics import median
from typing import Iterable, Sequence

from .ocr_extractor import OCRPage, OCRToken


# ============================================================
# CONSTANTS
# ============================================================


COLUMN_DATE = "date"
COLUMN_DESCRIPTION = "description"
COLUMN_REFERENCE = "reference"
COLUMN_DEBIT = "debit"
COLUMN_CREDIT = "credit"
COLUMN_BALANCE = "balance"

SEMANTIC_COLUMNS = (
    COLUMN_DATE,
    COLUMN_DESCRIPTION,
    COLUMN_REFERENCE,
    COLUMN_DEBIT,
    COLUMN_CREDIT,
    COLUMN_BALANCE,
)


# ============================================================
# RESULT MODELS
# ============================================================


@dataclass(frozen=True)
class LayoutToken:
    """
    OCR token after layout normalization.

    This intentionally copies the small amount of geometry required
    by the layout layer instead of exposing PaddleOCR structures.
    """

    text: str
    confidence: float | None

    x_min: float | None
    y_min: float | None
    x_max: float | None
    y_max: float | None

    x_center: float | None
    y_center: float | None

    width: float | None
    height: float | None

    x_min_norm: float | None
    y_min_norm: float | None
    x_max_norm: float | None
    y_max_norm: float | None

    x_center_norm: float | None
    y_center_norm: float | None

    @classmethod
    def from_ocr_token(
        cls,
        token: OCRToken,
    ) -> "LayoutToken":

        return cls(
            text=token.text,
            confidence=token.confidence,

            x_min=token.x_min,
            y_min=token.y_min,
            x_max=token.x_max,
            y_max=token.y_max,

            x_center=token.x_center,
            y_center=token.y_center,

            width=token.width,
            height=token.height,

            x_min_norm=token.x_min_norm,
            y_min_norm=token.y_min_norm,
            x_max_norm=token.x_max_norm,
            y_max_norm=token.y_max_norm,

            x_center_norm=token.x_center_norm,
            y_center_norm=token.y_center_norm,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayoutRow:
    """
    One reconstructed physical row.
    """

    row_number: int
    page_number: int

    text: str

    tokens: tuple[LayoutToken, ...]

    y_center_norm: float | None

    x_min_norm: float | None
    x_max_norm: float | None

    confidence: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HeaderColumn:
    """
    One inferred semantic table column.

    anchor_x:
        normalized horizontal location of the header concept.

    left_boundary/right_boundary:
        inferred horizontal region belonging to the semantic column.
    """

    semantic: str

    label: str

    anchor_x: float

    left_boundary: float
    right_boundary: float

    confidence: float

    def contains(
        self,
        x: float,
    ) -> bool:

        return (
            self.left_boundary
            <= x
            <= self.right_boundary
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TableHeader:
    """
    Detected transaction table header.
    """

    page_number: int

    row_start: int
    row_end: int

    text: str

    columns: tuple[HeaderColumn, ...]

    confidence: float

    @property
    def semantics(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            column.semantic
            for column in self.columns
        )

    def get_column(
        self,
        semantic: str,
    ) -> HeaderColumn | None:

        for column in self.columns:

            if column.semantic == semantic:
                return column

        return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssignedCell:
    """
    Tokens assigned to one semantic column in one physical row.
    """

    semantic: str

    text: str

    tokens: tuple[LayoutToken, ...]

    confidence: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StructuredLayoutRow:
    """
    Physical row after semantic column assignment.
    """

    row_number: int
    page_number: int

    raw_text: str

    cells: tuple[AssignedCell, ...]

    confidence: float | None

    def get(
        self,
        semantic: str,
    ) -> str | None:

        for cell in self.cells:

            if cell.semantic == semantic:
                return cell.text or None

        return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PageLayout:
    """
    Layout reconstruction result for one OCR page.
    """

    page_number: int

    rows: tuple[LayoutRow, ...]

    table_header: TableHeader | None

    structured_rows: tuple[
        StructuredLayoutRow,
        ...
    ]

    layout_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayoutReconstructionResult:
    """
    Complete OCR document layout reconstruction.
    """

    pages: tuple[PageLayout, ...]

    page_count: int

    row_count: int

    structured_row_count: int

    header_detected: bool

    header_page_number: int | None

    header_confidence: float

    inferred_columns: tuple[
        HeaderColumn,
        ...
    ]

    reconstructed_text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# INTERNAL HEADER CANDIDATE
# ============================================================


@dataclass(frozen=True)
class _HeaderCandidate:
    semantic: str
    label: str
    anchor_x: float
    confidence: float


# ============================================================
# LAYOUT RECONSTRUCTOR
# ============================================================


class LayoutReconstructor:
    """
    Generic OCR layout reconstruction service.

    Main stages:

        OCR tokens
            ↓
        physical rows
            ↓
        transaction header detection
            ↓
        semantic column inference
            ↓
        semantic row assignment

    No bank identity is used anywhere.
    """

    # --------------------------------------------------------
    # Generic header vocabulary
    # --------------------------------------------------------

    HEADER_PATTERNS = {
        COLUMN_DATE: (
            r"\bdate\b",
            r"\btxn\s*date\b",
            r"\btransaction\s*date\b",
            r"\bvalue\s*date\b",
            r"\bposting\s*date\b",
        ),

        COLUMN_DESCRIPTION: (
            r"\bdescription\b",
            r"\bparticulars?\b",
            r"\bnarration\b",
            r"\bdetails?\b",
            r"\btransaction\s*details?\b",
            r"\bremarks?\b",
        ),

        COLUMN_REFERENCE: (
            r"\bchq\b",
            r"\bcheque\b",
            r"\bref\b",
            r"\breference\b",
            r"\bchq\s*/?\s*ref\b",
            r"\bchq\s*no\b",
            r"\bref\s*no\b",
            r"\butr\b",
        ),

        COLUMN_DEBIT: (
            r"\bdebit\b",
            r"\bwithdrawal\b",
            r"\bwithdrawals\b",
            r"\bdr\b",
            r"\bdr\.\b",
            r"\bdebit\s*amount\b",
            r"\bwithdrawal\s*amount\b",
        ),

        COLUMN_CREDIT: (
            r"\bcredit\b",
            r"\bdeposit\b",
            r"\bdeposits\b",
            r"\bcr\b",
            r"\bcr\.\b",
            r"\bcredit\s*amount\b",
            r"\bdeposit\s*amount\b",
        ),

        COLUMN_BALANCE: (
            r"\bbalance\b",
            r"\bclosing\s*balance\b",
            r"\brunning\s*balance\b",
            r"\bavailable\s*balance\b",
        ),
    }

    # At minimum, a useful transaction header normally contains:
    #
    # date + description + balance
    #
    # or
    #
    # date + debit/credit + balance.
    MIN_HEADER_SEMANTICS = 3

    # Maximum number of neighboring rows combined while searching
    # for split/multi-line headers.
    MAX_HEADER_ROW_SPAN = 3

    # --------------------------------------------------------
    # Generic date patterns
    # --------------------------------------------------------

    DATE_PATTERNS = (
        re.compile(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\b\d{1,2}[-/\s]"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"[-/\s]\d{2,4}\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\b\d{1,2}\s+"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    )

    # --------------------------------------------------------
    # Generic money pattern
    # --------------------------------------------------------

    MONEY_PATTERN = re.compile(
        r"""
        ^\s*
        [₹$]?
        \(?
        [-+]?
        (?:
            \d{1,3}(?:,\d{2,3})*
            |
            \d+
        )
        (?:\.\d{1,2})?
        \)?
        \s*
        (?:CR|DR)?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def reconstruct(
        self,
        pages: Sequence[OCRPage],
    ) -> LayoutReconstructionResult:
        """
        Reconstruct layout from OCR pages.

        Header inference is document-level.

        Once a reliable transaction header is found, its normalized
        column positions may be reused on later pages where the table
        header is not repeated.
        """

        if not pages:

            return LayoutReconstructionResult(
                pages=(),
                page_count=0,
                row_count=0,
                structured_row_count=0,
                header_detected=False,
                header_page_number=None,
                header_confidence=0.0,
                inferred_columns=(),
                reconstructed_text="",
            )

        page_rows: list[
            tuple[int, tuple[LayoutRow, ...]]
        ] = []

        for page in pages:

            rows = self._build_rows(
                page
            )

            page_rows.append(
                (
                    page.page_number,
                    rows,
                )
            )

        # ----------------------------------------------------
        # Find best document-level header
        # ----------------------------------------------------

        best_header: TableHeader | None = None

        for page_number, rows in page_rows:

            header = self._detect_table_header(
                page_number=page_number,
                rows=rows,
            )

            if header is None:
                continue

            if (
                best_header is None
                or header.confidence
                > best_header.confidence
            ):
                best_header = header

        inferred_columns = (
            best_header.columns
            if best_header is not None
            else ()
        )

        # ----------------------------------------------------
        # Build page results
        # ----------------------------------------------------

        page_results: list[
            PageLayout
        ] = []

        document_text_parts: list[str] = []

        total_rows = 0
        total_structured_rows = 0

        for page_number, rows in page_rows:

            local_header = self._detect_table_header(
                page_number=page_number,
                rows=rows,
            )

            effective_header = (
                local_header
                if local_header is not None
                else best_header
            )

            structured_rows: tuple[
                StructuredLayoutRow,
                ...
            ] = ()

            if effective_header is not None:

                structured_rows = (
                    self._assign_rows_to_columns(
                        rows=rows,
                        header=effective_header,
                        local_header=local_header,
                    )
                )

            layout_available = any(
                token.x_center_norm is not None
                and token.y_center_norm is not None
                for row in rows
                for token in row.tokens
            )

            page_results.append(
                PageLayout(
                    page_number=page_number,
                    rows=rows,
                    table_header=local_header,
                    structured_rows=structured_rows,
                    layout_available=layout_available,
                )
            )

            total_rows += len(
                rows
            )

            total_structured_rows += len(
                structured_rows
            )

            if rows:

                document_text_parts.append(
                    "\n".join(
                        row.text
                        for row in rows
                        if row.text
                    )
                )

        reconstructed_text = "\n\n".join(
            part
            for part in document_text_parts
            if part
        )

        return LayoutReconstructionResult(
            pages=tuple(
                page_results
            ),
            page_count=len(
                page_results
            ),
            row_count=total_rows,
            structured_row_count=total_structured_rows,
            header_detected=best_header is not None,
            header_page_number=(
                best_header.page_number
                if best_header is not None
                else None
            ),
            header_confidence=(
                best_header.confidence
                if best_header is not None
                else 0.0
            ),
            inferred_columns=inferred_columns,
            reconstructed_text=reconstructed_text,
        )

    # ========================================================
    # ROW RECONSTRUCTION
    # ========================================================

    def _build_rows(
        self,
        page: OCRPage,
    ) -> tuple[LayoutRow, ...]:
        """
        Group OCR tokens by adaptive vertical proximity.

        We do NOT simply group boxes that overlap vertically.

        OCR detection boxes may occasionally be unusually tall.
        Center-Y is more stable.

        The grouping tolerance is derived from median token height
        where possible.
        """

        spatial_tokens = [
            LayoutToken.from_ocr_token(
                token
            )
            for token in page.tokens
            if (
                token.text
                and token.text.strip()
                and token.x_center_norm is not None
                and token.y_center_norm is not None
            )
        ]

        if not spatial_tokens:
            return ()

        spatial_tokens.sort(
            key=lambda token: (
                token.y_center_norm
                if token.y_center_norm is not None
                else 1.0,
                token.x_center_norm
                if token.x_center_norm is not None
                else 1.0,
            )
        )

        tolerance = self._row_tolerance(
            spatial_tokens
        )

        row_groups: list[
            list[LayoutToken]
        ] = []

        row_centers: list[
            float
        ] = []

        for token in spatial_tokens:

            token_y = token.y_center_norm

            if token_y is None:
                continue

            best_index: int | None = None
            best_distance: float | None = None

            # Search existing nearby rows.
            #
            # Number of rows per page is small enough that this
            # remains inexpensive and avoids fragile assumptions.
            for index in range(
                len(row_groups) - 1,
                -1,
                -1,
            ):

                center = row_centers[
                    index
                ]

                distance = abs(
                    token_y - center
                )

                # Because rows are vertically ordered, once the
                # distance is substantially beyond tolerance there
                # is little value searching much further upward.
                if (
                    token_y > center
                    and distance > tolerance * 2.5
                ):
                    break

                if distance <= tolerance:

                    if (
                        best_distance is None
                        or distance < best_distance
                    ):
                        best_index = index
                        best_distance = distance

            if best_index is None:

                row_groups.append(
                    [token]
                )

                row_centers.append(
                    token_y
                )

            else:

                row_groups[
                    best_index
                ].append(
                    token
                )

                valid_ys = [
                    item.y_center_norm
                    for item in row_groups[
                        best_index
                    ]
                    if item.y_center_norm is not None
                ]

                if valid_ys:

                    row_centers[
                        best_index
                    ] = median(
                        valid_ys
                    )

        # Sort groups by their final Y centers.
        ordered_groups = sorted(
            zip(
                row_centers,
                row_groups,
            ),
            key=lambda item: item[0],
        )

        rows: list[
            LayoutRow
        ] = []

        for row_index, (
            _,
            group,
        ) in enumerate(
            ordered_groups,
            start=1,
        ):

            group.sort(
                key=lambda token: (
                    token.x_center_norm
                    if token.x_center_norm is not None
                    else 1.0
                )
            )

            row = self._make_row(
                page_number=page.page_number,
                row_number=row_index,
                tokens=group,
            )

            rows.append(
                row
            )

        return tuple(
            rows
        )

    @staticmethod
    def _row_tolerance(
        tokens: Sequence[LayoutToken],
    ) -> float:
        """
        Calculate adaptive normalized vertical grouping tolerance.

        Typical OCR text height is roughly 1-2% of page height.

        We intentionally clamp tolerance to prevent:
        - tiny OCR boxes creating fragmented rows
        - unusually tall OCR boxes merging neighboring rows
        """

        heights = [
            (
                token.y_max_norm
                - token.y_min_norm
            )
            for token in tokens
            if (
                token.y_max_norm is not None
                and token.y_min_norm is not None
                and token.y_max_norm
                > token.y_min_norm
            )
        ]

        if not heights:
            return 0.008

        typical_height = median(
            heights
        )

        tolerance = (
            typical_height * 0.55
        )

        return max(
            0.004,
            min(
                tolerance,
                0.014,
            ),
        )

    @staticmethod
    def _make_row(
        page_number: int,
        row_number: int,
        tokens: Sequence[LayoutToken],
    ) -> LayoutRow:

        text = " ".join(
            token.text.strip()
            for token in tokens
            if token.text.strip()
        )

        x_mins = [
            token.x_min_norm
            for token in tokens
            if token.x_min_norm is not None
        ]

        x_maxs = [
            token.x_max_norm
            for token in tokens
            if token.x_max_norm is not None
        ]

        y_centers = [
            token.y_center_norm
            for token in tokens
            if token.y_center_norm is not None
        ]

        confidences = [
            token.confidence
            for token in tokens
            if token.confidence is not None
        ]

        return LayoutRow(
            row_number=row_number,
            page_number=page_number,
            text=text,
            tokens=tuple(
                tokens
            ),
            y_center_norm=(
                median(
                    y_centers
                )
                if y_centers
                else None
            ),
            x_min_norm=(
                min(
                    x_mins
                )
                if x_mins
                else None
            ),
            x_max_norm=(
                max(
                    x_maxs
                )
                if x_maxs
                else None
            ),
            confidence=(
                sum(
                    confidences
                )
                / len(
                    confidences
                )
                if confidences
                else None
            ),
        )

    # ========================================================
    # TABLE HEADER DETECTION
    # ========================================================

    def _detect_table_header(
        self,
        page_number: int,
        rows: Sequence[LayoutRow],
    ) -> TableHeader | None:
        """
        Search for the strongest transaction-header candidate.

        Supports headers contained in one row as well as split across
        two or three nearby rows.
        """

        if not rows:
            return None

        best_header: TableHeader | None = None

        row_count = len(
            rows
        )

        for start_index in range(
            row_count
        ):

            for span in range(
                1,
                self.MAX_HEADER_ROW_SPAN + 1,
            ):

                end_index = (
                    start_index + span
                )

                if end_index > row_count:
                    break

                candidate_rows = rows[
                    start_index:end_index
                ]

                # Prevent combining physically distant rows.
                if not self._rows_are_close(
                    candidate_rows
                ):
                    continue

                candidates = (
                    self._extract_header_candidates(
                        candidate_rows
                    )
                )

                unique_semantics = {
                    candidate.semantic
                    for candidate in candidates
                }

                if len(
                    unique_semantics
                ) < self.MIN_HEADER_SEMANTICS:
                    continue

                if not self._looks_like_transaction_header(
                    unique_semantics
                ):
                    continue

                columns = (
                    self._build_header_columns(
                        candidates
                    )
                )

                if len(
                    columns
                ) < self.MIN_HEADER_SEMANTICS:
                    continue

                confidence = (
                    self._header_confidence(
                        columns=columns,
                        semantics=unique_semantics,
                    )
                )

                header_text = " | ".join(
                    row.text
                    for row in candidate_rows
                )

                header = TableHeader(
                    page_number=page_number,
                    row_start=candidate_rows[
                        0
                    ].row_number,
                    row_end=candidate_rows[
                        -1
                    ].row_number,
                    text=header_text,
                    columns=columns,
                    confidence=confidence,
                )

                if (
                    best_header is None
                    or header.confidence
                    > best_header.confidence
                ):
                    best_header = header

        return best_header

    @staticmethod
    def _rows_are_close(
        rows: Sequence[LayoutRow],
    ) -> bool:

        if len(rows) <= 1:
            return True

        centers = [
            row.y_center_norm
            for row in rows
            if row.y_center_norm is not None
        ]

        if len(centers) <= 1:
            return True

        centers.sort()

        gaps = [
            centers[index]
            - centers[index - 1]
            for index in range(
                1,
                len(centers),
            )
        ]

        return all(
            gap <= 0.035
            for gap in gaps
        )

    def _extract_header_candidates(
        self,
        rows: Sequence[LayoutRow],
    ) -> tuple[_HeaderCandidate, ...]:
        """
        Detect semantic header concepts from spatial OCR tokens.

        Detection occurs both:
        - token-by-token
        - using neighboring token phrases

        This allows recognition of:

            "Transaction" + "Date"
            "Closing" + "Balance"
            "Chq" + "/" + "Ref" + "No"
        """

        tokens = [
            token
            for row in rows
            for token in row.tokens
            if (
                token.text
                and token.x_center_norm is not None
            )
        ]

        if not tokens:
            return ()

        tokens.sort(
            key=lambda token: (
                token.y_center_norm
                if token.y_center_norm is not None
                else 1.0,
                token.x_center_norm
                if token.x_center_norm is not None
                else 1.0,
            )
        )

        candidates: list[
            _HeaderCandidate
        ] = []

        # ----------------------------------------------------
        # Individual token matching
        # ----------------------------------------------------

        for token in tokens:

            semantic = (
                self._match_header_semantic(
                    token.text
                )
            )

            if semantic is None:
                continue

            candidates.append(
                _HeaderCandidate(
                    semantic=semantic,
                    label=token.text,
                    anchor_x=float(
                        token.x_center_norm
                    ),
                    confidence=(
                        token.confidence
                        if token.confidence is not None
                        else 0.85
                    ),
                )
            )

        # ----------------------------------------------------
        # Neighbor phrase matching
        # ----------------------------------------------------

        for row in rows:

            row_tokens = [
                token
                for token in row.tokens
                if token.x_center_norm is not None
            ]

            row_tokens.sort(
                key=lambda token: token.x_center_norm
            )

            for start in range(
                len(row_tokens)
            ):

                for phrase_size in (
                    2,
                    3,
                    4,
                ):

                    end = (
                        start + phrase_size
                    )

                    if end > len(
                        row_tokens
                    ):
                        break

                    phrase_tokens = row_tokens[
                        start:end
                    ]

                    # Do not merge tokens that are far apart
                    # horizontally.
                    if not self._tokens_are_neighbors(
                        phrase_tokens
                    ):
                        continue

                    phrase = " ".join(
                        token.text
                        for token in phrase_tokens
                    )

                    semantic = (
                        self._match_header_semantic(
                            phrase
                        )
                    )

                    if semantic is None:
                        continue

                    xs = [
                        token.x_center_norm
                        for token in phrase_tokens
                        if token.x_center_norm is not None
                    ]

                    scores = [
                        token.confidence
                        for token in phrase_tokens
                        if token.confidence is not None
                    ]

                    if not xs:
                        continue

                    candidates.append(
                        _HeaderCandidate(
                            semantic=semantic,
                            label=phrase,
                            anchor_x=sum(
                                xs
                            ) / len(
                                xs
                            ),
                            confidence=(
                                sum(
                                    scores
                                ) / len(
                                    scores
                                )
                                if scores
                                else 0.85
                            ),
                        )
                    )

        return self._deduplicate_header_candidates(
            candidates
        )

    @staticmethod
    def _tokens_are_neighbors(
        tokens: Sequence[LayoutToken],
    ) -> bool:

        if len(tokens) <= 1:
            return True

        ordered = sorted(
            tokens,
            key=lambda token: (
                token.x_center_norm
                if token.x_center_norm is not None
                else 1.0
            ),
        )

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):

            if (
                previous.x_max_norm is None
                or current.x_min_norm is None
            ):
                continue

            gap = (
                current.x_min_norm
                - previous.x_max_norm
            )

            if gap > 0.06:
                return False

        return True

    def _match_header_semantic(
        self,
        text: str,
    ) -> str | None:

        normalized = self._normalize_label(
            text
        )

        if not normalized:
            return None

        matches: list[
            tuple[str, int]
        ] = []

        for semantic, patterns in (
            self.HEADER_PATTERNS.items()
        ):

            for pattern in patterns:

                if re.search(
                    pattern,
                    normalized,
                    re.IGNORECASE,
                ):

                    # Longer matching phrases should generally beat
                    # short ambiguous terms such as "dr".
                    matches.append(
                        (
                            semantic,
                            len(
                                pattern
                            ),
                        )
                    )

        if not matches:
            return None

        matches.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return matches[
            0
        ][0]

    @staticmethod
    def _normalize_label(
        text: str,
    ) -> str:

        value = (
            text
            .strip()
            .lower()
        )

        value = re.sub(
            r"[_|]+",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    @staticmethod
    def _deduplicate_header_candidates(
        candidates: Sequence[_HeaderCandidate],
    ) -> tuple[_HeaderCandidate, ...]:
        """
        Keep strongest candidate for each semantic concept.

        If the same concept appears several times because both token
        and phrase matching recognized it, choose the candidate with
        the best combination of confidence and useful label length.
        """

        best: dict[
            str,
            _HeaderCandidate
        ] = {}

        for candidate in candidates:

            current = best.get(
                candidate.semantic
            )

            if current is None:

                best[
                    candidate.semantic
                ] = candidate

                continue

            candidate_score = (
                candidate.confidence
                + min(
                    len(
                        candidate.label
                    ),
                    30,
                ) / 300.0
            )

            current_score = (
                current.confidence
                + min(
                    len(
                        current.label
                    ),
                    30,
                ) / 300.0
            )

            if candidate_score > current_score:

                best[
                    candidate.semantic
                ] = candidate

        return tuple(
            best.values()
        )

    @staticmethod
    def _looks_like_transaction_header(
        semantics: set[str],
    ) -> bool:
        """
        Prevent arbitrary metadata labels from being mistaken for the
        transaction table.

        A transaction header should normally contain:
            date
        and:
            balance
        and at least one transaction-content/amount concept.
        """

        if COLUMN_DATE not in semantics:
            return False

        if COLUMN_BALANCE not in semantics:
            return False

        useful = {
            COLUMN_DESCRIPTION,
            COLUMN_REFERENCE,
            COLUMN_DEBIT,
            COLUMN_CREDIT,
        }

        return bool(
            semantics & useful
        )

    # ========================================================
    # COLUMN INFERENCE
    # ========================================================

    def _build_header_columns(
        self,
        candidates: Sequence[_HeaderCandidate],
    ) -> tuple[HeaderColumn, ...]:
        """
        Convert semantic header anchors into normalized column
        boundaries.

        Boundaries are midpoints between neighboring semantic header
        anchors.

        No fixed X coordinates are used.
        """

        if not candidates:
            return ()

        ordered = sorted(
            candidates,
            key=lambda candidate: candidate.anchor_x,
        )

        columns: list[
            HeaderColumn
        ] = []

        for index, candidate in enumerate(
            ordered
        ):

            if index == 0:

                left_boundary = 0.0

            else:

                previous = ordered[
                    index - 1
                ]

                left_boundary = (
                    previous.anchor_x
                    + candidate.anchor_x
                ) / 2.0

            if index == len(
                ordered
            ) - 1:

                right_boundary = 1.0

            else:

                following = ordered[
                    index + 1
                ]

                right_boundary = (
                    candidate.anchor_x
                    + following.anchor_x
                ) / 2.0

            columns.append(
                HeaderColumn(
                    semantic=candidate.semantic,
                    label=candidate.label,
                    anchor_x=candidate.anchor_x,
                    left_boundary=max(
                        0.0,
                        left_boundary,
                    ),
                    right_boundary=min(
                        1.0,
                        right_boundary,
                    ),
                    confidence=candidate.confidence,
                )
            )

        return tuple(
            columns
        )

    @staticmethod
    def _header_confidence(
        columns: Sequence[HeaderColumn],
        semantics: set[str],
    ) -> float:

        if not columns:
            return 0.0

        score = sum(
            column.confidence
            for column in columns
        ) / len(
            columns
        )

        coverage_bonus = min(
            len(
                semantics
            ) / len(
                SEMANTIC_COLUMNS
            ),
            1.0,
        )

        # Strong semantic coverage matters, but OCR confidence still
        # dominates.
        final_score = (
            score * 0.8
            + coverage_bonus * 0.2
        )

        return round(
            min(
                1.0,
                final_score,
            ),
            4,
        )

    # ========================================================
    # COLUMN ASSIGNMENT
    # ========================================================

    def _assign_rows_to_columns(
        self,
        rows: Sequence[LayoutRow],
        header: TableHeader,
        local_header: TableHeader | None,
    ) -> tuple[StructuredLayoutRow, ...]:
        """
        Assign tokens to inferred semantic columns.

        Only rows below the local transaction header are considered
        transaction-table candidates.

        When a page does not repeat the header, the document-level
        header geometry is reused and all page rows are considered.
        """

        if not rows:
            return ()

        if not header.columns:
            return ()

        if local_header is not None:

            candidate_rows = [
                row
                for row in rows
                if row.row_number
                > local_header.row_end
            ]

        else:

            candidate_rows = list(
                rows
            )

        structured: list[
            StructuredLayoutRow
        ] = []

        for row in candidate_rows:

            # Ignore repeated header-like rows that slipped through
            # due to a slightly different OCR reconstruction.
            row_semantics = (
                self._semantic_count_in_text(
                    row.text
                )
            )

            if row_semantics >= 3:
                continue

            cells = self._assign_row(
                row=row,
                columns=header.columns,
            )

            if not cells:
                continue

            if not self._looks_transaction_like(
                row=row,
                cells=cells,
            ):
                continue

            confidences = [
                cell.confidence
                for cell in cells
                if cell.confidence is not None
            ]

            structured.append(
                StructuredLayoutRow(
                    row_number=row.row_number,
                    page_number=row.page_number,
                    raw_text=row.text,
                    cells=cells,
                    confidence=(
                        sum(
                            confidences
                        ) / len(
                            confidences
                        )
                        if confidences
                        else None
                    ),
                )
            )

        return tuple(
            structured
        )

    @staticmethod
    def _is_page_marker(text: str) -> bool:
        """Generic page-number/footer marker rejection."""
        value = re.sub(r"\s+", " ", (text or "").strip().lower())
        return bool(re.fullmatch(r"(?:page|pg)\s*(?:no\.?|number)?\s*[:#.-]?\s*\d+(?:\s*(?:of|/)\s*\d+)?", value))

    def _token_semantic_score(
        self,
        token: LayoutToken,
        column: HeaderColumn,
    ) -> float:
        """Score a token/column pairing using geometry plus generic token shape.

        Geometry remains primary. Shape constraints only prevent obviously
        incompatible assignments (for example narrative text in an amount
        column or a CR/DR balance in a reference column).
        """
        x = token.x_center_norm
        if x is None:
            return float("-inf")

        span = max(column.right_boundary - column.left_boundary, 0.02)
        distance = abs(x - column.anchor_x) / span
        score = 1.0 - min(distance, 2.0) * 0.35

        text = (token.text or "").strip()
        upper = text.upper()
        money_like = self.MONEY_PATTERN.match(text) is not None
        date_like = self._contains_date(text)
        balance_suffix = bool(re.search(r"(?:CR|DR)\s*$", upper)) and money_like

        if date_like:
            score += 1.5 if column.semantic == COLUMN_DATE else -1.0

        if money_like:
            if column.semantic in (COLUMN_DEBIT, COLUMN_CREDIT, COLUMN_BALANCE):
                score += 0.8
            elif column.semantic in (COLUMN_DESCRIPTION, COLUMN_REFERENCE):
                score -= 0.45

            if balance_suffix:
                score += 1.4 if column.semantic == COLUMN_BALANCE else -0.6
        else:
            # Ordinary narrative/reference text should not populate numeric
            # amount columns merely because OCR geometry drifted horizontally.
            if column.semantic in (COLUMN_DEBIT, COLUMN_CREDIT, COLUMN_BALANCE):
                score -= 0.9
            elif column.semantic in (COLUMN_DESCRIPTION, COLUMN_REFERENCE):
                score += 0.25

        return score

    def _best_column_for_token(
        self,
        token: LayoutToken,
        columns: Sequence[HeaderColumn],
    ) -> HeaderColumn | None:
        if token.x_center_norm is None or not columns:
            return None

        return max(
            columns,
            key=lambda column: self._token_semantic_score(token, column),
        )

    def _is_financial_amount_token(
        self,
        token: LayoutToken,
    ) -> bool:
        """Return True for tokens that look like monetary amounts, not IDs.

        A long digit-only reference/account number technically matches the broad
        MONEY_PATTERN, so layout assignment must not automatically treat every
        numeric token as debit/credit/balance.  Decimal punctuation, grouping,
        currency symbols and CR/DR suffixes are strong generic amount signals.
        Short plain integers are retained because some statements omit decimals.
        """
        text = (token.text or "").strip()
        if not text or self.MONEY_PATTERN.match(text) is None:
            return False

        compact = re.sub(r"[₹$,+()\s-]", "", text, flags=re.IGNORECASE)
        compact = re.sub(r"(?:CR|DR)$", "", compact, flags=re.IGNORECASE)
        digits = re.sub(r"\D", "", compact)

        if re.search(r"[.,₹$]", text):
            return True
        if re.search(r"(?:CR|DR)\s*$", text, re.IGNORECASE):
            return True

        return bool(digits) and len(digits) <= 7

    def _rebalance_financial_tokens(
        self,
        row: LayoutRow,
        columns: Sequence[HeaderColumn],
        grouped: dict[str, list[LayoutToken]],
    ) -> None:
        """Resolve amount/balance collisions after first-pass assignment.

        OCR geometry can drift enough that both the transaction amount and the
        running balance fall into the inferred balance region.  The balance is
        usually the right-most financial value and is often explicitly marked
        CR/DR.  Once that token is identified, remaining financial values are
        assigned only among debit/credit columns.  This uses document-inferred
        semantics and token geometry; there are no bank names or fixed X values.
        """
        amount_columns = [
            column
            for column in columns
            if column.semantic in (COLUMN_DEBIT, COLUMN_CREDIT, COLUMN_BALANCE)
        ]
        debit_credit_columns = [
            column
            for column in columns
            if column.semantic in (COLUMN_DEBIT, COLUMN_CREDIT)
        ]

        if not amount_columns:
            return

        financial_tokens = [
            token
            for token in row.tokens
            if token.x_center_norm is not None
            and self._is_financial_amount_token(token)
        ]

        if not financial_tokens:
            return

        # Remove financial tokens from their first-pass buckets.  They are
        # reassigned below while narrative/date/reference tokens stay untouched.
        financial_ids = {id(token) for token in financial_tokens}
        for semantic in tuple(grouped):
            grouped[semantic] = [
                token
                for token in grouped[semantic]
                if id(token) not in financial_ids
            ]

        explicit_balances = [
            token
            for token in financial_tokens
            if re.search(r"(?:CR|DR)\s*$", (token.text or "").strip(), re.IGNORECASE)
        ]

        balance_token: LayoutToken | None = None
        if explicit_balances:
            balance_token = max(
                explicit_balances,
                key=lambda token: token.x_center_norm or -1.0,
            )
        elif len(financial_tokens) >= 2:
            # In running-balance tables the balance is generically the final
            # financial value on the row.  Restrict this inference to rows with
            # multiple monetary values so a lone amount is not stolen.
            balance_token = max(
                financial_tokens,
                key=lambda token: token.x_center_norm or -1.0,
            )

        remaining = list(financial_tokens)

        if balance_token is not None:
            grouped.setdefault(COLUMN_BALANCE, []).append(balance_token)
            remaining = [token for token in remaining if token is not balance_token]

        for token in remaining:
            candidates = debit_credit_columns or [
                column
                for column in amount_columns
                if column.semantic != COLUMN_BALANCE
            ]

            if not candidates:
                # No debit/credit semantics were inferred. Preserve the token in
                # its strongest amount column rather than discarding evidence.
                candidates = amount_columns

            column = max(
                candidates,
                key=lambda candidate: self._token_semantic_score(token, candidate),
            )
            grouped.setdefault(column.semantic, []).append(token)

    def _assign_row(
        self,
        row: LayoutRow,
        columns: Sequence[HeaderColumn],
    ) -> tuple[AssignedCell, ...]:

        grouped: dict[
            str,
            list[LayoutToken]
        ] = {
            column.semantic: []
            for column in columns
        }

        for token in row.tokens:

            x = token.x_center_norm

            if x is None:
                continue

            column = self._best_column_for_token(
                token=token,
                columns=columns,
            )

            if column is None:
                continue

            grouped[
                column.semantic
            ].append(
                token
            )

        self._rebalance_financial_tokens(
            row=row,
            columns=columns,
            grouped=grouped,
        )

        cells: list[
            AssignedCell
        ] = []

        for column in columns:

            tokens = grouped.get(
                column.semantic,
                [],
            )

            if not tokens:
                continue

            tokens.sort(
                key=lambda token: (
                    token.x_center_norm
                    if token.x_center_norm is not None
                    else 1.0
                )
            )

            text = " ".join(
                token.text.strip()
                for token in tokens
                if token.text.strip()
            )

            confidences = [
                token.confidence
                for token in tokens
                if token.confidence is not None
            ]

            cells.append(
                AssignedCell(
                    semantic=column.semantic,
                    text=text,
                    tokens=tuple(
                        tokens
                    ),
                    confidence=(
                        sum(
                            confidences
                        ) / len(
                            confidences
                        )
                        if confidences
                        else None
                    ),
                )
            )

        return tuple(
            cells
        )

    @staticmethod
    def _nearest_column(
        x: float,
        columns: Sequence[HeaderColumn],
    ) -> HeaderColumn | None:

        containing = [
            column
            for column in columns
            if column.contains(
                x
            )
        ]

        if containing:

            return min(
                containing,
                key=lambda column: abs(
                    x - column.anchor_x
                ),
            )

        if not columns:
            return None

        # Defensive fallback for tiny boundary rounding gaps.
        return min(
            columns,
            key=lambda column: abs(
                x - column.anchor_x
            ),
        )

    # ========================================================
    # TRANSACTION-LIKE FILTERING
    # ========================================================

    def _looks_transaction_like(
        self,
        row: LayoutRow,
        cells: Sequence[AssignedCell],
    ) -> bool:
        """
        Conservative physical-row filter.

        This is NOT the transaction parser.

        It only removes obvious non-transaction rows so downstream
        reconstruction is not flooded with page headers/footers.

        Multi-line transaction descriptions are expected and may be
        joined later by the transaction reconstruction layer.
        """

        if self._is_page_marker(row.text):
            return False

        values = {
            cell.semantic: cell.text
            for cell in cells
        }

        date_text = values.get(
            COLUMN_DATE,
            "",
        )

        has_date = self._contains_date(
            date_text
        ) or self._contains_date(
            row.text
        )

        amount_semantics = (
            COLUMN_DEBIT,
            COLUMN_CREDIT,
            COLUMN_BALANCE,
        )

        amount_count = 0

        for semantic in amount_semantics:

            value = values.get(
                semantic,
                "",
            )

            if self._contains_money(
                value
            ):
                amount_count += 1

        # Strong transaction row:
        # date + at least one numeric financial field.
        if has_date and amount_count >= 1:
            return True

        # Some layouts split the date and transaction values into
        # adjacent physical rows. Preserve numeric rows so a later
        # logical transaction assembler can attach them.
        if amount_count >= 2:
            return True

        # Description/reference continuation rows are also useful.
        description = values.get(
            COLUMN_DESCRIPTION,
            "",
        )

        reference = values.get(
            COLUMN_REFERENCE,
            "",
        )

        if (
            description
            or reference
        ) and len(
            row.text.strip()
        ) >= 3:

            return True

        return False

    def _semantic_count_in_text(
        self,
        text: str,
    ) -> int:

        semantics = set()

        normalized = self._normalize_label(
            text
        )

        for semantic, patterns in (
            self.HEADER_PATTERNS.items()
        ):

            for pattern in patterns:

                if re.search(
                    pattern,
                    normalized,
                    re.IGNORECASE,
                ):

                    semantics.add(
                        semantic
                    )

                    break

        return len(
            semantics
        )

    def _contains_date(
        self,
        text: str,
    ) -> bool:

        if not text:
            return False

        return any(
            pattern.search(
                text
            )
            is not None
            for pattern in self.DATE_PATTERNS
        )

    def _contains_money(
        self,
        text: str,
    ) -> bool:

        if not text:
            return False

        pieces = re.split(
            r"\s+",
            text.strip(),
        )

        return any(
            self.MONEY_PATTERN.match(
                piece
            )
            is not None
            for piece in pieces
            if piece
        )


# ============================================================
# SINGLETON
# ============================================================


layout_reconstructor = LayoutReconstructor()