"""
Generic PDF Revision Analyzer.

Purpose:
Detect structural evidence that a PDF may have been modified after
its original creation.

This analyzer does NOT:
- perform OCR
- identify a bank
- parse transactions
- decide whether the document is genuine
- prove that a document was tampered with

It produces independent PDF revision / object-structure evidence
for TamperDetector to consume.

Checks include:
- %%EOF occurrences
- startxref occurrences
- trailer occurrences
- classic xref sections
- /Prev references
- incremental-update indicators
- object generation numbers
- duplicate object definitions
- object numbering characteristics
- xref/object consistency observations

Important:
PDFs can legitimately contain incremental revisions.
A revision is evidence of modification history, NOT proof of fraud.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from io import BytesIO

import fitz


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True)
class PDFRevisionAnalysisResult:

    pdf_size_bytes: int

    eof_count: int
    startxref_count: int
    trailer_count: int
    xref_section_count: int

    prev_reference_count: int

    incremental_update_suspected: bool

    indirect_object_definition_count: int
    unique_object_number_count: int

    duplicate_object_numbers: tuple[int, ...]
    nonzero_generation_objects: tuple[int, ...]

    highest_object_number: int
    xref_length: int

    object_number_gap_count: int

    suspicious_revision_structure: bool

    strong_signals: tuple[str, ...]
    moderate_signals: tuple[str, ...]
    weak_signals: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Analyzer
# ============================================================


class PDFRevisionAnalyzer:

    """
    Generic PDF revision-history analyzer.

    The analyzer deliberately separates:

        structural observation
                from
        tamper verdict

    because legitimate PDF software can create:
    - incremental updates
    - multiple xref sections
    - multiple trailers
    - non-zero generation numbers

    TamperDetector should correlate these findings with other
    independent evidence.
    """

    # --------------------------------------------------------
    # Compiled byte patterns
    # --------------------------------------------------------

    OBJECT_PATTERN = re.compile(
        rb"(?m)(?:^|[\r\n])\s*(\d+)\s+(\d+)\s+obj\b"
    )

    PREV_PATTERN = re.compile(
        rb"/Prev\s+(\d+)"
    )

    STARTXREF_PATTERN = re.compile(
        rb"startxref\s+(\d+)",
        re.IGNORECASE,
    )

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        file_bytes: bytes,
    ) -> PDFRevisionAnalysisResult:

        if not file_bytes:
            raise ValueError(
                "PDF bytes are required."
            )

        if not file_bytes.startswith(
            b"%PDF-"
        ):
            raise ValueError(
                "Input does not have a valid PDF signature."
            )

        # ====================================================
        # Ensure PDF can actually be opened
        # ====================================================

        try:
            document = fitz.open(
                stream=BytesIO(file_bytes),
                filetype="pdf",
            )

        except Exception as exc:
            raise ValueError(
                "Unable to open PDF for revision analysis."
            ) from exc

        try:

            if document.needs_pass:
                raise ValueError(
                    "Password-protected PDF cannot be fully "
                    "analyzed for revision structure."
                )

            xref_length = int(
                document.xref_length()
            )

        finally:
            document.close()

        # ====================================================
        # Basic revision markers
        # ====================================================

        eof_count = file_bytes.count(
            b"%%EOF"
        )

        startxref_matches = list(
            self.STARTXREF_PATTERN.finditer(
                file_bytes
            )
        )

        startxref_count = len(
            startxref_matches
        )

        # Classic trailers only.
        #
        # PDF 1.5+ can use xref streams instead of traditional
        # "trailer" syntax, so absence is not automatically
        # suspicious.

        trailer_count = len(
            re.findall(
                rb"(?m)^\s*trailer\b",
                file_bytes,
            )
        )

        # Classic xref table sections only.
        #
        # Avoid matching "startxref".

        xref_section_count = len(
            re.findall(
                rb"(?m)^\s*xref\s*$",
                file_bytes,
            )
        )

        prev_matches = list(
            self.PREV_PATTERN.finditer(
                file_bytes
            )
        )

        prev_reference_count = len(
            prev_matches
        )

        # ====================================================
        # Indirect object definitions
        # ====================================================

        object_matches = list(
            self.OBJECT_PATTERN.finditer(
                file_bytes
            )
        )

        indirect_object_definition_count = len(
            object_matches
        )

        object_numbers: list[int] = []
        generations_by_object: dict[
            int,
            set[int],
        ] = {}

        definition_count_by_object: dict[
            int,
            int,
        ] = {}

        for match in object_matches:

            object_number = int(
                match.group(1)
            )

            generation = int(
                match.group(2)
            )

            object_numbers.append(
                object_number
            )

            generations_by_object.setdefault(
                object_number,
                set(),
            ).add(
                generation
            )

            definition_count_by_object[
                object_number
            ] = (
                definition_count_by_object.get(
                    object_number,
                    0,
                )
                + 1
            )

        unique_object_numbers = set(
            object_numbers
        )

        unique_object_number_count = len(
            unique_object_numbers
        )

        highest_object_number = (
            max(
                unique_object_numbers
            )
            if unique_object_numbers
            else 0
        )

        # ====================================================
        # Duplicate object definitions
        # ====================================================

        duplicate_object_numbers = tuple(
            sorted(
                object_number
                for object_number, count
                in definition_count_by_object.items()
                if count > 1
            )
        )

        # ====================================================
        # Non-zero generations
        # ====================================================

        nonzero_generation_objects = tuple(
            sorted(
                object_number
                for object_number, generations
                in generations_by_object.items()
                if any(
                    generation > 0
                    for generation
                    in generations
                )
            )
        )

        # ====================================================
        # Object-number gaps
        # ====================================================

        object_number_gap_count = (
            self._count_object_number_gaps(
                unique_object_numbers
            )
        )

        # ====================================================
        # Incremental update detection
        # ====================================================

        # Multiple EOF/startxref markers are common evidence
        # that content was appended as an incremental update.
        #
        # /Prev is particularly useful because it links a newer
        # trailer/xref structure to an earlier one.

        incremental_update_suspected = bool(
            prev_reference_count > 0
            or eof_count > 1
            or startxref_count > 1
        )

        # ====================================================
        # Evidence classification
        # ====================================================

        strong_signals: list[str] = []
        moderate_signals: list[str] = []
        weak_signals: list[str] = []
        warnings: list[str] = []

        # ----------------------------------------------------
        # /Prev
        # ----------------------------------------------------

        if prev_reference_count > 0:

            moderate_signals.append(
                f"{prev_reference_count} /Prev reference(s) "
                "detected, indicating prior PDF revision "
                "structure."
            )

        # ----------------------------------------------------
        # Multiple startxref
        # ----------------------------------------------------

        if startxref_count > 1:

            moderate_signals.append(
                f"{startxref_count} startxref markers "
                "detected."
            )

        # ----------------------------------------------------
        # Multiple EOF
        # ----------------------------------------------------

        if eof_count > 1:

            moderate_signals.append(
                f"{eof_count} %%EOF markers detected."
            )

        # ----------------------------------------------------
        # Multiple classic xref sections
        # ----------------------------------------------------

        if xref_section_count > 1:

            weak_signals.append(
                f"{xref_section_count} classic xref sections "
                "detected."
            )

        # ----------------------------------------------------
        # Multiple trailers
        # ----------------------------------------------------

        if trailer_count > 1:

            weak_signals.append(
                f"{trailer_count} classic trailer sections "
                "detected."
            )

        # ----------------------------------------------------
        # Duplicate object definitions
        # ----------------------------------------------------

        if duplicate_object_numbers:

            moderate_signals.append(
                f"{len(duplicate_object_numbers)} indirect "
                "object number(s) are defined more than once."
            )

        # ----------------------------------------------------
        # Non-zero generations
        # ----------------------------------------------------

        if nonzero_generation_objects:

            weak_signals.append(
                f"{len(nonzero_generation_objects)} object(s) "
                "use non-zero generation numbers."
            )

        # ----------------------------------------------------
        # Object numbering gaps
        # ----------------------------------------------------

        if object_number_gap_count > 0:

            weak_signals.append(
                f"{object_number_gap_count} gap(s) detected "
                "in observed indirect-object numbering."
            )

        # ====================================================
        # Strong correlated revision evidence
        # ====================================================

        # One marker alone is not enough.
        #
        # Strong revision evidence requires several independent
        # revision characteristics.

        revision_indicator_count = sum(
            [
                prev_reference_count > 0,
                eof_count > 1,
                startxref_count > 1,
                bool(
                    duplicate_object_numbers
                ),
            ]
        )

        if revision_indicator_count >= 3:

            strong_signals.append(
                "Multiple independent PDF revision indicators "
                "were detected."
            )

        # ====================================================
        # Structural sanity observations
        # ====================================================

        # xref_length normally includes object 0, so a modest
        # difference from the highest object number is normal.

        if (
            highest_object_number > 0
            and xref_length > 0
            and highest_object_number
            > xref_length + 10
        ):

            moderate_signals.append(
                "Observed object numbering extends materially "
                "beyond the active xref length."
            )

        # ====================================================
        # Suspicious revision structure
        # ====================================================

        suspicious_revision_structure = bool(
            strong_signals
            or len(
                moderate_signals
            ) >= 3
        )

        # ====================================================
        # Warnings
        # ====================================================

        if incremental_update_suspected:

            warnings.append(
                "Incremental PDF revision evidence does not "
                "by itself prove malicious tampering; valid "
                "PDF software, signatures, annotations, and "
                "workflow systems may create revisions."
            )

        if (
            xref_section_count == 0
            and trailer_count == 0
        ):

            warnings.append(
                "No classic xref/trailer structure was "
                "observed. The PDF may use xref streams, "
                "which are valid in newer PDF versions."
            )

        # ====================================================
        # Result
        # ====================================================

        return PDFRevisionAnalysisResult(

            pdf_size_bytes=len(
                file_bytes
            ),

            eof_count=(
                eof_count
            ),

            startxref_count=(
                startxref_count
            ),

            trailer_count=(
                trailer_count
            ),

            xref_section_count=(
                xref_section_count
            ),

            prev_reference_count=(
                prev_reference_count
            ),

            incremental_update_suspected=(
                incremental_update_suspected
            ),

            indirect_object_definition_count=(
                indirect_object_definition_count
            ),

            unique_object_number_count=(
                unique_object_number_count
            ),

            duplicate_object_numbers=(
                duplicate_object_numbers
            ),

            nonzero_generation_objects=(
                nonzero_generation_objects
            ),

            highest_object_number=(
                highest_object_number
            ),

            xref_length=(
                xref_length
            ),

            object_number_gap_count=(
                object_number_gap_count
            ),

            suspicious_revision_structure=(
                suspicious_revision_structure
            ),

            strong_signals=tuple(
                strong_signals
            ),

            moderate_signals=tuple(
                moderate_signals
            ),

            weak_signals=tuple(
                weak_signals
            ),

            warnings=tuple(
                warnings
            ),
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _count_object_number_gaps(
        object_numbers: set[int],
    ) -> int:

        if len(
            object_numbers
        ) < 2:

            return 0

        ordered = sorted(
            object_numbers
        )

        gaps = 0

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):

            if (
                current
                - previous
                > 1
            ):

                gaps += 1

        return gaps


# ============================================================
# Default Instance
# ============================================================


pdf_revision_analyzer = (
    PDFRevisionAnalyzer()
)