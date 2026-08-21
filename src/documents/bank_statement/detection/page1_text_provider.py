"""
Page-1 Text Provider
====================

Bank Statement V2 - Phase 1 Page-1 Detection / Metadata.

Responsibilities:
- extract Page 1 only
- detect Page-1 PDF mode
- prefer native PDF text
- evaluate native text quality
- use OCR only when native Page-1 text is insufficient
- OCR Page 1 only, never the complete statement
- normalize extracted text for generic metadata extraction
- remain bank-independent
- reuse the existing Phase-2 OCR extractor

Supported Page-1 modes:
- digital_pdf
- scanned_pdf
- hybrid_pdf

Important distinction:
- mode describes the structural composition of Page 1
- extraction_method describes how usable text was obtained

This module does NOT:
- extract bank metadata
- contain bank-specific regex
- classify a document as a bank statement
- parse transactions
- validate transactions
- calculate risk
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path

from src.documents.bank_statement.extraction.ocr_extractor import (
    ocr_extractor,
)


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass(frozen=True)
class Page1TextResult:
    """
    Standard result returned by the Page-1 text provider.
    """

    raw_text: str

    normalized_text: str

    # Structural Page-1 composition:
    #
    # digital_pdf
    # scanned_pdf
    # hybrid_pdf
    mode: str

    # Text acquisition route:
    #
    # native_pdf
    # native_pdf_low_quality
    # ocr
    # ocr_low_quality
    extraction_method: str

    text_quality_score: float

    ocr_used: bool

    native_text_quality_score: float

    native_char_count: int

    final_char_count: int

    ocr_confidence: float | None = None

    ocr_engine: str | None = None

    # --------------------------------------------------------
    # Mode diagnostics
    # --------------------------------------------------------

    page_image_count: int = 0

    page_image_coverage: float = 0.0

    mode_confidence: float = 0.0


# ============================================================
# INTERNAL PAGE ANALYSIS MODEL
# ============================================================


@dataclass(frozen=True)
class Page1Structure:
    """
    Structural characteristics of Page 1 used only for mode
    classification.
    """

    native_text: str

    image_count: int

    image_coverage: float

    page_area: float


# ============================================================
# PAGE-1 TEXT PROVIDER
# ============================================================


class Page1TextProvider:
    """
    Generic Page-1 text acquisition service.

    Routing:

        PDF
         |
         v
    Analyze Page 1
         |
         +----> classify mode
         |
         v
    Native Page-1 text
         |
         v
    Quality sufficient?
       /        \\
     yes         no
      |           |
      |       Page-1-only PDF
      |           |
      |          OCR
      |           |
      +-----+-----+
            |
            v
      normalized text

    No bank-specific logic belongs here.
    """

    # --------------------------------------------------------
    # Native text routing thresholds
    # --------------------------------------------------------

    MIN_NATIVE_CHAR_COUNT = 80

    MIN_NATIVE_QUALITY_SCORE = 0.45

    # --------------------------------------------------------
    # Quality scoring
    # --------------------------------------------------------

    GOOD_CHAR_COUNT = 250

    MIN_ALPHA_RATIO = 0.30

    MIN_ALNUM_RATIO = 0.45

    # --------------------------------------------------------
    # OCR acceptance
    # --------------------------------------------------------

    MIN_OCR_CHAR_COUNT = 40

    # --------------------------------------------------------
    # Mode classification
    # --------------------------------------------------------

    # Large raster coverage with little/no useful native text
    # strongly indicates a scanned page.
    SCANNED_IMAGE_COVERAGE = 0.70

    # Meaningful image coverage combined with useful native text
    # indicates a hybrid page.
    HYBRID_IMAGE_COVERAGE = 0.15

    # Native text needed before we consider the page to contain
    # meaningful digital text for mode classification.
    MODE_MIN_NATIVE_TEXT_CHARS = 40


    # ========================================================
    # PUBLIC API
    # ========================================================

    def extract(
        self,
        pdf_path: Path,
    ) -> Page1TextResult:
        """
        Extract usable text from Page 1 and classify Page-1 mode.

        Native PDF extraction is always attempted first.

        OCR is executed only when Page-1 native text is
        insufficient.
        """

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():
            raise ValueError(
                f"PDF file does not exist: {pdf_path}"
            )

        if not pdf_path.is_file():
            raise ValueError(
                f"PDF path is not a file: {pdf_path}"
            )

        file_bytes = pdf_path.read_bytes()

        if not file_bytes:
            raise ValueError(
                "PDF file is empty."
            )

        if not file_bytes.startswith(
            b"%PDF-"
        ):
            raise ValueError(
                "Page-1 text provider currently expects a PDF."
            )

        # ----------------------------------------------------
        # STEP 1
        # Analyze Page-1 structure.
        #
        # One PyMuPDF pass gives us:
        #
        # - native text
        # - image count
        # - approximate raster-image coverage
        #
        # These are used for mode classification.
        # ----------------------------------------------------

        structure = (
            self._analyze_page1_structure(
                file_bytes
            )
        )

        native_raw_text = (
            structure.native_text
        )

        native_normalized_text = (
            self.normalize_text(
                native_raw_text
            )
        )

        native_quality = (
            self._score_text_quality(
                native_normalized_text
            )
        )

        native_char_count = len(
            native_normalized_text
        )

        # ----------------------------------------------------
        # STEP 2
        # Determine structural PDF mode.
        # ----------------------------------------------------

        (
            mode,
            mode_confidence,
        ) = self._classify_mode(
            native_text=native_normalized_text,
            native_quality=native_quality,
            image_count=structure.image_count,
            image_coverage=structure.image_coverage,
        )

        # ----------------------------------------------------
        # STEP 3
        # Native text good enough -> STOP.
        #
        # Kotak / Canara should normally stop here.
        # ----------------------------------------------------

        if self._native_text_is_usable(
            text=native_normalized_text,
            quality_score=native_quality,
        ):

            return Page1TextResult(
                raw_text=native_raw_text,

                normalized_text=(
                    native_normalized_text
                ),

                mode=mode,

                extraction_method=(
                    "native_pdf"
                ),

                text_quality_score=(
                    native_quality
                ),

                ocr_used=False,

                native_text_quality_score=(
                    native_quality
                ),

                native_char_count=(
                    native_char_count
                ),

                final_char_count=len(
                    native_normalized_text
                ),

                ocr_confidence=None,

                ocr_engine=None,

                page_image_count=(
                    structure.image_count
                ),

                page_image_coverage=round(
                    structure.image_coverage,
                    4,
                ),

                mode_confidence=(
                    mode_confidence
                ),
            )

        # ----------------------------------------------------
        # STEP 4
        # Native Page-1 text is insufficient.
        #
        # Build a one-page PDF containing ONLY Page 1.
        #
        # We must NOT send the entire statement to OCR.
        # ----------------------------------------------------

        page1_pdf_bytes = (
            self._build_page1_pdf(
                file_bytes
            )
        )

        # ----------------------------------------------------
        # STEP 5
        # Existing generic OCR extractor.
        #
        # PaddleOCR remains owned by ocr_extractor.py.
        # ----------------------------------------------------

        try:

            ocr_result = (
                ocr_extractor.extract(
                    file_bytes=page1_pdf_bytes,
                    filename=(
                        f"{pdf_path.stem}_page1.pdf"
                    ),
                    detected_type="pdf",
                )
            )

        except Exception:

            # OCR failed.
            #
            # Return native text truthfully rather than claiming
            # OCR succeeded.

            return Page1TextResult(
                raw_text=native_raw_text,

                normalized_text=(
                    native_normalized_text
                ),

                mode=mode,

                extraction_method=(
                    "native_pdf_low_quality"
                ),

                text_quality_score=(
                    native_quality
                ),

                ocr_used=False,

                native_text_quality_score=(
                    native_quality
                ),

                native_char_count=(
                    native_char_count
                ),

                final_char_count=len(
                    native_normalized_text
                ),

                ocr_confidence=None,

                ocr_engine=None,

                page_image_count=(
                    structure.image_count
                ),

                page_image_coverage=round(
                    structure.image_coverage,
                    4,
                ),

                mode_confidence=(
                    mode_confidence
                ),
            )

        # ----------------------------------------------------
        # STEP 6
        # Normalize OCR output
        # ----------------------------------------------------

        ocr_raw_text = (
            ocr_result.text
            or ""
        )

        ocr_normalized_text = (
            self.normalize_text(
                ocr_raw_text
            )
        )

        ocr_quality = (
            self._score_text_quality(
                ocr_normalized_text
            )
        )

        # ----------------------------------------------------
        # STEP 7
        # OCR produced usable Page-1 text.
        # ----------------------------------------------------

        if len(
            ocr_normalized_text
        ) >= self.MIN_OCR_CHAR_COUNT:

            return Page1TextResult(
                raw_text=ocr_raw_text,

                normalized_text=(
                    ocr_normalized_text
                ),

                mode=mode,

                extraction_method="ocr",

                text_quality_score=(
                    ocr_quality
                ),

                ocr_used=True,

                native_text_quality_score=(
                    native_quality
                ),

                native_char_count=(
                    native_char_count
                ),

                final_char_count=len(
                    ocr_normalized_text
                ),

                ocr_confidence=(
                    ocr_result.average_confidence
                ),

                ocr_engine=(
                    ocr_result.engine
                ),

                page_image_count=(
                    structure.image_count
                ),

                page_image_coverage=round(
                    structure.image_coverage,
                    4,
                ),

                mode_confidence=(
                    mode_confidence
                ),
            )

        # ----------------------------------------------------
        # STEP 8
        # OCR ran but produced insufficient text.
        # ----------------------------------------------------

        return Page1TextResult(
            raw_text=ocr_raw_text,

            normalized_text=(
                ocr_normalized_text
            ),

            mode=mode,

            extraction_method=(
                "ocr_low_quality"
            ),

            text_quality_score=(
                ocr_quality
            ),

            ocr_used=True,

            native_text_quality_score=(
                native_quality
            ),

            native_char_count=(
                native_char_count
            ),

            final_char_count=len(
                ocr_normalized_text
            ),

            ocr_confidence=(
                ocr_result.average_confidence
            ),

            ocr_engine=(
                ocr_result.engine
            ),

            page_image_count=(
                structure.image_count
            ),

            page_image_coverage=round(
                structure.image_coverage,
                4,
            ),

            mode_confidence=(
                mode_confidence
            ),
        )


    # ========================================================
    # PAGE-1 STRUCTURE ANALYSIS
    # ========================================================

    @staticmethod
    def _analyze_page1_structure(
        file_bytes: bytes,
    ) -> Page1Structure:
        """
        Analyze Page 1 using PyMuPDF.

        Extract:
        - native text
        - raster image count
        - approximate union image coverage

        Image coverage is calculated from image rectangles on the
        page rather than from image count alone.

        This matters because:
        - a small bank logo should NOT make a digital PDF hybrid
        - a full-page scanned raster should strongly indicate scan
        """

        try:
            import fitz

        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is not installed. "
                "Install it with: "
                "python -m pip install pymupdf"
            ) from exc

        try:

            document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                "Unable to open PDF: "
                f"{exc}"
            ) from exc

        try:

            if document.page_count == 0:
                raise ValueError(
                    "PDF contains no pages."
                )

            page = document.load_page(
                0
            )

            native_text = (
                page.get_text(
                    "text"
                )
                or ""
            )

            page_rect = page.rect

            page_area = max(
                float(
                    page_rect.width
                    * page_rect.height
                ),
                1.0,
            )

            image_rectangles = []

            # ------------------------------------------------
            # Get raster images actually displayed on Page 1.
            # ------------------------------------------------

            try:

                images = page.get_images(
                    full=True
                )

            except Exception:

                images = []

            seen_xrefs = set()

            for image in images:

                if not image:
                    continue

                xref = image[0]

                if xref in seen_xrefs:
                    continue

                seen_xrefs.add(
                    xref
                )

                try:

                    rects = (
                        page.get_image_rects(
                            xref
                        )
                    )

                except Exception:

                    rects = []

                for rect in rects:

                    try:

                        clipped = (
                            rect
                            & page_rect
                        )

                    except Exception:

                        continue

                    if (
                        clipped.width <= 0
                        or clipped.height <= 0
                    ):
                        continue

                    image_rectangles.append(
                        clipped
                    )

            image_count = len(
                image_rectangles
            )

            image_coverage = (
                Page1TextProvider
                ._calculate_rectangle_union_coverage(
                    image_rectangles,
                    page_area,
                )
            )

            return Page1Structure(
                native_text=native_text,

                image_count=image_count,

                image_coverage=(
                    image_coverage
                ),

                page_area=page_area,
            )

        finally:

            document.close()


    # ========================================================
    # IMAGE COVERAGE
    # ========================================================

    @staticmethod
    def _calculate_rectangle_union_coverage(
        rectangles,
        page_area: float,
    ) -> float:
        """
        Calculate approximate union coverage of image rectangles.

        We intentionally avoid simply summing image areas because
        overlapping images could otherwise produce coverage > 100%.

        A sweep across unique x boundaries gives an exact union
        area for axis-aligned PDF image rectangles.
        """

        if (
            not rectangles
            or page_area <= 0
        ):
            return 0.0

        valid_rectangles = []

        for rect in rectangles:

            x0 = float(
                rect.x0
            )

            y0 = float(
                rect.y0
            )

            x1 = float(
                rect.x1
            )

            y1 = float(
                rect.y1
            )

            if (
                x1 <= x0
                or y1 <= y0
            ):
                continue

            valid_rectangles.append(
                (
                    x0,
                    y0,
                    x1,
                    y1,
                )
            )

        if not valid_rectangles:
            return 0.0

        x_points = sorted(
            {
                coordinate
                for rectangle
                in valid_rectangles
                for coordinate
                in (
                    rectangle[0],
                    rectangle[2],
                )
            }
        )

        union_area = 0.0

        for index in range(
            len(x_points) - 1
        ):

            left = x_points[
                index
            ]

            right = x_points[
                index + 1
            ]

            width = (
                right
                - left
            )

            if width <= 0:
                continue

            y_intervals = []

            for (
                x0,
                y0,
                x1,
                y1,
            ) in valid_rectangles:

                if (
                    x0 < right
                    and x1 > left
                ):
                    y_intervals.append(
                        (
                            y0,
                            y1,
                        )
                    )

            if not y_intervals:
                continue

            y_intervals.sort(
                key=lambda item: item[0]
            )

            merged_height = 0.0

            current_start = (
                y_intervals[0][0]
            )

            current_end = (
                y_intervals[0][1]
            )

            for (
                start,
                end,
            ) in y_intervals[1:]:

                if start <= current_end:

                    current_end = max(
                        current_end,
                        end,
                    )

                else:

                    merged_height += (
                        current_end
                        - current_start
                    )

                    current_start = start
                    current_end = end

            merged_height += (
                current_end
                - current_start
            )

            union_area += (
                width
                * merged_height
            )

        coverage = (
            union_area
            / page_area
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    coverage,
                ),
            ),
            4,
        )


    # ========================================================
    # MODE CLASSIFICATION
    # ========================================================

    def _classify_mode(
        self,
        native_text: str,
        native_quality: float,
        image_count: int,
        image_coverage: float,
    ) -> tuple[str, float]:
        """
        Classify structural Page-1 mode.

        DIGITAL PDF
        -----------
        Meaningful native text with little raster-image coverage.

        SCANNED PDF
        -----------
        Little/no meaningful native text and a large raster image
        covering most of the page.

        HYBRID PDF
        ----------
        Meaningful native text plus meaningful raster-image
        coverage.

        Important:
        A small logo/image does not make a page hybrid.
        """

        native_char_count = len(
            native_text or ""
        )

        has_meaningful_native_text = (
            native_char_count
            >= self.MODE_MIN_NATIVE_TEXT_CHARS
            and native_quality
            >= self.MIN_NATIVE_QUALITY_SCORE
        )

        has_images = (
            image_count > 0
        )

        # ----------------------------------------------------
        # SCANNED
        #
        # Large raster page + no meaningful native text.
        # ----------------------------------------------------

        if (
            not has_meaningful_native_text
            and has_images
            and image_coverage
            >= self.SCANNED_IMAGE_COVERAGE
        ):

            confidence = min(
                1.0,
                0.80
                + (
                    image_coverage
                    * 0.20
                ),
            )

            return (
                "scanned_pdf",
                round(
                    confidence,
                    4,
                ),
            )

        # ----------------------------------------------------
        # HYBRID
        #
        # Useful native text + substantial raster content.
        # ----------------------------------------------------

        if (
            has_meaningful_native_text
            and has_images
            and image_coverage
            >= self.HYBRID_IMAGE_COVERAGE
        ):

            confidence = min(
                1.0,
                0.70
                + (
                    image_coverage
                    * 0.20
                )
                + (
                    native_quality
                    * 0.10
                ),
            )

            return (
                "hybrid_pdf",
                round(
                    confidence,
                    4,
                ),
            )

        # ----------------------------------------------------
        # DIGITAL
        #
        # Useful native text and images are absent/small.
        # ----------------------------------------------------

        if has_meaningful_native_text:

            confidence = (
                0.80
                + (
                    native_quality
                    * 0.20
                )
            )

            return (
                "digital_pdf",
                round(
                    min(
                        confidence,
                        1.0,
                    ),
                    4,
                ),
            )

        # ----------------------------------------------------
        # FALLBACK
        #
        # Native text is poor but page-image coverage is not
        # large enough to confidently call it scanned.
        #
        # OCR routing still happens independently.
        # ----------------------------------------------------

        if has_images:

            # Image-heavy but below the strict scan threshold.
            #
            # Treat as scanned with reduced mode confidence.
            # This is preferable to incorrectly claiming digital
            # when native text is unusable.

            confidence = (
                0.55
                + (
                    image_coverage
                    * 0.25
                )
            )

            return (
                "scanned_pdf",
                round(
                    min(
                        confidence,
                        0.79,
                    ),
                    4,
                ),
            )

        # No useful native text and no detected raster images.
        #
        # Structurally ambiguous PDF. We retain digital_pdf as
        # the conservative PDF-object fallback but expose low
        # confidence so the API does not imply certainty.

        return (
            "digital_pdf",
            0.40,
        )


    # ========================================================
    # NATIVE PAGE-1 EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_native_page1(
        file_bytes: bytes,
    ) -> str:
        """
        Backward-compatible native Page-1 extraction helper.

        The main extract() path now uses
        _analyze_page1_structure(), but this method is retained
        because other code/tests may import it.
        """

        structure = (
            Page1TextProvider
            ._analyze_page1_structure(
                file_bytes
            )
        )

        return structure.native_text


    # ========================================================
    # PAGE-1 PDF CREATION
    # ========================================================

    @staticmethod
    def _build_page1_pdf(
        file_bytes: bytes,
    ) -> bytes:
        """
        Create an in-memory PDF containing only Page 1.

        The existing OCR extractor supports complete scanned PDFs
        and therefore may OCR every page.

        Phase 1 requires only Page 1.

        Creating a one-page PDF allows us to reuse the existing
        OCR component without changing Phase 2 and without OCRing
        a 20/40/100-page statement unnecessarily.
        """

        try:
            import fitz

        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is not installed."
            ) from exc

        try:

            source_document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                "Unable to open PDF while "
                "building Page-1 OCR input: "
                f"{exc}"
            ) from exc

        output_document = None

        try:

            if source_document.page_count == 0:
                raise ValueError(
                    "PDF contains no pages."
                )

            output_document = (
                fitz.open()
            )

            output_document.insert_pdf(
                source_document,
                from_page=0,
                to_page=0,
            )

            page1_bytes = (
                output_document.tobytes(
                    garbage=4,
                    deflate=True,
                )
            )

            if not page1_bytes:
                raise RuntimeError(
                    "Unable to create Page-1 PDF."
                )

            return page1_bytes

        finally:

            if output_document is not None:
                output_document.close()

            source_document.close()


    # ========================================================
    # NATIVE ROUTING
    # ========================================================

    def _native_text_is_usable(
        self,
        text: str,
        quality_score: float,
    ) -> bool:
        """
        Determine whether native Page-1 text is sufficiently
        usable to avoid OCR.
        """

        if not text:
            return False

        if len(
            text
        ) < self.MIN_NATIVE_CHAR_COUNT:
            return False

        if (
            quality_score
            < self.MIN_NATIVE_QUALITY_SCORE
        ):
            return False

        return True


    # ========================================================
    # TEXT QUALITY
    # ========================================================

    def _score_text_quality(
        self,
        text: str,
    ) -> float:
        """
        Generic text-quality score.

        This does NOT determine whether the document is a bank
        statement.

        It only determines whether extracted text appears
        sufficiently readable for downstream processing.
        """

        if not text:
            return 0.0

        char_count = len(
            text
        )

        if char_count == 0:
            return 0.0

        # ----------------------------------------------------
        # Character quantity
        # ----------------------------------------------------

        length_score = min(
            1.0,
            char_count
            / self.GOOD_CHAR_COUNT,
        )

        # ----------------------------------------------------
        # Alphabetic content
        # ----------------------------------------------------

        alpha_count = sum(
            character.isalpha()
            for character in text
        )

        alpha_ratio = (
            alpha_count
            / char_count
        )

        alpha_score = min(
            1.0,
            alpha_ratio
            / self.MIN_ALPHA_RATIO,
        )

        # ----------------------------------------------------
        # Alphanumeric content
        # ----------------------------------------------------

        alnum_count = sum(
            character.isalnum()
            for character in text
        )

        alnum_ratio = (
            alnum_count
            / char_count
        )

        alnum_score = min(
            1.0,
            alnum_ratio
            / self.MIN_ALNUM_RATIO,
        )

        # ----------------------------------------------------
        # Replacement / corruption characters
        # ----------------------------------------------------

        corruption_count = (
            text.count(
                "\ufffd"
            )
        )

        corruption_ratio = (
            corruption_count
            / char_count
        )

        corruption_score = max(
            0.0,
            1.0
            - (
                corruption_ratio
                * 10.0
            ),
        )

        # ----------------------------------------------------
        # Weighted generic score
        # ----------------------------------------------------

        score = (
            0.30 * length_score
            + 0.25 * alpha_score
            + 0.25 * alnum_score
            + 0.20 * corruption_score
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            ),
            4,
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Conservative generic normalization.

        Important:
        Do not destroy useful separators such as:

        /
        -
        :
        .
        @
        *

        They are important for:
        - dates
        - IFSC/account labels
        - masked account numbers
        - addresses
        - transaction references
        """

        if not text:
            return ""

        normalized = str(
            text
        )

        # ----------------------------------------------------
        # Unicode whitespace
        # ----------------------------------------------------

        normalized = (
            normalized
            .replace(
                "\u00a0",
                " ",
            )
            .replace(
                "\u2007",
                " ",
            )
            .replace(
                "\u202f",
                " ",
            )
        )

        # ----------------------------------------------------
        # Normalize line endings
        # ----------------------------------------------------

        normalized = (
            normalized
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        # ----------------------------------------------------
        # Remove control characters except newline/tab
        # ----------------------------------------------------

        normalized = "".join(
            character
            for character in normalized
            if (
                character in "\n\t"
                or ord(character) >= 32
            )
        )

        # ----------------------------------------------------
        # Collapse horizontal whitespace
        # ----------------------------------------------------

        normalized = re.sub(
            r"[ \t]+",
            " ",
            normalized,
        )

        # ----------------------------------------------------
        # Trim each line
        # ----------------------------------------------------

        lines = [
            line.strip()
            for line in normalized.split(
                "\n"
            )
        ]

        # ----------------------------------------------------
        # Remove excessive blank lines
        # ----------------------------------------------------

        cleaned_lines: list[str] = []

        previous_blank = False

        for line in lines:

            is_blank = (
                not line
            )

            if (
                is_blank
                and previous_blank
            ):
                continue

            cleaned_lines.append(
                line
            )

            previous_blank = (
                is_blank
            )

        normalized = "\n".join(
            cleaned_lines
        )

        return normalized.strip()


# ============================================================
# SINGLETON
# ============================================================


page1_text_provider = Page1TextProvider()