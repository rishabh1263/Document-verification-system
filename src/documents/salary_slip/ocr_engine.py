"""
OCR Engine abstraction layer.

Strategy:
1. For PDFs, try native PyMuPDF text extraction first.
2. Preserve native word bounding boxes for layout-aware extraction.
3. Preserve page number for every OCR/native-text word.
4. If the PDF has no usable native text, rasterize it and use Tesseract.
5. For image files, use Tesseract directly.

This allows downstream extraction to process multi-page documents
page-by-page instead of mixing all pages together.

PaddleOCR is intentionally not required.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import List

from src.documents.salary_slip import config


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class OCRWord:
    """
    One OCR/native-text word with its position.

    page_number is 1-based:
        first page  -> 1
        second page -> 2
        etc.
    """

    text: str
    confidence: float
    bbox: tuple
    page_number: int = 1


@dataclass
class OCRResult:
    """
    Standard OCR result returned by all extraction methods.
    """

    text: str
    words: List[OCRWord] = field(default_factory=list)

    source: str = "unknown"

    engine_time_sec: float = 0.0

    page_count: int = 1


# ============================================================
# NATIVE PDF EXTRACTION
# ============================================================

def extract_native_text(
    pdf_path: str,
) -> OCRResult | None:
    """
    Extract text and word bounding boxes directly from a PDF.

    Native PDF extraction is preferred because it is:
    - faster than OCR
    - usually more accurate
    - capable of returning exact word coordinates

    Every extracted word also receives its page number.

    Returns None when the PDF does not contain enough usable
    native text.
    """

    try:
        import fitz

    except ImportError:
        return None

    t0 = time.time()

    try:
        doc = fitz.open(
            pdf_path
        )

    except Exception:
        return None

    full_text = []

    all_words = []

    page_count = len(doc)

    try:

        # ====================================================
        # PROCESS EACH PDF PAGE
        # ====================================================

        for page_index, page in enumerate(
            doc,
            start=1,
        ):

            # ------------------------------------------------
            # PAGE TEXT
            # ------------------------------------------------

            page_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            if page_text:

                full_text.append(
                    page_text
                )

            else:

                # Preserve page separation even when a page
                # contains no readable native text.
                full_text.append(
                    ""
                )

            # ------------------------------------------------
            # WORDS + BOUNDING BOXES
            # ------------------------------------------------

            native_words = page.get_text(
                "words",
                sort=True,
            )

            for word in native_words:

                # PyMuPDF word format:
                #
                # x0
                # y0
                # x1
                # y1
                # text
                # block_no
                # line_no
                # word_no

                if len(word) < 5:
                    continue

                x0 = float(
                    word[0]
                )

                y0 = float(
                    word[1]
                )

                x1 = float(
                    word[2]
                )

                y1 = float(
                    word[3]
                )

                word_text = str(
                    word[4]
                ).strip()

                if not word_text:
                    continue

                all_words.append(
                    OCRWord(
                        text=word_text,

                        confidence=1.0,

                        bbox=(
                            x0,
                            y0,
                            x1,
                            y1,
                        ),

                        page_number=page_index,
                    )
                )

    finally:

        doc.close()

    # ========================================================
    # COMBINE TEXT
    # ========================================================

    joined = "\n".join(
        full_text
    ).strip()

    # ========================================================
    # NATIVE TEXT QUALITY CHECK
    # ========================================================

    if len(joined) < config.MIN_NATIVE_TEXT_CHARS:

        # Treat as scanned/image PDF.
        return None

    return OCRResult(
        text=joined,

        words=all_words,

        source="native_text_layer",

        engine_time_sec=round(
            time.time() - t0,
            3,
        ),

        page_count=page_count,
    )


# ============================================================
# TESSERACT BACKEND
# ============================================================

class _TesseractBackend:
    """
    Lightweight OCR backend.

    Used for:
    - PNG
    - JPG/JPEG
    - scanned PDFs after rasterization
    """

    def run(
        self,
        image_path: str,
        page_number: int = 1,
    ) -> OCRResult:

        import pytesseract

        from PIL import Image
        from pytesseract import Output

        # ====================================================
        # WINDOWS TESSERACT LOCATION
        # ====================================================

        default_tesseract = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        if os.path.exists(
            default_tesseract
        ):

            pytesseract.pytesseract.tesseract_cmd = (
                default_tesseract
            )

        # ====================================================
        # CHECK TESSERACT
        # ====================================================

        try:

            pytesseract.get_tesseract_version()

        except Exception as exc:

            raise RuntimeError(
                "Tesseract OCR is not available. "
                "Install Tesseract or configure "
                "pytesseract.pytesseract.tesseract_cmd."
            ) from exc

        t0 = time.time()

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            image_path
        )

        # ====================================================
        # OCR
        # ====================================================

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 6",
        )

        words = []

        lines_map = {}

        total = len(
            data["text"]
        )

        # ====================================================
        # PROCESS OCR WORDS
        # ====================================================

        for i in range(total):

            text = str(
                data["text"][i]
            ).strip()

            if not text:
                continue

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            try:

                raw_conf = float(
                    data["conf"][i]
                )

                if raw_conf >= 0:

                    confidence = (
                        raw_conf / 100.0
                    )

                else:

                    confidence = 0.0

            except (
                ValueError,
                TypeError,
            ):

                confidence = 0.0

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x = int(
                data["left"][i]
            )

            y = int(
                data["top"][i]
            )

            width = int(
                data["width"][i]
            )

            height = int(
                data["height"][i]
            )

            # ------------------------------------------------
            # STORE WORD
            # ------------------------------------------------

            words.append(
                OCRWord(
                    text=text,

                    confidence=confidence,

                    bbox=(
                        x,
                        y,
                        x + width,
                        y + height,
                    ),

                    page_number=page_number,
                )
            )

            # ------------------------------------------------
            # RECONSTRUCT TEXT LINES
            # ------------------------------------------------

            line_key = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            lines_map.setdefault(
                line_key,
                [],
            ).append(
                text
            )

        # ====================================================
        # BUILD FULL TEXT
        # ====================================================

        full_text = "\n".join(
            " ".join(
                words_in_line
            )
            for words_in_line
            in lines_map.values()
        )

        return OCRResult(
            text=full_text,

            words=words,

            source="tesseract",

            engine_time_sec=round(
                time.time() - t0,
                3,
            ),

            page_count=1,
        )


# ============================================================
# RUN OCR
# ============================================================

def run_ocr(
    image_path: str,
    page_number: int = 1,
) -> OCRResult:
    """
    Run Tesseract OCR on one image.

    page_number allows PDF rasterization to preserve which
    PDF page each OCR word came from.
    """

    backend = (
        _TesseractBackend()
    )

    return backend.run(
        image_path,
        page_number=page_number,
    )


# ============================================================
# MAIN PUBLIC ENTRY POINT
# ============================================================

def extract_text(
    file_path: str,
) -> OCRResult:
    """
    Extract text from PDF/image.

    PDF:

        native text
            â†“
        if usable
            â†“
        return text + bounding boxes + page numbers

        otherwise
            â†“
        rasterize each page
            â†“
        Tesseract OCR
            â†“
        return text + bounding boxes + page numbers

    Image:

        Tesseract OCR
    """

    ext = os.path.splitext(
        file_path
    )[1].lower()

    # ========================================================
    # PDF
    # ========================================================

    if ext == ".pdf":

        if config.PREFER_NATIVE_TEXT_LAYER:

            native = extract_native_text(
                file_path
            )

            if native is not None:

                # --------------------------------------------
                # Layout-aware extraction needs coordinates.
                # --------------------------------------------

                if native.words:

                    return native

                # Native text exists but no word coordinates.
                # Rasterize and OCR instead.

                return _ocr_scanned_pdf(
                    file_path
                )

        # No usable native text.

        return _ocr_scanned_pdf(
            file_path
        )

    # ========================================================
    # IMAGE
    # ========================================================

    return run_ocr(
        file_path,
        page_number=1,
    )


# ============================================================
# PDF -> IMAGE -> TESSERACT
# ============================================================

def _ocr_scanned_pdf(
    pdf_path: str,
) -> OCRResult:
    """
    Rasterize every PDF page and run Tesseract OCR.

    IMPORTANT:

    Every OCRWord receives its original PDF page number.

    Example:

        page 1 words -> page_number=1
        page 2 words -> page_number=2
        page 3 words -> page_number=3

    This allows pipeline.py to separate multiple salary slips
    contained inside one PDF.
    """

    import fitz

    t0 = time.time()

    doc = fitz.open(
        pdf_path
    )

    all_text = []

    all_words = []

    page_count = len(
        doc
    )

    # ========================================================
    # TEMP DIRECTORY
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix="doc_verify_"
    ) as tmp_dir:

        try:

            # =================================================
            # PROCESS EACH PAGE
            # =================================================

            for page_number, page in enumerate(
                doc,
                start=1,
            ):

                # --------------------------------------------
                # Rasterize
                #
                # 250 DPI is a reasonable balance for:
                # - small salary numbers
                # - table text
                # - OCR speed
                # --------------------------------------------

                pix = page.get_pixmap(
                    dpi=250,
                    alpha=False,
                )

                image_path = os.path.join(
                    tmp_dir,
                    f"page_{page_number}.png",
                )

                pix.save(
                    image_path
                )

                # --------------------------------------------
                # OCR PAGE
                # --------------------------------------------

                result = run_ocr(
                    image_path,
                    page_number=page_number,
                )

                # --------------------------------------------
                # PRESERVE PAGE TEXT
                # --------------------------------------------

                all_text.append(
                    result.text
                )

                # --------------------------------------------
                # PRESERVE PAGE-AWARE WORDS
                # --------------------------------------------

                all_words.extend(
                    result.words
                )

        finally:

            doc.close()

    # ========================================================
    # FINAL PDF OCR RESULT
    # ========================================================

    return OCRResult(
        text="\n".join(
            all_text
        ),

        words=all_words,

        source="tesseract_rasterized_pdf",

        engine_time_sec=round(
            time.time() - t0,
            3,
        ),

        page_count=page_count,
    )


# ============================================================
# PAGE HELPERS
# ============================================================

def get_words_for_page(
    ocr_result: OCRResult,
    page_number: int,
) -> List[OCRWord]:
    """
    Return only OCR words belonging to one page.

    Example:

        page_words = get_words_for_page(
            result,
            3,
        )
    """

    return [
        word
        for word in ocr_result.words
        if word.page_number == page_number
    ]


def get_page_numbers(
    ocr_result: OCRResult,
) -> List[int]:
    """
    Return sorted page numbers represented in OCRResult.words.

    Example:
        [1, 2, 3, 4, 5, 6]
    """

    return sorted(
        {
            word.page_number
            for word in ocr_result.words
        }
    )


def words_to_text(
    words: List[OCRWord],
    y_tolerance: float = 8.0,
) -> str:
    """
    Reconstruct approximate page text from OCR words.

    This helper is mainly intended for downstream page-by-page
    extraction.

    Words are grouped into lines using their Y coordinates and
    sorted left-to-right inside each line.
    """

    if not words:
        return ""

    # Sort top-to-bottom, then left-to-right.

    sorted_words = sorted(
        words,
        key=lambda word: (
            word.bbox[1],
            word.bbox[0],
        ),
    )

    lines = []

    current_line = []

    current_y = None

    for word in sorted_words:

        y = float(
            word.bbox[1]
        )

        if current_y is None:

            current_line = [
                word
            ]

            current_y = y

            continue

        # Same approximate line.

        if abs(
            y - current_y
        ) <= y_tolerance:

            current_line.append(
                word
            )

        else:

            # Finish previous line.

            current_line = sorted(
                current_line,
                key=lambda item: item.bbox[0],
            )

            lines.append(
                " ".join(
                    item.text
                    for item in current_line
                )
            )

            # Start new line.

            current_line = [
                word
            ]

            current_y = y

    # ========================================================
    # FINAL LINE
    # ========================================================

    if current_line:

        current_line = sorted(
            current_line,
            key=lambda item: item.bbox[0],
        )

        lines.append(
            " ".join(
                item.text
                for item in current_line
            )
        )

    return "\n".join(
        lines
    )
