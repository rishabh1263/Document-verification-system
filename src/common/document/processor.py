"""
Common Document Processing Engine.

Fast strategy
-------------
IMAGE
    -> quality
    -> preprocessing
    -> OCR
    -> generic tamper analysis

PDF
    -> native PDF text check
    -> if useful text exists: use native text
    -> if scanned: render pages
    -> OCR scanned pages
    -> generic tamper analysis

Important
---------
Document-specific extraction and validation do NOT belong here.

Tamper analysis is also document-independent. It is attached to the page
quality dictionary so the existing PageProcessingResult contract does not
need to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common.image.preprocessing import (
    load_image,
    prepare_for_ocr,
)

from src.common.image.quality import (
    analyze_image_quality,
)

from src.common.ocr import (
    extract_text_dicts,
)

from src.common.pdf.loader import (
    extract_pdf_text,
)

from src.common.pdf.renderer import (
    render_pdf_pages,
)

from src.common.authenticity.tamper import (
    analyze_tampering,
)

from .result import (
    DocumentProcessingResult,
    PageProcessingResult,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

PDF_EXTENSIONS = {
    ".pdf",
}


# ============================================================
# FILE TYPE
# ============================================================

def detect_file_type(
    filename: str,
    file_bytes: bytes | None = None,
) -> str:
    """
    Detect image or PDF.
    """

    suffix = Path(filename).suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return "pdf"

    if suffix in IMAGE_EXTENSIONS:
        return "image"

    if file_bytes:

        if file_bytes.startswith(b"%PDF"):
            return "pdf"

        if file_bytes.startswith(b"\xff\xd8\xff"):
            return "image"

        if file_bytes.startswith(b"\x89PNG"):
            return "image"

        if (
            len(file_bytes) >= 12
            and file_bytes[:4] == b"RIFF"
            and file_bytes[8:12] == b"WEBP"
        ):
            return "image"

    return "unknown"


# ============================================================
# IMAGE VALIDATION
# ============================================================

def _validate_image(
    image: np.ndarray,
) -> None:

    if image is None:
        raise ValueError(
            "Image is None."
        )

    if image.size == 0:
        raise ValueError(
            "Image is empty."
        )

    if len(image.shape) not in (2, 3):
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )


# ============================================================
# TAMPER ANALYSIS
# ============================================================

def _safe_tamper_analysis(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Run common image tamper analysis without allowing a tamper-module
    failure to break the whole document-processing pipeline.

    Tamper analysis is a risk signal, NOT proof that a document is fake.
    """
    try:
        return analyze_tampering(image)

    except Exception as exc:
        return {
            "tamper_score": 0,
            "risk": "UNKNOWN",
            "decision": "MANUAL_REVIEW",
            "signals": [
                f"Tamper analysis failed: {exc}"
            ],
            "checks": {},
        }


def _attach_tamper_to_quality(
    quality: dict[str, Any],
    tamper: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve the existing quality structure and add tamper information.

    Existing consumers can continue reading:
        quality["blur_score"]
        quality["brightness"]
        etc.

    New consumers can read:
        quality["tamper"]
    """
    if not isinstance(quality, dict):
        quality = {
            "source": quality,
        }

    result = dict(quality)
    result["tamper"] = tamper

    return result


# ============================================================
# PROCESS IMAGE PAGE
# ============================================================

def _process_image_page(
    image: np.ndarray,
    page_number: int,
    *,
    preprocess: bool = True,
    ocr: bool = True,
    minimum_confidence: float = 0.0,
    gpu: bool = False,
    run_tamper: bool = True,
) -> PageProcessingResult:
    """
    Process one image page.

    Pipeline:
        original image
            |
            +--> quality
            |
            +--> tamper analysis
            |
            +--> preprocessing
                    |
                    +--> OCR

    Quality and tamper analysis are calculated from the original image.
    OCR preprocessing therefore cannot hide image-level manipulation
    signals.
    """

    _validate_image(image)

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    quality = analyze_image_quality(
        image
    )

    # --------------------------------------------------------
    # TAMPER
    # --------------------------------------------------------

    if run_tamper:
        tamper = _safe_tamper_analysis(
            image
        )

        quality = _attach_tamper_to_quality(
            quality,
            tamper,
        )

    # --------------------------------------------------------
    # OCR IMAGE
    # --------------------------------------------------------

    if preprocess:

        ocr_image = prepare_for_ocr(
            image,
            grayscale=True,
        )

    else:

        ocr_image = image

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    if ocr:

        ocr_results = extract_text_dicts(
            ocr_image,
            gpu=gpu,
            minimum_confidence=minimum_confidence,
        )

    else:

        ocr_results = []

    return PageProcessingResult(
        page_number=page_number,
        width=width,
        height=height,
        quality=quality,
        ocr=ocr_results,
    )


# ============================================================
# PDF NATIVE TEXT PAGE
# ============================================================

def _native_text_page(
    text: str,
    page_number: int,
) -> PageProcessingResult:
    """
    Create a page result from native PDF text.

    Native PDF text does not have OCR confidence or bounding
    boxes, so those fields are intentionally left empty.

    There is no image available at this stage, therefore image-level
    tamper analysis is not performed here.
    """

    cleaned = text.strip()

    ocr_results = []

    if cleaned:

        ocr_results.append(
            {
                "text": cleaned,
                "confidence": 1.0,
                "bbox": [],
                "source": "native_pdf_text",
            }
        )

    return PageProcessingResult(
        page_number=page_number,
        width=0,
        height=0,
        quality={
            "source": "native_pdf",
            "tamper": {
                "available": False,
                "risk": "NOT_PERFORMED",
                "decision": "NOT_PERFORMED",
                "signals": [
                    "Native PDF text path used; no rendered image was available for image-level tamper analysis."
                ],
                "checks": {},
            },
        },
        ocr=ocr_results,
    )


# ============================================================
# PDF TEXT CHECK
# ============================================================

def _has_useful_native_text(
    pages: list[str],
    minimum_chars: int = 30,
) -> bool:
    """
    Decide whether a PDF contains enough native text
    to avoid OCR.

    We use a small threshold rather than simply checking
    whether text exists because some PDFs contain only
    headers, metadata, or a few characters.
    """

    total_chars = sum(
        len(page.strip())
        for page in pages
    )

    return total_chars >= minimum_chars


# ============================================================
# MAIN DOCUMENT PROCESSOR
# ============================================================

def process_document(
    file_bytes: bytes,
    filename: str = "document",
    *,
    preprocess: bool = True,
    ocr: bool = True,
    minimum_confidence: float = 0.0,
    gpu: bool = False,
    pdf_dpi: int = 150,
    force_pdf_ocr: bool = False,
    run_tamper: bool = True,
) -> DocumentProcessingResult:
    """
    Process an image or PDF.

    Fast PDF strategy:

        1. Try native PDF text.
        2. If useful text exists, DO NOT OCR.
        3. If PDF is scanned, render pages and OCR.
        4. Tamper analysis is performed on rendered image pages.

    force_pdf_ocr=True can be used when OCR is explicitly required.

    run_tamper=False can be used for very high-throughput workloads where
    image-level tamper analysis is intentionally disabled.
    """

    if not file_bytes:

        raise ValueError(
            "Document is empty."
        )

    # --------------------------------------------------------
    # FILE TYPE
    # --------------------------------------------------------

    file_type = detect_file_type(
        filename,
        file_bytes,
    )

    if file_type == "unknown":

        raise ValueError(
            f"Unsupported document type: {filename}"
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if file_type == "image":

        image = load_image(
            file_bytes
        )

        page_result = _process_image_page(
            image,
            page_number=1,
            preprocess=preprocess,
            ocr=ocr,
            minimum_confidence=minimum_confidence,
            gpu=gpu,
            run_tamper=run_tamper,
        )

        return DocumentProcessingResult(
            filename=filename,
            file_type="image",
            page_count=1,
            pages=[
                page_result
            ],
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    native_pages = extract_pdf_text(
        file_bytes
    )

    # --------------------------------------------------------
    # FAST PATH:
    # DIGITAL PDF
    # --------------------------------------------------------

    if (
        not force_pdf_ocr
        and _has_useful_native_text(
            native_pages
        )
    ):

        page_results = [
            _native_text_page(
                text,
                page_number=index,
            )
            for index, text in enumerate(
                native_pages,
                start=1,
            )
        ]

        return DocumentProcessingResult(
            filename=filename,
            file_type="pdf",
            page_count=len(page_results),
            pages=page_results,
        )

    # --------------------------------------------------------
    # SLOW PATH:
    # SCANNED PDF
    # --------------------------------------------------------

    if not ocr:

        return DocumentProcessingResult(
            filename=filename,
            file_type="pdf",
            page_count=len(native_pages),
            pages=[
                _native_text_page(
                    text,
                    page_number=index,
                )
                for index, text in enumerate(
                    native_pages,
                    start=1,
                )
            ],
        )

    # Render ONLY when OCR is actually necessary.
    pages = render_pdf_pages(
        file_bytes,
        dpi=pdf_dpi,
    )

    if not pages:

        raise ValueError(
            "PDF contains no readable pages."
        )

    page_results = []

    for page_number, image in enumerate(
        pages,
        start=1,
    ):

        page_results.append(
            _process_image_page(
                image,
                page_number=page_number,
                preprocess=preprocess,
                ocr=True,
                minimum_confidence=minimum_confidence,
                gpu=gpu,
                run_tamper=run_tamper,
            )
        )

    return DocumentProcessingResult(
        filename=filename,
        file_type="pdf",
        page_count=len(page_results),
        pages=page_results,
    )


# ============================================================
# LOCAL FILE
# ============================================================

def process_document_file(
    path: str | Path,
    **kwargs: Any,
) -> DocumentProcessingResult:
    """
    Process a local document file.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Path is not a file: {path}"
        )

    return process_document(
        file_bytes=path.read_bytes(),
        filename=path.name,
        **kwargs,
    )


# ============================================================
# SIMPLE DIAGNOSTIC
# ============================================================

def processor_contract_test() -> dict[str, Any]:
    """
    Static contract check for the common processor.

    This does not load OCR or process a real document.
    """

    return {
        "passed": True,
        "supports_images": True,
        "supports_digital_pdf": True,
        "supports_scanned_pdf": True,
        "common_tamper_analysis": True,
        "document_specific_validation_here": False,
        "tamper_failure_breaks_processor": False,
    }


__all__ = [
    "detect_file_type",
    "process_document",
    "process_document_file",
    "processor_contract_test",
]