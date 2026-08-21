from .engine import (
    get_ocr_reader,
    run_ocr,
    extract_text,
    extract_text_dicts,
    extract_text_lines,
    extract_full_text,
)

from .result import (
    OCRResult,
    normalize_text,
    normalize_ocr_results,
    filter_by_confidence,
)

__all__ = [
    "get_ocr_reader",
    "run_ocr",
    "extract_text",
    "extract_text_dicts",
    "extract_text_lines",
    "extract_full_text",
    "OCRResult",
    "normalize_text",
    "normalize_ocr_results",
    "filter_by_confidence",
]