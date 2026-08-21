"""
Common document processing package.
"""

from .processor import (
    detect_file_type,
    process_document,
    process_document_file,
)

from .result import (
    PageProcessingResult,
    DocumentProcessingResult,
)


__all__ = [
    "detect_file_type",
    "process_document",
    "process_document_file",
    "PageProcessingResult",
    "DocumentProcessingResult",
]