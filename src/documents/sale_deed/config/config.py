"""
Global configuration for the Sale Deed Extraction Pipeline.
"""

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME: str = "Sale Deed AI"

VERSION: str = "1.0.0"

# ==========================================================
# OCR SETTINGS
# ==========================================================

OCR_ENGINE: str = "surya"

# ==========================================================
# DOCUMENT SETTINGS
# ==========================================================

PDF_DPI: int = 300

SUPPORTED_DOCUMENT_FORMATS = [
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
]

# ==========================================================
# OUTPUT SETTINGS
# ==========================================================

SAVE_ENHANCED_IMAGES: bool = True

SAVE_OCR_TEXT: bool = True

SAVE_JSON: bool = True

SAVE_TABLE: bool = True

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL: str = "INFO"
