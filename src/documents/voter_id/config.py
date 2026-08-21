"""
Voter ID Configuration.

Single source of truth for Voter ID validation settings.
Validation is intentionally OCR-free.
"""

from pathlib import Path


# ============================================================
# DOCUMENT
# ============================================================

DOCUMENT_TYPE = "VOTER_ID"


# ============================================================
# FILE SUPPORT
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".pdf",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/pdf",
}


# ============================================================
# IMAGE / VISUAL VALIDATION
# ============================================================

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

MIN_CONTRAST = 10.0
MIN_EDGE_DENSITY = 0.006

MIN_GEOMETRY_SCORE = 55.0
MIN_VISUAL_FIELDS = 3
TOTAL_VISUAL_FIELDS = 5


# ============================================================
# ORIENTATION
# ============================================================

PORTRAIT_MIN_RATIO = 0.35
PORTRAIT_MAX_RATIO = 0.95

LANDSCAPE_MIN_RATIO = 1.05
LANDSCAPE_MAX_RATIO = 2.30


# ============================================================
# IMAGE QUALITY
# ============================================================

QUALITY_GOOD_THRESHOLD = 75.0
QUALITY_FAIR_THRESHOLD = 50.0


# ============================================================
# TAMPERING
# ============================================================

HIGH_TAMPER_RISK = {
    "HIGH",
    "CRITICAL",
}

TAMPER_VARIANCE_THRESHOLD = 5000.0


# ============================================================
# DECISION
# ============================================================

MIN_PASS_SCORE = 60.0


# ============================================================
# PERFORMANCE
# ============================================================

MAX_VALIDATION_TIME_SECONDS = 2.0


# ============================================================
# OCR / EXTRACTION
# ============================================================

# IMPORTANT:
# Validation must remain fast and OCR-free.
#
# Keep these disabled until a separate extraction stage is built.

OCR_ENABLED = False
EXTRACTION_ENABLED = False


# ============================================================
# PDF
# ============================================================

PDF_FIRST_PAGE_ONLY = True
PDF_RENDER_SCALE = 1.5


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent