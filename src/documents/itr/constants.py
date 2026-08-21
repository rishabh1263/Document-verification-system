"""
==============================================================
ITR Detection Constants
==============================================================

Centralized configuration constants.

Detector logic should NEVER contain hardcoded values.

Author : SBFC Document Intelligence
==============================================================
"""


# ==========================================================
# DOCUMENT
# ==========================================================

DOCUMENT_NAME = "Income Tax Return"


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# ==========================================================
# DETECTION THRESHOLDS
# ==========================================================

MIN_DETECTION_CONFIDENCE = 0.70

HIGH_CONFIDENCE = 0.90

MEDIUM_CONFIDENCE = 0.75

LOW_CONFIDENCE = 0.50


# ==========================================================
# METADATA SCORING
# ==========================================================

FILE_EXISTS_SCORE = 0.10

SUPPORTED_EXTENSION_SCORE = 0.10

FILE_SIZE_SCORE = 0.05

PDF_OPEN_SCORE = 0.05

PAGE_COUNT_SCORE = 0.10

DIGITAL_MODE_SCORE = 0.20

MIXED_MODE_SCORE = 0.15

SCANNED_MODE_SCORE = 0.10

METADATA_AVAILABLE_SCORE = 0.05


# ==========================================================
# DETECTION WEIGHTS
# ==========================================================

WEIGHTS = {
    "metadata": 0.20,
    "keyword": 0.45,
    "layout": 0.20,
    "structure": 0.15,
}


# ==========================================================
# PAGE LIMITS
# ==========================================================

MIN_PAGES = 1

MAX_REASONABLE_PAGES = 100


# ==========================================================
# DIGITAL PDF HEURISTICS
# ==========================================================

DIGITAL_TEXT_THRESHOLD = 500

MIXED_TEXT_THRESHOLD = 50

MAX_PAGES_TO_ANALYZE = 3


# ==========================================================
# PRIMARY KEYWORDS
# ==========================================================
#
# These are strong ITR identifiers.
#
# Do NOT put generic words here.
#
# ==========================================================

PRIMARY_KEYWORDS = {

    "income tax return": 25,

    "assessment year": 20,

    "acknowledgement": 20,

    "income tax department": 20,

    "department of income tax": 20,

}


# ==========================================================
# SECONDARY KEYWORDS
# ==========================================================
#
# These support ITR detection but should not independently
# establish the document type.
#
# ==========================================================

SECONDARY_KEYWORDS = {

    "pan": 10,

    "gross total income": 8,

    "total income": 8,

    "deduction": 5,

    "refund": 5,

    "tax payable": 8,

    "itr": 6,

    "efile": 5,

    "total tax": 6,

    "tax paid": 6,

}


# ==========================================================
# NEGATIVE KEYWORDS
# ==========================================================
#
# These indicate evidence for other document types.
#
# They are NOT automatic rejection rules.
#
# ==========================================================

NEGATIVE_KEYWORDS = {

    # ------------------------------------------------------
    # Bank Statement
    # ------------------------------------------------------

    "bank statement": 30,

    "statement period": 30,

    "account number": 30,

    "opening balance": 30,

    "closing balance": 30,

    "ifsc": 30,

    "transaction date": 25,

    "transaction details": 25,

    "available balance": 25,

    "account holder": 20,

    "debit": 15,

    "credit": 15,

    # ------------------------------------------------------
    # Salary Document
    # ------------------------------------------------------

    "salary slip": 20,

    "salary statement": 20,

    "payslip": 20,

    "pay slip": 20,

}


# ==========================================================
# COMMON ITR TYPES
# ==========================================================

ITR_TYPES = {
    "itr-1",
    "itr-2",
    "itr-3",
    "itr-4",
    "itr-5",
    "itr-6",
    "itr-7",
}


# ==========================================================
# REASONS
# ==========================================================

REASON_FILE_EXISTS = "File exists"

REASON_SUPPORTED_EXTENSION = "Supported extension"

REASON_EMPTY_FILE = "Empty file"

REASON_UNSUPPORTED_EXTENSION = "Unsupported extension"

REASON_CORRUPTED_PDF = "Corrupted PDF"

REASON_PASSWORD_PROTECTED = "Password protected PDF"

REASON_DIGITAL = "Digital PDF detected"

REASON_SCANNED = "Scanned PDF detected"

REASON_MIXED = "Mixed PDF detected"

REASON_METADATA_FOUND = "PDF metadata available"

REASON_PAGE_COUNT_VALID = "Valid page count"

REASON_KEYWORDS_FOUND = "ITR keywords detected"

REASON_LAYOUT_FOUND = "ITR layout detected"