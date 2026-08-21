"""
==============================================================
ITR Validation Constants
==============================================================

Centralized configuration for the ITR Validation Engine.

Validation logic should NOT contain hardcoded thresholds,
weights, limits, or validation messages.

Author : SBFC Document Intelligence
==============================================================
"""

from __future__ import annotations


# ==========================================================
# DOCUMENT
# ==========================================================

DOCUMENT_NAME = "Income Tax Return"


# ==========================================================
# VALIDATION THRESHOLDS
# ==========================================================

# Minimum overall confidence required to classify the
# document as VALID.

MIN_VALIDATION_CONFIDENCE = 0.70

# Confidence above which the document can be considered
# strongly valid.

HIGH_VALIDATION_CONFIDENCE = 0.90

# Confidence range requiring manual review.

REVIEW_VALIDATION_CONFIDENCE = 0.50


# ==========================================================
# INTEGRITY VALIDATION WEIGHTS
# ==========================================================

INTEGRITY_WEIGHTS = {
    "file_exists": 0.15,
    "valid_pdf": 0.20,
    "readable": 0.20,
    "not_encrypted": 0.15,
    "not_corrupted": 0.20,
    "valid_page_count": 0.10,
}


# ==========================================================
# CONTENT VALIDATION WEIGHTS
# ==========================================================

CONTENT_WEIGHTS = {
    "assessment_year": 0.15,
    "pan": 0.15,
    "taxpayer_information": 0.15,
    "income_information": 0.15,
    "tax_computation": 0.15,
    "verification": 0.10,
    "acknowledgement": 0.15,
}


# ==========================================================
# CONSISTENCY VALIDATION WEIGHTS
# ==========================================================

CONSISTENCY_WEIGHTS = {
    "assessment_year": 0.20,
    "pan": 0.20,
    "income": 0.20,
    "tax": 0.20,
    "acknowledgement": 0.20,
}


# ==========================================================
# FINAL VALIDATION WEIGHTS
# ==========================================================

VALIDATION_WEIGHTS = {
    "integrity": 0.30,
    "content": 0.40,
    "consistency": 0.30,
}


# ==========================================================
# PAGE LIMITS
# ==========================================================

MIN_PAGES = 1

MAX_REASONABLE_PAGES = 100


# ==========================================================
# FILE SIZE LIMITS
# ==========================================================

MIN_FILE_SIZE_BYTES = 1

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


# ==========================================================
# TEXT VALIDATION
# ==========================================================

MIN_TEXT_LENGTH = 100


# ==========================================================
# REQUIRED ITR CONTENT
# ==========================================================

REQUIRED_CONTENT = (
    "assessment_year",
    "pan",
    "taxpayer_information",
    "income_information",
    "tax_computation",
    "verification",
    "acknowledgement",
)


# ==========================================================
# ITR STRUCTURAL COMPONENTS
# ==========================================================

ITR_STRUCTURAL_COMPONENTS = (
    "assessment_year",
    "pan",
    "income",
    "tax_computation",
    "tax_paid",
    "verification",
    "itr_identity",
    "acknowledgement",
)


# ==========================================================
# VALIDATION REASONS
# ==========================================================

REASON_FILE_EXISTS = "File exists"

REASON_FILE_MISSING = "File does not exist"

REASON_VALID_PDF = "Valid PDF"

REASON_INVALID_PDF = "Invalid PDF"

REASON_READABLE = "Document is readable"

REASON_UNREADABLE = "Document is not readable"

REASON_ENCRYPTED = "Document is encrypted"

REASON_NOT_ENCRYPTED = "Document is not encrypted"

REASON_CORRUPTED = "Document is corrupted"

REASON_NOT_CORRUPTED = "Document is not corrupted"

REASON_VALID_PAGE_COUNT = "Valid page count"

REASON_INVALID_PAGE_COUNT = "Invalid page count"

REASON_VALID_FILE_SIZE = "Valid file size"

REASON_INVALID_FILE_SIZE = "Invalid file size"

REASON_TEXT_AVAILABLE = "Required document text is available"

REASON_TEXT_MISSING = "Required document text is missing"


# ==========================================================
# CONTENT REASONS
# ==========================================================

REASON_ASSESSMENT_YEAR_FOUND = (
    "Assessment year present"
)

REASON_ASSESSMENT_YEAR_MISSING = (
    "Assessment year missing"
)

REASON_PAN_FOUND = (
    "PAN present"
)

REASON_PAN_MISSING = (
    "PAN missing"
)

REASON_TAXPAYER_FOUND = (
    "Taxpayer information present"
)

REASON_TAXPAYER_MISSING = (
    "Taxpayer information missing"
)

REASON_INCOME_FOUND = (
    "Income information present"
)

REASON_INCOME_MISSING = (
    "Income information missing"
)

REASON_TAX_COMPUTATION_FOUND = (
    "Tax computation present"
)

REASON_TAX_COMPUTATION_MISSING = (
    "Tax computation missing"
)

REASON_VERIFICATION_FOUND = (
    "Verification section present"
)

REASON_VERIFICATION_MISSING = (
    "Verification section missing"
)

REASON_ACKNOWLEDGEMENT_FOUND = (
    "Acknowledgement present"
)

REASON_ACKNOWLEDGEMENT_MISSING = (
    "Acknowledgement missing"
)


# ==========================================================
# CONSISTENCY REASONS
# ==========================================================

REASON_ASSESSMENT_YEAR_CONSISTENT = (
    "Assessment year is consistent"
)

REASON_ASSESSMENT_YEAR_INCONSISTENT = (
    "Assessment year is inconsistent"
)

REASON_PAN_CONSISTENT = (
    "PAN is consistent"
)

REASON_PAN_INCONSISTENT = (
    "PAN is inconsistent"
)

REASON_INCOME_CONSISTENT = (
    "Income information is consistent"
)

REASON_INCOME_INCONSISTENT = (
    "Income information is inconsistent"
)

REASON_TAX_CONSISTENT = (
    "Tax information is consistent"
)

REASON_TAX_INCONSISTENT = (
    "Tax information is inconsistent"
)

REASON_ACKNOWLEDGEMENT_CONSISTENT = (
    "Acknowledgement information is consistent"
)

REASON_ACKNOWLEDGEMENT_INCONSISTENT = (
    "Acknowledgement information is inconsistent"
)


# ==========================================================
# VALIDATION DECISION REASONS
# ==========================================================

REASON_VALIDATION_PASSED = (
    "ITR validation passed"
)

REASON_VALIDATION_FAILED = (
    "ITR validation failed"
)

REASON_VALIDATION_REVIEW = (
    "ITR requires manual review"
)

REASON_VALIDATION_THRESHOLD_NOT_MET = (
    "Validation confidence threshold not satisfied"
)