"""
Central configuration for the Document Verification pipeline.
Keep all tunables here so the rest of the codebase stays generic.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ---------------------------------------------------------------------------
# OCR ENGINE
# ---------------------------------------------------------------------------
# "paddleocr" -> best accuracy/speed tradeoff, recommended for production.
# "tesseract" -> lightweight fallback, good for dev/offline/quick testing.
# "auto"      -> try paddleocr, silently fall back to tesseract if not installed.
OCR_ENGINE = os.environ.get("DOC_VERIFY_OCR_ENGINE", "auto")

# Use the lightweight ("mobile") PaddleOCR models instead of the heavier
# server models. Salary slips / bank statements are clean printed text,
# so the accuracy loss is negligible and inference is much faster.
PADDLEOCR_USE_LIGHTWEIGHT_MODELS = True
PADDLEOCR_LANG = "en"
PADDLEOCR_USE_GPU = os.environ.get("DOC_VERIFY_USE_GPU", "false").lower() == "true"

# ---------------------------------------------------------------------------
# SPEED OPTIMIZATION
# ---------------------------------------------------------------------------
# If a PDF already has a text layer (digitally generated payslip, e-statement,
# etc.) skip OCR entirely and read the text layer directly. This is the
# single biggest speed win for salary slips, since most payroll systems
# export text-based PDFs, not scanned images.
PREFER_NATIVE_TEXT_LAYER = True

# Minimum number of characters a PDF page's native text layer must contain
# before we trust it and skip OCR. Below this, we assume it's a scanned
# image with no usable text layer.
MIN_NATIVE_TEXT_CHARS = 40

# ---------------------------------------------------------------------------
# DOCUMENT TYPE CLASSIFICATION
# ---------------------------------------------------------------------------
# Simple, fast keyword-based classifier. Generic and extensible: add a new
# doc type by adding a keyword list + an extractor in src/extractors/.
DOC_TYPE_KEYWORDS = {
    "salary_slip": [
        "payslip", "pay slip", "salary slip", "net pay", "gross pay",
        "earnings", "deductions", "basic pay", "hra", "pf", "employee id",
        "ctc", "take home",
    ],
    "bank_statement": [
        "statement of account", "bank statement", "ifsc", "account number",
        "opening balance", "closing balance", "withdrawal", "deposit",
        "transaction date",
    ],
    "id_proof": [
        "permanent account number", "income tax department", "aadhaar",
        "unique identification authority", "election commission", "passport",
        "driving licence", "date of birth",
    ],
}

# Confidence below this -> route to generic/fallback extractor instead of
# the specialized template-based one.
DOC_TYPE_MIN_CONFIDENCE = 0.15

# ---------------------------------------------------------------------------
# FORGERY / RISK SCORING
# ---------------------------------------------------------------------------
# Weights for combining individual forgery signals into one risk score (0-100,
# higher = more suspicious). Keep them summing to 1.0 for readability.
RISK_WEIGHTS = {
    "metadata": 0.25,
    "ela": 0.35,
    "consistency": 0.40,
}

RISK_THRESHOLDS = {
    "low": 30,      # 0-30   -> likely genuine
    "medium": 60,   # 31-60  -> needs manual review
    # >60            -> high risk / likely fake
}

