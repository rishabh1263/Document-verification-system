"""
config.py
Central place for all settings, folder paths, and the regex patterns
used to pull fields out of an Indian Driving Licence.

If your licence format is different (different state / different card
version), this is the ONLY file you should need to tweak.
"""

import os

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------
OCR_LANGUAGES = ["en"]        # add more EasyOCR language codes if needed, e.g. ["en", "hi"]
OCR_GPU = False                # set True only if you have a working CUDA GPU + torch-gpu
MIN_CONFIDENCE = 0.35          # OCR lines below this confidence are ignored during field search

# ---------------------------------------------------------------------------
# PDF -> image conversion
# ---------------------------------------------------------------------------
PDF_DPI = 200       # higher DPI = sharper text = better OCR, but slower

# ---------------------------------------------------------------------------
# Known Indian state / UT RTO codes (used to sanity-check a detected DL number)
# Extend this list if you need states not listed here.
# ---------------------------------------------------------------------------
RTO_STATE_CODES = [
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB",
    "AN", "CH", "DN", "DD", "DL", "LD", "PY", "LA",
]

# ---------------------------------------------------------------------------
# Regex patterns
# Indian DL numbers commonly look like: MH14 20110012345  or  MH-14-20110012345
# i.e. 2 letters (state) + 2 digits (RTO code) + 11 digits (serial), with
# optional spaces / hyphens in between.
# ---------------------------------------------------------------------------
PATTERNS = {
    "dl_number": r"\b([A-Z]{2}[\s-]?\d{1,2}[\s-]?\d{10,11})\b",

    # dd-mm-yyyy, dd/mm/yyyy, dd.mm.yyyy  (2 or 4 digit year)
    "date": r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",

    "blood_group": r"\b(AB|A|B|O)[\s-]?(\+VE|-VE|POSITIVE|NEGATIVE|\+|-)",

    "pin_code": r"\b(\d{6})\b",
}

# Keywords used to anchor each field inside the OCR text (case-insensitive).
# The extractor looks for these words and then reads the nearest matching
# pattern (date, DL number, etc.) around them.
FIELD_KEYWORDS = {
    "name": ["name"],
    "relative_name": ["son of", "daughter of", "wife of", "s/o", "d/o", "w/o"],
    "dob": ["dob", "date of birth", "birth"],
    "blood_group": ["blood group", "bg"],
    "issue_date": ["date of issue", "issue"],
    "validity_nt": ["valid till", "validity", "nt", "non-transport"],
    "validity_tr": ["transport", "tr"],
    "cov": ["cov", "class of vehicle", "authorisation", "authorization"],
    "address": ["address"],
    "pin": ["pin"],
}
