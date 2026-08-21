"""
Passport-specific constants.

Keep generic document constants inside src.common.
Only passport-specific rules belong here.
"""

# ---------------------------------------------------------------------------
# SUPPORTED FILE FORMATS
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

SUPPORTED_DOCUMENT_FORMATS = {
    ".pdf",
}

SUPPORTED_FORMATS = (
    SUPPORTED_IMAGE_FORMATS
    |
    SUPPORTED_DOCUMENT_FORMATS
)


# ---------------------------------------------------------------------------
# PASSPORT DOCUMENT
# ---------------------------------------------------------------------------

PASSPORT_DOCUMENT_TYPE = "PASSPORT"

PASSPORT_ISSUING_COUNTRY = "IND"


# ---------------------------------------------------------------------------
# ICAO TD3 MRZ
# ---------------------------------------------------------------------------

MRZ_LINE_COUNT = 2

MRZ_LINE_LENGTH = 44

MRZ_TOTAL_LENGTH = (
    MRZ_LINE_COUNT
    *
    MRZ_LINE_LENGTH
)


# ---------------------------------------------------------------------------
# PASSPORT FIELDS
# ---------------------------------------------------------------------------

PASSPORT_REQUIRED_FIELDS = (
    "passport_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "date_of_expiry",
)


# ---------------------------------------------------------------------------
# LOS DECISION STATES
# ---------------------------------------------------------------------------

DOCUMENT_VERIFIED = "DOCUMENT_VERIFIED"

DOCUMENT_REVIEW = "DOCUMENT_REVIEW"

DOCUMENT_REJECTED = "DOCUMENT_REJECTED"


# ---------------------------------------------------------------------------
# COMMON VALIDATION STATES
# ---------------------------------------------------------------------------

# These should remain compatible with the common validation layer.

COMMON_DOCUMENT_PASS = "DOCUMENT_PASS"

COMMON_DOCUMENT_SUSPICIOUS = "DOCUMENT_SUSPICIOUS"

COMMON_MANUAL_REVIEW = "MANUAL_REVIEW"

COMMON_DOCUMENT_REJECT = "DOCUMENT_REJECT"


# ---------------------------------------------------------------------------
# PASSPORT-SPECIFIC VALIDATION WEIGHTS
# ---------------------------------------------------------------------------

# These are NOT credit-decision weights.
#
# They are only used to combine passport document evidence before passing
# the result to the common document-validation layer.

MRZ_STRUCTURE_WEIGHT = 10

PASSPORT_NUMBER_CHECKSUM_WEIGHT = 20

DOB_CHECKSUM_WEIGHT = 15

EXPIRY_CHECKSUM_WEIGHT = 15

PERSONAL_NUMBER_CHECKSUM_WEIGHT = 5

COMPOSITE_CHECKSUM_WEIGHT = 15

VISIBLE_MRZ_CONSISTENCY_WEIGHT = 20


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

MRZ_ALLOWED_CHARACTERS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "<"
)


# ---------------------------------------------------------------------------
# QUALITY
# ---------------------------------------------------------------------------

MIN_PASSPORT_IMAGE_WIDTH = 500

MIN_PASSPORT_IMAGE_HEIGHT = 500


# ---------------------------------------------------------------------------
# DATE
# ---------------------------------------------------------------------------

MRZ_DATE_LENGTH = 6


# ---------------------------------------------------------------------------
# RCU
# ---------------------------------------------------------------------------

# RCU is a separate deeper-risk stage.
#
# Do not treat these constants as proof of fraud.

RCU_TRIGGER_TAMPER = "TAMPER_SIGNAL"

RCU_TRIGGER_LOW_OCR = "LOW_OCR_CONFIDENCE"

RCU_TRIGGER_FIELD_MISMATCH = "FIELD_MISMATCH"

RCU_TRIGGER_MRZ_INCONSISTENCY = "MRZ_INCONSISTENCY"

RCU_TRIGGER_SUSPICIOUS_DOCUMENT = (
    "SUSPICIOUS_DOCUMENT"
)