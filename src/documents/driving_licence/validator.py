"""
src/validator.py

Validation layer for extracted Indian Driving Licence fields.

IMPORTANT
---------
This validator checks:
    - required fields
    - formats
    - date consistency
    - DL structure
    - blood-group validity
    - COV validity
    - PIN validity
    - basic name/address quality

It does NOT prove that a Driving Licence is genuine.

"is_valid" means:
    "The extracted data passed our structural/business validation."

It does NOT mean:
    "Government-verified authentic Driving Licence."
"""

import re
from datetime import datetime, date


# ============================================================
# CONSTANTS
# ============================================================

VALID_BLOOD_GROUPS = {
    "A+",
    "A-",
    "B+",
    "B-",
    "AB+",
    "AB-",
    "O+",
    "O-",
}


VALID_COV_CODES = {
    "LMV",
    "MCWG",
    "MCWOG",
    "HMV",
    "HGMV",
    "MGV",
    "LMV-NT",
    "LMV-TR",
    "TRANS",
    "PSV",
}


# Indian DL format used by our current extractor:
#
# Example:
# MH0320220045390
#
# MH = State
# 03 = RTO
# 2022 = year
# 0045390 = serial
DL_PATTERN = re.compile(
    r"^[A-Z]{2}\d{13}$"
)


PIN_PATTERN = re.compile(
    r"^[1-9]\d{5}$"
)


DATE_PATTERN = re.compile(
    r"^\d{2}-\d{2}-\d{4}$"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean(value):
    """
    Convert a value safely to stripped string.
    """

    if value is None:
        return ""

    return str(value).strip()


def _parse_date(value):
    """
    Parse DD-MM-YYYY.

    Returns datetime.date or None.
    """

    value = _clean(value)

    if not value:
        return None

    if not DATE_PATTERN.fullmatch(value):
        return None

    try:

        return datetime.strptime(
            value,
            "%d-%m-%Y",
        ).date()

    except ValueError:
        return None


# ============================================================
# DL NUMBER VALIDATION
# ============================================================

def _validate_dl_number(
    value,
    issues,
):
    """
    Validate normalized DL number.
    """

    value = _clean(
        value
    ).upper()

    if not value:

        issues.append(
            "DL number not found."
        )

        return

    if not DL_PATTERN.fullmatch(
        value
    ):

        issues.append(
            "DL number has invalid format."
        )

        return

    # --------------------------------------------------------
    # YEAR SANITY CHECK
    # --------------------------------------------------------

    # Format:
    #
    # MH 03 2022 0045390
    # 01 23 4567
    #
    # year starts at index 4.

    try:

        issue_year = int(
            value[4:8]
        )

        current_year = date.today().year

        if (
            issue_year < 1950
            or issue_year > current_year
        ):

            issues.append(
                "DL number contains an invalid issue year."
            )

    except ValueError:

        issues.append(
            "DL number year could not be validated."
        )


# ============================================================
# NAME VALIDATION
# ============================================================

def _validate_name(
    value,
    issues,
):
    """
    Basic person-name validation.

    This does NOT try to determine whether the spelling is correct.
    """

    value = _clean(
        value
    )

    if not value:

        issues.append(
            "Name not found."
        )

        return

    if len(value) < 3:

        issues.append(
            "Name is too short."
        )

        return

    letters = sum(
        char.isalpha()
        for char in value
    )

    ratio = (
        letters
        / max(
            len(value),
            1,
        )
    )

    if ratio < 0.70:

        issues.append(
            "Name contains too many non-alphabetic characters."
        )

    words = value.split()

    if len(words) > 8:

        issues.append(
            "Name contains an unusually large number of words."
        )

    # Reject obvious field labels accidentally extracted as name.

    bad_values = {
        "NAME",
        "DOB",
        "DOI",
        "ADDRESS",
        "ADD",
        "COV",
        "LMV",
        "MCWG",
        "SIGNATURE",
    }

    if value.upper() in bad_values:

        issues.append(
            "Extracted name appears to be a document label rather than a person name."
        )


# ============================================================
# RELATIVE NAME
# ============================================================

def _validate_relative_name(
    value,
    warnings,
):
    """
    Relative name is useful but not mandatory.

    Missing relative name should not make the entire extraction invalid.
    """

    value = _clean(
        value
    )

    if not value:

        warnings.append(
            "Relative name not found."
        )

        return

    letters = sum(
        char.isalpha()
        for char in value
    )

    if letters < 2:

        warnings.append(
            "Relative name appears unreliable."
        )


# ============================================================
# DATE VALIDATION
# ============================================================

def _validate_dob(
    value,
    issues,
):
    """
    Validate DOB and return parsed date.
    """

    if not value:

        issues.append(
            "Date of birth not found."
        )

        return None

    dob = _parse_date(
        value
    )

    if dob is None:

        issues.append(
            "Date of birth has invalid format or impossible date."
        )

        return None

    today = date.today()

    if dob >= today:

        issues.append(
            "Date of birth cannot be today or in the future."
        )

        return dob

    age = (
        today.year
        - dob.year
        - (
            (today.month, today.day)
            < (dob.month, dob.day)
        )
    )

    if age < 16:

        issues.append(
            "Date of birth implies an unusually low age for a Driving Licence."
        )

    if age > 100:

        issues.append(
            "Date of birth implies an unrealistic age."
        )

    return dob


def _validate_issue_date(
    value,
    issues,
):
    """
    Validate Date of Issue.
    """

    if not value:

        issues.append(
            "Date of issue not found."
        )

        return None

    issue_date = _parse_date(
        value
    )

    if issue_date is None:

        issues.append(
            "Date of issue has invalid format or impossible date."
        )

        return None

    if issue_date > date.today():

        issues.append(
            "Date of issue cannot be in the future."
        )

    return issue_date


def _validate_validity_date(
    value,
    field_name,
    issues,
):
    """
    Validate one validity date.

    Expiry in the past is NOT automatically a parsing failure.
    It means the licence may be expired, so this is handled separately
    by warnings.
    """

    if not value:
        return None

    parsed = _parse_date(
        value
    )

    if parsed is None:

        issues.append(
            f"{field_name} has invalid format or impossible date."
        )

    return parsed


# ============================================================
# DATE RELATIONSHIP VALIDATION
# ============================================================

def _validate_date_relationships(
    dob,
    issue_date,
    validity_nt,
    validity_tr,
    issues,
    warnings,
):
    """
    Cross-check chronological relationships.
    """

    if (
        dob is not None
        and issue_date is not None
    ):

        if issue_date <= dob:

            issues.append(
                "Date of issue must be after date of birth."
            )

        else:

            age_at_issue = (
                issue_date.year
                - dob.year
                - (
                    (issue_date.month, issue_date.day)
                    < (dob.month, dob.day)
                )
            )

            if age_at_issue < 16:

                issues.append(
                    "Licence issue date implies holder was under 16."
                )

    # --------------------------------------------------------
    # VALIDITY > ISSUE DATE
    # --------------------------------------------------------

    if (
        issue_date is not None
        and validity_nt is not None
        and validity_nt <= issue_date
    ):

        issues.append(
            "Non-transport validity date must be after date of issue."
        )

    if (
        issue_date is not None
        and validity_tr is not None
        and validity_tr <= issue_date
    ):

        issues.append(
            "Transport validity date must be after date of issue."
        )

    # --------------------------------------------------------
    # EXPIRED LICENCE
    # --------------------------------------------------------

    today = date.today()

    validity_dates = [
        value
        for value in [
            validity_nt,
            validity_tr,
        ]
        if value is not None
    ]

    if (
        validity_dates
        and all(
            value < today
            for value in validity_dates
        )
    ):

        warnings.append(
            "Driving Licence validity date has expired."
        )


# ============================================================
# VALIDITY PRESENCE
# ============================================================

def _validate_validity_presence(
    nt,
    tr,
    issues,
):
    """
    At least one licence validity date should normally exist.
    """

    if not nt and not tr:

        issues.append(
            "Licence validity date not found."
        )


# ============================================================
# BLOOD GROUP
# ============================================================

def _validate_blood_group(
    value,
    warnings,
):
    """
    Blood group is optional.

    Missing blood group should not invalidate the whole extraction.
    """

    value = _clean(
        value
    ).upper()

    if not value:

        warnings.append(
            "Blood group not found."
        )

        return

    if value not in VALID_BLOOD_GROUPS:

        warnings.append(
            "Blood group has an invalid value."
        )


# ============================================================
# CLASS OF VEHICLE
# ============================================================

def _validate_cov(
    value,
    issues,
):
    """
    Validate multiple vehicle classes.

    Expected:

        ["LMV", "MCWG"]
    """

    if value is None:

        issues.append(
            "Class of vehicle not found."
        )

        return

    # Backward compatibility with old extractor.
    if isinstance(
        value,
        str,
    ):

        value = [
            value
        ]

    if not isinstance(
        value,
        (list, tuple, set),
    ):

        issues.append(
            "Class of vehicle has invalid data type."
        )

        return

    values = [
        _clean(item).upper()
        for item in value
        if _clean(item)
    ]

    if not values:

        issues.append(
            "Class of vehicle not found."
        )

        return

    invalid = [
        item
        for item in values
        if item not in VALID_COV_CODES
    ]

    if invalid:

        issues.append(
            "Invalid class of vehicle detected: "
            + ", ".join(
                invalid
            )
            + "."
        )


# ============================================================
# PIN CODE
# ============================================================

def _validate_pin(
    value,
    issues,
):
    """
    Validate Indian six-digit PIN code.
    """

    value = _clean(
        value
    )

    if not value:

        issues.append(
            "PIN code not found."
        )

        return

    if not PIN_PATTERN.fullmatch(
        value
    ):

        issues.append(
            "PIN code has invalid format."
        )


# ============================================================
# ADDRESS
# ============================================================

def _validate_address(
    value,
    issues,
):
    """
    Basic address quality validation.
    """

    value = _clean(
        value
    )

    if not value:

        issues.append(
            "Address not found."
        )

        return

    if len(value) < 10:

        issues.append(
            "Address appears incomplete."
        )

        return

    alphanumeric = sum(
        char.isalnum()
        for char in value
    )

    ratio = (
        alphanumeric
        / max(
            len(value),
            1,
        )
    )

    if ratio < 0.50:

        issues.append(
            "Address contains excessive OCR noise."
        )


# ============================================================
# OCR CONFIDENCE INFORMATION
# ============================================================

def _calculate_ocr_confidence(fields):
    """
    Placeholder for field-level confidence architecture.

    Current field_extractor returns values, not per-field OCR confidence,
    so we cannot honestly calculate a reliable field confidence here.

    Keep this explicit instead of inventing a fake confidence score.
    """

    return None


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_fields(
    fields,
    return_details=False,
):
    """
    Validate extracted Driving Licence fields.

    Backward-compatible default:

        is_valid, issues = validate_fields(fields)

    Optional detailed mode:

        result = validate_fields(
            fields,
            return_details=True
        )

    Detailed result:

        {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "validation_status": "passed"
        }
    """

    issues = []
    warnings = []

    # --------------------------------------------------------
    # DL NUMBER
    # --------------------------------------------------------

    _validate_dl_number(
        fields.get(
            "dl_number"
        ),
        issues,
    )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    _validate_name(
        fields.get(
            "name"
        ),
        issues,
    )

    # --------------------------------------------------------
    # RELATIVE NAME
    # --------------------------------------------------------

    _validate_relative_name(
        fields.get(
            "relative_name"
        ),
        warnings,
    )

    # --------------------------------------------------------
    # DOB
    # --------------------------------------------------------

    dob = _validate_dob(
        fields.get(
            "date_of_birth"
        ),
        issues,
    )

    # --------------------------------------------------------
    # ISSUE DATE
    # --------------------------------------------------------

    issue_date = _validate_issue_date(
        fields.get(
            "date_of_issue"
        ),
        issues,
    )

    # --------------------------------------------------------
    # VALIDITY
    # --------------------------------------------------------

    validity_nt_raw = fields.get(
        "validity_non_transport"
    )

    validity_tr_raw = fields.get(
        "validity_transport"
    )

    _validate_validity_presence(
        validity_nt_raw,
        validity_tr_raw,
        issues,
    )

    validity_nt = _validate_validity_date(
        validity_nt_raw,
        "Non-transport validity date",
        issues,
    )

    validity_tr = _validate_validity_date(
        validity_tr_raw,
        "Transport validity date",
        issues,
    )

    # --------------------------------------------------------
    # DATE CROSS-CHECK
    # --------------------------------------------------------

    _validate_date_relationships(
        dob,
        issue_date,
        validity_nt,
        validity_tr,
        issues,
        warnings,
    )

    # --------------------------------------------------------
    # BLOOD GROUP
    # --------------------------------------------------------

    _validate_blood_group(
        fields.get(
            "blood_group"
        ),
        warnings,
    )

    # --------------------------------------------------------
    # COV
    # --------------------------------------------------------

    _validate_cov(
        fields.get(
            "class_of_vehicle"
        ),
        issues,
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    _validate_address(
        fields.get(
            "address"
        ),
        issues,
    )

    # --------------------------------------------------------
    # PIN
    # --------------------------------------------------------

    _validate_pin(
        fields.get(
            "pin_code"
        ),
        issues,
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE MESSAGES
    # --------------------------------------------------------

    issues = list(
        dict.fromkeys(
            issues
        )
    )

    warnings = list(
        dict.fromkeys(
            warnings
        )
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    is_valid = (
        len(issues) == 0
    )

    validation_status = (
        "passed"
        if is_valid
        else "failed"
    )

    # --------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # --------------------------------------------------------

    if not return_details:

        return (
            is_valid,
            issues,
        )

    # --------------------------------------------------------
    # DETAILED RESPONSE
    # --------------------------------------------------------

    return {
        "is_valid":
            is_valid,

        "validation_status":
            validation_status,

        "issues":
            issues,

        "warnings":
            warnings,

        "ocr_confidence":
            _calculate_ocr_confidence(
                fields
            ),
    }
