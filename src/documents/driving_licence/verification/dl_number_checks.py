"""
src/verification/dl_number_checks.py

Phase-1 structural checks for Indian Driving Licence numbers.

IMPORTANT:
These checks determine whether the extracted DL number looks
structurally plausible.

They DO NOT prove that the licence actually exists or that the
document is genuine.
"""

import re


# ============================================================
# INDIAN STATE / UT CODES
# ============================================================

VALID_STATE_CODES = {
    "AN",
    "AP",
    "AR",
    "AS",
    "BR",
    "CG",
    "CH",
    "DD",
    "DL",
    "DN",
    "GA",
    "GJ",
    "HP",
    "HR",
    "JH",
    "JK",
    "KA",
    "KL",
    "LA",
    "LD",
    "MH",
    "ML",
    "MN",
    "MP",
    "MZ",
    "NL",
    "OD",
    "OR",
    "PB",
    "PY",
    "RJ",
    "SK",
    "TN",
    "TR",
    "TS",
    "UK",
    "UP",
    "WB",
}


# ============================================================
# NORMALIZE DL NUMBER
# ============================================================

def normalize_dl_number(dl_number):
    """
    Normalize DL number before validation.

    Example:

        MH03 20220045390
            â†“
        MH0320220045390

        MH03-20220045390
            â†“
        MH0320220045390
    """

    if not dl_number:
        return None

    normalized = re.sub(
        r"[^A-Z0-9]",
        "",
        str(dl_number).upper(),
    )

    return normalized or None


# ============================================================
# MAIN DL NUMBER CHECK
# ============================================================

def check_dl_number(dl_number):
    """
    Run structural checks on an extracted DL number.

    Returns:

    {
        "passed": True,
        "normalized": "MH0320220045390",
        "state_code": "MH",
        "checks": [...]
    }

    NOTE:
    Passing these checks does NOT mean the licence is genuine.
    """

    checks = []

    normalized = normalize_dl_number(
        dl_number
    )

    # ========================================================
    # CHECK 1: DL NUMBER PRESENT
    # ========================================================

    if not normalized:

        checks.append({
            "check": "dl_number_present",
            "status": "failed",
            "message": "Driving licence number is missing.",
            "risk_points": 30,
        })

        return {
            "passed": False,
            "normalized": None,
            "state_code": None,
            "checks": checks,
        }

    checks.append({
        "check": "dl_number_present",
        "status": "passed",
        "message": "Driving licence number was extracted.",
        "risk_points": 0,
    })

    # ========================================================
    # CHECK 2: STATE / UT CODE
    # ========================================================

    state_code = normalized[:2]

    if state_code in VALID_STATE_CODES:

        checks.append({
            "check": "state_code",
            "status": "passed",
            "message": (
                f"Recognized state/UT code: "
                f"{state_code}."
            ),
            "risk_points": 0,
        })

    else:

        checks.append({
            "check": "state_code",
            "status": "failed",
            "message": (
                f"Unrecognized state/UT code: "
                f"{state_code}."
            ),
            "risk_points": 20,
        })

    # ========================================================
    # CHECK 3: GENERAL STRUCTURE
    # ========================================================
    #
    # We intentionally avoid one extremely strict regex here.
    # Indian DL formats can differ across states and older/newer
    # licence formats.
    #
    # Basic expectations:
    #
    # - reasonable total length
    # - starts with two letters
    # - contains numeric characters
    # ========================================================

    structure_valid = (
        10 <= len(normalized) <= 20
        and normalized[:2].isalpha()
        and any(
            char.isdigit()
            for char in normalized[2:]
        )
    )

    if structure_valid:

        checks.append({
            "check": "dl_number_structure",
            "status": "passed",
            "message": (
                "Driving licence number has a plausible "
                "alphanumeric structure."
            ),
            "risk_points": 0,
        })

    else:

        checks.append({
            "check": "dl_number_structure",
            "status": "failed",
            "message": (
                "Driving licence number structure is unusual."
            ),
            "risk_points": 20,
        })

    # ========================================================
    # CHECK 4: NUMERIC PORTION
    # ========================================================

    numeric_part = normalized[2:]

    digit_count = sum(
        char.isdigit()
        for char in numeric_part
    )

    if digit_count >= 8:

        checks.append({
            "check": "numeric_content",
            "status": "passed",
            "message": (
                "Driving licence number contains a plausible "
                "numeric portion."
            ),
            "risk_points": 0,
        })

    else:

        checks.append({
            "check": "numeric_content",
            "status": "warning",
            "message": (
                "Driving licence number contains fewer numeric "
                "characters than expected."
            ),
            "risk_points": 10,
        })

    # ========================================================
    # FINAL RESULT
    # ========================================================

    failed_checks = [
        check
        for check in checks
        if check["status"] == "failed"
    ]

    return {
        "passed": len(failed_checks) == 0,
        "normalized": normalized,
        "state_code": state_code,
        "checks": checks,
    }
