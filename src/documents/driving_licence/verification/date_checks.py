"""
src/verification/date_checks.py

Date-based verification checks for Driving Licences.

Checks:
    - DOB exists and is valid
    - DOB is not in the future
    - Issue date exists and is valid
    - Issue date is not in the future
    - Issue date occurs after DOB
    - Holder age at issue is plausible
    - Validity date occurs after issue date

IMPORTANT:
These are consistency/fraud-risk checks.
They do NOT prove that a document is genuine.
"""

from datetime import date, datetime


# ============================================================
# SUPPORTED DATE FORMATS
# ============================================================

DATE_FORMATS = (
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
)


# ============================================================
# DATE PARSER
# ============================================================

def parse_date(value):
    """
    Convert extracted date text into a Python date.

    Examples:

        12-06-2002
        12/06/2002
        12.06.2002

    Returns None when parsing fails.
    """

    if not value:
        return None

    value = str(value).strip()

    for fmt in DATE_FORMATS:

        try:
            return datetime.strptime(
                value,
                fmt,
            ).date()

        except ValueError:
            continue

    return None


# ============================================================
# AGE CALCULATION
# ============================================================

def calculate_age(dob, on_date):
    """
    Calculate age on a particular date.
    """

    years = (
        on_date.year
        - dob.year
    )

    if (
        on_date.month,
        on_date.day,
    ) < (
        dob.month,
        dob.day,
    ):
        years -= 1

    return years


# ============================================================
# MAIN DATE CHECKS
# ============================================================

def check_dates(fields):
    """
    Run date consistency checks.

    Expected fields:

        date_of_birth
        date_of_issue
        validity_non_transport
        validity_transport

    Returns a list of verification checks.
    """

    checks = []

    today = date.today()

    # ========================================================
    # PARSE DATES
    # ========================================================

    dob = parse_date(
        fields.get(
            "date_of_birth"
        )
    )

    issue_date = parse_date(
        fields.get(
            "date_of_issue"
        )
    )

    validity_nt = parse_date(
        fields.get(
            "validity_non_transport"
        )
    )

    validity_tr = parse_date(
        fields.get(
            "validity_transport"
        )
    )

    # ========================================================
    # CHECK 1: DOB
    # ========================================================

    if dob is None:

        checks.append({
            "check": "dob_present",
            "status": "failed",
            "message": (
                "Date of birth is missing "
                "or has an invalid format."
            ),
            "risk_points": 20,
        })

    elif dob > today:

        checks.append({
            "check": "dob_not_future",
            "status": "failed",
            "message": (
                "Date of birth is in the future."
            ),
            "risk_points": 40,
        })

    else:

        checks.append({
            "check": "dob_valid",
            "status": "passed",
            "message": (
                "Date of birth is structurally valid."
            ),
            "risk_points": 0,
        })

    # ========================================================
    # CHECK 2: ISSUE DATE
    # ========================================================

    if issue_date is None:

        checks.append({
            "check": "issue_date_present",
            "status": "failed",
            "message": (
                "Date of issue is missing "
                "or has an invalid format."
            ),
            "risk_points": 20,
        })

    elif issue_date > today:

        checks.append({
            "check": "issue_date_not_future",
            "status": "failed",
            "message": (
                "Date of issue is in the future."
            ),
            "risk_points": 40,
        })

    else:

        checks.append({
            "check": "issue_date_valid",
            "status": "passed",
            "message": (
                "Date of issue is structurally valid."
            ),
            "risk_points": 0,
        })

    # ========================================================
    # CHECK 3: DOB BEFORE ISSUE DATE
    # ========================================================

    if (
        dob is not None
        and issue_date is not None
    ):

        if issue_date <= dob:

            checks.append({
                "check": "dob_before_issue",
                "status": "failed",
                "message": (
                    "Date of issue is not after "
                    "the holder's date of birth."
                ),
                "risk_points": 40,
            })

        else:

            checks.append({
                "check": "dob_before_issue",
                "status": "passed",
                "message": (
                    "Date of birth occurs before "
                    "the issue date."
                ),
                "risk_points": 0,
            })

    # ========================================================
    # CHECK 4: AGE AT ISSUE
    # ========================================================

    if (
        dob is not None
        and issue_date is not None
        and issue_date > dob
    ):

        age_at_issue = calculate_age(
            dob,
            issue_date,
        )

        # Do not treat this alone as proof of fraud.
        # It is a review signal.

        if age_at_issue < 16:

            checks.append({
                "check": "age_at_issue",
                "status": "warning",
                "message": (
                    "Holder age at issue appears "
                    f"unusually low: {age_at_issue}."
                ),
                "risk_points": 15,
            })

        elif age_at_issue > 100:

            checks.append({
                "check": "age_at_issue",
                "status": "warning",
                "message": (
                    "Holder age at issue appears "
                    f"unusually high: {age_at_issue}."
                ),
                "risk_points": 15,
            })

        else:

            checks.append({
                "check": "age_at_issue",
                "status": "passed",
                "message": (
                    f"Holder age at issue: "
                    f"{age_at_issue}."
                ),
                "risk_points": 0,
            })

    # ========================================================
    # CHECK 5: NON-TRANSPORT VALIDITY
    # ========================================================

    if validity_nt is not None:

        if (
            issue_date is not None
            and validity_nt <= issue_date
        ):

            checks.append({
                "check": "non_transport_validity",
                "status": "failed",
                "message": (
                    "Non-transport validity date "
                    "is not after the issue date."
                ),
                "risk_points": 30,
            })

        else:

            checks.append({
                "check": "non_transport_validity",
                "status": "passed",
                "message": (
                    "Non-transport validity chronology "
                    "is plausible."
                ),
                "risk_points": 0,
            })

    # ========================================================
    # CHECK 6: TRANSPORT VALIDITY
    # ========================================================

    if validity_tr is not None:

        if (
            issue_date is not None
            and validity_tr <= issue_date
        ):

            checks.append({
                "check": "transport_validity",
                "status": "failed",
                "message": (
                    "Transport validity date "
                    "is not after the issue date."
                ),
                "risk_points": 30,
            })

        else:

            checks.append({
                "check": "transport_validity",
                "status": "passed",
                "message": (
                    "Transport validity chronology "
                    "is plausible."
                ),
                "risk_points": 0,
            })

    # ========================================================
    # CHECK 7: AT LEAST ONE VALIDITY DATE
    # ========================================================

    if (
        validity_nt is None
        and validity_tr is None
    ):

        checks.append({
            "check": "validity_present",
            "status": "warning",
            "message": (
                "No licence validity date "
                "was extracted."
            ),
            "risk_points": 10,
        })

    # ========================================================
    # RETURN
    # ========================================================

    return checks
