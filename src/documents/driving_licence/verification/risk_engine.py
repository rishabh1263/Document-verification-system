"""
src/verification/risk_engine.py

Combines verification checks and produces a risk assessment
for an extracted Driving Licence.

IMPORTANT:
This is heuristic verification.
It does not prove that a licence is genuine.
"""


from .dl_number_checks import check_dl_number
from .date_checks import check_dates


# ============================================================
# CRITICAL FIELDS
# ============================================================

CRITICAL_FIELDS = (
    "dl_number",
    "name",
    "date_of_birth",
    "date_of_issue",
)


# ============================================================
# CRITICAL FIELD CHECKS
# ============================================================

def _check_critical_fields(fields):
    """
    Check whether important Driving Licence fields are present.
    """

    checks = []

    for field_name in CRITICAL_FIELDS:

        value = fields.get(field_name)

        if value:

            checks.append({
                "check": f"{field_name}_present",
                "status": "passed",
                "message": f"{field_name} is present.",
                "risk_points": 0,
            })

        else:

            checks.append({
                "check": f"{field_name}_present",
                "status": "failed",
                "message": f"{field_name} is missing.",
                "risk_points": 20,
            })

    return checks


# ============================================================
# RISK SCORE
# ============================================================

def _calculate_risk_score(checks):
    """
    Add risk points from all checks.

    Maximum score = 100.
    """

    score = sum(
        check.get("risk_points", 0)
        for check in checks
    )

    return min(score, 100)


# ============================================================
# RISK DECISION
# ============================================================

def _get_decision(risk_score):
    """
    Convert risk score into document risk level.

    0 - 14
        LOW_RISK

    15 - 49
        REVIEW

    50 - 100
        SUSPICIOUS
    """

    if risk_score >= 50:
        return "SUSPICIOUS"

    if risk_score >= 15:
        return "REVIEW"

    return "LOW_RISK"


# ============================================================
# MAIN VERIFICATION FUNCTION
# ============================================================

def verify_document(fields):
    """
    Main Driving Licence verification function.

    Current Phase-1 verification:

        1. Critical field presence
        2. DL number structural validation
        3. State/UT code validation
        4. Date consistency
        5. Age-at-issue plausibility
        6. Licence validity chronology

    Returns a risk assessment.

    LOW_RISK does not mean officially verified genuine.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(fields, dict):

        raise TypeError(
            "fields must be a dictionary."
        )

    all_checks = []

    # ========================================================
    # 1. CRITICAL FIELDS
    # ========================================================

    critical_checks = _check_critical_fields(
        fields
    )

    all_checks.extend(
        critical_checks
    )

    # ========================================================
    # 2. DL NUMBER CHECKS
    # ========================================================

    dl_result = check_dl_number(
        fields.get("dl_number")
    )

    dl_checks = dl_result.get(
        "checks",
        [],
    )

    all_checks.extend(
        dl_checks
    )

    # ========================================================
    # 3. DATE CHECKS
    # ========================================================

    date_results = check_dates(
        fields
    )

    all_checks.extend(
        date_results
    )

    # ========================================================
    # 4. CALCULATE RISK SCORE
    # ========================================================

    risk_score = _calculate_risk_score(
        all_checks
    )

    # ========================================================
    # 5. GET DECISION
    # ========================================================

    decision = _get_decision(
        risk_score
    )

    # ========================================================
    # 6. GROUP CHECK RESULTS
    # ========================================================

    passed_checks = [
        check
        for check in all_checks
        if check.get("status") == "passed"
    ]

    warning_checks = [
        check
        for check in all_checks
        if check.get("status") == "warning"
    ]

    failed_checks = [
        check
        for check in all_checks
        if check.get("status") == "failed"
    ]

    # ========================================================
    # 7. BUILD REASONS
    # ========================================================

    reasons = []

    for check in failed_checks + warning_checks:

        message = check.get("message")

        if message:
            reasons.append(message)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "decision": decision,

        "risk_score": risk_score,

        # We currently do not query an official government
        # source-of-truth.
        "authoritative_verification": False,

        "summary": {

            "total_checks": len(
                all_checks
            ),

            "passed": len(
                passed_checks
            ),

            "warnings": len(
                warning_checks
            ),

            "failed": len(
                failed_checks
            ),
        },

        "checks": all_checks,

        "reasons": reasons,
    }
