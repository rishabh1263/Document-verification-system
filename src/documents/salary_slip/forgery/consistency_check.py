"""
Internal consistency forgery signal.

Doesn't look at pixels/metadata at all â€” looks at whether the *extracted
data itself* makes sense. This catches a common, cheap type of fraud:
someone edits the "net pay" number in an image editor but doesn't bother
(or doesn't know how) to keep gross - deductions = net consistent.

Generic by design: each doc_type gets its own small set of sanity checks,
and unknown doc types just get skipped gracefully (score 0, not penalized).
"""

from typing import Any, Dict

TOLERANCE = 1.0  # rupees, allow tiny rounding differences


def check(doc_type: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    if doc_type == "salary_slip":
        return _check_salary_slip(fields)
    if doc_type == "bank_statement":
        return _check_bank_statement(fields)
    return {"score": 0, "reasons": ["No consistency rules defined for this document type."], "checked": False}


def _check_salary_slip(fields: Dict[str, Any]) -> Dict[str, Any]:
    numeric = fields.get("_numeric", {})

    reasons = []
    score = 0
    checked = False

    # ============================================================
    # 1. Extraction coverage / required fields
    # ============================================================

    critical_fields = [
        "employee_name",
        "employee_id",
        "pay_period",
        "gross_pay",
        "net_pay",
    ]

    missing_fields = [
        field
        for field in critical_fields
        if not fields.get(field)
    ]

    missing_count = len(missing_fields)

    if missing_count:
        checked = True

        # Missing information means verification confidence is lower.
        # It does NOT automatically mean fraud.
        if missing_count == 1:
            score += 5
        elif missing_count == 2:
            score += 15
        elif missing_count == 3:
            score += 25
        else:
            score += 40

        reasons.append(
            "Could not extract critical field(s): "
            + ", ".join(missing_fields)
            + ". Verification confidence is reduced."
        )

    # ============================================================
    # 2. Gross - deductions = net
    # ============================================================

    gross = numeric.get("gross_pay")
    deductions = numeric.get("total_deductions")
    net = numeric.get("net_pay")

    if (
        gross is not None
        and deductions is not None
        and net is not None
    ):
        checked = True

        expected_net = gross - deductions
        diff = abs(expected_net - net)

        if diff > TOLERANCE:
            score += 70

            reasons.append(
                f"Gross ({gross}) - Deductions ({deductions}) "
                f"= {expected_net:.2f}, but stated Net Pay is "
                f"{net} â€” mismatch of {diff:.2f}."
            )

        else:
            reasons.append(
                "Gross - Deductions matches stated Net Pay."
            )

    else:
        missing_numeric = []

        if gross is None:
            missing_numeric.append("gross_pay")

        if deductions is None:
            missing_numeric.append("total_deductions")

        if net is None:
            missing_numeric.append("net_pay")

        if missing_numeric:
            checked = True

            reasons.append(
                "Salary arithmetic could not be fully verified because "
                "these numeric field(s) were unavailable: "
                + ", ".join(missing_numeric)
                + "."
            )

    # ============================================================
    # 3. Basic <= Gross
    # ============================================================

    basic = numeric.get("basic_pay")

    if gross is not None and basic is not None:
        checked = True

        if basic > gross:
            score += 50

            reasons.append(
                f"Basic pay ({basic}) exceeds gross pay ({gross}) "
                "â€” logically inconsistent."
            )

        else:
            reasons.append(
                "Basic pay does not exceed gross pay."
            )

    # ============================================================
    # 4. Negative salary values
    # ============================================================

    for field_name, value in [
        ("gross_pay", gross),
        ("total_deductions", deductions),
        ("net_pay", net),
        ("basic_pay", basic),
    ]:

        if value is not None:
            checked = True

            if value < 0:
                score += 40

                reasons.append(
                    f"{field_name} has a negative value ({value}), "
                    "which is unusual."
                )

    # ============================================================
    # 5. No checks possible
    # ============================================================

    if not reasons:
        reasons.append(
            "Not enough extracted information to run salary-slip "
            "consistency checks."
        )

    return {
        "score": min(100, score),
        "reasons": reasons,
        "checked": checked,
    }


def _check_bank_statement(fields: Dict[str, Any]) -> Dict[str, Any]:
    reasons = []
    score = 0
    checked = False

    ifsc = fields.get("ifsc_code")
    if ifsc and (len(ifsc) != 11 or ifsc[4] != "0"):
        score += 40
        reasons.append(f"IFSC code '{ifsc}' does not match the standard format (4 letters + 0 + 6 chars).")
        checked = True

    if not reasons:
        reasons.append("No consistency issues detected in extracted fields.")

    return {"score": min(100, score), "reasons": reasons, "checked": checked}

