"""
PAN final decision layer.

Purpose
-------
Separates:

1. Local document authenticity
2. External/database verification
3. Final verification decision

IMPORTANT
---------
A PAN image looking structurally correct does NOT prove that the PAN is
genuine. Without an authoritative external/database check, the result can
only be DOCUMENT_PASS / DOCUMENT_FAIL / MANUAL_REVIEW.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

LOCAL_PASS_SCORE = 75
LOCAL_REVIEW_SCORE = 55

EXTERNAL_VERIFICATION_REQUIRED = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _normalise_name(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).upper()

    value = re.sub(
        r"[^A-Z ]",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _names_match(
    extracted: Any,
    authoritative: Any,
) -> bool:
    """
    Conservative name comparison.

    We do not require exact OCR spelling because OCR may introduce
    insignificant differences.

    Example:
        "Amit Akhilesh Pandey"
        "AMIT AKHILESH PANDEY"

    should match.

    But completely different names should not.
    """
    a = _normalise_name(extracted)
    b = _normalise_name(authoritative)

    if not a or not b:
        return False

    if a == b:
        return True

    a_words = set(a.split())
    b_words = set(b.split())

    if not a_words or not b_words:
        return False

    intersection = a_words.intersection(b_words)

    # Strong enough for OCR variations but not overly permissive.
    smaller_count = min(
        len(a_words),
        len(b_words),
    )

    if smaller_count == 0:
        return False

    overlap = len(intersection) / smaller_count

    return overlap >= 0.67


def _dob_normalise(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    match = re.search(
        r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})",
        value,
    )

    if not match:
        return ""

    day, month, year = match.groups()

    return (
        f"{int(day):02d}/"
        f"{int(month):02d}/"
        f"{year}"
    )


# ---------------------------------------------------------------------------
# Local authenticity
# ---------------------------------------------------------------------------

def evaluate_local_authenticity(
    validation: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate whether the uploaded document looks like a structurally valid
    PAN document.

    This function does NOT claim that the PAN belongs to a real person.
    """

    checks = validation.get(
        "checks",
        {},
    )

    score = _safe_float(
        validation.get("score"),
        0.0,
    )

    reasons: list[str] = []

    # ---------------------------------------------------------------
    # Required structural checks
    # ---------------------------------------------------------------

    if not _safe_bool(
        checks.get("pan_format")
    ):
        reasons.append(
            "PAN format is invalid."
        )

    if not _safe_bool(
        checks.get("pan_structure")
    ):
        reasons.append(
            "PAN structure is invalid."
        )

    if not _safe_bool(
        checks.get("required_fields")
    ):
        reasons.append(
            "Required PAN fields could not be reliably extracted."
        )

    if not _safe_bool(
        checks.get("name_quality")
    ):
        reasons.append(
            "Extracted name quality is insufficient."
        )

    if not _safe_bool(
        checks.get("dob_format")
    ):
        reasons.append(
            "Date of birth format is invalid."
        )

    if not _safe_bool(
        checks.get("is_pan_document")
    ):
        reasons.append(
            "Document does not contain enough PAN-card markers."
        )

    # ---------------------------------------------------------------
    # OCR quality
    # ---------------------------------------------------------------

    average_confidence = _safe_float(
        checks.get("average_ocr_confidence")
    )

    if average_confidence < 0.50:
        reasons.append(
            "OCR confidence is too low."
        )

    # ---------------------------------------------------------------
    # Score decision
    # ---------------------------------------------------------------

    if reasons:
        if score >= LOCAL_REVIEW_SCORE:
            decision = "MANUAL_REVIEW"
        else:
            decision = "DOCUMENT_FAIL"

    elif score >= LOCAL_PASS_SCORE:
        decision = "DOCUMENT_PASS"

    elif score >= LOCAL_REVIEW_SCORE:
        decision = "MANUAL_REVIEW"

    else:
        decision = "DOCUMENT_FAIL"

    return {
        "decision": decision,
        "score": round(score),
        "reasons": reasons,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# External / database verification
# ---------------------------------------------------------------------------

def evaluate_external_verification(
    extracted: dict[str, Any],
    database_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Evaluate authoritative PAN verification.

    database_result must come from a real authoritative source.

    Expected shape:

    {
        "status": "VERIFIED",
        "pan_exists": True,
        "active": True,
        "name_match": True,
        "dob_match": True
    }

    If verification has not been performed, this function deliberately
    refuses to call the PAN genuine.
    """

    if not database_result:
        return {
            "status": "NOT_PERFORMED",
            "verified": False,
            "reasons": [
                "Authoritative PAN verification was not performed."
            ],
        }

    status = str(
        database_result.get(
            "status",
            "NOT_PERFORMED",
        )
    ).upper()

    if status in {
        "NOT_PERFORMED",
        "UNAVAILABLE",
        "NOT_AVAILABLE",
    }:
        return {
            "status": status,
            "verified": False,
            "reasons": [
                "Authoritative PAN verification was not completed."
            ],
        }

    pan_exists = database_result.get(
        "pan_exists"
    )

    active = database_result.get(
        "active"
    )

    name_match = database_result.get(
        "name_match"
    )

    dob_match = database_result.get(
        "dob_match"
    )

    reasons: list[str] = []

    if pan_exists is False:
        reasons.append(
            "PAN does not exist in the authoritative source."
        )

    if active is False:
        reasons.append(
            "PAN is not active."
        )

    if name_match is False:
        reasons.append(
            "Name does not match the authoritative record."
        )

    if dob_match is False:
        reasons.append(
            "Date of birth does not match the authoritative record."
        )

    if reasons:
        return {
            "status": "FAILED",
            "verified": False,
            "reasons": reasons,
        }

    # Do not accept partial external responses as verification.
    if not all(
        value is True
        for value in (
            pan_exists,
            active,
            name_match,
            dob_match,
        )
    ):
        return {
            "status": "INCOMPLETE",
            "verified": False,
            "reasons": [
                "Authoritative verification returned incomplete fields."
            ],
        }

    return {
        "status": "VERIFIED",
        "verified": True,
        "reasons": [],
    }


# ---------------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------------

def make_final_decision(
    *,
    local_validation: dict[str, Any],
    extracted: dict[str, Any],
    database_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Produce the final PAN verification decision.

    Decision rules
    --------------

    Local document failure
        -> DOCUMENT_FAIL

    Local document uncertain
        -> MANUAL_REVIEW

    Local document passes but external verification was not performed
        -> DOCUMENT_PASS

    Local document passes + authoritative verification passes
        -> VERIFIED

    Local document passes + authoritative verification fails
        -> DOCUMENT_FAIL

    This prevents a structurally convincing fake PAN from being reported
    as genuinely verified.
    """

    local = evaluate_local_authenticity(
        local_validation
    )

    # ---------------------------------------------------------------
    # Hard local failure
    # ---------------------------------------------------------------

    if local["decision"] == "DOCUMENT_FAIL":
        return {
            "verified": False,
            "decision": "DOCUMENT_FAIL",
            "verification_stage": "LOCAL_DOCUMENT_AUTHENTICITY",
            "score": local["score"],
            "reasons": local["reasons"],
            "external_verification": {
                "status": "NOT_REQUIRED",
                "verified": False,
            },
        }

    # ---------------------------------------------------------------
    # Local uncertainty
    # ---------------------------------------------------------------

    if local["decision"] == "MANUAL_REVIEW":
        return {
            "verified": False,
            "decision": "MANUAL_REVIEW",
            "verification_stage": "LOCAL_DOCUMENT_AUTHENTICITY",
            "score": local["score"],
            "reasons": local["reasons"],
            "external_verification": {
                "status": "NOT_REQUIRED",
                "verified": False,
            },
        }

    # ---------------------------------------------------------------
    # Local document passed.
    #
    # Now check authoritative source.
    # ---------------------------------------------------------------

    external = evaluate_external_verification(
        extracted,
        database_result,
    )

    # ---------------------------------------------------------------
    # No authoritative verification
    # ---------------------------------------------------------------

    if external["status"] in {
        "NOT_PERFORMED",
        "UNAVAILABLE",
        "NOT_AVAILABLE",
        "INCOMPLETE",
    }:
        return {
            "verified": False,
            "decision": "DOCUMENT_PASS",
            "verification_stage": "LOCAL_DOCUMENT_AUTHENTICITY",
            "score": local["score"],
            "reasons": [
                "Document passed local authenticity checks.",
                "PAN authenticity cannot be confirmed without "
                "authoritative verification.",
            ],
            "external_verification": external,
        }

    # ---------------------------------------------------------------
    # External verification failed
    # ---------------------------------------------------------------

    if not external["verified"]:
        return {
            "verified": False,
            "decision": "DOCUMENT_FAIL",
            "verification_stage": "AUTHORITATIVE_VERIFICATION",
            "score": local["score"],
            "reasons": external["reasons"],
            "external_verification": external,
        }

    # ---------------------------------------------------------------
    # Fully verified
    # ---------------------------------------------------------------

    return {
        "verified": True,
        "decision": "VERIFIED",
        "verification_stage": "AUTHORITATIVE_VERIFICATION",
        "score": 100,
        "reasons": [],
        "external_verification": external,
    }


# ---------------------------------------------------------------------------
# Main integration helper
# ---------------------------------------------------------------------------

def build_pan_result(
    *,
    extracted: dict[str, Any],
    validation: dict[str, Any],
    quality_metrics: dict[str, Any] | None = None,
    database_result: dict[str, Any] | None = None,
    ocr_used: bool = True,
    extraction_method: str = "paddleocr_v5_mobile",
) -> dict[str, Any]:
    """
    Build the final API response.

    This is the function the existing PAN verification file should call
    after extraction + local validation.
    """

    final = make_final_decision(
        local_validation=validation,
        extracted=extracted,
        database_result=database_result,
    )

    return {
        "verified": final["verified"],
        "decision": final["decision"],
        "final_score": final["score"],
        "verification_stage": final["verification_stage"],
        "pan_number": extracted.get("pan"),
        "name": extracted.get("name"),
        "father_name": extracted.get("father_name"),
        "dob": extracted.get("dob"),
        "quality_metrics": quality_metrics or {},
        "validation": validation,
        "database_verification": {
            **(
                database_result
                if database_result
                else {
                    "status": "NOT_PERFORMED",
                    "pan_exists": None,
                    "active": None,
                    "name_match": None,
                    "dob_match": None,
                }
            )
        },
        "final_reasons": final["reasons"],
        "ocr_used": ocr_used,
        "extraction_method": extraction_method,
    }


__all__ = [
    "evaluate_local_authenticity",
    "evaluate_external_verification",
    "make_final_decision",
    "build_pan_result",
]