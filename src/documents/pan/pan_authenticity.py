from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .pan_verification import (
    PAN_PATTERN,
    _normal,
    _compact,
    _clean_date,
)


@dataclass
class PanValidationResult:
    decision: str
    score: int
    checks: dict[str, Any]
    reasons: list[str]


def _check_pan_format(pan: str | None) -> bool:
    if not pan:
        return False

    return bool(
        PAN_PATTERN.fullmatch(
            pan.upper().strip()
        )
    )


def _check_pan_structure(pan: str | None) -> bool:
    """
    PAN structure:

    5 letters
    4 digits
    1 letter

    This is only a structural check.
    It does NOT prove the PAN exists.
    """
    if not _check_pan_format(pan):
        return False

    pan = pan.upper().strip()

    return (
        pan[:5].isalpha()
        and pan[5:9].isdigit()
        and pan[9].isalpha()
    )


def _check_required_fields(
    pan: str | None,
    name: str | None,
    dob: str | None,
) -> bool:
    return bool(
        pan
        and name
        and dob
    )


def _check_ocr_confidence(
    results: list[dict[str, Any]],
) -> tuple[float, int]:
    """
    Calculate average confidence and convert it to a score.

    This is a quality signal, NOT proof of authenticity.
    """
    values = []

    for item in results:
        try:
            confidence = float(
                item.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            continue

        if confidence > 0:
            values.append(confidence)

    if not values:
        return 0.0, 0

    average = sum(values) / len(values)

    if average >= 0.90:
        score = 15
    elif average >= 0.80:
        score = 12
    elif average >= 0.70:
        score = 9
    elif average >= 0.60:
        score = 6
    else:
        score = 2

    return round(average, 3), score


def _check_pan_markers(
    results: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    text = _normal(
        " ".join(
            str(item.get("text", ""))
            for item in results
        )
    )

    compact = _compact(text)

    score = 0
    reasons = []

    if "INCOMETAX" in compact:
        score += 10
    else:
        reasons.append(
            "Income Tax Department marker not detected."
        )

    if "GOVTOFINDIA" in compact:
        score += 8
    elif "GOVT" in compact and "INDIA" in compact:
        score += 5
    else:
        reasons.append(
            "Government of India marker not detected."
        )

    if "PERMANENTACCOUNT" in compact:
        score += 7
    else:
        reasons.append(
            "Permanent Account marker not detected."
        )

    return score, reasons


def _check_dob(dob: str | None) -> bool:
    if not dob:
        return False

    return _clean_date(dob) is not None


def validate_pan_document(
    *,
    results: list[dict[str, Any]],
    pan: str | None,
    name: str | None,
    father_name: str | None,
    dob: str | None,
    quality_metrics: dict[str, Any] | None = None,
) -> PanValidationResult:

    score = 0
    reasons = []
    checks = {}

    # ---------------------------------------------------------
    # 1. PAN format
    # ---------------------------------------------------------

    format_ok = _check_pan_format(pan)

    checks["pan_format"] = format_ok

    if format_ok:
        score += 20
    else:
        reasons.append(
            "PAN format is invalid."
        )

    # ---------------------------------------------------------
    # 2. PAN structure
    # ---------------------------------------------------------

    structure_ok = _check_pan_structure(pan)

    checks["pan_structure"] = structure_ok

    if structure_ok:
        score += 10
    else:
        reasons.append(
            "PAN character structure is invalid."
        )

    # ---------------------------------------------------------
    # 3. Required fields
    # ---------------------------------------------------------

    fields_ok = _check_required_fields(
        pan,
        name,
        dob,
    )

    checks["required_fields"] = fields_ok

    if fields_ok:
        score += 15
    else:
        reasons.append(
            "Required PAN fields could not be reliably extracted."
        )

    # ---------------------------------------------------------
    # 4. DOB
    # ---------------------------------------------------------

    dob_ok = _check_dob(dob)

    checks["dob_format"] = dob_ok

    if dob_ok:
        score += 5
    else:
        reasons.append(
            "Date of birth is invalid or missing."
        )

    # ---------------------------------------------------------
    # 5. PAN-specific visual/text markers
    # ---------------------------------------------------------

    marker_score, marker_reasons = _check_pan_markers(
        results
    )

    score += marker_score
    reasons.extend(marker_reasons)

    checks["pan_markers_score"] = marker_score

    # ---------------------------------------------------------
    # 6. OCR confidence
    # ---------------------------------------------------------

    average_confidence, confidence_score = (
        _check_ocr_confidence(results)
    )

    score += confidence_score

    checks["average_ocr_confidence"] = (
        average_confidence
    )

    checks["ocr_confidence_score"] = (
        confidence_score
    )

    # ---------------------------------------------------------
    # 7. Image quality
    # ---------------------------------------------------------

    if quality_metrics:
        blur_score = float(
            quality_metrics.get(
                "blur_score",
                0,
            )
        )

        checks["blur_score"] = blur_score

        if blur_score >= 300:
            score += 5
        elif blur_score >= 100:
            score += 3
        else:
            reasons.append(
                "Image sharpness is poor."
            )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # This score means:
    #
    # "Does this document strongly resemble a PAN card?"
    #
    # It does NOT mean:
    #
    # "The PAN exists in the government database."
    # ---------------------------------------------------------

    if score >= 85:
        decision = "DOCUMENT_PASS"

    elif score >= 65:
        decision = "MANUAL_REVIEW"

    else:
        decision = "DOCUMENT_REJECT"

    return PanValidationResult(
        decision=decision,
        score=min(score, 100),
        checks=checks,
        reasons=reasons,
    )