"""
Common document validation layer.

Document-agnostic local authenticity validation.
PAN/Bank/ITR modules only provide evidence; this module decides the
local document status. It does NOT perform government/bank verification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.common.verification.image_quality import calculate_quality_score

DOCUMENT_PASS = "DOCUMENT_PASS"
DOCUMENT_SUSPICIOUS = "DOCUMENT_SUSPICIOUS"
MANUAL_REVIEW = "MANUAL_REVIEW"
DOCUMENT_REJECT = "DOCUMENT_REJECT"

PASS_SCORE = 80.0
REVIEW_SCORE = 60.0
MEDIUM_TAMPER_SCORE = 50.0
HIGH_TAMPER_SCORE = 75.0
MIN_OCR_CONFIDENCE = 0.60

WEIGHTS = {
    "document_type": 10.0,
    "required_fields": 15.0,
    "field_format": 10.0,
    "ocr": 10.0,
    "field_consistency": 15.0,
    "text_coherence": 10.0,
    "layout": 10.0,
    "quality": 10.0,
    "tamper": 10.0,
}


@dataclass
class ValidationEvidence:
    document_type_detected: bool = False
    required_fields_present: int = 0
    required_fields_total: int = 0
    field_format_valid: bool = True
    ocr_confidence: float = 0.0
    field_consistency: float = 0.0
    text_coherence: float = 0.0
    layout_score: float = 0.0
    quality_score: float = 0.0
    tamper_score: float = 0.0
    tamper_risk: str = "LOW"
    extraction_successful: bool = False
    # Raw image-quality evidence is common across document types.
    # Examples: blur, brightness, contrast, resolution, aspect ratio.
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    optional: dict[str, Any] = field(default_factory=dict)


def _score(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _required_score(present: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return _score((present / total) * 100.0)


def _resolved_tamper_risk(score: float, explicit: str = "") -> str:
    score = _score(score)
    explicit = str(explicit or "").upper().strip()

    if score >= HIGH_TAMPER_SCORE:
        return "HIGH"
    if score >= MEDIUM_TAMPER_SCORE:
        return "MEDIUM"
    if explicit in {"LOW", "MEDIUM", "HIGH"}:
        return explicit
    return "LOW"


def _weighted_score(values: dict[str, float]) -> float:
    total = sum(
        values[key] * WEIGHTS[key] / 100.0
        for key in WEIGHTS
    )
    return round(max(0.0, min(100.0, total)), 2)



def _quality_score_from_metrics(
    metrics: Mapping[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    """Compatibility wrapper around the common image-quality module."""
    return calculate_quality_score(metrics)

def validate_document(
    evidence: ValidationEvidence | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """
    Validate generic document evidence.

    Required evidence is intentionally generic so this same function can
    validate PAN, bank passbook, ITR, Aadhaar, salary slips, etc.

    Local PASS is NOT authoritative verification.
    """

    if evidence is None:
        data: dict[str, Any] = {}
    elif isinstance(evidence, ValidationEvidence):
        data = evidence.__dict__.copy()
    elif isinstance(evidence, Mapping):
        data = dict(evidence)
    else:
        raise TypeError(
            "evidence must be ValidationEvidence, Mapping, or None"
        )

    data.update({
        k: v for k, v in overrides.items()
        if v is not None
    })

    ev = ValidationEvidence(
        document_type_detected=bool(
            data.get("document_type_detected", False)
        ),
        required_fields_present=max(
            0, int(data.get("required_fields_present", 0) or 0)
        ),
        required_fields_total=max(
            0, int(data.get("required_fields_total", 0) or 0)
        ),
        field_format_valid=bool(
            data.get("field_format_valid", True)
        ),
        ocr_confidence=_confidence(
            data.get("ocr_confidence", 0.0)
        ),
        field_consistency=_score(
            data.get("field_consistency", 0.0)
        ),
        text_coherence=_score(
            data.get("text_coherence", 0.0)
        ),
        layout_score=_score(
            data.get("layout_score", 0.0)
        ),
        quality_score=_score(
            data.get("quality_score", 0.0)
        ),
        tamper_score=_score(
            data.get("tamper_score", 0.0)
        ),
        tamper_risk=str(
            data.get("tamper_risk", "LOW")
        ),
        extraction_successful=bool(
            data.get("extraction_successful", False)
        ),
        quality_metrics=dict(
            data.get("quality_metrics", {}) or {}
        ),
        optional=dict(
            data.get("optional", {}) or {}
        ),
    )

    # If raw quality metrics are supplied, the COMMON layer owns the quality
    # calculation. Document-specific modules must not duplicate this logic.
    common_quality_score, quality_details = _quality_score_from_metrics(
        ev.quality_metrics
    )
    if ev.quality_metrics:
        ev.quality_score = common_quality_score

    required_score = _required_score(
        ev.required_fields_present,
        ev.required_fields_total,
    )
    ocr_score = ev.ocr_confidence * 100.0
    tamper_risk = _resolved_tamper_risk(
        ev.tamper_score,
        ev.tamper_risk,
    )

    components = {
        "document_type": 100.0 if ev.document_type_detected else 0.0,
        "required_fields": required_score,
        "field_format": 100.0 if ev.field_format_valid else 0.0,
        "ocr": ocr_score,
        "field_consistency": ev.field_consistency,
        "text_coherence": ev.text_coherence,
        "layout": ev.layout_score,
        "quality": ev.quality_score,
        "tamper": 100.0 - ev.tamper_score,
    }

    final_score = _weighted_score(components)

    reasons: list[str] = []
    warnings: list[str] = []

    if not ev.document_type_detected:
        reasons.append(
            "Document type could not be reliably identified."
        )

    if ev.required_fields_total <= 0:
        reasons.append(
            "No required-field definition was supplied."
        )
    elif ev.required_fields_present < ev.required_fields_total:
        reasons.append(
            "Required document extraction is incomplete."
        )

    if not ev.field_format_valid:
        reasons.append(
            "One or more extracted fields failed format validation."
        )

    if not ev.extraction_successful:
        reasons.append(
            "Required document extraction did not complete reliably."
        )

    if tamper_risk == "HIGH":
        reasons.append(
            "Strong image-level tampering indicators were detected."
        )
    elif tamper_risk == "MEDIUM":
        warnings.append(
            "Moderate image-level tampering indicators were detected."
        )

    if ev.ocr_confidence < MIN_OCR_CONFIDENCE:
        warnings.append(
            f"OCR confidence is low ({ev.ocr_confidence:.2f})."
        )

    if ev.field_consistency < 80:
        warnings.append(
            "Extracted fields show weak internal consistency."
        )

    if ev.text_coherence < 80:
        warnings.append(
            "OCR text has weak coherence."
        )

    if ev.layout_score < 80:
        warnings.append(
            "Document layout/structure is below the preferred threshold."
        )

    if ev.quality_score < 80:
        warnings.append(
            "Document image/file quality is below the preferred threshold."
        )

    if quality_details.get("available"):
        if quality_details.get("blur_score", 0.0) < 100:
            warnings.append(
                "Image sharpness is below the preferred threshold."
            )
        if not (30 <= quality_details.get("brightness", 0.0) <= 245):
            warnings.append(
                "Image brightness is outside the preferred range."
            )
        if quality_details.get("contrast", 0.0) < 20:
            warnings.append(
                "Image contrast is below the preferred threshold."
            )
        if not quality_details.get("resolution_ok", True):
            warnings.append(
                "Image resolution is below the common minimum."
            )

    # Critical failures are hard gates. A high weighted score cannot hide
    # these failures.
    if not ev.document_type_detected:
        decision = DOCUMENT_REJECT
    elif tamper_risk == "HIGH":
        decision = DOCUMENT_SUSPICIOUS
    elif not ev.extraction_successful:
        decision = (
            MANUAL_REVIEW
            if final_score >= REVIEW_SCORE
            else DOCUMENT_REJECT
        )
    elif ev.required_fields_total <= 0:
        decision = DOCUMENT_REJECT
    elif ev.required_fields_present < ev.required_fields_total:
        decision = (
            MANUAL_REVIEW
            if final_score >= REVIEW_SCORE
            else DOCUMENT_REJECT
        )
    elif not ev.field_format_valid:
        decision = (
            MANUAL_REVIEW
            if final_score >= REVIEW_SCORE
            else DOCUMENT_REJECT
        )
    elif final_score >= PASS_SCORE:
        decision = DOCUMENT_PASS
    elif final_score >= REVIEW_SCORE:
        decision = MANUAL_REVIEW
    else:
        decision = DOCUMENT_REJECT

    if decision == MANUAL_REVIEW:
        warnings.append(
            f"Local validation score ({final_score:.2f}) requires manual review."
        )

    if decision == DOCUMENT_REJECT:
        reasons.append(
            f"Local validation score ({final_score:.2f}) is below the review threshold."
        )

    reasons = list(dict.fromkeys(reasons))
    warnings = list(dict.fromkeys(warnings))

    return {
        "decision": decision,
        "score": final_score,
        "verification_stage": "LOCAL_DOCUMENT_AUTHENTICITY",
        "checks": {
            "document_type_detected": ev.document_type_detected,
            "required_fields_present": ev.required_fields_present,
            "required_fields_total": ev.required_fields_total,
            "required_fields_score": round(required_score, 2),
            "field_format_valid": ev.field_format_valid,
            "ocr_confidence": round(ev.ocr_confidence, 4),
            "ocr_quality_score": round(ocr_score, 2),
            "field_consistency_score": round(ev.field_consistency, 2),
            "text_coherence_score": round(ev.text_coherence, 2),
            "layout_score": round(ev.layout_score, 2),
            "quality_score": round(ev.quality_score, 2),
            "quality_metrics": quality_details,
            "tamper_score": round(ev.tamper_score, 2),
            "tamper_risk": tamper_risk,
            "extraction_successful": ev.extraction_successful,
        },
        "reasons": reasons,
        "warnings": warnings,
        "authoritative_verification": {
            "status": "NOT_PERFORMED",
            "message": (
                "Local validation does not prove official validity. "
                "Authoritative verification must be performed separately."
            ),
        },
        "optional": ev.optional,
    }


def validate_from_existing_result(
    *,
    document_type_detected: bool,
    required_fields_present: int,
    required_fields_total: int,
    field_format_valid: bool,
    average_ocr_confidence: float,
    field_consistency_score: float,
    text_coherence_score: float,
    layout_score: float,
    quality_score: float | None = None,
    tamper_score: float = 0.0,
    tamper_risk: str = "LOW",
    extraction_successful: bool = True,
    quality_metrics: Mapping[str, Any] | None = None,
    optional: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapter for the evidence already produced by the current project."""
    return validate_document(
        document_type_detected=document_type_detected,
        required_fields_present=required_fields_present,
        required_fields_total=required_fields_total,
        field_format_valid=field_format_valid,
        ocr_confidence=average_ocr_confidence,
        field_consistency=field_consistency_score,
        text_coherence=text_coherence_score,
        layout_score=layout_score,
        quality_score=quality_score if quality_score is not None else 100.0,
        tamper_score=tamper_score,
        quality_metrics=quality_metrics,
        tamper_risk=tamper_risk,
        extraction_successful=extraction_successful,
        optional=optional,
    )


def module_test() -> dict[str, Any]:
    """Deterministic tests; no OCR, filesystem, GPU or network required."""

    clean = validate_document(
        document_type_detected=True,
        required_fields_present=3,
        required_fields_total=3,
        field_format_valid=True,
        ocr_confidence=0.95,
        field_consistency=95,
        text_coherence=95,
        layout_score=95,
        quality_score=100,
        tamper_score=4,
        extraction_successful=True,
    )

    review = validate_document(
        document_type_detected=True,
        required_fields_present=2,
        required_fields_total=3,
        field_format_valid=True,
        ocr_confidence=0.68,
        field_consistency=65,
        text_coherence=70,
        layout_score=70,
        quality_score=70,
        tamper_score=35,
        extraction_successful=True,
    )

    suspicious = validate_document(
        document_type_detected=True,
        required_fields_present=1,
        required_fields_total=3,
        field_format_valid=False,
        ocr_confidence=0.45,
        field_consistency=25,
        text_coherence=40,
        layout_score=55,
        quality_score=65,
        tamper_score=85,
        tamper_risk="HIGH",
        extraction_successful=False,
    )

    reject = validate_document(
        document_type_detected=False,
        required_fields_present=0,
        required_fields_total=3,
        field_format_valid=False,
        ocr_confidence=0.20,
        field_consistency=10,
        text_coherence=10,
        layout_score=10,
        quality_score=20,
        tamper_score=10,
        extraction_successful=False,
    )

    expected = {
        "clean": DOCUMENT_PASS,
        "review": MANUAL_REVIEW,
        "suspicious": DOCUMENT_SUSPICIOUS,
        "reject": DOCUMENT_REJECT,
    }
    actual = {
        "clean": clean["decision"],
        "review": review["decision"],
        "suspicious": suspicious["decision"],
        "reject": reject["decision"],
    }

    return {
        "passed": actual == expected,
        "actual_decisions": actual,
        "expected_decisions": expected,
        "clean_case": clean,
        "review_case": review,
        "suspicious_case": suspicious,
        "reject_case": reject,
    }


__all__ = [
    "DOCUMENT_PASS",
    "DOCUMENT_SUSPICIOUS",
    "MANUAL_REVIEW",
    "DOCUMENT_REJECT",
    "ValidationEvidence",
    "validate_document",
    "validate_from_existing_result",
    "module_test",
]