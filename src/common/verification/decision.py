"""
Common document verification decision engine.

Purpose
-------
Combine document-independent signals into one consistent decision.

This module does NOT decide that a document is legally genuine. It evaluates
the evidence available locally:

    document markers
    field extraction
    OCR confidence
    image quality
    tamper signals
    field consistency

Document-specific validators (PAN, Aadhaar, Voter ID, passbook, etc.) should
provide the extracted fields and document-specific checks to this engine.

Decision levels
---------------
DOCUMENT_PASS
    Strong local evidence and no major contradiction.

MANUAL_REVIEW
    Evidence is incomplete, weak, or conflicting.

DOCUMENT_SUSPICIOUS
    Multiple strong local risk signals or a critical contradiction.

Important
---------
A local image cannot prove government-record authenticity by itself. If an
authoritative database/API verification is available, it must remain a
separate verification stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# ============================================================================
# CONFIGURATION
# ============================================================================

PASS_THRESHOLD = 75
REVIEW_THRESHOLD = 50

# A high tamper score is treated as a strong warning, but not as automatic
# proof of forgery unless corroborated by another serious signal.
HIGH_TAMPER_SCORE = 75
MEDIUM_TAMPER_SCORE = 50

# Very low OCR confidence means extraction itself is unreliable.
LOW_OCR_CONFIDENCE = 0.45
MEDIUM_OCR_CONFIDENCE = 0.65

# Required local evidence.
MIN_REQUIRED_FIELDS = 2


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class DecisionEvidence:
    """
    Normalized evidence supplied to the decision engine.

    All fields are optional so different document types can use the same
    engine without forcing PAN-specific assumptions.
    """

    document_markers_score: float = 0.0
    layout_score: float = 0.0
    average_ocr_confidence: float | None = None

    required_fields_present: int = 0
    required_fields_total: int = 0

    field_consistency_score: float = 0.0

    tamper_score: float = 0.0
    tamper_risk: str = "UNKNOWN"

    blur_score: float | None = None
    brightness: float | None = None
    contrast: float | None = None

    document_type_detected: bool = False
    extraction_successful: bool = False

    critical_conflict: bool = False

    authoritative_status: str = "NOT_PERFORMED"
    authoritative_match: bool | None = None


# ============================================================================
# HELPERS
# ============================================================================

def _number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 100.0,
) -> float:
    return max(
        low,
        min(high, value),
    )


def _normalize_risk(value: Any) -> str:
    return str(value or "UNKNOWN").upper().strip()


def _normalize_authoritative_status(value: Any) -> str:
    return str(
        value or "NOT_PERFORMED"
    ).upper().strip()


def _field_presence_ratio(
    present: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    return _clamp(
        present / total * 100.0
    )


# ============================================================================
# EVIDENCE EXTRACTION
# ============================================================================

def evidence_from_document_result(
    result: Mapping[str, Any],
    *,
    required_fields: Sequence[str] | None = None,
    document_markers_score: float | None = None,
    layout_score: float | None = None,
    field_consistency_score: float | None = None,
    critical_conflict: bool = False,
    authoritative_status: str = "NOT_PERFORMED",
    authoritative_match: bool | None = None,
) -> DecisionEvidence:
    """
    Convert an existing document-verification result into DecisionEvidence.

    This accepts the structure already produced by the project's validators,
    for example:

        {
            "pan_number": "...",
            "name": "...",
            "dob": "...",
            "validation": {
                "checks": {...}
            },
            "quality_metrics": {...}
        }

    It also accepts the common processor structure where page quality contains
    the generic tamper result.
    """

    result = dict(result)

    validation = result.get("validation")
    if not isinstance(validation, Mapping):
        validation = {}

    checks = validation.get("checks")
    if not isinstance(checks, Mapping):
        checks = {}

    quality = result.get("quality_metrics")
    if not isinstance(quality, Mapping):
        quality = {}

    # Some callers may pass:
    # result["pages"][0]["quality"]["tamper"]
    tamper = result.get("tamper")

    if not isinstance(tamper, Mapping):
        pages = result.get("pages")

        if isinstance(pages, Sequence) and not isinstance(
            pages,
            (str, bytes, bytearray),
        ):
            for page in pages:
                if not isinstance(page, Mapping):
                    continue

                page_quality = page.get("quality")
                if not isinstance(page_quality, Mapping):
                    continue

                page_tamper = page_quality.get("tamper")

                if isinstance(page_tamper, Mapping):
                    tamper = page_tamper
                    break

    if not isinstance(tamper, Mapping):
        tamper = quality.get("tamper")

    if not isinstance(tamper, Mapping):
        tamper = {}

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    if required_fields is None:
        # Generic default. Document-specific validators should pass their own
        # required fields.
        required_fields = (
            "name",
            "dob",
        )

    present = 0

    for field in required_fields:
        value = result.get(field)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        present += 1

    # --------------------------------------------------------
    # OCR confidence
    # --------------------------------------------------------

    average_ocr_confidence = checks.get(
        "average_ocr_confidence"
    )

    if average_ocr_confidence is None:
        average_ocr_confidence = result.get(
            "average_ocr_confidence"
        )

    # --------------------------------------------------------
    # Generic scores
    # --------------------------------------------------------

    markers = (
        document_markers_score
        if document_markers_score is not None
        else checks.get(
            "pan_markers_score",
            checks.get(
                "document_markers_score",
                0.0,
            ),
        )
    )

    layout = (
        layout_score
        if layout_score is not None
        else checks.get(
            "layout_score",
            0.0,
        )
    )

    consistency = (
        field_consistency_score
        if field_consistency_score is not None
        else checks.get(
            "field_consistency_score",
            0.0,
        )
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    blur = quality.get(
        "blur_score",
        result.get("blur_score"),
    )

    brightness = quality.get(
        "brightness",
        result.get("brightness"),
    )

    contrast = quality.get(
        "contrast",
        result.get("contrast"),
    )

    # --------------------------------------------------------
    # Document detection / extraction
    # --------------------------------------------------------

    document_detected = checks.get(
        "is_pan_document",
        checks.get(
            "is_document",
            result.get(
                "document_type_detected",
                False,
            ),
        ),
    )

    extraction_successful = checks.get(
        "required_fields",
        result.get(
            "extraction_successful",
            present >= MIN_REQUIRED_FIELDS,
        ),
    )

    return DecisionEvidence(
        document_markers_score=_clamp(
            _number(markers)
        ),
        layout_score=_clamp(
            _number(layout)
        ),
        average_ocr_confidence=(
            None
            if average_ocr_confidence is None
            else _clamp(
                _number(average_ocr_confidence),
                0.0,
                1.0,
            )
        ),
        required_fields_present=present,
        required_fields_total=len(required_fields),
        field_consistency_score=_clamp(
            _number(consistency)
        ),
        tamper_score=_clamp(
            _number(
                tamper.get(
                    "tamper_score",
                    0.0,
                )
            )
        ),
        tamper_risk=_normalize_risk(
            tamper.get(
                "risk",
                "UNKNOWN",
            )
        ),
        blur_score=(
            None
            if blur is None
            else _number(blur)
        ),
        brightness=(
            None
            if brightness is None
            else _number(brightness)
        ),
        contrast=(
            None
            if contrast is None
            else _number(contrast)
        ),
        document_type_detected=bool(
            document_detected
        ),
        extraction_successful=bool(
            extraction_successful
        ),
        critical_conflict=bool(
            critical_conflict
        ),
        authoritative_status=_normalize_authoritative_status(
            authoritative_status
        ),
        authoritative_match=authoritative_match,
    )


# ============================================================================
# COMPONENT SCORING
# ============================================================================

def score_document_markers(
    evidence: DecisionEvidence,
) -> float:
    return _clamp(
        evidence.document_markers_score
    )


def score_layout(
    evidence: DecisionEvidence,
) -> float:
    return _clamp(
        evidence.layout_score
    )


def score_ocr(
    evidence: DecisionEvidence,
) -> float:
    confidence = evidence.average_ocr_confidence

    if confidence is None:
        return 50.0

    if confidence >= 0.90:
        return 100.0

    if confidence >= 0.80:
        return 90.0

    if confidence >= 0.70:
        return 80.0

    if confidence >= MEDIUM_OCR_CONFIDENCE:
        return 65.0

    if confidence >= LOW_OCR_CONFIDENCE:
        return 45.0

    return 20.0


def score_required_fields(
    evidence: DecisionEvidence,
) -> float:
    return _field_presence_ratio(
        evidence.required_fields_present,
        evidence.required_fields_total,
    )


def score_field_consistency(
    evidence: DecisionEvidence,
) -> float:
    return _clamp(
        evidence.field_consistency_score
    )


def score_tamper(
    evidence: DecisionEvidence,
) -> float:
    """
    Convert tamper evidence into an authenticity contribution.

    LOW tamper risk does not add a bonus beyond the neutral score.
    HIGH tamper risk substantially reduces confidence.
    """
    score = _clamp(
        evidence.tamper_score
    )

    if score >= HIGH_TAMPER_SCORE:
        return 0.0

    if score >= MEDIUM_TAMPER_SCORE:
        return 25.0

    if score >= 25:
        return 70.0

    return 100.0


def score_quality(
    evidence: DecisionEvidence,
) -> float:
    """
    Generic quality score.

    No hard brightness/contrast threshold is used because valid documents
    can legitimately vary in lighting and scanning conditions.
    """
    values = []

    if evidence.blur_score is not None:
        # Very conservative blur normalization.
        if evidence.blur_score >= 1000:
            values.append(100.0)
        elif evidence.blur_score >= 300:
            values.append(80.0)
        elif evidence.blur_score >= 100:
            values.append(60.0)
        else:
            values.append(20.0)

    if evidence.contrast is not None:
        contrast = evidence.contrast

        if 25 <= contrast <= 90:
            values.append(100.0)
        elif 15 <= contrast <= 110:
            values.append(75.0)
        else:
            values.append(50.0)

    if not values:
        return 70.0

    return sum(values) / len(values)


# ============================================================================
# DECISION ENGINE
# ============================================================================

def _weighted_local_score(
    evidence: DecisionEvidence,
) -> float:
    """
    Weighted score for local document evidence.

    The weights deliberately prevent tamper analysis from overpowering
    otherwise coherent document evidence.
    """

    components = [
        (
            score_document_markers(evidence),
            0.20,
        ),
        (
            score_layout(evidence),
            0.15,
        ),
        (
            score_ocr(evidence),
            0.15,
        ),
        (
            score_required_fields(evidence),
            0.15,
        ),
        (
            score_field_consistency(evidence),
            0.15,
        ),
        (
            score_tamper(evidence),
            0.10,
        ),
        (
            score_quality(evidence),
            0.10,
        ),
    ]

    return sum(
        score * weight
        for score, weight in components
    )


def _decision_from_score(
    score: float,
) -> str:
    if score >= PASS_THRESHOLD:
        return "DOCUMENT_PASS"

    if score >= REVIEW_THRESHOLD:
        return "MANUAL_REVIEW"

    return "DOCUMENT_SUSPICIOUS"


def evaluate_document(
    evidence: DecisionEvidence,
) -> dict[str, Any]:
    """
    Produce the common local authenticity decision.

    This function intentionally separates:
        LOCAL_DOCUMENT_AUTHENTICITY
    from:
        AUTHORITATIVE_VERIFICATION

    A document can pass local checks while still having no database
    verification.
    """

    score = _weighted_local_score(
        evidence
    )

    reasons: list[str] = []
    warnings: list[str] = []

    # --------------------------------------------------------
    # Critical conditions
    # --------------------------------------------------------

    if not evidence.document_type_detected:
        reasons.append(
            "Document type was not reliably identified."
        )

    if evidence.required_fields_total > 0:
        if (
            evidence.required_fields_present
            < MIN_REQUIRED_FIELDS
        ):
            reasons.append(
                "Too few required fields were extracted."
            )

    if not evidence.extraction_successful:
        reasons.append(
            "Required document extraction did not complete reliably."
        )

    if evidence.critical_conflict:
        reasons.append(
            "A critical field conflict was detected."
        )

    if (
        evidence.tamper_risk == "HIGH"
        or evidence.tamper_score >= HIGH_TAMPER_SCORE
    ):
        reasons.append(
            "Strong image-level tampering indicators were detected."
        )

    elif (
        evidence.tamper_risk == "MEDIUM"
        or evidence.tamper_score >= MEDIUM_TAMPER_SCORE
    ):
        warnings.append(
            "Moderate image-level tampering indicators were detected."
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    if (
        evidence.average_ocr_confidence is not None
        and evidence.average_ocr_confidence
        < LOW_OCR_CONFIDENCE
    ):
        warnings.append(
            "OCR confidence is low; extracted fields require review."
        )

    # --------------------------------------------------------
    # Field consistency
    # --------------------------------------------------------

    if (
        evidence.field_consistency_score > 0
        and evidence.field_consistency_score < 50
    ):
        warnings.append(
            "Extracted fields show weak internal consistency."
        )

    # --------------------------------------------------------
    # Initial decision
    # --------------------------------------------------------

    decision = _decision_from_score(
        score
    )

    # Critical conflicts should never pass local verification.
    if evidence.critical_conflict:
        decision = "DOCUMENT_SUSPICIOUS"

    # Strong tamper evidence plus weak extraction is suspicious.
    if (
        evidence.tamper_score >= HIGH_TAMPER_SCORE
        and (
            evidence.required_fields_present
            < evidence.required_fields_total
            or (
                evidence.average_ocr_confidence is not None
                and evidence.average_ocr_confidence
                < MEDIUM_OCR_CONFIDENCE
            )
        )
    ):
        decision = "DOCUMENT_SUSPICIOUS"

    # A weak local score cannot pass.
    if score < REVIEW_THRESHOLD:
        decision = "DOCUMENT_SUSPICIOUS"

    # --------------------------------------------------------
    # Authoritative verification
    # --------------------------------------------------------

    authoritative_status = _normalize_authoritative_status(
        evidence.authoritative_status
    )

    authoritative_decision = "NOT_PERFORMED"

    if evidence.authoritative_match is not None:
        # This is intentionally explicit rather than folded into local score.
        if evidence.authoritative_match is True:
            authoritative_decision = "MATCH"
        else:
            authoritative_decision = "MISMATCH"
            decision = "DOCUMENT_SUSPICIOUS"
            reasons.append(
                "Authoritative verification did not match the extracted identity."
            )

    elif authoritative_status in {
        "NOT_PERFORMED",
        "PENDING",
        "UNAVAILABLE",
    }:
        authoritative_decision = authoritative_status

    return {
        "decision": decision,
        "score": round(
            _clamp(score),
            2,
        ),
        "verification_stage": "LOCAL_DOCUMENT_AUTHENTICITY",
        "authoritative_verification": {
            "status": authoritative_status,
            "decision": authoritative_decision,
            "match": evidence.authoritative_match,
        },
        "checks": {
            "document_markers_score": round(
                score_document_markers(evidence),
                2,
            ),
            "layout_score": round(
                score_layout(evidence),
                2,
            ),
            "ocr_score": round(
                score_ocr(evidence),
                2,
            ),
            "required_fields_score": round(
                score_required_fields(evidence),
                2,
            ),
            "field_consistency_score": round(
                score_field_consistency(evidence),
                2,
            ),
            "tamper_score": round(
                evidence.tamper_score,
                2,
            ),
            "tamper_risk": evidence.tamper_risk,
            "quality_score": round(
                score_quality(evidence),
                2,
            ),
            "average_ocr_confidence": evidence.average_ocr_confidence,
            "required_fields_present": evidence.required_fields_present,
            "required_fields_total": evidence.required_fields_total,
            "document_type_detected": evidence.document_type_detected,
            "extraction_successful": evidence.extraction_successful,
        },
        "reasons": reasons,
        "warnings": warnings,
    }


# ============================================================================
# CONVENIENCE API
# ============================================================================

def decide_from_result(
    result: Mapping[str, Any],
    *,
    required_fields: Sequence[str] | None = None,
    document_markers_score: float | None = None,
    layout_score: float | None = None,
    field_consistency_score: float | None = None,
    critical_conflict: bool = False,
    authoritative_status: str = "NOT_PERFORMED",
    authoritative_match: bool | None = None,
) -> dict[str, Any]:
    """
    One-call API for document validators.

    Example:

        decision = decide_from_result(
            validation_result,
            required_fields=("name", "dob"),
            document_markers_score=90,
            layout_score=85,
        )
    """

    evidence = evidence_from_document_result(
        result,
        required_fields=required_fields,
        document_markers_score=document_markers_score,
        layout_score=layout_score,
        field_consistency_score=field_consistency_score,
        critical_conflict=critical_conflict,
        authoritative_status=authoritative_status,
        authoritative_match=authoritative_match,
    )

    return evaluate_document(
        evidence
    )


# ============================================================================
# TESTS
# ============================================================================

def module_test() -> dict[str, Any]:
    """
    Deterministic unit test for the common decision engine.

    No OCR and no external services are used.
    """

    clean = DecisionEvidence(
        document_markers_score=95,
        layout_score=90,
        average_ocr_confidence=0.90,
        required_fields_present=3,
        required_fields_total=3,
        field_consistency_score=95,
        tamper_score=4,
        tamper_risk="LOW",
        blur_score=10000,
        contrast=55,
        document_type_detected=True,
        extraction_successful=True,
    )

    clean_result = evaluate_document(
        clean
    )

    suspicious = DecisionEvidence(
        document_markers_score=85,
        layout_score=75,
        average_ocr_confidence=0.55,
        required_fields_present=1,
        required_fields_total=3,
        field_consistency_score=25,
        tamper_score=85,
        tamper_risk="HIGH",
        blur_score=500,
        contrast=20,
        document_type_detected=True,
        extraction_successful=False,
    )

    suspicious_result = evaluate_document(
        suspicious
    )

    passed = (
        clean_result["decision"] == "DOCUMENT_PASS"
        and suspicious_result["decision"]
        == "DOCUMENT_SUSPICIOUS"
    )

    return {
        "passed": passed,
        "clean_case": clean_result,
        "suspicious_case": suspicious_result,
    }


__all__ = [
    "DecisionEvidence",
    "evidence_from_document_result",
    "evaluate_document",
    "decide_from_result",
    "score_document_markers",
    "score_layout",
    "score_ocr",
    "score_required_fields",
    "score_field_consistency",
    "score_tamper",
    "score_quality",
    "module_test",
]