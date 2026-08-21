"""
Risk aggregation for document verification.

Keeps two concepts separate:

1. risk_score
   How suspicious the evidence that actually ran is.
   0 = no suspicious signals detected, 100 = highly suspicious.

2. verification_confidence
   How much useful verification evidence was available.
   A low risk score with weak evidence must NOT be labelled genuine.

Supported signals:
    metadata
    pdf_structure
    ela
    consistency
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


# Relative importance when a signal is applicable.
DEFAULT_RISK_WEIGHTS = {
    "metadata": 0.20,
    "pdf_structure": 0.30,
    "ela": 0.20,
    "consistency": 0.30,
}

# Evidence contribution to verification confidence.
# These intentionally sum to 100.
CONFIDENCE_WEIGHTS = {
    "metadata": 20,
    "pdf_structure": 30,
    "ela": 20,
    "consistency": 30,
}

LOW_RISK_MAX = 30.0
MEDIUM_RISK_MAX = 60.0

# We require meaningful evidence before making a positive authenticity-style
# verdict. This is deliberately stricter than merely having one check run.
MIN_CONFIDENCE_FOR_VERDICT = 70


def combine(
    metadata_result: Dict[str, Any],
    ela_result: Dict[str, Any],
    consistency_result: Dict[str, Any],
    pdf_structure_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Combine forgery signals into a clean verification result.

    Backward compatible with the old 3-argument call:
        combine(metadata, ela, consistency)

    New pipeline should pass:
        combine(metadata, ela, consistency, pdf_structure)
    """

    signals: Dict[str, Dict[str, Any]] = {
        "metadata": metadata_result or {},
        "pdf_structure": pdf_structure_result or {},
        "ela": ela_result or {},
        "consistency": consistency_result or {},
    }

    applicable = {
        name: result
        for name, result in signals.items()
        if _is_checked(result)
    }

    # ------------------------------------------------------------
    # RISK SCORE
    # ------------------------------------------------------------
    # Re-normalize weights over checks that actually ran. An N/A check must
    # neither dilute nor increase another signal's suspicion score.
    risk_weights = {
        name: DEFAULT_RISK_WEIGHTS[name]
        for name in applicable
    }

    weight_sum = sum(risk_weights.values())

    if weight_sum > 0:
        risk_score = sum(
            _bounded_score(applicable[name].get("score", 0))
            * (weight / weight_sum)
            for name, weight in risk_weights.items()
        )
    else:
        risk_score = 0.0

    risk_score = round(risk_score, 1)

    # ------------------------------------------------------------
    # VERIFICATION CONFIDENCE
    # ------------------------------------------------------------
    confidence = 0.0
    confidence_details: Dict[str, Any] = {}

    for name, result in signals.items():
        base = CONFIDENCE_WEIGHTS[name]

        if not _is_checked(result):
            confidence_details[name] = {
                "available": False,
                "contribution": 0,
                "max_contribution": base,
                "status": result.get("status", "not_checked"),
            }
            continue

        quality = _signal_quality(name, result)
        contribution = base * quality
        confidence += contribution

        confidence_details[name] = {
            "available": True,
            "quality": round(quality, 2),
            "contribution": round(contribution, 1),
            "max_contribution": base,
            "status": result.get("status", "checked"),
        }

    verification_confidence = int(round(min(100.0, confidence)))

    # ------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------
    if verification_confidence < MIN_CONFIDENCE_FOR_VERDICT:
        verdict = "INSUFFICIENT EVIDENCE"
    elif risk_score <= LOW_RISK_MAX:
        verdict = "NO SUSPICIOUS SIGNALS DETECTED"
    elif risk_score <= MEDIUM_RISK_MAX:
        verdict = "NEEDS MANUAL REVIEW"
    else:
        verdict = "HIGH RISK / LIKELY TAMPERED"

    # ------------------------------------------------------------
    # REASONS
    # ------------------------------------------------------------
    reasons = []

    for name in ("metadata", "pdf_structure", "ela", "consistency"):
        result = signals[name]
        for reason in result.get("reasons", []) or []:
            reasons.append(f"[{name}] {reason}")

    checked_names = [
        name for name, result in signals.items()
        if _is_checked(result)
    ]
    skipped_names = [
        name for name, result in signals.items()
        if not _is_checked(result)
    ]

    reasons.append(
        "[verification] Checks completed: "
        + (", ".join(checked_names) if checked_names else "none")
        + "."
    )

    if skipped_names:
        reasons.append(
            "[verification] Checks not completed/applicable: "
            + ", ".join(skipped_names)
            + "."
        )

    reasons.append(
        f"[verification] Verification confidence: "
        f"{verification_confidence}/100."
    )

    return {
        "risk_score": risk_score,
        "verification_confidence": verification_confidence,
        "verdict": verdict,
        "signals": signals,
        "confidence_details": confidence_details,
        "reasons": reasons,
    }


def _is_checked(result: Mapping[str, Any]) -> bool:
    return bool(result.get("checked") is True)


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, score))


def _signal_quality(name: str, result: Mapping[str, Any]) -> float:
    """
    Estimate how complete/useful a successfully-run signal was.

    This affects confidence only, never suspicion risk.
    """

    if not _is_checked(result):
        return 0.0

    if name == "metadata":
        details = result.get("details", {}) or {}

        quality = 0.55  # PDF was opened and metadata container inspected.

        if details.get("producer") or details.get("creator"):
            quality += 0.15

        if (
            details.get("creation_date_raw")
            or details.get("modification_date_raw")
        ):
            quality += 0.15

        if details.get("page_count"):
            quality += 0.15

        return min(1.0, quality)

    if name == "pdf_structure":
        details = result.get("details", {}) or {}

        # Image-only PDFs still technically run the checker, but native-text
        # structural evidence is substantially weaker.
        total_spans = details.get("total_text_spans", 0) or 0

        if total_spans <= 0:
            return 0.45

        quality = 0.75

        if details.get("unique_font_count") is not None:
            quality += 0.10

        if details.get("page_count"):
            quality += 0.10

        if details.get("overlap_ratio") is not None:
            quality += 0.05

        return min(1.0, quality)

    if name == "ela":
        # If ELA ran on a raster image, it provides its full evidence share.
        return 1.0

    if name == "consistency":
        # `checked=True` means at least one extraction/business-rule check ran.
        # Missing fields can reduce usefulness, but the current checker does
        # not expose a structured coverage ratio, so use reasons conservatively.
        reasons = " ".join(
            str(r).lower()
            for r in (result.get("reasons", []) or [])
        )

        quality = 1.0

        if "could not extract critical field" in reasons:
            quality -= 0.25

        if "could not be fully verified" in reasons:
            quality -= 0.20

        if "not enough extracted information" in reasons:
            quality -= 0.35

        return max(0.35, min(1.0, quality))

    return 1.0
