"""
Bank Statement Validation API.

Phase 1 ONLY:
    Fast document validation before OCR/extraction.

Kept validation layers:
    1. Physical file/signature validation
    2. Bank-statement document classification
    3. PDF structural/integrity validation
    4. Page consistency
    5. Content-stream analysis
    6. Tamper detection
    7. Document-origin analysis
    8. Authenticity assessment

Intentionally NOT executed here:
    - OCR
    - transaction extraction
    - transaction parsing
    - balance-chain validation
    - statement-period validation based on extracted transactions
    - Phase-4 transaction risk analysis

Those components remain in the project for the later OCR/extraction phase.

Public response is intentionally limited to four top-level fields:
    document_type
    decision
    score
    validation
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.documents.bank_statement.services.bank_statement_verification_service import (
    bank_statement_verification_service,
)


router = APIRouter(
    prefix="/bank-statement",
    tags=["Bank Statement"],
)


def _classify_document(result: dict[str, Any]) -> str:
    """Return the detected document type without guessing."""

    detection = result.get("bank_statement_detection") or {}

    if result.get("is_bank_statement") is True:
        return "BANK_STATEMENT"

    # The Phase-1 detector only knows whether the document is a bank
    # statement. It must not invent another document type.
    # A later shared document classifier can replace this safely.
    if result.get("requires_ocr"):
        return "UNKNOWN"

    if result.get("is_bank_statement") is False:
        return "NON_BANK_STATEMENT"

    return "UNKNOWN"


def _decision(
    result: dict[str, Any],
) -> str:
    """
    Map validation state to LOS decision.

    Tampering policy:
        LOW    -> can continue
        MEDIUM -> HARD REJECT
        HIGH   -> HARD REJECT

    Tampering is evaluated before authenticity/manual-review logic so a
    suspicious document can never become DOCUMENT_VERIFIED.
    """

    status = str(
        result.get("status", "")
    ).lower()

    # ================================================================
    # HARD TAMPER REJECTION
    # ================================================================

    risk = str(
        result.get("risk_level") or "LOW"
    ).strip().upper()

    tamper_suspected = bool(
        result.get("tamper_suspected")
    )

    if tamper_suspected or risk in {
        "MEDIUM",
        "MODERATE",
        "HIGH",
        "CRITICAL",
    }:
        return "DOCUMENT_REJECTED"

    # ================================================================
    # NORMAL VALIDATION STATES
    # ================================================================

    if status == "rejected":
        return "DOCUMENT_REJECTED"

    if status in {
        "requires_ocr",
        "unsupported_for_integrity",
    }:
        return "DOCUMENT_REVIEW"

    if bool(
        result.get("manual_review_required")
    ):
        return "DOCUMENT_REVIEW"

    if (
        result.get("authenticity_assessment")
        in {"REVIEW_REQUIRED", "WEAK"}
    ):
        return "DOCUMENT_REVIEW"

    return "DOCUMENT_VERIFIED"


def _score(
    result: dict[str, Any],
    decision: str,
) -> int:
    """Create a deterministic 0-100 Phase-1 validation score."""

    if decision == "DOCUMENT_REJECTED":
        return 0

    if decision == "DOCUMENT_REVIEW":
        # Preserve useful evidence when available, but never make a
        # review state look verified.
        authenticity = result.get(
            "authenticity_score"
        )

        if authenticity is not None:
            try:
                return max(
                    0,
                    min(
                        99,
                        int(round(float(authenticity))),
                    ),
                )
            except (TypeError, ValueError):
                pass

        return 0

    authenticity = result.get(
        "authenticity_score"
    )

    if authenticity is not None:
        try:
            return max(
                0,
                min(
                    100,
                    int(round(float(authenticity))),
                ),
            )
        except (TypeError, ValueError):
            pass

    return 100


def _validation(
    result: dict[str, Any],
    decision: str,
    processing_time_ms: float,
) -> dict[str, Any]:
    """Build compact validation evidence plus processing timing."""

    if decision == "DOCUMENT_REJECTED":
        structural = result.get(
            "file_detection",
            {},
        )
        structural_pass = bool(
            structural.get("supported")
            and structural.get("signature_valid")
        )
    else:
        structural_pass = True

    if result.get("requires_ocr"):
        quality = "NOT_CHECKED"
    else:
        quality = "GOOD"

    if result.get("tamper_suspected"):
        tampering = "HIGH"
    else:
        risk = str(
            result.get("risk_level") or "LOW"
        ).strip().upper()

        tampering = (
            "HIGH"
            if risk in {"HIGH", "CRITICAL"}
            else "MEDIUM"
            if risk in {"MEDIUM", "MODERATE"}
            else "LOW"
        )

    return {
        "document_detected": (
            result.get("is_bank_statement") is True
        ),
        "image_quality": quality,
        "tampering_risk": tampering,
        "structural_validation": (
            "PASS"
            if structural_pass
            else "FAIL"
        ),
        "processing_time_ms": processing_time_ms,
    }


@router.post(
    "/verify",
    summary="Validate bank statement",
)
async def verify_bank_statement(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Phase-1 validation only.

    IMPORTANT:
        Extraction/OCR code is intentionally not called here.

    The existing extraction implementation is retained in the project
    for the next phase, but this endpoint stops after document/integrity
    validation.
    """

    started = perf_counter()

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded.",
        )

    filename = (
        file.filename or "uploaded_document"
    ).strip()

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unable to read uploaded file.",
                "error": str(exc),
            },
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        # ============================================================
        # VALIDATION-ONLY SERVICE
        # ============================================================
        #
        # This service performs:
        #   file detection
        #   bank-statement detection
        #   PDF integrity
        #   page consistency
        #   content-stream analysis
        #   tamper detection
        #   document-origin analysis
        #   authenticity analysis
        #
        # It deliberately does NOT perform OCR or transaction extraction.
        # ============================================================
        result_obj = bank_statement_verification_service.fast_validate(
            file_bytes=file_bytes,
            filename=filename,
        )

        result = result_obj.to_dict()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Bank Statement validation failed.",
                "error": str(exc),
            },
        ) from exc

    # ================================================================
    # DECISION + TIMING
    # ================================================================

    decision = _decision(result)
    document_type = _classify_document(result)
    score = _score(result, decision)

    processing_time_ms = round(
        (perf_counter() - started) * 1000.0,
        2,
    )

    validation = _validation(
        result,
        decision,
        processing_time_ms,
    )

    # Keep exactly four top-level LOS fields.
    return {
        "document_type": document_type,
        "decision": decision,
        "score": score,
        "validation": validation,
    }


# Backward-compatible endpoint.
# Existing clients using /analyze can continue to call the same
# validation-only implementation. Extraction remains disabled.
@router.post(
    "/analyze",
    summary="Validate bank statement (legacy alias)",
)
async def analyze_bank_statement(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    return await verify_bank_statement(file)