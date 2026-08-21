"""
Voter ID Validation API Router.

Public endpoint:
    POST /voter-id/verify

Validation-only path:
    Upload
      -> Common security
      -> Fast visual Voter ID validation
      -> Clean public response

OCR/extraction is intentionally not exposed or executed by this endpoint.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .voter_verification import verify_voter_card

from src.common.security.file_security import (
    validate_upload,
    MaliciousFileError,
    VirusScannerUnavailableError,
    UnsupportedFileTypeError,
    InvalidFileSignatureError,
    FileTooLargeError,
    InvalidFileError,
)


router = APIRouter(
    prefix="/voter-id",
    tags=["Voter ID Verification"],
)


class VoterValidation(BaseModel):
    document_detected: bool
    image_quality: str
    tampering_risk: str
    visual_fields: str


class VoterVerificationResponse(BaseModel):
    document_type: str
    decision: str
    score: float
    validation: VoterValidation
    security: dict[str, Any]
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    detail: Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quality(validation: dict[str, Any]) -> str:
    explicit = validation.get("image_quality")
    if explicit:
        return str(explicit).upper()

    metrics = validation.get("quality_metrics") or {}
    if not isinstance(metrics, dict):
        return "UNKNOWN"

    if not metrics.get("available", False):
        return "UNKNOWN"

    if not metrics.get("resolution_ok", True):
        return "POOR"

    blur = _safe_float(metrics.get("blur_score"))
    contrast = _safe_float(metrics.get("contrast"))
    brightness = _safe_float(metrics.get("brightness"))

    if blur < 100 or contrast < 20 or not (30 <= brightness <= 245):
        return "POOR"

    if blur < 300 or contrast < 30:
        return "FAIR"

    return "GOOD"


def _fields(validation: dict[str, Any]) -> str:
    value = validation.get("visual_fields")
    if value:
        return str(value)

    value = validation.get("visual_fields_present")
    total = validation.get("visual_fields_total")

    if value is not None and total is not None:
        return f"{value}/{total}"

    value = validation.get("required_fields")
    if value:
        return str(value)

    return "0/5"


def _final_decision(validation: dict[str, Any]) -> str:
    """
    IMPORTANT:
    Do not trust an old/internal OCR decision.

    The public decision is calculated only from the validation evidence.
    """

    detected = bool(validation.get("document_detected", False))
    quality = _quality(validation)
    tamper = str(
        validation.get("tampering_risk", "UNKNOWN")
    ).upper()

    fields = _fields(validation)

    try:
        present, total = fields.split("/", 1)
        present = int(present)
        total = int(total)
    except (ValueError, TypeError):
        present = 0
        total = 5

    # Critical failures always reject.
    if not detected:
        return "DOCUMENT_REJECTED"

    if quality == "POOR":
        return "DOCUMENT_REJECTED"

    if tamper in {"HIGH", "CRITICAL"}:
        return "DOCUMENT_REJECTED"

    # Visual validation is the acceptance gate.
    if present >= 3 and quality in {"GOOD", "FAIR"}:
        return "DOCUMENT_VERIFIED_SUCCESSFULLY"

    return "DOCUMENT_REJECTED"


@router.post(
    "/verify",
    response_model=VoterVerificationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Validate a Voter ID",
)
async def verify_voter_id(
    file: UploadFile = File(
        ...,
        description="Voter ID card (JPG/JPEG/PNG/PDF).",
    ),
) -> VoterVerificationResponse:

    started_at = time.perf_counter()

    # ============================================================
    # 1. READ UPLOAD
    # ============================================================

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UPLOAD_READ_FAILED",
                "message": "Unable to read uploaded file.",
            },
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_FILE",
                "message": "Uploaded file is empty.",
            },
        )

    # ============================================================
    # 2. COMMON SECURITY
    # ============================================================

    try:
        security_result = validate_upload(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )

    except MaliciousFileError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "MALICIOUS_FILE",
                "message": (
                    "Suspicious or executable content detected. "
                    "Upload rejected."
                ),
            },
        ) from exc

    except VirusScannerUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "VIRUS_SCAN_UNAVAILABLE",
                "message": "Common document security service is unavailable.",
            },
        ) from exc

    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": str(exc),
            },
        ) from exc

    except InvalidFileSignatureError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE_SIGNATURE",
                "message": str(exc),
            },
        ) from exc

    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": str(exc),
            },
        ) from exc

    except InvalidFileError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE",
                "message": str(exc),
            },
        ) from exc

    content_type = (
        security_result.get("content_type")
        or file.content_type
        or ""
    ).lower().strip()

    # ============================================================
    # 3. FAST VOTER VALIDATION
    # ============================================================
    #
    # Exactly TWO arguments.
    # No OCR/extraction is called by this endpoint.
    # ============================================================

    try:
        result = await run_in_threadpool(
            verify_voter_card,
            file_bytes,
            content_type,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VOTER_VALIDATION_FAILED",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "VOTER_PROCESSING_FAILED",
                "message": "Voter ID validation failed.",
            },
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INVALID_VALIDATION_RESPONSE",
                "message": "Voter ID validation returned an invalid response.",
            },
        )

    validation = result.get("validation") or {}

    if not isinstance(validation, dict):
        validation = {}

    document_detected = bool(
        validation.get("document_detected", False)
    )

    image_quality = _quality(validation)

    tampering_risk = str(
        validation.get("tampering_risk", "UNKNOWN")
    ).upper()

    visual_fields = _fields(validation)

    score = round(
        _safe_float(
            validation.get(
                "score",
                result.get(
                    "score",
                    result.get("final_score", 0),
                ),
            )
        ),
        2,
    )

    # DO NOT use result["decision"].
    decision = _final_decision(validation)

    # ============================================================
    # 4. COMMON SECURITY RESPONSE
    # ============================================================

    security = security_result.get("virus_scan")

    if not isinstance(security, dict):
        security = {
            "safe": True,
            "status": "BASIC_SECURITY_PASS",
        }

    # ============================================================
    # 5. RESPONSE
    # ============================================================

    return VoterVerificationResponse(
        document_type="VOTER_ID",
        decision=decision,
        score=score,
        validation=VoterValidation(
            document_detected=document_detected,
            image_quality=image_quality,
            tampering_risk=tampering_risk,
            visual_fields=visual_fields,
        ),
        security=security,
        processing_time_seconds=round(
            time.perf_counter() - started_at,
            3,
        ),
    )