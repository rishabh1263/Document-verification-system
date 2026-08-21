"""
CIBIL Verification API.

Endpoint:
    POST /cibil/verify

Validation order:
    1. File security / PDF structure
    2. Native/scanned detection
    3. Strong document classification
    4. Tampering / structural validation
    5. Existing CIBIL credit analysis

OCR is intentionally disabled.
"""

from pathlib import Path
from typing import Optional
from time import perf_counter

import fitz

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .cibil_verification import verify_cibil_document


router = APIRouter(
    prefix="/cibil",
    tags=["CIBIL Verification"],
)


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
}

ALLOWED_EXTENSION = ".pdf"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


class ValidationResponse(BaseModel):
    document_detected: bool
    image_quality: str
    tampering_risk: str
    structural_validation: str
    processing_time_ms: float


class CibilVerificationResponse(BaseModel):
    # Standard document-verification contract
    document_type: str = "CIBIL"
    decision: str
    score: int = 0
    validation: ValidationResponse

    # Existing CIBIL business output is retained
    verified: bool = False
    overall_status: Optional[str] = None
    applicant: Optional[dict] = None
    score_analysis: Optional[dict] = None
    account_summary: Optional[dict] = None
    risk_summary: Optional[dict] = None
    decision_reason: Optional[str] = None
    report_freshness: Optional[dict] = None
    credit_vintage: Optional[dict] = None
    utilisation: Optional[dict] = None
    debt_analysis: Optional[dict] = None
    enquiries: Optional[dict] = None
    document_type_check: Optional[dict] = None
    raw_result: Optional[dict] = None


class ErrorResponse(BaseModel):
    detail: str


def _safe_pdf_structure(file_bytes: bytes) -> tuple[bool, str]:
    """Fast PDF structural check without extracting all content."""
    try:
        if not file_bytes.startswith(b"%PDF"):
            return False, "FAIL"

        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            if len(pdf) <= 0:
                return False, "FAIL"

            # Force page access. This catches a number of malformed PDFs
            # without doing expensive rendering.
            _ = pdf[0].rect

        return True, "PASS"

    except Exception:
        return False, "FAIL"


def _fast_pdf_tampering_risk(file_bytes: bytes) -> str:
    """
    Fast PDF-level authenticity heuristic.

    This is deliberately conservative: normal PDF metadata is not treated
    as tampering. Active-content features that are unusual for a credit
    report are treated as MEDIUM and therefore rejected by this API.
    Malformed PDFs are already HIGH through the structural check.
    """
    try:
        raw = file_bytes[:8 * 1024 * 1024]
        upper = raw.upper()

        suspicious = (
            b"/JAVASCRIPT" in upper
            or b"/JS " in upper
            or b"/JS/" in upper
            or b"/LAUNCH" in upper
            or b"/SUBMITFORM" in upper
            or b"/RICHMEDIA" in upper
        )

        if suspicious:
            return "MEDIUM"

        return "LOW"
    except Exception:
        return "MEDIUM"


def _tampering_from_result(result: dict, file_bytes: bytes | None = None) -> str:
    """
    Current CIBIL core performs PDF structural validation. Keep this
    normalized to the same contract used by the other document APIs.

    Future dedicated authenticity engine output can be plugged into the
    result here without changing the API contract.
    """
    risk = str(
        result.get("tampering_risk")
        or result.get("file_check", {}).get("tampering_risk")
        or ""
    ).upper()

    if not risk and file_bytes is not None:
        risk = _fast_pdf_tampering_risk(file_bytes)

    if not risk:
        risk = "LOW"

    if risk in {"HIGH", "CRITICAL"}:
        return "HIGH"

    if risk == "MEDIUM":
        return "MEDIUM"

    return "LOW"


def _image_quality(result: dict) -> str:
    # Native CIBIL reports are text PDFs. We do not render pages merely
    # to calculate a cosmetic image score because that would slow the
    # native path. Scanned documents are handled as REVIEW because OCR
    # is disabled.
    if not result.get("native_text", False):
        return "NOT_CHECKED"

    return "GOOD"


def _decision_from_result(result: dict) -> str:
    status = str(
        result.get("overall_status") or "REJECTED"
    ).upper()

    if status == "APPROVED":
        return "DOCUMENT_VERIFIED"

    if status in {
        "APPROVED_WITH_NOTES",
        "MANUAL_REVIEW",
        "REVIEW",
    }:
        return "DOCUMENT_REVIEW"

    return "DOCUMENT_REJECTED"


def _score_from_result(result: dict) -> int:
    predicted = (
        result.get("score_analysis", {})
        .get("predicted", {})
    )

    value = predicted.get("predicted_score")

    if value is None:
        value = result.get("score_analysis", {}).get("score")

    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0

    # Preserve special CIBIL values (-1, 0, 1-5) as zero for the generic
    # document score. The detailed CIBIL result still contains the actual
    # bureau score.
    if value < 0 or value > 100:
        return 0

    return value


@router.post(
    "/verify",
    response_model=CibilVerificationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Verify a CIBIL report PDF",
)
async def verify_cibil(
    file: UploadFile = File(
        ...,
        description="CIBIL/credit report PDF",
    ),
    applicant_pan: Optional[str] = Form(
        default=None,
        description="Optional applicant PAN for validation",
    ),
) -> CibilVerificationResponse:

    started = perf_counter()

    # ============================================================
    # 1. FILE SECURITY
    # ============================================================

    filename = Path(
        str(file.filename or "")
    ).name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(filename).suffix.lower()

    if extension != ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=400,
            detail="Upload only PDF files.",
        )

    content_type = (
        file.content_type or ""
    ).lower().strip()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                "Only PDF CIBIL reports are supported."
            ),
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds 50 MB.",
        )

    # ============================================================
    # 2. FAST STRUCTURAL CHECK
    # ============================================================

    structurally_valid, structural_status = _safe_pdf_structure(
        file_bytes
    )

    if not structurally_valid:
        elapsed = round(
            (perf_counter() - started) * 1000,
            2,
        )

        return CibilVerificationResponse(
            document_type="CIBIL",
            decision="DOCUMENT_REJECTED",
            score=0,
            validation=ValidationResponse(
                document_detected=False,
                image_quality="NOT_CHECKED",
                tampering_risk="HIGH",
                structural_validation="FAIL",
                processing_time_ms=elapsed,
            ),
            verified=False,
            overall_status="REJECTED",
            decision_reason="Invalid or unreadable PDF structure.",
        )

    # ============================================================
    # 3. EXISTING CIBIL ENGINE + CLASSIFICATION
    # ============================================================

    try:
        result = await run_in_threadpool(
            verify_cibil_document,
            file_bytes,
            applicant_pan,
            filename,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CIBIL verification failed: {exc}",
        ) from exc

    # ============================================================
    # 4. STANDARD DECISION
    # ============================================================

    decision = _decision_from_result(result)

    detected_type = (
        result.get("document_type_check", {})
        .get("detected", "UNKNOWN")
    )

    # A known wrong document is always rejected.
    if detected_type != "CIBIL":
        decision = "DOCUMENT_REJECTED"

    # Scanned PDF without OCR is review, not false verification.
    if not result.get("native_text", False):
        decision = "DOCUMENT_REVIEW"

    # ============================================================
    # 5. VALIDATION OUTPUT
    # ============================================================

    tampering_risk = _tampering_from_result(result, file_bytes)

    # Medium/high authenticity risk must not be approved.
    if tampering_risk == "HIGH":
        decision = "DOCUMENT_REJECTED"
    elif tampering_risk == "MEDIUM":
        decision = "DOCUMENT_REJECTED"

    elapsed = result.get("processing_time_ms")

    if elapsed is None:
        elapsed = round(
            (perf_counter() - started) * 1000,
            2,
        )

    score = _score_from_result(result)

    return CibilVerificationResponse(
        document_type="CIBIL",
        decision=decision,
        score=score,
        validation=ValidationResponse(
            document_detected=(
                detected_type == "CIBIL"
                and bool(result.get("native_text", False))
            ),
            image_quality=_image_quality(result),
            tampering_risk=tampering_risk,
            structural_validation=structural_status,
            processing_time_ms=float(elapsed),
        ),
        verified=(decision == "DOCUMENT_VERIFIED"),
        overall_status=result.get("overall_status"),
        applicant=result.get("applicant"),
        score_analysis=result.get("score_analysis"),
        account_summary=result.get("account_summary"),
        risk_summary=result.get("risk_summary"),
        decision_reason=result.get("decision_reason"),
        report_freshness=result.get("freshness_check"),
        credit_vintage=(
            result.get("credit_summary", {})
            .get("credit_vintage")
        ),
        utilisation=(
            result.get("credit_summary", {})
            .get("credit_utilization")
        ),
        debt_analysis=(
            result.get("credit_summary", {})
            .get("debt_trend")
        ),
        enquiries=result.get("enquiry_analysis"),
        document_type_check=result.get("document_type_check"),
        raw_result=result,
    )