"""
CRIF Verification API.

Single endpoint:
    POST /crif/verify
"""

from typing import Optional, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .crif_verification import verify_crif_document


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/crif",
    tags=["CRIF Verification"],
)


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
}


# ============================================================
# RESPONSE MODEL
# ============================================================

class CrifVerificationResponse(BaseModel):
    verified: bool
    decision: str
    final_score: Optional[int] = None

    overall_status: Optional[str] = None

    applicant: Optional[dict[str, Any]] = None

    score_analysis: Optional[dict[str, Any]] = None

    account_summary: Optional[dict[str, Any]] = None

    risk_summary: Optional[dict[str, Any]] = None

    decision_reason: Optional[str] = None

    report_freshness: Optional[dict[str, Any]] = None

    credit_vintage: Optional[dict[str, Any]] = None

    utilisation: Optional[dict[str, Any]] = None

    debt_analysis: Optional[dict[str, Any]] = None

    enquiries: Optional[dict[str, Any]] = None

    raw_result: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    detail: str


# ============================================================
# VERIFY CRIF
# ============================================================

@router.post(
    "/verify",
    response_model=CrifVerificationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Verify a CRIF credit report PDF",
)
async def verify_crif(
    file: UploadFile = File(
        ...,
        description="CRIF credit report PDF",
    ),
    applicant_pan: Optional[str] = Form(
        default=None,
        description="Optional applicant PAN for identity validation",
    ),
) -> CrifVerificationResponse:

    # --------------------------------------------------------
    # FILE TYPE
    # --------------------------------------------------------

    content_type = (
        file.content_type or ""
    ).lower().strip()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: "
                f"{file.content_type}. "
                "Only PDF CRIF reports are supported."
            ),
        )

    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    try:
        result = await run_in_threadpool(
            verify_crif_document,
            file_bytes,
            applicant_pan=applicant_pan,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CRIF verification failed: {exc}",
        ) from exc

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = str(
        result.get("decision")
        or result.get("overall_status")
        or "REJECTED"
    ).upper()

    verified = decision == "APPROVED"

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return CrifVerificationResponse(
        verified=verified,
        decision=decision,

        final_score=result.get(
            "final_score"
        ),

        overall_status=result.get(
            "overall_status"
        ),

        applicant=result.get(
            "applicant"
        ),

        score_analysis=result.get(
            "score_analysis"
        ),

        account_summary=result.get(
            "account_summary"
        ),

        risk_summary=result.get(
            "risk_summary"
        ),

        decision_reason=result.get(
            "decision_reason"
        ),

        report_freshness=result.get(
            "report_freshness"
        ),

        credit_vintage=result.get(
            "credit_vintage"
        ),

        utilisation=result.get(
            "utilisation"
        ),

        debt_analysis=result.get(
            "debt_analysis"
        ),

        enquiries=result.get(
            "enquiries"
        ),

        raw_result=result,
    )