"""
PAN EXTRACTION API ROUTER

Public endpoint:
    POST /pan/extract

This router is intentionally separate from PAN validation.

Validation:
    POST /pan/verify-pan
    -> pan/router.py
    -> validation only

Extraction:
    POST /pan/extract
    -> pan/extraction_router.py
    -> pan_extraction.py
    -> currently extracts ONLY the PAN holder name

The common upload-security gate is still used here so extraction does not
bypass the application's upload security controls.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .pan_extraction import extract_pan_name

from src.common.security.file_security import (
    validate_upload,
    MaliciousFileError,
    VirusScannerUnavailableError,
    UnsupportedFileTypeError,
    InvalidFileSignatureError,
    FileTooLargeError,
    InvalidFileError,
)


# ======================================================================
# ROUTER
# ======================================================================

router = APIRouter(
    prefix="/pan",
    tags=["PAN Extraction"],
)


# ======================================================================
# RESPONSE MODELS
# ======================================================================

class PanExtractedData(BaseModel):
    """
    Current extraction contract.

    Only name is implemented for Phase 1.
    """

    name: str | None = Field(
        default=None,
        description="PAN holder name.",
    )


class PanExtractionResponse(BaseModel):
    document_type: str = "PAN"

    extracted_data: PanExtractedData

    confidence: float = Field(
        description="Confidence of the extracted name.",
    )

    method: str = Field(
        description="Extraction method used.",
    )

    ocr_used: bool = Field(
        description="Whether OCR was used.",
    )

    processing_time_seconds: float = Field(
        description="Total extraction processing time.",
    )

    virus_scan: dict[str, Any] = Field(
        description="Result of the common upload-security gate.",
    )


class ErrorResponse(BaseModel):
    detail: Any


# ======================================================================
# ENDPOINT
# ======================================================================

@router.post(
    "/extract",
    response_model=PanExtractionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Extract PAN holder name",
    description=(
        "PAN extraction endpoint. Currently extracts only the PAN holder "
        "name. Validation and authenticity decisions are handled by the "
        "separate /pan/verify-pan endpoint."
    ),
)
async def extract_pan(
    file: UploadFile = File(
        ...,
        description="PAN card image (JPG/JPEG/PNG) or PDF.",
    ),
) -> PanExtractionResponse:

    # ==================================================================
    # 1. READ UPLOAD
    # ==================================================================

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_READ_FAILED",
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

    # ==================================================================
    # 2. COMMON SECURITY GATE
    # ==================================================================
    #
    # This is NOT document validation.
    #
    # It only protects the extraction API from unsupported, malformed,
    # oversized or malicious uploads.
    #
    # Supported:
    #     JPG
    #     JPEG
    #     PNG
    #     PDF
    # ==================================================================

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
                    "Malicious content detected. Upload rejected."
                ),
            },
        ) from exc

    except VirusScannerUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "VIRUS_SCAN_UNAVAILABLE",
                "message": (
                    "Document cannot be processed because the common "
                    "antivirus scanner is unavailable."
                ),
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

    # Use the normalized type returned by the common security layer.
    content_type = security_result.get(
        "content_type"
    )

    # ==================================================================
    # 3. EXTRACTION ONLY
    # ==================================================================
    #
    # No:
    #     - PAN validation
    #     - document score
    #     - tamper decision
    #     - face validation
    #     - authenticity decision
    #
    # Only:
    #     file -> name
    # ==================================================================

    try:
        result = await run_in_threadpool(
            extract_pan_name,
            file_bytes,
            content_type,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "PAN_EXTRACTION_FAILED",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PAN_EXTRACTION_PROCESSING_FAILED",
                "message": "PAN name extraction failed.",
            },
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INVALID_EXTRACTION_RESPONSE",
                "message": (
                    "PAN extraction returned an invalid response."
                ),
            },
        )

    # ==================================================================
    # 4. CLEAN RESPONSE
    # ==================================================================

    return PanExtractionResponse(
        document_type="PAN",
        extracted_data=PanExtractedData(
            name=result.get("name"),
        ),
        confidence=round(
            float(
                result.get(
                    "confidence",
                    0.0,
                )
            ),
            4,
        ),
        method=str(
            result.get(
                "method",
                "UNKNOWN",
            )
        ),
        ocr_used=bool(
            result.get(
                "ocr_used",
                False,
            )
        ),
        processing_time_seconds=round(
            float(
                result.get(
                    "processing_time_seconds",
                    0.0,
                )
            ),
            3,
        ),
        virus_scan={
            "safe": True,
            "status": "BASIC_SECURITY_PASS",
        },
    )