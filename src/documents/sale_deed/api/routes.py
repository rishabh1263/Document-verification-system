"""
Sale Deed Upload / Verification API.

Phase 1:
    Fast validation for LOS.

Responsibilities:
    1. Validate upload
    2. Read uploaded bytes
    3. Run common security validation
    4. Persist uploaded file
    5. Run SaleDeedPipeline
    6. Return clean LOS response

OCR / extraction is handled by later phases.

Supported endpoints:
    POST /sale-deed/upload
    POST /sale-deed/verify

Both endpoints execute exactly the same validation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from src.common.security.file_security import (
    validate_upload,
)

from src.documents.sale_deed.pipeline import (
    SaleDeedPipeline,
)


# ======================================================================
# ROUTER
# ======================================================================

router = APIRouter()


# ======================================================================
# SUPPORTED FORMATS
# ======================================================================

SUPPORTED_FORMATS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}


# ======================================================================
# STORAGE
# ======================================================================

UPLOAD_DIRECTORY = Path(
    "src/documents/sale_deed/uploads"
)


# ======================================================================
# INTERNAL SECURITY RESULT NORMALIZER
# ======================================================================

def _security_passed(
    security_result: Any,
) -> bool:
    """
    Normalize the different possible return formats of
    common validate_upload().
    """

    if isinstance(
        security_result,
        bool,
    ):
        return security_result

    if isinstance(
        security_result,
        dict,
    ):

        return bool(
            security_result.get(
                "safe",
                security_result.get(
                    "valid",
                    security_result.get(
                        "passed",
                        True,
                    ),
                ),
            )
        )

    return bool(security_result)


# ======================================================================
# INTERNAL VALIDATION HANDLER
# ======================================================================

async def _process_sale_deed_upload(
    file: UploadFile,
) -> dict[str, Any]:

    started = perf_counter()

    # ================================================================
    # 1. FILE NAME
    # ================================================================

    original_filename = (
        file.filename or ""
    ).strip()

    if not original_filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # Prevent path traversal.
    filename = Path(
        original_filename
    ).name

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_FORMATS:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Unsupported file format.",

                "supported_formats":
                    sorted(
                        SUPPORTED_FORMATS
                    ),
            },
        )

    # ================================================================
    # 2. READ FILE
    # ================================================================

    try:

        contents = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Unable to read uploaded file.",

                "error":
                    str(exc),
            },
        ) from exc

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # ================================================================
    # 3. COMMON SECURITY VALIDATION
    # ================================================================
    #
    # IMPORTANT:
    #
    # validate_upload() is synchronous.
    #
    # DO NOT:
    #
    #     await validate_upload(...)
    #
    # It expects:
    #
    #     file_bytes
    #     filename
    #     content_type
    #
    # ================================================================

    try:

        security_result = validate_upload(
            file_bytes=contents,
            filename=filename,
            content_type=file.content_type,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "File security validation failed.",

                "error":
                    str(exc),
            },
        ) from exc

    security_safe = _security_passed(
        security_result
    )

    # ================================================================
    # 4. SECURITY FAILURE
    # ================================================================

    if not security_safe:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Uploaded file failed security validation.",

                "virus_scan": {

                    "safe":
                        False,

                    "status":
                        "BASIC_SECURITY_FAIL",

                },
            },
        )

    # ================================================================
    # 5. SAVE FILE
    # ================================================================

    try:

        UPLOAD_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved_filename = (
            f"{Path(filename).stem}_"
            f"{uuid4().hex[:10]}"
            f"{extension}"
        )

        saved_path = (
            UPLOAD_DIRECTORY
            /
            saved_filename
        )

        saved_path.write_bytes(
            contents
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Could not store uploaded file.",

                "error":
                    str(exc),
            },
        ) from exc

    # ================================================================
    # 6. SALE DEED PIPELINE
    # ================================================================

    try:

        result = (
            SaleDeedPipeline.verify_document(
                file_path=str(
                    saved_path
                ),
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Sale Deed validation failed.",

                "error":
                    str(exc),
            },
        ) from exc

    # ================================================================
    # 7. SAFETY NORMALIZATION
    # ================================================================

    if not isinstance(
        result,
        dict,
    ):

        result = {}

    # ================================================================
    # 8. BASIC RESPONSE FIELDS
    # ================================================================

    document_type = result.get(
        "document_type",
        "SALE_DEED",
    )

    decision = result.get(
        "decision",
        "DOCUMENT_REVIEW",
    )

    score = result.get(
        "score",
        0,
    )

    # ================================================================
    # 9. VALIDATION
    # ================================================================

    validation = result.get(
        "validation",
        {},
    )

    if not isinstance(
        validation,
        dict,
    ):

        validation = {}

    # Keep the public LOS response clean.
    validation = {

        "document_detected":
            bool(
                validation.get(
                    "document_detected",
                    False,
                )
            ),

        "image_quality":
            validation.get(
                "image_quality",
                "NOT_CHECKED",
            ),

        "tampering_risk":
            validation.get(
                "tampering_risk",
                "NOT_CHECKED",
            ),

        "structural_validation":
            validation.get(
                "structural_validation",
                "NOT_CHECKED",
            ),

    }

    # ================================================================
    # 10. LOS
    # ================================================================

    pipeline_los = result.get(
        "los",
        {},
    )

    if not isinstance(
        pipeline_los,
        dict,
    ):

        pipeline_los = {}

    # ---------------------------------------------------------------
    # Important:
    #
    # A confirmed wrong document should never proceed to OCR.
    #
    # The pipeline is still the authority for document classification.
    # This is only a safety normalization at the API boundary.
    # ---------------------------------------------------------------

    if decision == "DOCUMENT_REJECTED":

        requires_ocr_phase = False
        requires_rcu_review = False

    else:

        requires_ocr_phase = bool(
            pipeline_los.get(
                "requires_ocr_phase",
                True,
            )
        )

        requires_rcu_review = bool(
            pipeline_los.get(
                "requires_rcu_review",
                decision
                ==
                "DOCUMENT_REVIEW",
            )
        )

    # ================================================================
    # 11. PROCESSING TIME
    # ================================================================

    elapsed = round(
        perf_counter()
        -
        started,
        3,
    )

    # ================================================================
    # 12. CLEAN LOS RESPONSE
    # ================================================================

    return {

        "document_type":
            document_type,

        "decision":
            decision,

        "score":
            score,

        "validation":
            validation,

        "virus_scan": {

            "safe":
                security_safe,

            "status":
                (
                    "BASIC_SECURITY_PASS"
                    if security_safe
                    else
                    "BASIC_SECURITY_FAIL"
                ),

        },

        "los": {

            "document_validation":
                pipeline_los.get(
                    "document_validation",
                    decision,
                ),

            "requires_ocr_phase":
                requires_ocr_phase,

            "requires_rcu_review":
                requires_rcu_review,

        },

        "processing_time_seconds":
            elapsed,

    }


# ======================================================================
# UPLOAD ENDPOINT
# ======================================================================

@router.post(
    "/upload",
    tags=["Sale Deed"],
)
async def upload_sale_deed(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    return await _process_sale_deed_upload(
        file
    )


# ======================================================================
# VERIFY ENDPOINT
# ======================================================================

@router.post(
    "/verify",
    tags=["Sale Deed"],
)
async def verify_sale_deed(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    return await _process_sale_deed_upload(
        file
    )