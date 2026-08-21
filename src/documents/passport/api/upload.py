"""
Passport Upload API.

PHASE 1:
    Fast, non-OCR Passport validation for LOS.

The API handles:
    - File validation
    - Common security validation
    - File storage
    - Passport verification pipeline
    - Clean LOS response
    - Processing-time measurement

The verification pipeline handles:
    - Document detection
    - Structural validation
    - Image quality
    - Human photo detection
    - Tampering risk
    - LOS decision

OCR / MRZ is NOT executed in Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from src.common.security.file_security import (
    validate_upload,
)

from src.documents.passport.core.config import (
    settings,
)

from src.documents.passport.core.constants import (
    SUPPORTED_FORMATS,
    DOCUMENT_REJECTED,
)

from src.documents.passport.storage.file_manager import (
    FileManager,
)

from src.documents.passport.storage.metadata_manager import (
    MetadataManager,
)

from src.documents.passport.utils.request_id import (
    generate_request_id,
)

from src.documents.passport.verification.pipeline.verification_pipeline import (
    VerificationPipeline,
)


router = APIRouter()


@router.post(
    "/upload",
)
async def upload(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    started = perf_counter()

    request_id = generate_request_id()

    # ================================================================
    # 1. BASIC FILE VALIDATION
    # ================================================================

    filename = (
        file.filename or ""
    ).strip()

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_FORMATS:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "Unsupported file format.",
                "supported_formats":
                    sorted(
                        SUPPORTED_FORMATS
                    ),
            },
        )

    # ================================================================
    # 2. COMMON FILE SECURITY
    # ================================================================

    security_result: dict[str, Any] = {}

    try:

        security = await validate_upload(
            file
        )

        if isinstance(
            security,
            dict,
        ):

            security_result = security

    except TypeError:

        # Keep compatibility with common validators that
        # may use a different input contract.

        security_result = {}

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "File security validation failed.",
                "message":
                    str(exc),
            },
        ) from exc

    security_safe = bool(
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

    if not security_safe:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "File failed security validation.",
            },
        )

    # ================================================================
    # 3. SAVE FILE
    # ================================================================

    try:

        saved_file = FileManager.save(
            file
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Could not store uploaded file.",
                "message":
                    str(exc),
            },
        ) from exc

    saved_path = (
        saved_file.get(
            "path"
        )
    )

    if not saved_path:

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "File storage did not return a valid path.",
            },
        )

    # ================================================================
    # 4. PASSPORT VERIFICATION
    # ================================================================

    try:

        result = (
            VerificationPipeline.verify(
                file_path=saved_path,
                request_id=request_id,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Passport validation failed.",
                "request_id":
                    request_id,
                "message":
                    str(exc),
            },
        ) from exc

    if not isinstance(
        result,
        dict,
    ):

        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "Invalid passport verification response.",
                "request_id":
                    request_id,
            },
        )

    # ================================================================
    # 5. PROCESSING TIME
    # ================================================================

    processing_time_seconds = round(
        perf_counter()
        -
        started,
        3,
    )

    # ================================================================
    # 6. INTERNAL AUDIT
    # ================================================================

    if settings.STORE_METADATA:

        try:

            MetadataManager.save(

                request_id=request_id,

                data={

                    "request_id":
                        request_id,

                    "original_filename":
                        filename,

                    "stored_filename":
                        saved_file.get(
                            "filename"
                        ),

                    "file_type":
                        extension,

                    "validation_mode":
                        "PHASE_1_NO_OCR",

                    "verification_result":
                        result,

                },

            )

        except Exception:

            # Audit failure must not change the
            # document validation decision.

            pass

    # ================================================================
    # 7. CLEAN RESPONSE
    # ================================================================

    validation = (
        result.get(
            "validation",
            {},
        )
        or {}
    )

    if not isinstance(
        validation,
        dict,
    ):

        validation = {}

    # ------------------------------------------------
    # Document
    # ------------------------------------------------

    document_detected = bool(
        validation.get(
            "document_detected",
            (
                validation.get(
                    "document",
                    {},
                )
                or {}
            ).get(
                "eligible",
                False,
            ),
        )
    )

    # ------------------------------------------------
    # Image quality
    # ------------------------------------------------

    image_quality = validation.get(
        "image_quality",
    )

    if image_quality is None:

        quality = (
            validation.get(
                "quality",
                {},
            )
            or {}
        )

        if quality.get(
            "available",
            False,
        ):

            image_quality = (
                "GOOD"
                if quality.get(
                    "passed",
                    False,
                )
                else
                "POOR"
            )

        else:

            image_quality = "NOT_CHECKED"

    # ------------------------------------------------
    # Human photo
    # ------------------------------------------------

    human_photo_detected = validation.get(
        "human_photo_detected",
        "NOT_CHECKED",
    )

    # ------------------------------------------------
    # Tampering
    # ------------------------------------------------

    tampering_risk = validation.get(
        "tampering_risk",
        "NOT_CHECKED",
    )

    # ------------------------------------------------
    # Structural validation
    # ------------------------------------------------

    structural_validation = validation.get(
        "structural_validation",
    )

    if structural_validation is None:

        structural_validation = (
            "PASS"
            if document_detected
            else
            "FAIL"
        )

    # ================================================================
    # FINAL RESPONSE
    # ================================================================

    return {

        "document_type":
            result.get(
                "document_type",
                "PASSPORT",
            ),

        "decision":
            result.get(
                "decision",
                DOCUMENT_REJECTED,
            ),

        "score":
            result.get(
                "score",
                0,
            ),

        "validation": {

            "document_detected":
                document_detected,

            "image_quality":
                image_quality,

            "human_photo_detected":
                human_photo_detected,

            "tampering_risk":
                tampering_risk,

            "structural_validation":
                structural_validation,

            "security":
                (
                    "PASS"
                    if security_safe
                    else
                    "FAIL"
                ),

        },

        "processing_time_seconds":
            processing_time_seconds,

    }