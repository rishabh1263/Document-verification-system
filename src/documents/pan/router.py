"""
PAN Validation API Router.

Production flow
---------------
1. Read upload.
2. Run the COMMON upload-security gate.
3. Validate the PAN visually.
4. Return a minimal business response.

IMPORTANT
---------
This endpoint is VALIDATION ONLY.

It intentionally does NOT run:
    - OCR
    - PAN number extraction
    - Name extraction
    - Father's-name extraction
    - DOB extraction
    - Government/source-of-truth verification

The extraction implementation can remain in pan_verification.py and be
enabled later without changing the common upload-security architecture.

Supported files are controlled centrally by:
    src.common.security.file_security

Allowed:
    JPG
    JPEG
    PNG
    PDF
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pymupdf

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .pan_verification import (
    PanVerificationError,
    verify_pan_card,
)

from src.common.security.file_security import (
    validate_upload,
    MaliciousFileError,
    VirusScannerUnavailableError,
    UnsupportedFileTypeError,
    InvalidFileSignatureError,
    FileTooLargeError,
    InvalidFileError,
)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/pan",
    tags=["PAN Verification"],
)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PanValidation(BaseModel):
    """Minimal PAN local-validation result."""

    document_detected: bool = Field(
        description="Whether the uploaded document was detected as PAN."
    )

    image_quality: str = Field(
        description="High-level image quality."
    )

    tampering_risk: str = Field(
        description="High-level local tampering risk."
    )

    visual_fields: str = Field(
        description="Detected PAN visual regions, for example 5/5."
    )

    photo_present: bool = Field(
        description="Whether the expected PAN photo region is present."
    )

    face_detected: bool = Field(
        description="Whether a face was detected in the expected photo region."
    )


class PanValidationResponse(BaseModel):
    """
    Clean business-facing PAN validation response.

    No OCR/extracted fields are returned because this endpoint is validation
    only.
    """

    document_type: str = "PAN"

    decision: str = Field(
        description=(
            "DOCUMENT_VERIFIED_SUCCESSFULLY or DOCUMENT_REJECTED."
        )
    )

    score: float = Field(
        description="Local PAN validation score."
    )

    validation: PanValidation

    virus_scan: dict[str, Any] = Field(
        description="Result of the common upload-security/virus scan."
    )

    processing_time_seconds: float = Field(
        description="Total PAN validation processing time in seconds."
    )


class ErrorResponse(BaseModel):
    detail: Any


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _clean_decision(result: dict[str, Any]) -> str:
    """
    AUTHORITATIVE PAN DECISION.

    VERIFIED is allowed ONLY when every mandatory PAN validation passes.

    Mandatory:
        document_detected == True
        image_quality in GOOD/FAIR
        tampering_risk == LOW
        visual_fields == complete
        photo_present == True
        face_detected == True

    No previous/internal decision is allowed to override a failed gate.
    """

    validation = result.get(
        "validation",
        {},
    )

    if not isinstance(validation, dict):
        validation = {}

    document_detected = bool(
        validation.get(
            "document_detected",
            False,
        )
    )

    quality = str(
        validation.get(
            "image_quality",
            "UNKNOWN",
        )
    ).strip().upper()

    tamper = str(
        validation.get(
            "tampering_risk",
            "UNKNOWN",
        )
    ).strip().upper()

    visual_fields = str(
        validation.get(
            "visual_fields",
            "0/0",
        )
    ).strip()

    photo_present = bool(
        validation.get(
            "photo_present",
            False,
        )
    )

    # --------------------------------------------------------------
    # FACE IS MANDATORY
    # --------------------------------------------------------------

    face_detected = bool(
        validation.get(
            "face_detected",
            False,
        )
    )

    # STRICT PAN IDENTITY IS MANDATORY.
    #
    # The underlying validator must prove this is visually PAN-like,
    # not merely a landscape document containing a face.
    pan_identity = bool(
        validation.get(
            "pan_identity",
            False,
        )
    )

    security_feature = bool(
        validation.get(
            "security_feature",
            False,
        )
    )

    # --------------------------------------------------------------
    # VISUAL FIELD CHECK
    # --------------------------------------------------------------

    try:
        present, total = visual_fields.split(
            "/",
            1,
        )

        fields_ok = (
            int(present)
            == int(total)
            and
            int(total) > 0
        )

    except (
        ValueError,
        TypeError,
    ):
        fields_ok = False

    # --------------------------------------------------------------
    # AUTHORITATIVE AND GATE
    # --------------------------------------------------------------

    validator_verified = bool(
        result.get(
            "verified",
            False,
        )
    )

    all_tests_pass = (
        validator_verified
        and document_detected
        and quality in {
            "GOOD",
            "FAIR",
        }
        and tamper == "LOW"
        and fields_ok
        and photo_present
        and face_detected
        and pan_identity
        and security_feature
    )

    if all_tests_pass:
        return (
            "DOCUMENT_VERIFIED_SUCCESSFULLY"
        )

    return "DOCUMENT_REJECTED"


def _build_clean_response(
    result: dict[str, Any],
) -> dict[str, bool]:
    """
    Minimal PAN business response.

    TRUE is returned only when ALL PAN validation gates pass.

    In particular:
        face_detected=False -> verified=False
    """

    decision = _clean_decision(
        result
    )

    return {
        "verified": (
            decision
            ==
            "DOCUMENT_VERIFIED_SUCCESSFULLY"
        )
    }


# ============================================================================
# FAST WRONG-DOCUMENT GEOMETRY GATE
# ============================================================================

def _pan_input_geometry_ok(
    file_bytes: bytes,
    content_type: str | None,
) -> bool:
    """
    Fast PAN input gate.

    IMPORTANT: do not measure the whole camera frame only. A user may upload
    a phone photograph where the PAN card occupies only part of a 4:3/16:9
    frame. In that case the full-frame ratio can be > 1.90 even though the
    card itself has the correct PAN-card geometry.

    This remains only a defense-in-depth gate. The authoritative PAN
    validator performs the actual PAN-specific visual/security checks.
    """
    if not file_bytes:
        return False

    normalized = (
        content_type.lower().strip()
        if content_type
        else ""
    )

    try:
        if normalized == "application/pdf" or file_bytes[:4] == b"%PDF":
            document = pymupdf.open(
                stream=file_bytes,
                filetype="pdf",
            )
            try:
                # Do not reject scanned PDFs just because the PDF page is A4
                # or portrait. The PAN validator handles the rendered page.
                return document.page_count > 0
            finally:
                document.close()

        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return False

        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            return False

        full_ratio = w / h
        if 1.30 <= full_ratio <= 1.90:
            return True

        # Phone-photo fallback: detect a large light/low-saturation card
        # against the surrounding scene. This is deliberately lightweight and
        # runs on a downscaled image.
        scale = min(1.0, 1100.0 / max(w, h))
        work = image if scale >= 1.0 else cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

        wh, ww = work.shape[:2]
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 45], dtype=np.uint8),
            np.array([179, 115, 255], dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = float(ww * wh)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.08:
                continue

            _, _, bw, bh = cv2.boundingRect(contour)
            if bw < 160 or bh < 90:
                continue

            ratio = bw / max(float(bh), 1.0)
            if 1.30 <= ratio <= 1.90:
                return True

        return False

    except Exception:
        return False

# ============================================================================

@router.post(
    "/verify-pan",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid upload.",
        },
        422: {
            "model": ErrorResponse,
            "description": "PAN validation could not be completed.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Required common security service unavailable.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Unexpected validation error.",
        },
    },
    summary="Validate a PAN document",
    description=(
        "Fast local PAN document validation. "
        "OCR and extraction are intentionally disabled."
    ),
)
async def verify_pan(
    file: UploadFile = File(
        ...,
        description="PAN card image (JPG/JPEG/PNG) or PDF.",
    ),
) -> dict[str, bool]:

    print("\n========== PAN REQUEST DEBUG ==========")
    print("ENDPOINT: /pan/verify-pan")
    print("UPLOAD NAME:", file.filename)
    print("UPLOAD CONTENT TYPE:", file.content_type)

    # ========================================================================
    # 1. READ UPLOAD
    # ========================================================================

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_FILE",
                "message": "Uploaded file is empty.",
            },
        )

    # ========================================================================
    # 2. COMMON SECURITY GATE
    # ========================================================================
    #
    # This is shared by PAN, Driving Licence, Voter ID, Passport, etc.
    #
    # No PAN-specific image decoding, validation, OCR, extraction or tamper
    # processing happens before this gate.
    #
    # The common module controls:
    #     - file size
    #     - extension
    #     - MIME type
    #     - magic-byte/signature
    #     - malware/virus scan
    #
    # Supported formats:
    #     JPG / JPEG / PNG / PDF
    # ========================================================================

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
                "message": "Malicious content detected. Upload rejected.",
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

    # Use the common security layer's normalized content type.
    content_type = security_result.get(
        "content_type"
    )

    # Defense-in-depth wrong-document gate.
    # This specifically catches portrait passbooks/images before the
    # PAN visual validator gets a chance to score generic page structure.
    geometry_ok = _pan_input_geometry_ok(
        file_bytes,
        content_type,
    )

    print("NORMALIZED CONTENT TYPE:", content_type)
    print("FILE SIZE BYTES:", len(file_bytes))
    print("PAN GEOMETRY OK:", geometry_ok)

    if not geometry_ok:
        print("FINAL VERIFIED: False")
        print("REASON: PAN GEOMETRY GATE FAILED")
        print("====================================")
        return {
            "verified": False
        }

    # ========================================================================
    # 3. PAN VALIDATION ONLY
    # ========================================================================
    #
    # verify_pan_card() is the validation-only implementation.
    #
    # It performs:
    #     - image/PDF handling
    #     - image quality
    #     - visual PAN-region validation
    #     - photo-region detection
    #     - face detection
    #     - tamper-risk analysis
    #     - score
    #     - processing time
    #
    # It does NOT perform OCR/extraction.
    # ========================================================================

    try:
        result = await run_in_threadpool(
            verify_pan_card,
            file_bytes,
            content_type=content_type,
        )

    except PanVerificationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "PAN_VALIDATION_FAILED",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PAN_PROCESSING_FAILED",
                "message": (str(exc)),
            },
        ) from exc

    # ========================================================================
    # 4. DEBUG + CLEAN FINAL RESPONSE
    # ========================================================================

    print("---------- RAW VALIDATOR RESULT ----------")
    print(result)
    print("------------------------------------------")

    validation = result.get(
        "validation",
        {},
    )

    if not isinstance(validation, dict):
        validation = {}

    debug_gates = {
        "validator_verified": bool(result.get("verified", False)),
        "document_detected": bool(
            validation.get("document_detected", False)
        ),
        "image_quality": validation.get(
            "image_quality",
            "MISSING",
        ),
        "tampering_risk": validation.get(
            "tampering_risk",
            "MISSING",
        ),
        "visual_fields": validation.get(
            "visual_fields",
            "MISSING",
        ),
        "photo_present": bool(
            validation.get("photo_present", False)
        ),
        "face_detected": bool(
            validation.get("face_detected", False)
        ),
        "pan_identity": bool(
            validation.get("pan_identity", False)
        ),
        "security_feature": bool(
            validation.get("security_feature", False)
        ),
    }

    print("---------- AUTHORITATIVE GATES ----------")
    for gate_name, gate_value in debug_gates.items():
        print(f"{gate_name}: {gate_value}")
    print("------------------------------------------")

    try:
        clean_response = _build_clean_response(
            result
        )

        print("---------- FINAL API RESPONSE -----------")
        print(clean_response)
        print("==========================================")

        return clean_response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RESPONSE_FORMAT_ERROR",
                "message": "Failed to format PAN validation response.",
            },
        ) from exc


# ============================================================================
# CONTRACT TEST
# ============================================================================

def router_contract_test() -> dict[str, Any]:
    """
    Lightweight static contract test.

    Confirms that this endpoint is validation-only and does not expose
    extraction fields.
    """

    response_fields = set(
        PanValidationResponse.model_fields.keys()
    )

    validation_fields = set(
        PanValidation.model_fields.keys()
    )

    expected_response_fields = {
        "document_type",
        "decision",
        "score",
        "validation",
        "virus_scan",
        "processing_time_seconds",
    }

    expected_validation_fields = {
        "document_detected",
        "image_quality",
        "tampering_risk",
        "visual_fields",
        "photo_present",
        "face_detected",
        "pan_identity",
        "security_feature",
    }

    return {
        "passed": (
            response_fields == expected_response_fields
            and validation_fields == expected_validation_fields
        ),
        "ocr_used": False,
        "extraction_used": False,
        "common_security_used": True,
        "supported_files": [
            "JPG",
            "JPEG",
            "PNG",
            "PDF",
        ],
    }

# Authoritative router version.
PAN_ROUTER_VERSION = "PAN-AUTHORITATIVE-V3-DEBUG"

print("========== PAN ROUTER DEBUG LOADED ==========")
print("PAN ROUTER FILE:", __file__)
print("PAN ROUTER VERSION:", PAN_ROUTER_VERSION)
print("VERIFY FUNCTION:", verify_pan_card)
print(
    "VERIFY FUNCTION FILE:",
    getattr(
        getattr(verify_pan_card, "__code__", None),
        "co_filename",
        "UNKNOWN",
    ),
)
print(
    "VERIFY FUNCTION LINE:",
    getattr(
        getattr(verify_pan_card, "__code__", None),
        "co_firstlineno",
        "UNKNOWN",
    ),
)
print("============================================")

