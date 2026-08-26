"""
FastAPI endpoint for signature verification.

Supported input modes:

    1. upload
    2. draw
    3. capture

The API accepts an image and passes the selected input mode
to the signature validation layer.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

import cv2
import numpy as np

from src.documents.signature.services.signature_validator import (
    validate_signature,
)


# =========================================================
# Router
# =========================================================

router = APIRouter(
    prefix="/signature",
    tags=["Signature Verification"],
)


# =========================================================
# Configuration
# =========================================================

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


ALLOWED_INPUT_MODES = {
    "upload",
    "draw",
    "capture",
}


# =========================================================
# Verify Signature
# =========================================================

@router.post("/verify")
async def verify_signature(
    file: UploadFile = File(...),
    input_mode: str = Form(
        default="upload"
    ),
):
    """
    Verify whether an uploaded image is a valid
    signature candidate.

    Supported input modes:

        upload
        draw
        capture
    """

    # =====================================================
    # 1. Validate upload
    # =====================================================

    if not file:

        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="File name is missing.",
        )

    # =====================================================
    # 2. Validate input mode
    # =====================================================

    normalized_input_mode = (
        input_mode.strip().lower()
    )

    if (
        normalized_input_mode
        not in ALLOWED_INPUT_MODES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid input_mode. "
                "Allowed values are: "
                "upload, draw, capture."
            ),
        )

    # =====================================================
    # 3. Validate content type
    # =====================================================

    if (
        file.content_type
        not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Only JPEG, PNG and WEBP images "
                "are supported."
            ),
        )

    # =====================================================
    # 4. Read bytes
    # =====================================================

    try:

        contents = await file.read()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded file.",
        )

    # =====================================================
    # 5. Validate empty file
    # =====================================================

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # =====================================================
    # 6. Validate file size
    # =====================================================

    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded file is too large. "
                "Maximum allowed size is 5 MB."
            ),
        )

    # =====================================================
    # 7. Decode image
    # =====================================================

    try:

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not decode image.",
        )

    # =====================================================
    # 8. Validate decoded image
    # =====================================================

    if image is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is not a valid image."
            ),
        )

    # =====================================================
    # 9. Run signature validator
    # =====================================================

    try:

        result = validate_signature(
            image=image,
            input_mode=normalized_input_mode,
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Signature verification failed.",
        )

    # =====================================================
    # 10. Build response
    # =====================================================

    response = {

        "document_type": "SIGNATURE",

        "input_mode": (
            normalized_input_mode
        ),

        "decision": (
            result.decision.value
        ),

        "confidence": (
            result.confidence
        ),

        "reason_code": (
            result.reason_code
        ),

        "message": (
            result.message
        ),
    }

    # =====================================================
    # 11. Features
    # =====================================================

    if result.features:

        response["features"] = {

            "image_width": (
                result.features.image_width
            ),

            "image_height": (
                result.features.image_height
            ),

            "foreground_density": (
                result.features.foreground_density
            ),

            "bbox_x": (
                result.features.bbox_x
            ),

            "bbox_y": (
                result.features.bbox_y
            ),

            "bbox_width": (
                result.features.bbox_width
            ),

            "bbox_height": (
                result.features.bbox_height
            ),

            "aspect_ratio": (
                result.features.aspect_ratio
            ),

            "occupancy_ratio": (
                result.features.occupancy_ratio
            ),

            "connected_components": (
                result.features.connected_components
            ),

            "contour_count": (
                result.features.contour_count
            ),

            "largest_contour_area": (
                result.features.largest_contour_area
            ),

            "total_contour_area": (
                result.features.total_contour_area
            ),
        }

    else:

        response["features"] = None

    # =====================================================
    # 12. MobileNet classifier information
    # =====================================================

    if result.classifier:

        response["classifier"] = {

            "predicted_class": (
                result.classifier.predicted_class
            ),

            "signature_probability": (
                result.classifier.signature_probability
            ),

            "non_signature_probability": (
                result.classifier.non_signature_probability
            ),

            "confidence": (
                result.classifier.confidence
            ),

            "is_signature": (
                result.classifier.is_signature
            ),
        }

    else:

        response["classifier"] = None

    # =====================================================
    # 13. Whole-image context
    # =====================================================

    if result.context:

        response["context"] = {

            "foreground_density": (
                result.context.foreground_density
            ),

            "edge_density": (
                result.context.edge_density
            ),

            "connected_components": (
                result.context.connected_components
            ),

            "large_rectangle_count": (
                result.context.large_rectangle_count
            ),

            "largest_rectangle_area_ratio": (
                result.context
                .largest_rectangle_area_ratio
            ),

            "document_like": (
                result.context.document_like
            ),

            "signature_area_ratio": (
                result.context.signature_area_ratio
            ),

            "signature_count": (
                result.context.signature_count
            ),

            "multiple_signatures": (
                result.context.multiple_signatures
            ),
        }

    else:

        response["context"] = None

    # =====================================================
    # 14. Return
    # =====================================================

    return response