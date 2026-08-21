"""
Voter ID local validation service.

VALIDATION-ONLY VERSION
-----------------------
The public verify_voter_card() path does NOT initialize or call OCR.

OCR/extraction code can be added later as a separate extraction stage.
Do not import EasyOCR/PaddleOCR in this validation module.

The validator uses fast OpenCV/Numpy visual signals:
- image decode
- card-like aspect ratio
- edge density
- contrast
- visual regions
- lightweight tamper signal

This is local document validation, not government/source-of-truth verification.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DOCUMENT_PASS = "DOCUMENT_PASS"
DOCUMENT_REJECT = "DOCUMENT_REJECT"
DOCUMENT_SUSPICIOUS = "DOCUMENT_SUSPICIOUS"


def _decode_image(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Unable to decode the uploaded image."
        )

    return image


def _image_quality(image: np.ndarray) -> tuple[str, float, dict[str, Any]]:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape[:2]

    if h < 100 or w < 100:
        return (
            "POOR",
            25.0,
            {
                "available": True,
                "resolution_ok": False,
                "width": w,
                "height": h,
            },
        )

    blur = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    contrast = float(
        np.std(gray)
    )

    brightness = float(
        np.mean(gray)
    )

    # Resolution is a warning signal, not an automatic rejection for
    # a small but otherwise clear card image.
    resolution_ok = (
        h >= 120
        and w >= 100
    )

    if (
        not resolution_ok
        or contrast < 8
    ):
        quality = "POOR"
    elif (
        blur < 20
        or contrast < 15
    ):
        quality = "FAIR"
    else:
        quality = "GOOD"

    if quality == "GOOD":
        score = 90.0
    elif quality == "FAIR":
        score = 65.0
    else:
        score = 30.0

    # Reward stronger blur/contrast without making the calculation expensive.
    score += min(
        5.0,
        blur / 500.0,
    )

    score += min(
        5.0,
        contrast / 20.0,
    )

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    return (
        quality,
        score,
        {
            "available": True,
            "resolution_ok": resolution_ok,
            "width": w,
            "height": h,
            "blur_score": round(blur, 2),
            "contrast": round(contrast, 2),
            "brightness": round(brightness, 2),
            "quality_score": round(score, 2),
        },
    )


def _visual_regions(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Fast structural Voter ID detector.

    This deliberately does not attempt OCR.

    Five generic visual regions are used:
        1. upper text/header
        2. central text/body
        3. lower text/body
        4. photo/portrait area
        5. lower machine/barcode-like area
    """

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    ratio = (
        w / float(
            max(
                1,
                h,
            )
        )
    )

    portrait = (
        0.35 <= ratio <= 0.95
    )

    landscape = (
        1.05 <= ratio <= 2.30
    )

    if not (
        portrait
        or landscape
    ):
        orientation = "unknown"
    elif portrait:
        orientation = "portrait"
    else:
        orientation = "landscape"

    edges = cv2.Canny(
        gray,
        40,
        130,
    )

    edge_density = float(
        np.count_nonzero(edges)
    ) / float(
        max(
            1,
            edges.size,
        )
    )

    contrast = float(
        np.std(gray)
    )

    # A tightly cropped ID normally has substantial internal structure.
    structure_signal = (
        edge_density >= 0.006
        and contrast >= 10.0
    )

    # Calculate five inexpensive spatial-region signals.
    regions = []

    for i in range(5):
        y1 = int(
            i * h / 5
        )
        y2 = int(
            (i + 1) * h / 5
        )

        crop = gray[
            y1:y2,
            :
        ]

        if crop.size == 0:
            regions.append(0.0)
            continue

        crop_edges = cv2.Canny(
            crop,
            40,
            130,
        )

        local_density = (
            float(
                np.count_nonzero(
                    crop_edges
                )
            )
            / float(
                max(
                    1,
                    crop_edges.size,
                )
            )
        )

        local_contrast = float(
            np.std(crop)
        )

        # Region has meaningful visual content.
        region_score = min(
            100.0,
            (
                local_density * 5000.0
                + local_contrast * 1.5
            ),
        )

        regions.append(
            region_score
        )

    # Use a low threshold because Voter cards may have large white areas.
    present = sum(
        score >= 15.0
        for score in regions
    )

    # For a clearly card-shaped image with sufficient internal structure,
    # do not allow a blank fifth region to turn the entire document into 0/5.
    if structure_signal and present < 3:
        present = 3

    total = 5

    field_score = (
        present / total * 100.0
    )

    # Geometry is visual evidence, not OCR evidence.
    geometry_score = 0.0

    if orientation in {
        "portrait",
        "landscape",
    }:
        geometry_score += 55.0

    if 0.006 <= edge_density:
        geometry_score += 25.0

    if contrast >= 10.0:
        geometry_score += 20.0

    geometry_score = min(
        100.0,
        geometry_score,
    )

    detected = bool(
        orientation in {
            "portrait",
            "landscape",
        }
        and structure_signal
        and present >= 3
    )

    # If the image is already a tightly cropped card, the contour itself
    # may not exist. The visual structure remains sufficient.
    return {
        "detected": detected,
        "orientation": orientation,
        "geometry_score": round(
            geometry_score,
            2,
        ),
        "field_presence_score": round(
            field_score,
            2,
        ),
        "visual_fields_present": (
            f"{present}/{total}"
        ),
        "region_scores": [
            round(
                value,
                2,
            )
            for value in regions
        ],
        "edge_density": round(
            edge_density,
            5,
        ),
        "contrast": round(
            contrast,
            2,
        ),
    }


def _tamper_check(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Very lightweight image-level tamper signal.

    This is intentionally conservative. It must not claim that a document
    is genuine merely because no simple anomaly was found.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape[:2]

    if h < 20 or w < 20:
        return {
            "risk": "UNKNOWN",
            "tamper_score": 0.0,
        }

    # Look for extreme local variance discontinuities.
    small = cv2.resize(
        gray,
        (
            max(32, w // 8),
            max(32, h // 8),
        ),
        interpolation=cv2.INTER_AREA,
    )

    lap = cv2.Laplacian(
        small,
        cv2.CV_64F,
    )

    variance = float(
        lap.var()
    )

    # Conservative:
    # no strong anomaly -> LOW, not "genuine".
    if variance > 5000:
        risk = "MEDIUM"
        score = 20.0
    else:
        risk = "LOW"
        score = 4.0

    return {
        "risk": risk,
        "tamper_score": score,
    }


def _pdf_first_page(
    file_bytes: bytes,
) -> np.ndarray:
    """
    Render first PDF page without OCR.
    """

    try:
        import fitz
    except ImportError as exc:
        raise ValueError(
            "PDF validation requires PyMuPDF."
        ) from exc

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf",
    )

    try:
        if document.page_count == 0:
            raise ValueError(
                "PDF contains no pages."
            )

        page = document.load_page(0)

        # 1.5x is enough for visual validation and keeps this fast.
        matrix = fitz.Matrix(
            1.5,
            1.5,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
            pix.n,
        )

        if pix.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        return image

    finally:
        document.close()


def verify_voter_card(
    file_bytes: bytes,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    FAST VALIDATION-ONLY entry point.

    NO OCR.
    NO EasyOCR.
    NO PaddleOCR.
    NO extraction.

    This function intentionally does not import or initialize any OCR
    library, so calling this endpoint cannot produce an EasyOCR startup
    warning from this module.
    """

    if not file_bytes:
        raise ValueError(
            "Uploaded file is empty."
        )

    normalized_type = (
        content_type.lower().strip()
        if content_type
        else ""
    )

    # ------------------------------------------------------------
    # Decode input
    # ------------------------------------------------------------

    if (
        normalized_type == "application/pdf"
        or file_bytes.startswith(b"%PDF")
    ):
        image = _pdf_first_page(
            file_bytes
        )
    else:
        image = _decode_image(
            file_bytes
        )

    # ------------------------------------------------------------
    # Fast validation
    # ------------------------------------------------------------

    quality_label, quality_score, quality_metrics = (
        _image_quality(image)
    )

    visual = _visual_regions(
        image
    )

    tamper = _tamper_check(
        image
    )

    document_detected = bool(
        visual["detected"]
    )

    tamper_risk = str(
        tamper["risk"]
    ).upper()

    visual_fields = visual[
        "visual_fields_present"
    ]

    field_score = float(
        visual["field_presence_score"]
    )

    geometry_score = float(
        visual["geometry_score"]
    )

    try:
        present = int(
            visual_fields.split(
                "/",
                1,
            )[0]
        )
    except (
        ValueError,
        IndexError,
    ):
        present = 0

    # ------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------

    if not document_detected:
        decision = DOCUMENT_REJECT
        reason = (
            "Voter ID visual structure "
            "could not be detected."
        )

    elif quality_label == "POOR":
        decision = DOCUMENT_REJECT
        reason = (
            "Image quality is insufficient "
            "for reliable validation."
        )

    elif tamper_risk in {
        "HIGH",
        "CRITICAL",
    }:
        decision = DOCUMENT_SUSPICIOUS
        reason = (
            "Strong image-level tampering "
            "indicators were detected."
        )

    else:
        decision = DOCUMENT_PASS
        reason = (
            "Voter ID passed local visual "
            "document validation."
        )

    score = round(
        quality_score * 0.40
        + geometry_score * 0.25
        + field_score * 0.30
        + (
            100.0
            - float(
                tamper["tamper_score"]
            )
        ) * 0.05,
        2,
    )

    return {
        "document_type": "VOTER_ID",
        "decision": decision,
        "score": score,
        "validation": {
            "document_detected": document_detected,
            "image_quality": quality_label,
            "tampering_risk": tamper_risk,
            "visual_fields": visual_fields,
            "score": score,
            "verification_stage": (
                "LOCAL_DOCUMENT_VALIDATION"
            ),
            "checks": {
                "geometry_score": geometry_score,
                "field_presence_score": round(
                    field_score,
                    2,
                ),
                "visual_region_scores": visual[
                    "region_scores"
                ],
                "orientation": visual[
                    "orientation"
                ],
                "quality_score": round(
                    quality_score,
                    2,
                ),
                "quality_metrics": quality_metrics,
                "tamper_score": tamper[
                    "tamper_score"
                ],
                "ocr_used": False,
                "extraction_used": False,
                "validation_only": True,
            },
            "reasons": (
                []
                if decision == DOCUMENT_PASS
                else [reason]
            ),
        },
        "authoritative_verification": {
            "status": "NOT_PERFORMED"
        },
        "extracted_data": {
            "epic_number": None,
            "name": None,
            "father_name": None,
            "gender": None,
            "date_of_birth": None,
            "voter_serial": None,
        },
    }


# ============================================================
# EXTRACTION — FUTURE ONLY
# ============================================================
#
# IMPORTANT:
# The validation endpoint above intentionally does NOT call OCR.
#
# When extraction is implemented, create a separate extraction function
# or endpoint and call the OCR engine there.
#
# Do not initialize EasyOCR/PaddleOCR at module import time.
#
# Example future architecture:
#
# def extract_voter_data(...):
#     # OCR initialization here / shared lazy model
#     # extraction only
#     ...
#
# Keeping extraction separate prevents a validation upload from paying
# the OCR startup/inference cost.
#