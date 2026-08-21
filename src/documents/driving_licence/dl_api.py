"""
Driving Licence Validation API.

VALIDATION-ONLY BUILD
---------------------
This endpoint intentionally does NOT run OCR and does NOT perform field
extraction.

Pipeline:

    Upload PDF/Image
          |
          v
    In-memory image decode
      - no temp-file round trip for images
          |
          v
    File validation
          |
          v
    PDF/Image -> page images
          |
          v
    Image quality
      - blur
      - brightness
      - contrast
      - resolution
      - aspect ratio
          |
          v
    Visual DL structure validation
      - card geometry
      - header/content regions
      - identity region
      - validity/date region
      - vehicle-class region
      - address region
      - photo/security region
      - lightweight holder-face presence check
          |
          v
    Tamper evidence
          |
          v
    Common/local validation decision

IMPORTANT
---------
Without OCR or extraction, this service can verify that the uploaded image
visually resembles a Driving Licence and that the expected visual regions
contain usable content.

It CANNOT verify:
    - the actual person's name
    - the actual DL number
    - the actual DOB
    - whether the licence exists in a government database
    - whether the printed text is semantically correct

Therefore DOCUMENT_PASS means LOCAL_DOCUMENT_VALIDATION only.
"""

from __future__ import annotations

import os
import time
import uuid
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.documents.driving_licence import config
from src.documents.driving_licence import utils

from src.common.verification.image_quality import analyze_image_quality

from src.common.security.file_security import (
    validate_upload,
    MaliciousFileError,
    VirusScannerUnavailableError,
    UnsupportedFileTypeError,
    InvalidFileSignatureError,
    FileTooLargeError,
    InvalidFileError,
)

try:
    from src.common.authenticity.tamper import analyze_tampering
except Exception:  # pragma: no cover
    analyze_tampering = None


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    tags=["Driving Licence"],
)


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "DL_MAX_UPLOAD_BYTES",
        str(10 * 1024 * 1024),
    )
)

MAX_PAGES = int(
    os.getenv(
        "DL_MAX_PAGES",
        "2",
    )
)

DOCUMENT_PASS = "DOCUMENT_PASS"
DOCUMENT_SUSPICIOUS = "DOCUMENT_SUSPICIOUS"
MANUAL_REVIEW = "MANUAL_REVIEW"
DOCUMENT_REJECT = "DOCUMENT_REJECT"

PASS_SCORE = float(
    os.getenv(
        "DL_PASS_SCORE",
        "75",
    )
)

MANUAL_REVIEW_SCORE = float(
    os.getenv(
        "DL_MANUAL_REVIEW_SCORE",
        "55",
    )
)

# Visual-region thresholds.
MIN_VISUAL_REGIONS_FOR_PASS = int(
    os.getenv(
        "DL_MIN_VISUAL_REGIONS_FOR_PASS",
        "6",
    )
)

TOTAL_VISUAL_REGIONS = 8

# Lightweight face detector. Haar cascade is used intentionally instead of
# a deep-learning face model so validation stays fast on CPU.
try:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
except Exception:  # pragma: no cover
    _FACE_CASCADE = None


# ============================================================
# FAST ANALYSIS CONFIGURATION
# ============================================================

# Heavy image operations are performed on a bounded working image.
# This keeps validation fast on CPU without changing the uploaded file.
MAX_ANALYSIS_DIMENSION = int(
    os.getenv(
        "DL_MAX_ANALYSIS_DIMENSION",
        "900",
    )
)

# Tamper analysis is the most expensive validation stage. Keep its working
# image smaller than the structural-analysis image.
MAX_TAMPER_ANALYSIS_DIMENSION = int(
    os.getenv(
        "DL_MAX_TAMPER_ANALYSIS_DIMENSION",
        "700",
    )
)

# PDFs are rasterized only as much as needed for visual validation.
# Image uploads are decoded directly from memory and do not touch disk.
DL_PDF_DPI = int(
    os.getenv(
        "DL_PDF_DPI",
        "150",
    )
)


# ============================================================
# OPTIONAL EXTRACTION — DISABLED / PRESERVED FOR LATER
# ============================================================
#
# OCR and field extraction are intentionally NOT executed by:
#
#     POST /driving-licence/verify
#
# The old extraction layer can be enabled later without changing
# the validation architecture.
#
# Future integration:
#
#     # from src.paddle_ocr_engine import run_ocr
#     # from src.documents.driving_licence.field_extractor import (
#     #     extract_driving_licence_fields,
#     # )
#     #
#     # ocr_lines = run_ocr(
#     #     image,
#     #     min_confidence=0.30,
#     # )
#     #
#     # extracted_data = extract_driving_licence_fields(
#     #     ocr_lines
#     # )
#
# DO NOT uncomment this for the fast validation endpoint unless
# OCR/extraction is explicitly required.
#
# OCR previously caused very high processing time, so validation
# deliberately remains OCR-free.
#
#
# ============================================================
# GENERIC HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 100.0,
) -> float:
    return max(
        low,
        min(high, value),
    )


def _quality_score(
    metrics: Any,
) -> float:
    """
    Read/derive the common image-quality score robustly.

    The common quality layer can expose either:
      - an explicit quality score
      - component scores
      - raw blur/brightness/contrast/resolution metrics
      - nested quality_metrics

    A zero/empty top-level score is NOT treated as a real quality failure
    when valid raw quality evidence is available underneath it.
    """

    if not isinstance(metrics, dict):
        return 0.0

    # --------------------------------------------------------
    # 1. Explicit non-zero quality score.
    # --------------------------------------------------------

    for key in (
        "quality_score",
        "overall_score",
        "image_quality_score",
        "score",
    ):
        raw_value = metrics.get(key)

        if raw_value is None:
            continue

        numeric = _safe_float(
            raw_value,
            -1.0,
        )

        # Important:
        # A common-layer adapter may return score=0 while still providing
        # valid blur/brightness/contrast evidence. Do not stop here.
        if 0.0 < numeric <= 100.0:
            return round(
                _clamp(numeric),
                2,
            )

    # --------------------------------------------------------
    # 2. Component scores.
    # --------------------------------------------------------

    component_values = []

    component_keys = (
        "blur_score_component",
        "brightness_component",
        "contrast_component",
        "resolution_component",
        "aspect_ratio_component",
    )

    for key in component_keys:
        value = metrics.get(key)

        if value is None:
            continue

        numeric = _safe_float(
            value,
            -1.0,
        )

        if 0.0 <= numeric <= 100.0:
            component_values.append(
                numeric
            )

    if component_values:
        return round(
            _clamp(
                sum(component_values)
                / len(component_values)
            ),
            2,
        )

    # --------------------------------------------------------
    # 3. Find nested common quality metrics.
    # --------------------------------------------------------

    def _find_quality_dict(
        value: Any,
        depth: int = 0,
    ) -> dict[str, Any]:
        if (
            depth > 5
            or not isinstance(value, dict)
        ):
            return {}

        interesting = {
            "blur_score",
            "brightness",
            "contrast",
            "width",
            "height",
            "resolution_ok",
            "aspect_ratio_ok",
            "blur_score_component",
            "brightness_component",
            "contrast_component",
        }

        if any(
            key in value
            for key in interesting
        ):
            return value

        for nested in value.values():
            if isinstance(nested, dict):
                result = _find_quality_dict(
                    nested,
                    depth + 1,
                )

                if result:
                    return result

        return {}

    raw = _find_quality_dict(
        metrics
    )

    if not raw:
        return 0.0

    # --------------------------------------------------------
    # 4. Nested component scores.
    # --------------------------------------------------------

    nested_components = []

    for key in component_keys:
        value = raw.get(key)

        if value is None:
            continue

        numeric = _safe_float(
            value,
            -1.0,
        )

        if 0.0 <= numeric <= 100.0:
            nested_components.append(
                numeric
            )

    if nested_components:
        return round(
            _clamp(
                sum(nested_components)
                / len(nested_components)
            ),
            2,
        )

    # --------------------------------------------------------
    # 5. Derive score from RAW common image-quality metrics.
    #
    # This does NOT run another quality engine. These are the metrics
    # already produced by the common quality layer.
    # --------------------------------------------------------

    scores = []

    # Blur / sharpness.
    #
    # A Laplacian-variance-style blur score increases with sharpness.
    # 100+ is treated as strong enough for the quality component.
    if raw.get("blur_score") is not None:
        blur = _safe_float(
            raw.get("blur_score"),
            -1.0,
        )

        if blur >= 0:
            blur_component = min(
                100.0,
                blur,
            )

            scores.append(
                blur_component
            )

    # Brightness.
    if raw.get("brightness") is not None:
        brightness = _safe_float(
            raw.get("brightness"),
            -1.0,
        )

        if 0.0 <= brightness <= 255.0:

            if 70.0 <= brightness <= 210.0:
                brightness_component = 100.0

            elif brightness < 70.0:
                brightness_component = (
                    brightness
                    / 70.0
                    * 100.0
                )

            else:
                brightness_component = (
                    (255.0 - brightness)
                    / 45.0
                    * 100.0
                )

            scores.append(
                _clamp(
                    brightness_component
                )
            )

    # Contrast.
    if raw.get("contrast") is not None:
        contrast = _safe_float(
            raw.get("contrast"),
            -1.0,
        )

        if contrast >= 0:
            contrast_component = min(
                100.0,
                contrast
                / 30.0
                * 100.0,
            )

            scores.append(
                contrast_component
            )

    # Resolution.
    if raw.get("resolution_ok") is not None:
        scores.append(
            100.0
            if bool(
                raw.get(
                    "resolution_ok"
                )
            )
            else 0.0
        )

    # Aspect ratio.
    if raw.get("aspect_ratio_ok") is not None:
        scores.append(
            100.0
            if bool(
                raw.get(
                    "aspect_ratio_ok"
                )
            )
            else 0.0
        )

    if scores:
        return round(
            _clamp(
                sum(scores)
                / len(scores)
            ),
            2,
        )

    return 0.0


def _tamper_result(
    image: np.ndarray,
) -> dict[str, Any]:
    if analyze_tampering is None:
        return {
            "available": False,
            "risk": "UNKNOWN",
            "tamper_score": 0.0,
            "decision": "NOT_PERFORMED",
            "signals": [],
        }

    try:
        result = analyze_tampering(image)

        if isinstance(result, dict):
            result.setdefault(
                "available",
                True,
            )
            return result

    except Exception as exc:
        return {
            "available": False,
            "risk": "UNKNOWN",
            "tamper_score": 0.0,
            "decision": "NOT_PERFORMED",
            "signals": [
                "Tamper analysis unavailable."
            ],
            "error": str(exc),
        }

    return {
        "available": False,
        "risk": "UNKNOWN",
        "tamper_score": 0.0,
        "decision": "NOT_PERFORMED",
        "signals": [],
    }


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def _to_bgr(
    image: Any,
) -> np.ndarray:
    """
    Convert common image representations to OpenCV BGR.
    """
    if image is None:
        raise ValueError(
            "Image is empty."
        )

    if isinstance(
        image,
        np.ndarray,
    ):
        array = image

    else:
        try:
            array = np.asarray(
                image
            )
        except Exception as exc:
            raise ValueError(
                "Unable to convert image to an array."
            ) from exc

    if array.size == 0:
        raise ValueError(
            "Image is empty."
        )

    if array.ndim == 2:
        return cv2.cvtColor(
            array,
            cv2.COLOR_GRAY2BGR,
        )

    if array.ndim == 3:
        if array.shape[2] == 4:
            return cv2.cvtColor(
                array,
                cv2.COLOR_BGRA2BGR,
            )

        if array.shape[2] == 3:
            return array

    raise ValueError(
        "Unsupported image format."
    )


# ============================================================
# FAST WORKING IMAGE
# ============================================================

def _prepare_fast_analysis_image(
    image: np.ndarray,
    max_dimension: int = MAX_ANALYSIS_DIMENSION,
) -> np.ndarray:
    """
    Create one bounded in-memory image for CPU-heavy validation.

    The uploaded image is never modified.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image

    scale = max_dimension / float(longest_side)

    return cv2.resize(
        image,
        (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# CARD GEOMETRY
# ============================================================

def _detect_card_geometry(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Detect whether the image contains a card-like rectangular document.

    This is intentionally visual. No OCR is used.
    """
    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return {
            "detected": False,
            "score": 0.0,
            "aspect_ratio": 0.0,
            "rectangle_score": 0.0,
        }

    aspect_ratio = width / height

    # Indian driving-licence cards commonly fall around this range, but
    # we deliberately allow a broad range for scans, screenshots and crops.
    aspect_score = 100.0

    if aspect_ratio < 1.20:
        aspect_score = max(
            0.0,
            100.0 - (
                (1.20 - aspect_ratio) * 220.0
            ),
        )

    elif aspect_ratio > 2.20:
        aspect_score = max(
            0.0,
            100.0 - (
                (aspect_ratio - 2.20) * 100.0
            ),
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rectangle_score = 0.0
    image_area = float(
        width * height
    )

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approximation) == 4:
            rectangle_score = max(
                rectangle_score,
                min(
                    100.0,
                    (area / image_area) * 120.0,
                ),
            )

    # A cropped card often has no visible outer contour. In that case,
    # aspect ratio still provides useful evidence.
    if rectangle_score < 50.0:
        rectangle_score = max(
            rectangle_score,
            aspect_score * 0.70,
        )

    geometry_score = (
        aspect_score * 0.45
        + rectangle_score * 0.55
    )

    detected = (
        geometry_score >= 55.0
        and aspect_ratio >= 1.15
        and aspect_ratio <= 2.40
    )

    return {
        "detected": detected,
        "score": round(
            _clamp(
                geometry_score
            ),
            2,
        ),
        "aspect_ratio": round(
            aspect_ratio,
            3,
        ),
        "rectangle_score": round(
            _clamp(
                rectangle_score
            ),
            2,
        ),
        "width": width,
        "height": height,
    }


# ============================================================
# LIGHTWEIGHT HOLDER PHOTO / FACE VALIDATION
# ============================================================

def _detect_face_in_photo_regions(
    image: np.ndarray,
    preferred_regions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check the expected DL photo regions for a human face.

    This is deliberately lightweight:
      - OpenCV Haar cascade only
      - no PaddleOCR
      - no embedding model
      - no face recognition / identity matching

    The goal is only to catch obvious cases where the expected holder-photo
    region is empty, a placeholder, or contains no detectable human face.
    """

    if (
        _FACE_CASCADE is None
        or _FACE_CASCADE.empty()
    ):
        return {
            "photo_region_present": False,
            "face_detected": False,
            "face_region": None,
            "method": "opencv_haar",
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape[:2]

    # Existing broad DL photo locations. Supporting both avoids tying the
    # validator to one card generation/layout.
    photo_regions = {
        "left": (
            0.00,
            0.18,
            0.32,
            0.72,
        ),
        "right": (
            0.68,
            0.18,
            1.00,
            0.72,
        ),
    }

    detected_faces: list[dict[str, Any]] = []

    region_items = list(
        photo_regions.items()
    )

    if preferred_regions:
        preferred = [
            item
            for item in region_items
            if item[0] in preferred_regions
        ]
        fallback = [
            item
            for item in region_items
            if item[0] not in preferred_regions
        ]
        region_items = preferred + fallback

    # In the common case only the first candidate region is scanned.
    # The second region is used only when the first candidate has no face.
    for index, (region_name, (x1, y1, x2, y2)) in enumerate(
        region_items
    ):
        left = max(
            0,
            int(width * x1),
        )
        top = max(
            0,
            int(height * y1),
        )
        right = min(
            width,
            int(width * x2),
        )
        bottom = min(
            height,
            int(height * y2),
        )

        roi = gray[top:bottom, left:right]

        if roi.size == 0:
            continue

        # Equalization improves detection on common ID-card lighting while
        # remaining extremely cheap compared with OCR.
        roi = cv2.equalizeHist(roi)

        min_face = max(
            32,
            int(min(roi.shape[:2]) * 0.12),
        )

        try:
            faces = _FACE_CASCADE.detectMultiScale(
                roi,
                scaleFactor=1.12,
                minNeighbors=6,
                minSize=(
                    min_face,
                    min_face,
                ),
            )
        except Exception:
            faces = ()

        for fx, fy, fw, fh in faces:
            face_area_ratio = (
                float(fw * fh)
                / float(max(roi.shape[0] * roi.shape[1], 1))
            )

            # Ignore tiny detections that are usually text/icons/noise.
            if face_area_ratio < 0.008:
                continue

            detected_faces.append(
                {
                    "region": region_name,
                    "area_ratio": round(
                        face_area_ratio,
                        4,
                    ),
                }
            )

        # Do not scan another photo region once a valid face is found.
        if detected_faces:
            break

    face_detected = bool(
        detected_faces
    )

    return {
        "photo_region_present": face_detected,
        "face_detected": face_detected,
        "face_region": (
            detected_faces[0]["region"]
            if detected_faces
            else None
        ),
        "method": "opencv_haar",
    }


# ============================================================
# VISUAL FIELD-REGION DETECTION
# ============================================================

def _region_content_score(
    gray: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Estimate whether a region contains meaningful visual content.

    This does NOT understand the text. It only checks whether the region
    contains enough edge/texture/contrast to plausibly contain printed
    content, a photo, or another expected card element.
    """
    height, width = gray.shape[:2]

    left = max(
        0,
        min(
            width - 1,
            int(width * x1),
        ),
    )
    top = max(
        0,
        min(
            height - 1,
            int(height * y1),
        ),
    )
    right = max(
        left + 1,
        min(
            width,
            int(width * x2),
        ),
    )
    bottom = max(
        top + 1,
        min(
            height,
            int(height * y2),
        ),
    )

    crop = gray[
        top:bottom,
        left:right,
    ]

    if crop.size == 0:
        return 0.0

    crop = cv2.GaussianBlur(
        crop,
        (3, 3),
        0,
    )

    edges = cv2.Canny(
        crop,
        40,
        120,
    )

    edge_density = (
        float(
            np.count_nonzero(edges)
        )
        / float(
            edges.size
        )
    )

    std = float(
        np.std(crop)
    )

    # Both local texture and edge density indicate useful visual content.
    texture_score = min(
        100.0,
        std * 2.0,
    )

    edge_score = min(
        100.0,
        edge_density * 500.0,
    )

    return round(
        (
            texture_score * 0.55
            + edge_score * 0.45
        ),
        2,
    )


def _detect_visual_fields(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Detect expected DL visual regions without OCR.

    The regions are deliberately broad because Indian DL layouts vary.
    This is a structural/visual check, not semantic field extraction.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Broad layout-independent regions.
    #
    # Each region is intentionally larger than the actual printed field so
    # minor differences between old/new DL layouts do not cause false rejects.
    region_definitions = {
        "header": (
            0.05,
            0.02,
            0.95,
            0.22,
        ),
        "identity": (
            0.05,
            0.20,
            0.72,
            0.50,
        ),
        "date_validity": (
            0.05,
            0.45,
            0.72,
            0.68,
        ),
        "vehicle_class": (
            0.05,
            0.62,
            0.72,
            0.84,
        ),
        "address": (
            0.05,
            0.68,
            0.95,
            0.95,
        ),
        "photo_left": (
            0.00,
            0.18,
            0.32,
            0.72,
        ),
        "photo_right": (
            0.68,
            0.18,
            1.00,
            0.72,
        ),
        "security_bottom": (
            0.05,
            0.82,
            0.95,
            0.99,
        ),
    }

    scores = {
        name: _region_content_score(
            gray,
            *bounds,
        )
        for name, bounds in region_definitions.items()
    }

    present = {
        name: (
            score >= 12.0
        )
        for name, score in scores.items()
    }

    # Photo can exist on either side depending on card design.
    photo_present = (
        present["photo_left"]
        or present["photo_right"]
    )

    # The two photo regions represent one logical requirement.
    logical_fields = {
        "header": present["header"],
        "identity": present["identity"],
        "date_validity": present["date_validity"],
        "vehicle_class": present["vehicle_class"],
        "address": present["address"],
        "photo": photo_present,
        "security_bottom": present["security_bottom"],
    }

    logical_present = sum(
        logical_fields.values()
    )
    logical_total = len(
        logical_fields
    )

    # Additional structural evidence.
    mean_region_score = (
        sum(scores.values())
        / max(len(scores), 1)
    )

    return {
        "regions": scores,
        "region_present": present,
        "logical_fields": logical_fields,
        "logical_fields_present": logical_present,
        "logical_fields_total": logical_total,
        "mean_region_score": round(
            mean_region_score,
            2,
        ),
        "photo_detected": photo_present,
    }


# ============================================================
# DL VISUAL VALIDATION
# ============================================================

def _validate_page(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Fast validation pipeline.

    Expensive stages are gated:
      1. common quality
      2. lightweight structure/regions
      3. face check
      4. tamper only for viable candidates

    OCR and extraction remain disabled.
    """
    page_start = time.perf_counter()

    # Use the common quality layer once. It is the source of the quality
    # decision; no second quality implementation is run here.
    quality_metrics = analyze_image_quality(
        image
    )

    quality_score = _quality_score(
        quality_metrics
    )

    analysis_image = _prepare_fast_analysis_image(
        image,
        MAX_ANALYSIS_DIMENSION,
    )

    geometry = _detect_card_geometry(
        analysis_image
    )

    visual_fields = _detect_visual_fields(
        analysis_image
    )

    field_ratio = (
        visual_fields["logical_fields_present"]
        / max(
            visual_fields["logical_fields_total"],
            1,
        )
    )

    field_presence_score = field_ratio * 100.0

    geometry_score = _safe_float(
        geometry.get("score"),
        0.0,
    )

    visual_region_score = _clamp(
        visual_fields["mean_region_score"]
    )

    # Determine which photo region is most likely before running Haar.
    photo_scores = visual_fields.get("regions", {})
    preferred_regions = sorted(
        (
            (
                name,
                _safe_float(
                    photo_scores.get(name),
                    0.0,
                ),
            )
            for name in (
                "photo_left",
                "photo_right",
            )
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    preferred_regions = [
        name
        for name, score_value in preferred_regions
        if score_value >= 12.0
    ]

    # If visual regions are too weak, there is no reason to spend CPU on
    # face/tamper analysis. Face is required only for a viable candidate.
    preliminary_structure = bool(
        geometry["detected"]
        or (
            visual_fields["logical_fields_present"]
            >= MIN_VISUAL_REGIONS_FOR_PASS
            and quality_score >= 50.0
        )
    )

    photo_validation = {
        "photo_region_present": bool(
            visual_fields.get("photo_detected")
        ),
        "face_detected": False,
        "face_region": None,
        "method": "opencv_haar",
    }

    if preliminary_structure:
        photo_validation = _detect_face_in_photo_regions(
            analysis_image,
            preferred_regions=preferred_regions,
        )

    visual_structure_fallback = bool(
        visual_fields["logical_fields_present"]
        >= MIN_VISUAL_REGIONS_FOR_PASS
        and quality_score >= 50.0
        and photo_validation["photo_region_present"]
        and photo_validation["face_detected"]
    )

    document_structure_detected = bool(
        geometry["detected"]
        or visual_structure_fallback
    )

    effective_geometry_score = (
        geometry_score
        if geometry["detected"]
        else min(
            visual_region_score,
            80.0,
        )
    )

    # Only viable document candidates pay the tamper-analysis cost.
    # Use a smaller working image for the expensive common tamper checks.
    tamper_candidate = bool(
        document_structure_detected
        and quality_score >= 50.0
        and visual_fields["logical_fields_present"]
        >= MIN_VISUAL_REGIONS_FOR_PASS
    )

    if tamper_candidate:
        tamper_image = _prepare_fast_analysis_image(
            analysis_image,
            MAX_TAMPER_ANALYSIS_DIMENSION,
        )
        tamper = _tamper_result(
            tamper_image
        )
    else:
        tamper = {
            "available": False,
            "risk": "UNKNOWN",
            "tamper_score": 0.0,
            "decision": "NOT_PERFORMED",
            "signals": [],
        }

    tamper_score = _clamp(
        _safe_float(
            tamper.get(
                "tamper_score"
            ),
            0.0,
        )
    )

    tamper_risk = str(
        tamper.get(
            "risk",
            "UNKNOWN",
        )
    ).upper().strip()

    photo_score = (
        100.0
        if photo_validation["face_detected"]
        else 0.0
    )

    score = (
        quality_score * 0.27
        + effective_geometry_score * 0.18
        + field_presence_score * 0.30
        + visual_region_score * 0.10
        + photo_score * 0.10
        + (100.0 - tamper_score) * 0.05
    )

    score = round(
        _clamp(score),
        2,
    )

    reasons: list[str] = []
    warnings: list[str] = []

    if not document_structure_detected:
        reasons.append(
            "Driving Licence card structure could not be reliably detected."
        )
    elif not geometry["detected"]:
        warnings.append(
            "Strict card-edge geometry was inconclusive; DL visual-region "
            "and holder-photo evidence were used as the document-detection "
            "fallback."
        )

    if (
        visual_fields["logical_fields_present"]
        < MIN_VISUAL_REGIONS_FOR_PASS
    ):
        reasons.append(
            "Expected Driving Licence visual regions are incomplete."
        )

    if not photo_validation["face_detected"]:
        reasons.append(
            "Holder photo/face could not be detected in the expected photo region."
        )

    if quality_score < 60.0:
        reasons.append(
            "Image quality is below the minimum validation threshold."
        )
    elif quality_score < 75.0:
        warnings.append(
            "Image quality is below the preferred threshold."
        )

    if tamper_risk == "HIGH":
        reasons.append(
            "Strong image-level tampering indicators were detected."
        )
    elif tamper_risk in {"MEDIUM", "REVIEW"}:
        warnings.append(
            "Image-level tamper evidence requires review."
        )

    if (
        not document_structure_detected
        or visual_fields["logical_fields_present"]
        < MIN_VISUAL_REGIONS_FOR_PASS
        or quality_score < 50.0
    ):
        decision = DOCUMENT_REJECT

    elif tamper_risk == "HIGH":
        decision = DOCUMENT_SUSPICIOUS

    elif tamper_risk in {"UNKNOWN", ""}:
        decision = MANUAL_REVIEW

    elif (
        not photo_validation["face_detected"]
        and quality_score >= 75.0
    ):
        decision = DOCUMENT_REJECT

    elif (
        not photo_validation["face_detected"]
        and quality_score < 75.0
    ):
        decision = MANUAL_REVIEW

    elif score >= PASS_SCORE:
        decision = DOCUMENT_PASS

    elif score >= MANUAL_REVIEW_SCORE:
        decision = MANUAL_REVIEW

    else:
        decision = DOCUMENT_REJECT

    page_elapsed = time.perf_counter() - page_start

    return {
        "decision": decision,
        "score": score,
        "verification_stage": "LOCAL_DOCUMENT_VALIDATION",
        "checks": {
            "document_type_detected": document_structure_detected,
            "visual_fields_present": (
                f"{visual_fields['logical_fields_present']}/"
                f"{visual_fields['logical_fields_total']}"
            ),
            "photo_present": bool(
                photo_validation["photo_region_present"]
            ),
            "face_detected": bool(
                photo_validation["face_detected"]
            ),
            "field_presence_score": round(
                field_presence_score,
                2,
            ),
            "format_validation": (
                "PASS"
                if document_structure_detected
                else "FAIL"
            ),
            "image_quality": (
                "GOOD"
                if quality_score >= 75.0
                else (
                    "FAIR"
                    if quality_score >= 50.0
                    else "POOR"
                )
            ),
            "quality_score": quality_score,
            "geometry_score": round(
                effective_geometry_score,
                2,
            ),
            "visual_region_score": round(
                visual_region_score,
                2,
            ),
            "tamper_score": round(
                tamper_score,
                2,
            ),
            "tamper_risk": tamper_risk,
        },
        "reasons": reasons,
        "warnings": warnings,
        "visual_structure": {
            "card_geometry": geometry,
            "fields": visual_fields,
            "photo_validation": photo_validation,
        },
        "quality_metrics": quality_metrics,
        "tamper_analysis": tamper,
        "authoritative_verification": {
            "status": "NOT_PERFORMED",
            "message": (
                "No government/source-of-truth verification "
                "was performed."
            ),
        },
        "_processing_time_seconds": round(
            page_elapsed,
            4,
        ),
    }


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(
    file_path: str,
) -> list[dict[str, Any]]:
    """
    Process a PDF without OCR/extraction.

    PDF rasterization is intentionally kept at a validation-appropriate DPI.
    Image uploads use the direct in-memory path in the API route.
    """
    pages = utils.load_file_as_images(
        file_path,
        pdf_dpi=DL_PDF_DPI,
    )

    if not pages:
        raise ValueError(
            "No readable pages found in the uploaded document."
        )

    if len(pages) > MAX_PAGES:
        raise ValueError(
            f"Driving Licence documents are limited "
            f"to {MAX_PAGES} pages."
        )

    page_results: list[dict[str, Any]] = []

    for page_number, (
        _page_label,
        image,
    ) in enumerate(
        pages,
        start=1,
    ):
        if image is None:
            page_results.append(
                {
                    "page_number": page_number,
                    "validation": {
                        "decision": DOCUMENT_REJECT,
                        "score": 0.0,
                        "verification_stage": (
                            "LOCAL_DOCUMENT_VALIDATION"
                        ),
                        "checks": {
                            "document_type_detected": False,
                            "visual_fields_present": "0/7",
                            "image_quality": "POOR",
                        },
                        "reasons": [
                            "Page image could not be loaded."
                        ],
                        "warnings": [],
                    },
                }
            )
            continue

        image_bgr = _to_bgr(image)

        page_results.append(
            {
                "page_number": page_number,
                "validation": _validate_page(
                    image_bgr
                ),
            }
        )

    return page_results


def process_image_bytes(
    file_bytes: bytes,
) -> list[dict[str, Any]]:
    """
    Decode an image directly from memory.

    This avoids the disk write -> disk read cycle used by the old path.
    """
    if not file_bytes:
        raise ValueError(
            "Uploaded file is empty."
        )

    encoded = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Uploaded image could not be decoded."
        )

    return [
        {
            "page_number": 1,
            "validation": _validate_page(
                image
            ),
        }
    ]


# ============================================================
# ROUTES
# ============================================================

@router.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "success",
        "message": (
            "Driving Licence Validation API is running."
        ),
        "version": "4.0.0",
        "mode": "validation_only",
        "ocr": "disabled",
        "extraction": "disabled",
    }


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "document": "driving_licence",
        "mode": "validation_only",
        "ocr": "disabled",
        "extraction": "disabled",
    }


@router.post(
    "/verify",
    summary="Validate Driving Licence",
    description=(
        "Performs fast local Driving Licence validation without OCR "
        "or field extraction. Validation uses image quality, card "
        "geometry, visual field-region presence, holder-face presence "
        "and tamper evidence. Government/source-of-truth verification "
        "is not performed."
    ),
)
async def verify_driving_licence(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    request_id = uuid.uuid4().hex
    request_start = time.perf_counter()
    temp_path: str | None = None

    try:
        # --------------------------------------------------------
        # FILE VALIDATION
        # --------------------------------------------------------
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "MISSING_FILENAME",
                    "message": "Filename is missing.",
                    "request_id": request_id,
                },
            )

        extension = Path(
            file.filename
        ).suffix.lower()

        # --------------------------------------------------------
        # READ UPLOAD ONCE
        # --------------------------------------------------------
        #
        # Images are decoded directly from memory. PDFs still use a temp
        # file because the existing PDF-to-image utility expects a path.
        file_bytes = await file.read(
            MAX_UPLOAD_BYTES + 1
        )

        total_bytes = len(file_bytes)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "EMPTY_FILE",
                    "message": "Uploaded file is empty.",
                    "request_id": request_id,
                },
            )

        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": (
                        "Uploaded file exceeds "
                        "the configured size limit."
                    ),
                    "request_id": request_id,
                },
            )

        # --------------------------------------------------------
        # COMMON SECURITY GATE — FIRST SECURITY PROCESSING STEP
        # --------------------------------------------------------
        #
        # Every document type uses this same security layer.
        # No PDF rasterization, OpenCV validation, OCR, extraction or
        # tamper analysis happens before this gate.
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
                    "request_id": request_id,
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
                    "request_id": request_id,
                },
            ) from exc
        except (
            UnsupportedFileTypeError,
            InvalidFileSignatureError,
            FileTooLargeError,
            InvalidFileError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_UPLOAD",
                    "message": str(exc),
                    "request_id": request_id,
                },
            ) from exc

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------
        if extension == ".pdf":
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(file_bytes)

            page_results = process_document(
                temp_path
            )
        else:
            page_results = process_image_bytes(
                file_bytes
            )


        if not page_results:
            raise ValueError(
                "No pages were processed."
            )

        page_scores = [
            _safe_float(
                page["validation"].get(
                    "score"
                ),
                0.0,
            )
            for page in page_results
        ]

        overall_score = min(
            page_scores
        )

        decisions = [
            str(
                page["validation"].get(
                    "decision",
                    DOCUMENT_REJECT,
                )
            ).upper()
            for page in page_results
        ]

        # Conservative multi-page aggregation.
        if DOCUMENT_REJECT in decisions:
            overall_decision = DOCUMENT_REJECT

        elif DOCUMENT_SUSPICIOUS in decisions:
            overall_decision = DOCUMENT_SUSPICIOUS

        elif MANUAL_REVIEW in decisions:
            overall_decision = MANUAL_REVIEW

        else:
            overall_decision = DOCUMENT_PASS

        elapsed = (
            time.perf_counter()
            - request_start
        )

        # --------------------------------------------------------
        # CLEAN PRODUCTION RESPONSE
        # --------------------------------------------------------
        # ====================================================
        # CLEAN PRODUCTION RESPONSE
        #
        # Public API intentionally exposes only the four fields
        # required by the document-verification contract.
        # Internal quality/tamper/geometry calculations remain
        # unchanged and are used for the decision and score.
        # ====================================================

        quality_statuses = [
            str(
                page["validation"]["checks"].get(
                    "image_quality",
                    "POOR",
                )
            ).upper()
            for page in page_results
        ]

        if all(
            status == "GOOD"
            for status in quality_statuses
        ):
            public_image_quality = "GOOD"
        elif all(
            status in {"GOOD", "FAIR"}
            for status in quality_statuses
        ):
            public_image_quality = "FAIR"
        else:
            public_image_quality = "POOR"

        tamper_statuses = [
            str(
                page["validation"]["checks"].get(
                    "tamper_risk",
                    "UNKNOWN",
                )
            ).upper()
            for page in page_results
        ]

        if "HIGH" in tamper_statuses:
            public_tampering_risk = "HIGH"
        elif any(
            status in {"MEDIUM", "REVIEW"}
            for status in tamper_statuses
        ):
            public_tampering_risk = "MEDIUM"
        elif all(
            status == "LOW"
            for status in tamper_statuses
        ):
            public_tampering_risk = "LOW"
        else:
            public_tampering_risk = "UNKNOWN"

        field_counts = []

        for page in page_results:
            field_value = str(
                page["validation"]["checks"].get(
                    "visual_fields_present",
                    "0/0",
                )
            )

            try:
                present, total = field_value.split(
                    "/",
                    1,
                )
                field_counts.append(
                    (
                        int(present),
                        int(total),
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                field_counts.append(
                    (0, 0)
                )

        if field_counts:
            visual_fields = (
                f"{min(item[0] for item in field_counts)}/"
                f"{min(item[1] for item in field_counts)}"
            )
        else:
            visual_fields = "0/0"

        # Public decision wording is intentionally business-friendly.
        # DOCUMENT_VERIFIED_SUCCESSFULLY is only reachable after all
        # automatic safety gates pass, including LOW tamper risk.
        public_decision = {
            DOCUMENT_PASS: "DOCUMENT_VERIFIED_SUCCESSFULLY",
            DOCUMENT_REJECT: "DOCUMENT_REJECTED",
            DOCUMENT_SUSPICIOUS: "DOCUMENT_SUSPICIOUS",
            MANUAL_REVIEW: "MANUAL_REVIEW",
        }.get(
            overall_decision,
            overall_decision,
        )

        return {
            "document_type": "DRIVING_LICENCE",
            "decision": public_decision,
            "score": round(
                overall_score,
                2,
            ),
            "validation": {
                "document_detected": all(
                    bool(
                        page["validation"]["checks"].get(
                            "document_type_detected",
                            False,
                        )
                    )
                    for page in page_results
                ),
                "image_quality": public_image_quality,
                "tampering_risk": public_tampering_risk,
                "visual_fields": visual_fields,
                "photo_present": all(
                    bool(
                        page["validation"]["checks"].get(
                            "photo_present",
                            False,
                        )
                    )
                    for page in page_results
                ),
                "face_detected": all(
                    bool(
                        page["validation"]["checks"].get(
                            "face_detected",
                            False,
                        )
                    )
                    for page in page_results
                ),
            },
            "processing_time_seconds": round(
                elapsed,
                3,
            ),
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DOCUMENT_VALIDATION_FAILED",
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc

    except Exception as exc:
        # Do not expose stack traces or implementation details.
        print(
            f"[DL API] request_id={request_id} "
            f"processing_error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_PROCESSING_ERROR",
                "message": (
                    "Driving Licence processing failed."
                ),
                "request_id": request_id,
            },
        ) from exc

    finally:
        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):
            try:
                os.remove(
                    temp_path
                )
            except OSError:
                pass

        try:
            await file.close()
        except Exception:
            pass