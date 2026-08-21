"""
Salary Slip Validation API - Phase 1.

Validation only.

Supports:
    - Native/text PDFs
    - Scanned/image-only PDFs
    - JPG/JPEG/PNG

Intentionally NOT used:
    - OCR
    - Tesseract
    - Salary extraction
    - LLM
    - Face recognition

Phase-1 checks:
    1. Common upload security
    2. PDF readability / structure
    3. Native salary-slip semantic evidence when text exists
    4. Visual document/layout evidence for scanned PDFs
    5. Image quality
    6. Page integrity / blank-page detection
    7. Duplicate representative-page detection
    8. Tampering analysis
    9. Validation score
    10. Decision

Endpoint:
    POST /salary-slip/verify

The master router owns /salary-slip.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.common.security.file_security import validate_upload

from src.common.authenticity.tamper import (
    analyze_tampering,
)


# ======================================================================
# ROUTER
# ======================================================================

router = APIRouter(
    tags=["Salary Slip"],
)


# ======================================================================
# CONFIGURATION
# ======================================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
}

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
}

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

# Fast representative-page validation.
MAX_REPRESENTATIVE_PAGES = 2

# Low rendering DPI keeps validation fast.
PREVIEW_DPI = 55

MIN_PAGE_WIDTH = 300
MIN_PAGE_HEIGHT = 400

BLANK_STD_THRESHOLD = 7.0

# Representative-page duplicate threshold.
DUPLICATE_THRESHOLD = 0.985


# ======================================================================
# SALARY-SLIP NATIVE TEXT SIGNALS
# ======================================================================

TITLE_TERMS = (
    "SALARY SLIP",
    "SALARYSLIP",
    "PAY SLIP",
    "PAYSLIP",
    "PAYSLIP",
)

EMPLOYEE_TERMS = (
    "EMPLOYEE",
    "EMPLOYEE ID",
    "EMPLOYEE NUMBER",
    "EMPLOYEE CODE",
    "EMP CODE",
    "ID / NAME",
    "DESIGNATION",
    "DATE OF JOINING",
    "DOJ",
)

EARNING_TERMS = (
    "EARNING DETAILS",
    "EARNINGS",
    "BASIC SALARY",
    "BASIC PAY",
    "BASIC",
    "HRA",
    "CONVEYANCE",
    "ALLOWANCE",
    "OVERTIME",
    "GROSS SALARY",
    "TOTAL EARNINGS",
)

DEDUCTION_TERMS = (
    "DEDUCTION DETAILS",
    "DEDUCTIONS",
    "PF",
    "EPF",
    "ESIC",
    "ESI",
    "PROFESSIONAL TAX",
    "PROF. TAX",
    "PROF TAX",
)

NET_PAY_TERMS = (
    "NET PAYABLE",
    "NET PAY",
    "NET SALARY",
    "TAKE HOME",
    "TAKE-HOME",
)

PAY_PERIOD_TERMS = (
    "SALARY SLIP FOR THE MONTH",
    "SALARY SLIP FOR",
    "FOR THE MONTH OF",
    "PAY PERIOD",
    "MONTH OF",
)

OTHER_DOCUMENT_TERMS = {
    "CIBIL": (
        "CIBIL",
        "TRANSUNION CIBIL",
        "CREDIT INFORMATION REPORT",
        "CREDIT INFORMATION REPORT (CIR)",
        "CREDIT REPORT",
        "CREDIT SCORE",
        "CIBIL SCORE",
        "CONSUMER CREDIT INFORMATION",
    ),
    "BANK_STATEMENT": (
        "BANK STATEMENT",
        "ACCOUNT STATEMENT",
        "TRANSACTION DATE",
        "WITHDRAWAL",
        "DEPOSIT",
        "CLOSING BALANCE",
    ),
    "PAN": (
        "INCOME TAX DEPARTMENT",
        "PERMANENT ACCOUNT NUMBER",
        "PAN CARD",
    ),
    "VOTER_ID": (
        "ELECTION COMMISSION",
        "ELECTOR PHOTO ID",
        "EPIC",
    ),
    "DRIVING_LICENCE": (
        "DRIVING LICENCE",
        "DRIVING LICENSE",
        "DL NO",
        "TRANSPORT DEPARTMENT",
    ),
    "ITR": (
        "INCOME TAX RETURN",
        "ITR-V",
        "ACKNOWLEDGEMENT NUMBER",
    ),
    "SALE_DEED": (
        "SALE DEED",
        "CONVEYANCE DEED",
        "PURCHASER",
        "SELLER",
    ),
}


# ======================================================================
# HELPERS
# ======================================================================

def normalize_text(value: str) -> str:
    value = str(value or "").upper()

    value = value.replace(
        "\u2013",
        "-",
    ).replace(
        "\u2014",
        "-",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def security_is_safe(
    result: Any,
) -> bool:

    if isinstance(
        result,
        bool,
    ):
        return result

    if isinstance(
        result,
        dict,
    ):
        return bool(
            result.get(
                "safe",
                result.get(
                    "valid",
                    result.get(
                        "passed",
                        True,
                    ),
                ),
            )
        )

    return True


def count_hits(
    text: str,
    terms: tuple[str, ...],
) -> int:

    text = normalize_text(
        text
    )

    return sum(
        1
        for term in terms
        if term in text
    )


# ======================================================================
# NATIVE TEXT CLASSIFICATION
# ======================================================================

def native_classification(
    text: str,
) -> dict[str, Any]:

    normalized = normalize_text(
        text
    )

    groups = {
        "title":
            count_hits(
                normalized,
                TITLE_TERMS,
            ),
        "employee":
            count_hits(
                normalized,
                EMPLOYEE_TERMS,
            ),
        "earnings":
            count_hits(
                normalized,
                EARNING_TERMS,
            ),
        "deductions":
            count_hits(
                normalized,
                DEDUCTION_TERMS,
            ),
        "net_pay":
            count_hits(
                normalized,
                NET_PAY_TERMS,
            ),
        "pay_period":
            count_hits(
                normalized,
                PAY_PERIOD_TERMS,
            ),
    }

    group_count = sum(
        1
        for value in groups.values()
        if value > 0
    )

    score = 0

    if groups["title"]:
        score += 30

    if groups["employee"]:
        score += 15

    if groups["earnings"]:
        score += 15

    if groups["deductions"]:
        score += 15

    if groups["net_pay"]:
        score += 15

    if groups["pay_period"]:
        score += 10

    score = min(
        100,
        score,
    )

    detected = (
        (
            groups["title"] > 0
            and group_count >= 2
        )
        or
        (
            groups["earnings"] > 0
            and groups["deductions"] > 0
            and groups["net_pay"] > 0
        )
        or
        (
            groups["employee"] > 0
            and groups["earnings"] > 0
            and group_count >= 3
        )
    )

    competitors: dict[str, int] = {}

    for document_type, terms in (
        OTHER_DOCUMENT_TERMS.items()
    ):

        hits = count_hits(
            normalized,
            terms,
        )

        if hits:
            competitors[
                document_type
            ] = hits

    strongest_competitor_type = (
        max(
            competitors,
            key=competitors.get,
        )
        if competitors
        else None
    )

    strongest_competitor = (
        competitors.get(
            strongest_competitor_type,
            0,
        )
        if strongest_competitor_type
        else 0
    )

    # A clear competing document wins if salary evidence is weak.
    if (
        strongest_competitor >= 2
        and score < 70
    ):
        detected = False
        score = 0

    return {
        "detected":
            detected,
        "score":
            score,
        "groups":
            groups,
        "competitors":
            competitors,
        "strongest_competitor":
            strongest_competitor_type,
        "strongest_competitor_hits":
            strongest_competitor,
    }


# ======================================================================
# PDF REPRESENTATIVE PAGES
# ======================================================================

def representative_page_indexes(
    page_count: int,
) -> list[int]:

    if page_count <= 0:
        return []

    if page_count == 1:
        return [0]

    # First and last pages give coverage without rendering every page.
    indexes = [
        0,
        page_count - 1,
    ]

    return indexes[
        :MAX_REPRESENTATIVE_PAGES
    ]


def render_page(
    page: Any,
) -> np.ndarray | None:

    scale = (
        PREVIEW_DPI
        /
        72.0
    )

    matrix = pymupdf.Matrix(
        scale,
        scale,
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    if pix.n == 4:

        array = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
            4,
        )

        image = cv2.cvtColor(
            array,
            cv2.COLOR_RGBA2BGR,
        )

    else:

        array = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
            3,
        )

        image = cv2.cvtColor(
            array,
            cv2.COLOR_RGB2BGR,
        )

    return image.copy()


# ======================================================================
# IMAGE QUALITY
# ======================================================================

def quality_score(
    image: np.ndarray,
) -> float:

    if (
        image is None
        or image.size == 0
    ):
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    score = 0.0

    if sharpness >= 100:
        score += 40
    elif sharpness >= 40:
        score += 25
    else:
        score += 10

    if 35 <= brightness <= 235:
        score += 30
    elif 20 <= brightness <= 245:
        score += 20
    else:
        score += 5

    if contrast >= 25:
        score += 30
    elif contrast >= 15:
        score += 20
    else:
        score += 10

    return score


def analyze_quality(
    pages: list[
        tuple[int, np.ndarray]
    ],
) -> dict[str, Any]:

    scores = [
        quality_score(
            image
        )
        for _, image in pages
    ]

    if not scores:

        return {
            "status":
                "NOT_CHECKED",
            "score":
                0.0,
        }

    average = (
        sum(scores)
        /
        len(scores)
    )

    minimum = min(
        scores
    )

    if minimum < 40:
        status = "POOR"
    elif average >= 70:
        status = "GOOD"
    else:
        status = "FAIR"

    return {
        "status":
            status,
        "score":
            round(
                average,
                2,
            ),
        "minimum":
            round(
                minimum,
                2,
            ),
    }


# ======================================================================
# PAGE INTEGRITY
# ======================================================================

def page_integrity(
    pages: list[
        tuple[int, np.ndarray]
    ],
) -> dict[str, Any]:

    thumbnails = []

    blank_pages = 0

    for index, image in pages:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        std = float(
            np.std(gray)
        )

        if std < BLANK_STD_THRESHOLD:
            blank_pages += 1

        thumbnail = cv2.resize(
            gray,
            (32, 32),
            interpolation=cv2.INTER_AREA,
        )

        thumbnails.append(
            (
                index,
                thumbnail,
            )
        )

    duplicate_found = False

    for i in range(
        len(thumbnails)
    ):

        for j in range(
            i + 1,
            len(thumbnails),
        ):

            first = (
                thumbnails[i][1]
                .astype(
                    np.float32
                )
                .flatten()
            )

            second = (
                thumbnails[j][1]
                .astype(
                    np.float32
                )
                .flatten()
            )

            if (
                np.std(first) < 1e-6
                or
                np.std(second) < 1e-6
            ):
                continue

            similarity = np.corrcoef(
                first,
                second,
            )[0, 1]

            if (
                np.isfinite(
                    similarity
                )
                and
                similarity >= DUPLICATE_THRESHOLD
            ):
                duplicate_found = True

    return {
        "suspicious":
            (
                blank_pages > 0
                or duplicate_found
            ),
        "blank_pages":
            blank_pages,
        "duplicate_pages":
            duplicate_found,
    }


# ======================================================================
# VISUAL SALARY-SLIP CLASSIFICATION
# ======================================================================
#
# This is NOT OCR.
#
# It checks whether a scanned page has the visual structure expected
# from a salary slip:
#
#     - rectangular/tabular layout
#     - multiple horizontal/vertical table lines
#     - concentrated document region
#     - text-like edge density
#
# This is intentionally conservative.
# ======================================================================

def visual_salary_structure(
    image: np.ndarray,
) -> dict[str, Any]:

    if (
        image is None
        or image.size == 0
    ):
        return {
            "passed":
                False,
            "score":
                0,
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape

    if (
        width < MIN_PAGE_WIDTH
        or height < MIN_PAGE_HEIGHT
    ):
        return {
            "passed":
                False,
            "score":
                0,
        }

    # Adaptive threshold makes scanned forms easier to analyze.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(15, width // 20),
            1,
        ),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            max(15, height // 20),
        ),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    horizontal_pixels = int(
        np.count_nonzero(
            horizontal
        )
    )

    vertical_pixels = int(
        np.count_nonzero(
            vertical
        )
    )

    total_pixels = (
        width
        *
        height
    )

    horizontal_ratio = (
        horizontal_pixels
        /
        max(
            1,
            total_pixels,
        )
    )

    vertical_ratio = (
        vertical_pixels
        /
        max(
            1,
            total_pixels,
        )
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_ratio = (
        float(
            np.count_nonzero(
                edges
            )
        )
        /
        max(
            1,
            total_pixels,
        )
    )

    # Salary slips are usually form/table-heavy.
    line_score = 0

    if horizontal_ratio >= 0.004:
        line_score += 25
    elif horizontal_ratio >= 0.002:
        line_score += 15

    if vertical_ratio >= 0.002:
        line_score += 25
    elif vertical_ratio >= 0.001:
        line_score += 15

    if edge_ratio >= 0.015:
        line_score += 25
    elif edge_ratio >= 0.008:
        line_score += 15

    # Detect whether the page has a concentrated content rectangle.
    # This avoids treating a nearly blank page as a salary slip.
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    largest_area = 0.0

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = (
            w
            *
            h
        )

        if (
            w >= width * 0.35
            and h >= height * 0.10
        ):
            largest_area = max(
                largest_area,
                area,
            )

    occupied_ratio = (
        largest_area
        /
        max(
            1,
            total_pixels,
        )
    )

    if occupied_ratio >= 0.20:
        line_score += 25
    elif occupied_ratio >= 0.10:
        line_score += 10

    line_score = min(
        100,
        line_score,
    )

    return {
        "passed":
            line_score >= 50,
        "score":
            line_score,
        "horizontal_ratio":
            round(
                horizontal_ratio,
                6,
            ),
        "vertical_ratio":
            round(
                vertical_ratio,
                6,
            ),
        "edge_ratio":
            round(
                edge_ratio,
                6,
            ),
        "occupied_ratio":
            round(
                occupied_ratio,
                4,
            ),
    }


# ======================================================================
# TAMPERING
# ======================================================================

def analyze_page_tampering(
    pages: list[
        tuple[int, np.ndarray]
    ],
) -> dict[str, Any]:

    risks = []

    scores = []

    for _, image in pages:

        try:

            result = analyze_tampering(
                image
            )

            if not isinstance(
                result,
                dict,
            ):
                continue

            risk = str(
                result.get(
                    "risk",
                    "NOT_CHECKED",
                )
            ).upper()

            risks.append(
                risk
            )

            try:

                scores.append(
                    float(
                        result.get(
                            "tamper_score",
                            0,
                        )
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        except Exception:
            continue

    if not risks:

        return {
            "risk":
                "NOT_CHECKED",
            "score":
                0,
        }

    # HIGH always wins.
    if "HIGH" in risks:
        risk = "HIGH"

    elif "MEDIUM" in risks:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    return {
        "risk":
            risk,
        "score":
            round(
                max(scores)
                if scores
                else 0,
                2,
            ),
    }


# ======================================================================
# VALIDATION SCORE
# ======================================================================

def validation_score(
    *,
    document_detected: bool,
    structural_pass: bool,
    visual_pass: bool,
    quality_status: str,
    page_integrity_pass: bool,
    tamper_risk: str,
    native_classification_score: int,
    native_text_available: bool,
) -> int:

    if not document_detected:
        return 0

    score = 0

    # File/document detection.
    score += 15

    # PDF/image structural validity.
    if structural_pass:
        score += 20

    # Salary-slip semantic evidence for native PDFs.
    if native_text_available:

        score += round(
            native_classification_score
            * 0.25
        )

    else:

        # Scanned documents use visual layout validation.
        if visual_pass:
            score += 25

    # Image quality.
    if quality_status == "GOOD":
        score += 20
    elif quality_status == "FAIR":
        score += 12

    # Page integrity.
    if page_integrity_pass:
        score += 10

    # Tampering.
    if tamper_risk == "LOW":
        score += 10
    elif tamper_risk == "MEDIUM":
        score += 0

    return max(
        0,
        min(
            100,
            score,
        ),
    )


# ======================================================================
# ROUTE
# ======================================================================

@router.post(
    "/verify",
)
async def verify_salary_slip(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    started = perf_counter()

    # ================================================================
    # 1. FILE NAME
    # ================================================================

    filename = Path(
        str(
            file.filename or ""
        ).strip()
    ).name

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

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Upload only JPG, JPEG, PNG or PDF files."
            ),
        )

    # ================================================================
    # 2. READ
    # ================================================================

    try:

        file_bytes = await file.read()

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

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "File exceeds the maximum allowed "
                "size of 15 MB."
            ),
        )

    # ================================================================
    # 3. COMMON SECURITY
    # ================================================================

    content_type = (
        str(
            file.content_type or ""
        )
        .split(";")[0]
        .strip()
        .lower()
    )

    extension_mime = {
        ".pdf":
            "application/pdf",
        ".jpg":
            "image/jpeg",
        ".jpeg":
            "image/jpeg",
        ".png":
            "image/png",
    }

    if content_type not in SUPPORTED_MIME_TYPES:

        content_type = (
            extension_mime[
                extension
            ]
        )

    try:

        security_result = (
            validate_upload(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )
        )

        if hasattr(
            security_result,
            "__await__",
        ):
            security_result = (
                await security_result
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

    if not security_is_safe(
        security_result
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Uploaded file failed security validation.",
            },
        )

    # ================================================================
    # 4. IMAGE FILE
    # ================================================================

    if extension != ".pdf":

        array = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:

            elapsed_ms = round(
                (
                    perf_counter()
                    -
                    started
                )
                * 1000,
                2,
            )

            return {
                "document_type":
                    "SALARY_SLIP",
                "decision":
                    "DOCUMENT_REJECTED",
                "score":
                    0,
                "validation": {
                    "document_detected":
                        False,
                    "image_quality":
                        "POOR",
                    "tampering_risk":
                        "HIGH",
                    "structural_validation":
                        "FAIL",
                    "processing_time_ms":
                        elapsed_ms,
                },
            }

        pages = [
            (
                0,
                image,
            )
        ]

        quality = analyze_quality(
            pages
        )

        integrity = page_integrity(
            pages
        )

        tamper = analyze_page_tampering(
            pages
        )

        visual = visual_salary_structure(
            image
        )

        document_detected = (
            quality["status"] != "POOR"
            and visual["passed"]
        )

        structural_pass = True

        score = validation_score(
            document_detected=document_detected,
            structural_pass=structural_pass,
            visual_pass=visual["passed"],
            quality_status=quality["status"],
            page_integrity_pass=(
                not integrity["suspicious"]
            ),
            tamper_risk=tamper["risk"],
            native_classification_score=0,
            native_text_available=False,
        )

        if tamper["risk"] in {
            "MEDIUM",
            "HIGH",
        }:
            decision = (
                "DOCUMENT_REJECTED"
            )

        elif not document_detected:
            decision = (
                "DOCUMENT_REVIEW"
            )

        elif (
            quality["status"] == "POOR"
            or integrity["suspicious"]
        ):
            decision = (
                "DOCUMENT_REVIEW"
            )

        else:
            decision = (
                "DOCUMENT_VERIFIED"
            )

        elapsed_ms = round(
            (
                perf_counter()
                -
                started
            )
            * 1000,
            2,
        )

        return {
            "document_type":
                "SALARY_SLIP",
            "decision":
                decision,
            "score":
                score,
            "validation": {
                "document_detected":
                    document_detected,
                "image_quality":
                    quality["status"],
                "tampering_risk":
                    tamper["risk"],
                "structural_validation":
                    (
                        "PASS"
                        if structural_pass
                        else "FAIL"
                    ),
                "processing_time_ms":
                    elapsed_ms,
            },
        }

    # ================================================================
    # 5. OPEN PDF
    # ================================================================

    document = None

    try:

        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )

        page_count = len(
            document
        )

        if page_count <= 0:
            raise ValueError(
                "PDF contains no pages."
            )

    except Exception:

        elapsed_ms = round(
            (
                perf_counter()
                -
                started
            )
            * 1000,
            2,
        )

        return {
            "document_type":
                "SALARY_SLIP",
            "decision":
                "DOCUMENT_REJECTED",
            "score":
                0,
            "validation": {
                "document_detected":
                    False,
                "image_quality":
                    "POOR",
                "tampering_risk":
                    "HIGH",
                "structural_validation":
                    "FAIL",
                "processing_time_ms":
                    elapsed_ms,
            },
        }

    try:

        # ============================================================
        # 6. NATIVE TEXT
        # ============================================================

        text_parts: list[str] = []

        for page in document:

            try:

                text = page.get_text(
                    "text"
                )

            except Exception:

                text = ""

            if text:

                text_parts.append(
                    text
                )

        native_text = "\n".join(
            text_parts
        ).strip()

        native_text_available = bool(
            native_text
        )

        if native_text_available:

            classification = (
                native_classification(
                    native_text
                )
            )

            native_detected = bool(
                classification[
                    "detected"
                ]
            )

            native_score = int(
                classification[
                    "score"
                ]
            )

        else:

            classification = {
                "detected":
                    False,
                "score":
                    0,
                "groups":
                    {},
                "competitors":
                    {},
                "strongest_competitor":
                    None,
                "strongest_competitor_hits":
                    0,
            }

            native_detected = False
            native_score = 0

        # ============================================================
        # 7. REPRESENTATIVE PAGES
        # ============================================================

        page_indexes = (
            representative_page_indexes(
                page_count
            )
        )

        rendered_pages = []

        for index in page_indexes:

            try:

                image = render_page(
                    document[index]
                )

                if image is not None:

                    rendered_pages.append(
                        (
                            index,
                            image,
                        )
                    )

            except Exception:

                continue

        if not rendered_pages:

            elapsed_ms = round(
                (
                    perf_counter()
                    -
                    started
                )
                * 1000,
                2,
            )

            return {
                "document_type":
                    "SALARY_SLIP",
                "decision":
                    "DOCUMENT_REVIEW",
                "score":
                    30,
                "validation": {
                    "document_detected":
                        False,
                    "image_quality":
                        "NOT_CHECKED",
                    "tampering_risk":
                        "NOT_CHECKED",
                    "structural_validation":
                        "FAIL",
                    "processing_time_ms":
                        elapsed_ms,
                },
            }

        # ============================================================
        # 8. IMAGE QUALITY
        # ============================================================

        quality = analyze_quality(
            rendered_pages
        )

        # ============================================================
        # 9. PAGE INTEGRITY
        # ============================================================

        integrity = page_integrity(
            rendered_pages
        )

        # ============================================================
        # 10. TAMPERING
        # ============================================================

        tamper = analyze_page_tampering(
            rendered_pages
        )

        # ============================================================
        # 11. VISUAL CLASSIFICATION
        # ============================================================
        #
        # Only required for scanned PDFs.
        # No OCR is performed.
        #
        # A salary slip is generally a compact table/form with strong
        # horizontal and vertical line structure.
        #

        visual_scores = []

        for _, image in rendered_pages:

            visual = visual_salary_structure(
                image
            )

            visual_scores.append(
                visual
            )

        visual_pass = bool(
            visual_scores
            and
            sum(
                item["passed"]
                for item in visual_scores
            )
            >=
            max(
                1,
                len(visual_scores)
                // 2,
            )
        )

        visual_average = round(
            sum(
                item["score"]
                for item in visual_scores
            )
            /
            len(visual_scores),
            2,
        )

        # ============================================================
        # 12. DOCUMENT DETECTION
        # ============================================================

        detected_wrong_document = False
        detected_wrong_document_type = None

        if native_text_available:

            # Native PDF:
            # semantic salary-slip evidence is authoritative.
            document_detected = (
                native_detected
            )

            # IMPORTANT:
            # If a known other document type is present, reject it
            # instead of returning REVIEW with score 0.
            competitor_type = (
                classification.get(
                    "strongest_competitor"
                )
            )

            competitor_hits = int(
                classification.get(
                    "strongest_competitor_hits",
                    0,
                )
            )

            if (
                not native_detected
                and competitor_type
                and competitor_hits >= 2
            ):
                detected_wrong_document = True
                detected_wrong_document_type = (
                    competitor_type
                )

        else:

            # Scanned document:
            # no OCR is used. Visual validation is still performed.
            document_detected = (
                visual_pass
                and
                quality["status"]
                !=
                "POOR"
            )

        # ============================================================
        # 13. STRUCTURAL VALIDATION
        # ============================================================

        structural_pass = (
            page_count > 0
            and
            len(rendered_pages) > 0
            and
            quality["status"]
            !=
            "POOR"
        )

        # Blank/duplicate representative pages are integrity failures.
        if integrity["suspicious"]:
            structural_pass = False

        # ============================================================
        # 14. SCORE
        # ============================================================

        score = validation_score(
            document_detected=(
                document_detected
            ),
            structural_pass=(
                structural_pass
            ),
            visual_pass=(
                visual_pass
            ),
            quality_status=(
                quality["status"]
            ),
            page_integrity_pass=(
                not integrity["suspicious"]
            ),
            tamper_risk=(
                tamper["risk"]
            ),
            native_classification_score=(
                native_score
            ),
            native_text_available=(
                native_text_available
            ),
        )

        # ============================================================
        # 15. DECISION
        # ============================================================

        if tamper["risk"] in {
            "MEDIUM",
            "HIGH",
        }:

            decision = (
                "DOCUMENT_REJECTED"
            )

            score = 0

        elif detected_wrong_document:

            # Known wrong document uploaded to the Salary Slip endpoint.
            decision = (
                "DOCUMENT_REJECTED"
            )

            score = 0

        elif not document_detected:

            decision = (
                "DOCUMENT_REVIEW"
            )

        elif not structural_pass:

            decision = (
                "DOCUMENT_REVIEW"
            )

        elif quality["status"] == "POOR":

            decision = (
                "DOCUMENT_REVIEW"
            )

        elif native_text_available:

            # Native PDF must have strong salary evidence.
            if native_score >= 70:

                decision = (
                    "DOCUMENT_VERIFIED"
                )

            else:

                decision = (
                    "DOCUMENT_REVIEW"
                )

        else:

            # Scanned PDF:
            # Phase 1 validates the document itself, not extraction.
            if visual_pass:

                decision = (
                    "DOCUMENT_VERIFIED"
                )

            else:

                decision = (
                    "DOCUMENT_REVIEW"
                )

        # ============================================================
        # 16. CLEANUP
        # ============================================================

        try:
            document.close()
        except Exception:
            pass

        # ============================================================
        # 17. PROCESSING TIME
        # ============================================================

        elapsed_ms = round(
            (
                perf_counter()
                -
                started
            )
            * 1000,
            2,
        )

        # ============================================================
        # 18. CLEAN RESPONSE
        # ============================================================

        return {
            "document_type":
                (
                    detected_wrong_document_type
                    if detected_wrong_document
                    else "SALARY_SLIP"
                ),

            "decision":
                decision,

            "score":
                score,

            "validation": {
                "document_detected":
                    document_detected,

                "image_quality":
                    quality["status"],

                "tampering_risk":
                    tamper["risk"],

                "structural_validation":
                    (
                        "PASS"
                        if structural_pass
                        else "FAIL"
                    ),

                "processing_time_ms":
                    elapsed_ms,
            },
        }

    finally:

        if document is not None:

            try:
                document.close()
            except Exception:
                pass