"""
ITR DOCUMENT VERIFICATION ROUTER

Purpose:
    - ITR document classification
    - ITR validation
    - Native PDF validation
    - Scanned PDF validation without OCR
    - Wrong-document rejection
    - Tampering detection
    - Image quality
    - Structural validation
    - Processing-time reporting

IMPORTANT:

The API is an ITR endpoint.

Therefore the response always exposes:

    "document_type": "ITR"

Internally, however, we classify the uploaded document.

Example:

    Salary Slip
        ->
    detected_class = SALARY_SLIP
        ->
    DOCUMENT_REJECTED

    Genuine ITR
        ->
    detected_class = ITR
        ->
    Continue ITR validation

OCR:
    Disabled.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import fitz
import numpy as np

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..validation.validation_engine import ValidationEngine
from ..authenticity.pdf_integrity import analyze_pdf_integrity


# ======================================================================
# ROUTER
# ======================================================================

router = APIRouter(
    prefix="/api/v1/itr",
    tags=["ITR"],
)


# ======================================================================
# CONFIGURATION
# ======================================================================

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

FAST_DPI = 50

MAX_NATIVE_TEXT_PAGES = 6

MAX_VISUAL_PAGES = 3

MIN_PAGE_WIDTH = 250

MIN_PAGE_HEIGHT = 350

BLANK_STD_THRESHOLD = 5.0

RUN_DEEP_NATIVE_VALIDATION = True


# ======================================================================
# VALIDATION ENGINE
# ======================================================================

_validator = ValidationEngine()


# ======================================================================
# ITR SIGNALS
# ======================================================================

ITR_STRONG = (
    "INCOME TAX RETURN",
    "FORM ITR",
    "ITR-V",
    "ITR V",
    "RETURN OF INCOME",
    "ACKNOWLEDGEMENT NUMBER",
    "ACKNOWLEDGEMENT NO",
    "ACKNOWLEDGEMENT",
    "ACKNOWLEDGMENT",
    "E-FILING ACKNOWLEDGEMENT",
    "E-FILING ACKNOWLEDGMENT",
)


ITR_MEDIUM = (
    "ASSESSMENT YEAR",
    "INCOME TAX DEPARTMENT",
    "TOTAL INCOME",
    "GROSS TOTAL INCOME",
    "TAXABLE INCOME",
    "TAX PAYABLE",
    "TAX PAID",
    "TAX COMPUTATION",
    "COMPUTATION OF INCOME",
    "COMPUTATION OF TAX",
    "E-VERIFY",
    "E-FILING",
    "E-FILING PORTAL",
)


ITR_INCOME_HEADS = (
    "INCOME FROM SALARY",
    "INCOME FROM HOUSE PROPERTY",
    "CAPITAL GAINS",
    "INCOME FROM OTHER SOURCES",
)


# ======================================================================
# OTHER DOCUMENT SIGNALS
# ======================================================================

DOCUMENT_SIGNATURES = {

    # ------------------------------------------------------------------
    # SALARY SLIP
    # ------------------------------------------------------------------

    "SALARY_SLIP": {

        "strong": (
            "SALARY SLIP",
            "SALARYSLIP",
            "PAY SLIP",
            "PAYSLIP",
            "SALARY STATEMENT",
            "MONTHLY PAYSLIP",
            "MONTHLY SALARY",
        ),

        "medium": (
            "SALARY DETAILS",
            "BASIC SALARY",
            "BASIC PAY",
            "GROSS SALARY",
            "GROSS PAY",
            "NET SALARY",
            "NET PAY",
            "NET PAYABLE",
            "NET SALARY PAYABLE",
            "EARNINGS",
            "EARNING DETAILS",
            "TOTAL EARNINGS",
            "DEDUCTIONS",
            "DEDUCTION DETAILS",
            "TAXES & DEDUCTIONS",
            "TAXES AND DEDUCTIONS",
            "EMPLOYEE ID",
            "EMPLOYEE NUMBER",
            "EMPLOYEE NO",
            "EMPLOYEE CODE",
            "EMPLOYEE NAME",
            "DATE JOINED",
            "DATE OF JOINING",
            "JOINING DATE",
            "DEPARTMENT",
            "SUB DEPARTMENT",
            "DESIGNATION",
            "PAYMENT MODE",
            "PAY PERIOD",
            "PAY MONTH",
            "SALARY MONTH",
            "PAYABLE DAYS",
            "WORKING DAYS",
            "LOSS OF PAY",
            "PROVIDENT FUND",
            "PF EMPLOYEE",
            "PF EMPLOYER",
            "PROFESSIONAL TAX",
            "EMPLOYEE CONTRIBUTION",
            "EMPLOYER CONTRIBUTION",
            "TDS",
            "ESIC",
            "ESI",
            "UAN",
            "PF NUMBER",
            "PAN NUMBER",
            "NET SALARY IN WORDS",
            "COMPUTER GENERATED STATEMENT",
        ),
    },


    # ------------------------------------------------------------------
    # BANK STATEMENT
    # ------------------------------------------------------------------

    "BANK_STATEMENT": {

        "strong": (
            "BANK STATEMENT",
            "ACCOUNT STATEMENT",
            "TRANSACTION STATEMENT",
        ),

        "medium": (
            "TRANSACTION DATE",
            "WITHDRAWAL",
            "DEPOSIT",
            "CLOSING BALANCE",
            "AVAILABLE BALANCE",
            "OPENING BALANCE",
            "VALUE DATE",
            "DEBIT",
            "CREDIT",
            "ACCOUNT NUMBER",
            "ACCOUNT NO",
            "BALANCE",
            "IFSC",
            "NEFT",
            "RTGS",
            "IMPS",
        ),
    },


    # ------------------------------------------------------------------
    # CIBIL
    # ------------------------------------------------------------------

    "CIBIL": {

        "strong": (
            "CIBIL",
            "TRANSUNION CIBIL",
            "CIBIL TRANSUNION",
            "CREDIT INFORMATION REPORT",
            "CREDIT INFORMATION REPORT (CIR)",
        ),

        "medium": (
            "CREDIT REPORT",
            "CIBIL SCORE",
            "CREDIT SCORE",
            "CREDIT HISTORY",
            "CREDIT ACCOUNT",
            "CREDIT ACCOUNTS",
            "DAYS PAST DUE",
            "DPD",
            "OVERDUE",
            "ENQUIRY",
            "ENQUIRIES",
            "REPAYMENT HISTORY",
            "LOAN ACCOUNT",
            "ACCOUNT STATUS",
            "PAYMENT HISTORY",
            "HIGH CREDIT",
            "CURRENT BALANCE",
            "SANCTIONED AMOUNT",
        ),
    },


    # ------------------------------------------------------------------
    # PAN
    # ------------------------------------------------------------------

    "PAN": {

        "strong": (
            "PAN CARD",
            "PERMANENT ACCOUNT NUMBER",
        ),

        "medium": (
            "DATE OF BIRTH",
            "FATHER'S NAME",
            "FATHER NAME",
        ),
    },


    # ------------------------------------------------------------------
    # PASSPORT
    # ------------------------------------------------------------------

    "PASSPORT": {

        "strong": (
            "PASSPORT",
            "REPUBLIC OF INDIA",
        ),

        "medium": (
            "NATIONALITY",
            "DATE OF EXPIRY",
            "DATE OF ISSUE",
            "PLACE OF BIRTH",
            "PASSPORT NO",
            "PASSPORT NUMBER",
        ),
    },


    # ------------------------------------------------------------------
    # DRIVING LICENCE
    # ------------------------------------------------------------------

    "DRIVING_LICENCE": {

        "strong": (
            "DRIVING LICENCE",
            "DRIVING LICENSE",
        ),

        "medium": (
            "DL NO",
            "LICENCE NO",
            "LICENSE NO",
            "TRANSPORT DEPARTMENT",
            "VALID TILL",
            "VALIDITY",
        ),
    },


    # ------------------------------------------------------------------
    # SALE DEED
    # ------------------------------------------------------------------

    "SALE_DEED": {

        "strong": (
            "SALE DEED",
            "CONVEYANCE DEED",
        ),

        "medium": (
            "PURCHASER",
            "SELLER",
            "PROPERTY SCHEDULE",
            "CONSIDERATION",
            "WITNESS",
            "REGISTRATION",
            "REGISTRATION NUMBER",
        ),
    },
}


# ======================================================================
# TEXT NORMALIZATION
# ======================================================================

def normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    ).upper()

    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ======================================================================
# TERM SEARCH
# ======================================================================

def find_hits(
    text: str,
    terms: tuple[str, ...],
) -> list[str]:

    return [
        term
        for term in terms
        if term in text
    ]


# ======================================================================
# SALARY SLIP CLASSIFICATION
# ======================================================================

def classify_salary_slip(
    text: str,
) -> dict[str, Any]:
    """
    Strong salary-slip classifier.

    Generic words such as TAX, TDS, PAN and INCOME are deliberately
    ignored as standalone salary evidence. Salary classification is
    based on salary-specific structure.
    """

    text = normalize_text(text)
    signature = DOCUMENT_SIGNATURES["SALARY_SLIP"]

    strong_hits = find_hits(text, signature["strong"])
    medium_hits = find_hits(text, signature["medium"])

    has_payslip = any(
        term in text
        for term in (
            "PAYSLIP",
            "PAY SLIP",
            "SALARY SLIP",
            "SALARYSLIP",
            "SALARY STATEMENT",
            "MONTHLY PAYSLIP",
        )
    )

    has_salary_details = "SALARY DETAILS" in text

    has_earnings = any(
        term in text
        for term in (
            "EARNINGS",
            "EARNING DETAILS",
            "TOTAL EARNINGS",
            "GROSS SALARY",
            "GROSS PAY",
        )
    )

    has_deductions = any(
        term in text
        for term in (
            "DEDUCTIONS",
            "DEDUCTION DETAILS",
            "TAXES & DEDUCTIONS",
            "TAXES AND DEDUCTIONS",
        )
    )

    has_net_salary = any(
        term in text
        for term in (
            "NET SALARY PAYABLE",
            "NET SALARY",
            "NET PAYABLE",
            "NET PAY",
            "TAKE HOME",
            "TAKE-HOME",
        )
    )

    has_employee_details = any(
        term in text
        for term in (
            "EMPLOYEE NUMBER",
            "EMPLOYEE NO",
            "EMPLOYEE ID",
            "EMPLOYEE CODE",
            "EMPLOYEE NAME",
        )
    )

    has_pay_period = any(
        term in text
        for term in (
            "PAY PERIOD",
            "PAY MONTH",
            "SALARY MONTH",
            "SALARY SLIP FOR THE MONTH",
            "FOR THE MONTH OF",
            "PAYABLE DAYS",
            "WORKING DAYS",
        )
    )

    has_pf_or_contribution = any(
        term in text
        for term in (
            "PROVIDENT FUND",
            "PF EMPLOYEE",
            "PF EMPLOYER",
            "PF NUMBER",
            "EMPLOYEE CONTRIBUTION",
            "EMPLOYER CONTRIBUTION",
            "UAN",
            "ESIC",
            "ESI",
        )
    )

    # Salary-specific score only.
    score = 0

    if has_payslip:
        score += 40
    if has_salary_details:
        score += 15
    if has_earnings:
        score += 15
    if has_deductions:
        score += 10
    if has_net_salary:
        score += 15
    if has_employee_details:
        score += 5
    if has_pay_period:
        score += 5
    if has_pf_or_contribution:
        score += 5

    score = min(100, score)

    strong_combination = (
        has_payslip
        and has_earnings
        and has_net_salary
    )

    complete_salary_structure = (
        has_salary_details
        and has_earnings
        and has_deductions
        and has_net_salary
    )

    employee_salary_structure = (
        has_employee_details
        and has_earnings
        and has_net_salary
    )

    salary_without_title = (
        has_earnings
        and has_deductions
        and has_net_salary
        and (
            has_employee_details
            or has_pay_period
            or has_salary_details
        )
    )

    confident = bool(
        strong_combination
        or complete_salary_structure
        or employee_salary_structure
        or salary_without_title
    )

    # OCR/text extraction may damage the title. A payslip title plus
    # multiple salary-specific fields is still enough to reject it.
    if (
        not confident
        and has_payslip
        and len(medium_hits) >= 2
    ):
        confident = True

    return {
        "score": int(score),
        "confident": confident,
        "strong_hits": strong_hits,
        "medium_hits": medium_hits,
        "flags": {
            "payslip": has_payslip,
            "salary_details": has_salary_details,
            "earnings": has_earnings,
            "deductions": has_deductions,
            "net_salary": has_net_salary,
            "employee_details": has_employee_details,
            "pay_period": has_pay_period,
            "pf_or_contribution": has_pf_or_contribution,
        },
    }


# ======================================================================
# ITR CLASSIFICATION
# ======================================================================

def classify_itr(
    text: str,
) -> dict[str, Any]:

    text = normalize_text(
        text
    )

    strong_hits = find_hits(
        text,
        ITR_STRONG,
    )

    medium_hits = find_hits(
        text,
        ITR_MEDIUM,
    )

    income_hits = find_hits(
        text,
        ITR_INCOME_HEADS,
    )

    has_itr_identity = bool(
        "INCOME TAX RETURN" in text
        or
        "FORM ITR" in text
        or
        "ITR-V" in text
        or
        "ITR V" in text
        or
        "RETURN OF INCOME" in text
    )

    has_assessment_year = (
        "ASSESSMENT YEAR" in text
    )

    has_tax_context = bool(
        "TOTAL INCOME" in text
        or
        "TAXABLE INCOME" in text
        or
        "TAX PAYABLE" in text
        or
        "TAX COMPUTATION" in text
        or
        "COMPUTATION OF INCOME" in text
    )

    has_acknowledgement = bool(
        "ACKNOWLEDGEMENT" in text
        or
        "ACKNOWLEDGMENT" in text
    )

    has_verification = bool(
        "VERIFICATION" in text
        or
        "E-VERIFY" in text
    )

    score = 0

    if has_itr_identity:
        score += 45

    if has_assessment_year:
        score += 25

    if has_tax_context:
        score += 15

    if has_acknowledgement:
        score += 10

    score += min(
        10,
        len(income_hits) * 5,
    )

    if has_verification:
        score += 5

    score = min(
        100,
        score,
    )

    # --------------------------------------------------------------
    # Strong ITR combinations
    # --------------------------------------------------------------

    combination_1 = (
        has_itr_identity
        and
        has_assessment_year
    )

    combination_2 = (
        has_assessment_year
        and
        has_tax_context
        and
        (
            len(income_hits) >= 1
            or
            has_acknowledgement
            or
            has_verification
        )
    )

    combination_3 = (
        has_acknowledgement
        and
        has_assessment_year
        and
        has_tax_context
    )

    confident = bool(
        combination_1
        or
        combination_2
        or
        combination_3
    )

    return {
        "score":
            int(score),

        "confident":
            confident,

        "strong_hits":
            strong_hits,

        "medium_hits":
            medium_hits,

        "income_hits":
            income_hits,

        "flags": {
            "itr_identity":
                has_itr_identity,

            "assessment_year":
                has_assessment_year,

            "tax_context":
                has_tax_context,

            "acknowledgement":
                has_acknowledgement,

            "verification":
                has_verification,
        },
    }


# ======================================================================
# GENERIC DOCUMENT CLASSIFICATION
# ======================================================================

def classify_generic_document(
    text: str,
    document_type: str,
) -> dict[str, Any]:

    signature = DOCUMENT_SIGNATURES[
        document_type
    ]

    strong_hits = find_hits(
        text,
        signature["strong"],
    )

    medium_hits = find_hits(
        text,
        signature["medium"],
    )

    score = 0

    score += min(
        60,
        len(strong_hits) * 30,
    )

    score += min(
        40,
        len(medium_hits) * 6,
    )

    return {
        "score":
            min(
                100,
                score,
            ),

        "strong_hits":
            strong_hits,

        "medium_hits":
            medium_hits,
    }


# ======================================================================
# MASTER CLASSIFIER
# ======================================================================

def classify_document(
    text: str,
) -> dict[str, Any]:
    """
    Master multi-document classifier.

    The ITR route classifies the actual uploaded document first.
    Only a document whose winning classification is ITR is allowed
    to continue into ITR validation.
    """

    text = normalize_text(text)

    scores: dict[str, dict[str, Any]] = {}

    itr_result = classify_itr(text)

    scores["ITR"] = {
        "score": itr_result["score"],
        "strong_hits": itr_result["strong_hits"],
        "medium_hits": itr_result["medium_hits"],
    }

    salary_result = classify_salary_slip(text)

    scores["SALARY_SLIP"] = {
        "score": salary_result["score"],
        "strong_hits": salary_result["strong_hits"],
        "medium_hits": salary_result["medium_hits"],
    }

    for document_type in (
        "BANK_STATEMENT",
        "CIBIL",
        "PAN",
        "PASSPORT",
        "DRIVING_LICENCE",
        "SALE_DEED",
    ):
        scores[document_type] = classify_generic_document(
            text,
            document_type,
        )

    ranking = sorted(
        (
            (document_type, data["score"])
            for document_type, data in scores.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    best_type = ranking[0][0]
    best_score = ranking[0][1]
    second_score = ranking[1][1] if len(ranking) > 1 else 0
    margin = best_score - second_score

    # HARD PRIORITY:
    # A genuine payslip commonly contains TAX, TDS, PAN and INCOME.
    # Those generic terms must never turn a payslip into an ITR.
    if salary_result["confident"]:
        detected_type = "SALARY_SLIP"
        best_score = salary_result["score"]

    elif (
        best_type == "ITR"
        and itr_result["confident"]
        and (
            margin >= 10
            or best_score >= 90
        )
    ):
        detected_type = "ITR"

    elif (
        best_type != "ITR"
        and best_score >= 35
        and margin >= 8
    ):
        detected_type = best_type

    else:
        detected_type = "UNKNOWN"

    return {
        "detected_type": detected_type,
        "classification_score": int(best_score),
        "second_score": int(second_score),
        "margin": int(margin),
        "ranking": ranking,
        "scores": scores,
        "itr": itr_result,
        "salary_slip": salary_result,
    }


# ======================================================================
# FAST PDF IMAGE RENDERING
# ======================================================================

def render_fast(
    page: fitz.Page,
) -> np.ndarray | None:

    try:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(
                FAST_DPI / 72.0,
                FAST_DPI / 72.0,
            ),
            alpha=False,
        )

    except Exception:

        return None

    if pix.n == 4:

        array = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
            4,
        )

        return cv2.cvtColor(
            array,
            cv2.COLOR_RGBA2BGR,
        )

    array = np.frombuffer(
        pix.samples,
        dtype=np.uint8,
    ).reshape(
        pix.height,
        pix.width,
        3,
    )

    return cv2.cvtColor(
        array,
        cv2.COLOR_RGB2BGR,
    )


# ======================================================================
# IMAGE QUALITY
# ======================================================================

def calculate_image_quality(
    image: np.ndarray,
) -> tuple[
    float,
    str,
]:

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
        gray.mean()
    )

    contrast = float(
        gray.std()
    )

    score = 0

    if sharpness >= 80:

        score += 40

    elif sharpness >= 25:

        score += 25

    elif sharpness >= 10:

        score += 15

    else:

        score += 5

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

    if score >= 70:

        quality = "GOOD"

    elif score >= 45:

        quality = "FAIR"

    else:

        quality = "POOR"

    return (
        float(score),
        quality,
    )


# ======================================================================
# SCANNED VISUAL CHECK
# ======================================================================

def visual_itr_score(
    image: np.ndarray,
) -> int:

    if (
        image is None
        or
        image.size == 0
    ):

        return 0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = (
        gray.shape
    )

    if (
        width < MIN_PAGE_WIDTH
        or
        height < MIN_PAGE_HEIGHT
    ):

        return 0

    target_width = 500

    target_height = max(
        500,
        int(
            target_width
            *
            height
            /
            max(
                1,
                width,
            )
        ),
    )

    gray = cv2.resize(
        gray,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    height, width = (
        gray.shape
    )

    total = float(
        height * width
    )

    edges = cv2.Canny(
        gray,
        60,
        160,
    )

    edge_ratio = (
        np.count_nonzero(
            edges
        )
        /
        total
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    15,
                    width // 22,
                ),
                1,
            ),
        )
    )

    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    horizontal_ratio = (
        np.count_nonzero(
            horizontal
        )
        /
        total
    )

    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(
                    15,
                    height // 22,
                ),
            ),
        )
    )

    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    vertical_ratio = (
        np.count_nonzero(
            vertical
        )
        /
        total
    )

    score = 0

    if (
        0.006
        <= edge_ratio
        <= 0.18
    ):

        score += 25

    elif edge_ratio >= 0.003:

        score += 12

    if horizontal_ratio >= 0.0025:

        score += 30

    elif horizontal_ratio >= 0.001:

        score += 18

    if vertical_ratio >= 0.001:

        score += 25

    elif vertical_ratio >= 0.0005:

        score += 15

    if (
        horizontal_ratio >= 0.001
        and
        vertical_ratio >= 0.0005
    ):

        score += 20

    return min(
        100,
        score,
    )


# ======================================================================
# SCANNED ITR CLASSIFICATION
# ======================================================================

def classify_scanned_itr(
    pdf: fitz.Document,
) -> tuple[
    str,
    int,
    np.ndarray | None,
]:

    page_count = len(
        pdf
    )

    if page_count <= 0:

        return (
            "UNKNOWN",
            0,
            None,
        )

    first_image = render_fast(
        pdf[0]
    )

    if first_image is None:

        return (
            "UNKNOWN",
            0,
            None,
        )

    first_score = visual_itr_score(
        first_image
    )

    if first_score >= 85:

        return (
            "ITR",
            first_score,
            first_image,
        )

    scores = [
        first_score
    ]

    indexes = []

    if page_count > 1:

        indexes.append(
            page_count // 2
        )

    if page_count > 2:

        indexes.append(
            page_count - 1
        )

    seen = {0}

    for index in indexes:

        if index in seen:

            continue

        seen.add(index)

        try:

            image = render_fast(
                pdf[index]
            )

            if image is not None:

                scores.append(
                    visual_itr_score(
                        image
                    )
                )

        except Exception:

            continue

        if len(scores) >= MAX_VISUAL_PAGES:

            break

    average_score = int(
        round(
            sum(scores)
            /
            len(scores)
        )
    )

    if average_score >= 75:

        return (
            "ITR",
            average_score,
            first_image,
        )

    return (
        "UNKNOWN",
        average_score,
        first_image,
    )


# ======================================================================
# TAMPERING
# ======================================================================

def check_tampering(
    file_bytes: bytes,
) -> str:

    try:

        result = analyze_pdf_integrity(
            file_bytes
        )

        risk = str(
            getattr(
                result,
                "risk_level",
                "LOW",
            )
        ).upper()

        if (
            "CRITICAL" in risk
            or
            "HIGH" in risk
        ):

            return "HIGH"

        if "MEDIUM" in risk:

            return "MEDIUM"

        return "LOW"

    except Exception:

        return "LOW"


# ======================================================================
# SCORE
# ======================================================================

def calculate_score(
    classification_score: int,
    quality_score: float,
    structural_pass: bool,
    tampering: str,
) -> int:

    score = (
        classification_score
        * 0.50
    )

    score += (
        quality_score
        * 0.25
    )

    if structural_pass:

        score += 15

    if tampering == "LOW":

        score += 10

    elif tampering == "MEDIUM":

        score -= 20

    return max(
        0,
        min(
            100,
            int(
                round(
                    score
                )
            ),
        ),
    )


# ======================================================================
# RESPONSE
# ======================================================================

def make_response(
    *,
    decision: str,
    score: int,
    detected: bool,
    quality: str,
    tampering: str,
    structural: str,
    started: float,
) -> dict[str, Any]:

    return {
        "document_type":
            "ITR",

        "decision":
            decision,

        "score":
            max(
                0,
                min(
                    100,
                    int(score),
                ),
            ),

        "validation": {

            "document_detected":
                detected,

            "image_quality":
                quality,

            "tampering_risk":
                tampering,

            "structural_validation":
                structural,

            "processing_time_ms":
                round(
                    (
                        perf_counter()
                        -
                        started
                    )
                    * 1000,
                    2,
                ),
        },
    }


# ======================================================================
# VERIFY ITR
# ======================================================================

@router.post(
    "/verify",
)
async def verify_itr(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    started = perf_counter()

    # ================================================================
    # FILE
    # ================================================================

    filename = Path(
        str(
            file.filename
            or
            ""
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

    if extension != ".pdf":

        raise HTTPException(
            status_code=400,
            detail="Upload only PDF files.",
        )

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if (
        len(file_bytes)
        >
        MAX_FILE_SIZE_BYTES
    ):

        raise HTTPException(
            status_code=413,
            detail="PDF exceeds 50 MB.",
        )

    # ================================================================
    # OPEN PDF
    # ================================================================

    try:

        pdf = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )

    except Exception:

        return make_response(
            decision="DOCUMENT_REJECTED",
            score=0,
            detected=False,
            quality="POOR",
            tampering="HIGH",
            structural="FAIL",
            started=started,
        )

    try:

        page_count = len(
            pdf
        )

        if page_count <= 0:

            return make_response(
                decision="DOCUMENT_REJECTED",
                score=0,
                detected=False,
                quality="POOR",
                tampering="HIGH",
                structural="FAIL",
                started=started,
            )

        # ============================================================
        # NATIVE TEXT
        # ============================================================

        text_parts = []

        for index in range(
            min(
                page_count,
                MAX_NATIVE_TEXT_PAGES,
            )
        ):

            try:

                page_text = pdf[
                    index
                ].get_text(
                    "text"
                )

                if page_text:

                    text_parts.append(
                        page_text
                    )

            except Exception:

                continue

        native_text = normalize_text(
            "\n".join(
                text_parts
            )
        )

        native_pdf = bool(
            native_text
        )

        # ============================================================
        # CLASSIFICATION
        # ============================================================

        if native_pdf:

            classification = (
                classify_document(
                    native_text
                )
            )

            detected_type = (
                classification[
                    "detected_type"
                ]
            )

            classification_score = int(
                classification[
                    "classification_score"
                ]
            )

        else:

            (
                detected_type,
                classification_score,
                first_image,
            ) = classify_scanned_itr(
                pdf
            )

        # ============================================================
        # WRONG DOCUMENT
        # ============================================================

        #
        # THIS IS THE IMPORTANT PART.
        #
        # If uploaded PDF is a salary slip:
        #
        #     detected_type = SALARY_SLIP
        #
        # therefore:
        #
        #     DOCUMENT_REJECTED
        #
        # It will NOT run:
        #
        #     image quality
        #     tampering
        #     structural validation
        #     ITR validation
        #
        # This keeps the response fast.
        #

        # detected_type is the ACTUAL uploaded document.
        # This endpoint is /itr, so any non-ITR document is rejected
        # before image quality, tampering or ITR validation.
        if detected_type != "ITR":

            return make_response(
                decision="DOCUMENT_REJECTED",
                score=0,
                detected=False,
                quality="NOT_CHECKED",
                tampering="LOW",
                structural="NOT_CHECKED",
                started=started,
            )

        # ============================================================
        # ONLY ITR CONTINUES
        # ============================================================

        if native_pdf:

            first_image = render_fast(
                pdf[0]
            )

        if first_image is None:

            return make_response(
                decision="DOCUMENT_REVIEW",
                score=0,
                detected=False,
                quality="NOT_CHECKED",
                tampering="LOW",
                structural="FAIL",
                started=started,
            )

        # ============================================================
        # IMAGE QUALITY
        # ============================================================

        quality_score, quality_label = (
            calculate_image_quality(
                first_image
            )
        )

        # ============================================================
        # STRUCTURAL VALIDATION
        # ============================================================

        gray = cv2.cvtColor(
            first_image,
            cv2.COLOR_BGR2GRAY,
        )

        height, width = (
            gray.shape
        )

        structural_pass = bool(
            width >= MIN_PAGE_WIDTH
            and
            height >= MIN_PAGE_HEIGHT
            and
            float(
                gray.std()
            )
            >
            BLANK_STD_THRESHOLD
        )

        # ============================================================
        # TAMPERING
        # ============================================================

        tampering = check_tampering(
            file_bytes
        )

        # ============================================================
        # HIGH TAMPERING
        # ============================================================

        if tampering == "HIGH":

            return make_response(
                decision="DOCUMENT_REJECTED",
                score=0,
                detected=True,
                quality=quality_label,
                tampering="HIGH",
                structural=(
                    "PASS"
                    if structural_pass
                    else "FAIL"
                ),
                started=started,
            )

        # ============================================================
        # SCORE
        # ============================================================

        score = calculate_score(
            classification_score=(
                classification_score
            ),
            quality_score=(
                quality_score
            ),
            structural_pass=(
                structural_pass
            ),
            tampering=(
                tampering
            ),
        )

        # ============================================================
        # STRUCTURE
        # ============================================================

        if not structural_pass:

            return make_response(
                decision="DOCUMENT_REVIEW",
                score=score,
                detected=True,
                quality=quality_label,
                tampering=tampering,
                structural="FAIL",
                started=started,
            )

        # ============================================================
        # QUALITY
        # ============================================================

        if quality_label == "POOR":

            return make_response(
                decision="DOCUMENT_REVIEW",
                score=score,
                detected=True,
                quality=quality_label,
                tampering=tampering,
                structural="PASS",
                started=started,
            )

        # ============================================================
        # MEDIUM TAMPERING
        # ============================================================

        if tampering == "MEDIUM":

            return make_response(
                decision="DOCUMENT_REVIEW",
                score=score,
                detected=True,
                quality=quality_label,
                tampering="MEDIUM",
                structural="PASS",
                started=started,
            )

        # ============================================================
        # EXISTING ITR VALIDATION
        # ============================================================

        validation_ok = True

        if (
            native_pdf
            and
            RUN_DEEP_NATIVE_VALIDATION
            and
            classification_score >= 85
        ):

            try:

                validation_result = (
                    _validator.validate_file(
                        file_bytes
                    )
                )

                validation_ok = bool(
                    getattr(
                        validation_result,
                        "valid",
                        False,
                    )
                )

            except TypeError:

                validation_ok = True

            except Exception:

                validation_ok = False

        # ============================================================
        # FINAL DECISION
        # ============================================================

        if native_pdf:

            if (
                classification_score >= 85
                and
                validation_ok
                and
                tampering == "LOW"
                and
                quality_label != "POOR"
            ):

                decision = (
                    "DOCUMENT_VERIFIED"
                )

            elif (
                classification_score >= 92
                and
                structural_pass
                and
                tampering == "LOW"
            ):

                decision = (
                    "DOCUMENT_VERIFIED"
                )

            else:

                decision = (
                    "DOCUMENT_REVIEW"
                )

        else:

            if (
                classification_score >= 80
                and
                structural_pass
                and
                tampering == "LOW"
                and
                quality_label != "POOR"
            ):

                decision = (
                    "DOCUMENT_VERIFIED"
                )

            else:

                decision = (
                    "DOCUMENT_REVIEW"
                )

        # ============================================================
        # RESPONSE
        # ============================================================

        return make_response(
            decision=decision,
            score=score,
            detected=True,
            quality=quality_label,
            tampering=tampering,
            structural=(
                "PASS"
                if structural_pass
                else "FAIL"
            ),
            started=started,
        )

    finally:

        try:

            pdf.close()

        except Exception:

            pass