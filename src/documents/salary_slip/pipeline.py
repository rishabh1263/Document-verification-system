"""
End-to-end document verification pipeline.

Flow:
    document
        ->
    OCR / native text extraction
        ->
    document classification
        ->
    page-aware structured field extraction
        ->
    forgery checks
        ->
    combined risk score

Important:
- Single-page salary slip -> normal extracted_fields response.
- Multi-page salary-slip PDF -> each page is extracted independently.
- We do NOT mix salary values from different pages/months.
- No company/template-specific hardcoding is done here.

This function is used by:
- CLI
- FastAPI
- future UI / .NET integration
"""

import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

from src.documents.salary_slip import classifier, ocr_engine
from src.documents.salary_slip.extractors import get_extractor
from src.documents.salary_slip.forgery import (
    consistency_check,
    ela_check,
    metadata_check,
    pdf_structure_check,
    risk_score,
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_document(file_path: str) -> Dict[str, Any]:
    """
    Process one document through the complete verification pipeline.

    Salary-slip behaviour:

    Single page:
        extracted_fields = {...}

    Multiple pages:
        salary_slips = [
            {
                "page": 1,
                "extracted_fields": {...},
                ...
            },
            ...
        ]

    Each page is extracted independently so values from different
    salary-slip months/pages are never mixed.
    """

    # ========================================================
    # VALIDATE FILE
    # ========================================================

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    if not os.path.isfile(file_path):
        raise ValueError(
            f"Expected a document file, got: {file_path}"
        )

    t_start = time.time()

    timing: Dict[str, float] = {}

    # ========================================================
    # 1. OCR / NATIVE TEXT EXTRACTION
    # ========================================================

    t0 = time.time()

    ocr_result = ocr_engine.extract_text(
        file_path
    )

    timing["text_extraction_sec"] = round(
        time.time() - t0,
        3,
    )

    if not ocr_result.text.strip():
        raise ValueError(
            "No readable text could be extracted from the document."
        )

    # ========================================================
    # 2. DOCUMENT CLASSIFICATION
    # ========================================================

    t0 = time.time()

    doc_type, confidence = classifier.classify(
        ocr_result.text
    )

    timing["classification_sec"] = round(
        time.time() - t0,
        3,
    )

    # ========================================================
    # 3. STRUCTURED FIELD EXTRACTION
    # ========================================================

    t0 = time.time()

    extractor = get_extractor(
        doc_type
    )

    # --------------------------------------------------------
    # Salary slips need page-aware extraction.
    # --------------------------------------------------------

    if (
        doc_type == "salary_slip"
        and ocr_result.page_count > 1
    ):

        salary_slips = _extract_salary_slips_by_page(
            extractor=extractor,
            ocr_result=ocr_result,
        )

        timing["field_extraction_sec"] = round(
            time.time() - t0,
            3,
        )

        # ====================================================
        # 4. DOCUMENT-LEVEL VERIFICATION
        # ====================================================

        t0 = time.time()

        meta_result = metadata_check.check(
            file_path
        )

        ela_result = _run_ela(
            file_path
        )

        pdf_structure_result = pdf_structure_check.check(
            file_path
        )

        # ----------------------------------------------------
        # Run consistency independently for every salary slip.
        # ----------------------------------------------------

        page_consistency_results = []

        for slip in salary_slips:

            fields = slip.get(
                "extracted_fields",
                {},
            )

            consistency = consistency_check.check(
                "salary_slip",
                fields,
            )

            slip["consistency"] = consistency

            page_consistency_results.append(
                consistency
            )

        # ----------------------------------------------------
        # Aggregate page-level consistency into one
        # document-level consistency signal.
        # ----------------------------------------------------

        consistency_result = _combine_page_consistency(
            page_consistency_results
        )

        timing["forgery_checks_sec"] = round(
            time.time() - t0,
            3,
        )

        # ====================================================
        # 5. COMBINED RISK
        # ====================================================

        t0 = time.time()

        risk = risk_score.combine(
            meta_result,
            ela_result,
            consistency_result,
            pdf_structure_result,
        )

        timing["risk_aggregation_sec"] = round(
            time.time() - t0,
            3,
        )

        # ====================================================
        # 6. TOTAL TIMING
        # ====================================================

        timing["total_sec"] = round(
            time.time() - t_start,
            3,
        )

        # ====================================================
        # 7. MULTI-PAGE RESULT
        # ====================================================

        return {
            "file": os.path.basename(
                file_path
            ),

            "doc_type": doc_type,

            "doc_type_confidence": round(
                confidence,
                2,
            ),

            "ocr_source": ocr_result.source,

            "page_count": ocr_result.page_count,

            "salary_slips": salary_slips,

            "risk_assessment": risk,

            "timing": timing,
        }

    # ========================================================
    # SINGLE DOCUMENT / SINGLE PAGE EXTRACTION
    # ========================================================

    fields = _extract_fields(
        extractor=extractor,
        text=ocr_result.text,
        words=ocr_result.words,
        doc_type=doc_type,
    )

    timing["field_extraction_sec"] = round(
        time.time() - t0,
        3,
    )

    # ========================================================
    # 4. FORGERY CHECKS
    # ========================================================

    t0 = time.time()

    meta_result = metadata_check.check(
        file_path
    )

    ela_result = _run_ela(
        file_path
    )

    pdf_structure_result = pdf_structure_check.check(
        file_path
    )

    consistency_result = consistency_check.check(
        doc_type,
        fields,
    )

    timing["forgery_checks_sec"] = round(
        time.time() - t0,
        3,
    )

    # ========================================================
    # 5. COMBINED RISK SCORE
    # ========================================================

    t0 = time.time()

    risk = risk_score.combine(
        meta_result,
        ela_result,
        consistency_result,
        pdf_structure_result,
    )

    timing["risk_aggregation_sec"] = round(
        time.time() - t0,
        3,
    )

    # ========================================================
    # 6. TOTAL TIMING
    # ========================================================

    timing["total_sec"] = round(
        time.time() - t_start,
        3,
    )

    # ========================================================
    # 7. FINAL RESULT
    # ========================================================

    return {
        "file": os.path.basename(
            file_path
        ),

        "doc_type": doc_type,

        "doc_type_confidence": round(
            confidence,
            2,
        ),

        "ocr_source": ocr_result.source,

        "page_count": ocr_result.page_count,

        "extracted_fields": fields,

        "risk_assessment": risk,

        "timing": timing,
    }


# ============================================================
# GENERIC FIELD EXTRACTION
# ============================================================

def _extract_fields(
    extractor,
    text: str,
    words: List,
    doc_type: str,
) -> Dict[str, Any]:
    """
    Extract fields using layout-aware extraction when supported.

    Salary slips:
        extract_with_layout(text, words)

    Other documents:
        extract(text)

    This keeps pipeline.py generic.
    """

    if (
        doc_type == "salary_slip"
        and hasattr(
            extractor,
            "extract_with_layout",
        )
        and words
    ):

        try:

            return extractor.extract_with_layout(
                text,
                words,
            )

        except Exception:

            # Layout extraction should not kill the entire API.
            # Fall back to normal text extraction.
            return extractor.extract(
                text
            )

    return extractor.extract(
        text
    )


# ============================================================
# MULTI-PAGE SALARY-SLIP EXTRACTION
# ============================================================

def _extract_salary_slips_by_page(
    extractor,
    ocr_result,
) -> List[Dict[str, Any]]:
    """
    Extract each salary-slip page independently.

    This is critical for PDFs containing multiple monthly salary slips.

    BAD:
        Page 1 + Page 2 + Page 3 text
                    ->
              one extraction

    GOOD:
        Page 1 -> extraction
        Page 2 -> extraction
        Page 3 -> extraction

    No salary values can leak between pages.
    """

    words_by_page = defaultdict(
        list
    )

    # ========================================================
    # GROUP WORDS BY PAGE
    # ========================================================

    for word in ocr_result.words:

        page_number = getattr(
            word,
            "page_number",
            1,
        )

        words_by_page[
            page_number
        ].append(
            word
        )

    salary_slips = []

    # ========================================================
    # PROCESS EVERY PAGE
    # ========================================================

    for page_number in range(
        1,
        ocr_result.page_count + 1,
    ):

        page_words = words_by_page.get(
            page_number,
            [],
        )

        # ----------------------------------------------------
        # Reconstruct text only from this page.
        # ----------------------------------------------------

        page_text = _words_to_page_text(
            page_words
        )

        if not page_text.strip():
            continue

        # ----------------------------------------------------
        # Confirm this page actually resembles a salary slip.
        #
        # We classify per page because a multi-page document
        # could theoretically contain unrelated attachments.
        # ----------------------------------------------------

        page_doc_type, page_confidence = classifier.classify(
            page_text
        )

        # ----------------------------------------------------
        # If document-level classification says salary slip,
        # we still allow low-confidence salary pages.
        #
        # But if the page is strongly classified as something
        # completely different, don't force salary extraction.
        # ----------------------------------------------------

        if page_doc_type != "salary_slip":

            continue

        # ----------------------------------------------------
        # Extract this page independently.
        # ----------------------------------------------------

        fields = _extract_fields(
            extractor=extractor,
            text=page_text,
            words=page_words,
            doc_type="salary_slip",
        )

        salary_slips.append(
            {
                "page": page_number,

                "document_type": page_doc_type,

                "classification_confidence": round(
                    page_confidence,
                    2,
                ),

                "extracted_fields": fields,
            }
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    # If page-level classification was too strict and rejected
    # everything, process each readable page as a salary slip
    # because the whole document was already classified as one.

    if not salary_slips:

        for page_number in range(
            1,
            ocr_result.page_count + 1,
        ):

            page_words = words_by_page.get(
                page_number,
                [],
            )

            page_text = _words_to_page_text(
                page_words
            )

            if not page_text.strip():
                continue

            fields = _extract_fields(
                extractor=extractor,
                text=page_text,
                words=page_words,
                doc_type="salary_slip",
            )

            salary_slips.append(
                {
                    "page": page_number,

                    "document_type": "salary_slip",

                    "classification_confidence": None,

                    "extracted_fields": fields,
                }
            )

    # ========================================================
    # DOCUMENT-LEVEL STABLE IDENTITY CONSENSUS
    # ========================================================
    #
    # Salary amounts and pay periods MUST remain page-specific.
    # Only stable employee identity fields may be propagated.
    #
    # A value is propagated only when:
    #   1. it is present on at least one page, and
    #   2. all non-empty normalized values agree.
    #
    # This avoids copying conflicting OCR values across pages.
    salary_slips = _propagate_stable_salary_identity(
        salary_slips
    )

    return salary_slips


# ============================================================
# STABLE MULTI-PAGE IDENTITY PROPAGATION
# ============================================================

def _propagate_stable_salary_identity(
    salary_slips: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fill missing stable employee identity fields across pages of the
    same multi-page salary-slip document.

    Never propagated:
        pay_period
        gross_pay
        basic_pay
        net_pay
        total_deductions
        designation

    Safe propagation requires unanimous agreement among all non-empty
    normalized values observed in the document. If pages disagree, the
    field is left untouched everywhere it is missing.
    """

    if not salary_slips:
        return salary_slips

    stable_fields = (
        "employee_name",
        "employee_id",
        "pan",
        "bank_account",
    )

    def normalize(field: str, value: Any) -> str:
        if value is None:
            return ""

        text = str(value).strip()

        if not text:
            return ""

        if field == "employee_name":
            return re.sub(
                r"[^A-Z0-9]",
                "",
                text.upper(),
            )

        return re.sub(
            r"[^A-Z0-9]",
            "",
            text.upper(),
        )

    consensus: Dict[str, Any] = {}

    for field in stable_fields:

        observed = []

        for slip in salary_slips:

            fields = slip.get(
                "extracted_fields",
                {},
            )

            value = fields.get(
                field
            )

            normalized = normalize(
                field,
                value,
            )

            if normalized:
                observed.append(
                    (
                        normalized,
                        value,
                    )
                )

        if not observed:
            continue

        unique_normalized = {
            normalized
            for normalized, _ in observed
        }

        # Conflicting values -> do not propagate.
        if len(unique_normalized) != 1:
            continue

        # Preserve the most common original representation.
        original_values = [
            str(value).strip()
            for _, value in observed
            if value is not None
        ]

        if not original_values:
            continue

        consensus[field] = Counter(
            original_values
        ).most_common(1)[0][0]

    # Fill only missing fields. Never overwrite page-level extraction.
    for slip in salary_slips:

        fields = slip.get(
            "extracted_fields",
            {},
        )

        for field, consensus_value in consensus.items():

            current = fields.get(
                field
            )

            if current is None or not str(current).strip():

                fields[field] = consensus_value

    return salary_slips


# ============================================================
# PAGE TEXT RECONSTRUCTION
# ============================================================

def _words_to_page_text(
    words: List,
) -> str:
    """
    Convert one page's OCR words back into approximate text.

    Prefer the helper from ocr_engine.py.

    A local fallback is kept here so the pipeline remains robust
    if the helper changes later.
    """

    if not words:
        return ""

    # ========================================================
    # USE OCR ENGINE HELPER
    # ========================================================

    if hasattr(
        ocr_engine,
        "words_to_text",
    ):

        try:

            return ocr_engine.words_to_text(
                words
            )

        except Exception:
            pass

    # ========================================================
    # FALLBACK RECONSTRUCTION
    # ========================================================

    sorted_words = sorted(
        words,
        key=lambda word: (
            word.bbox[1],
            word.bbox[0],
        ),
    )

    lines = []

    current_line = []

    current_y = None

    y_tolerance = 10.0

    for word in sorted_words:

        y = float(
            word.bbox[1]
        )

        if current_y is None:

            current_line = [
                word
            ]

            current_y = y

            continue

        if abs(
            y - current_y
        ) <= y_tolerance:

            current_line.append(
                word
            )

        else:

            current_line = sorted(
                current_line,
                key=lambda item: item.bbox[0],
            )

            lines.append(
                " ".join(
                    item.text
                    for item in current_line
                )
            )

            current_line = [
                word
            ]

            current_y = y

    # ========================================================
    # FINAL LINE
    # ========================================================

    if current_line:

        current_line = sorted(
            current_line,
            key=lambda item: item.bbox[0],
        )

        lines.append(
            " ".join(
                item.text
                for item in current_line
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# MULTI-PAGE CONSISTENCY AGGREGATION
# ============================================================

def _combine_page_consistency(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine salary-slip consistency results from multiple pages.

    We use the highest page-level risk score.

    Reason:
    If 5 salary slips are internally consistent but one page contains
    a serious arithmetic mismatch, averaging all six scores would hide
    the suspicious page.

    Example:

        page 1 = 0
        page 2 = 0
        page 3 = 70

    Document consistency risk should remain 70, not 23.
    """

    if not results:

        return {
            "score": 0,
            "reasons": [
                "No salary-slip pages were available for consistency verification."
            ],
            "checked": False,
        }

    checked_results = [
        result
        for result in results
        if result.get(
            "checked"
        )
    ]

    if not checked_results:

        reasons = []

        for page_number, result in enumerate(
            results,
            start=1,
        ):

            for reason in result.get(
                "reasons",
                [],
            ):

                reasons.append(
                    f"Page {page_number}: {reason}"
                )

        return {
            "score": 0,

            "reasons": reasons or [
                "No page had enough extracted data for consistency verification."
            ],

            "checked": False,
        }

    # ========================================================
    # HIGHEST PAGE RISK
    # ========================================================

    highest_score = max(
        float(
            result.get(
                "score",
                0,
            )
        )
        for result in checked_results
    )

    # ========================================================
    # PAGE-AWARE REASONS
    # ========================================================

    reasons = []

    for page_number, result in enumerate(
        results,
        start=1,
    ):

        for reason in result.get(
            "reasons",
            [],
        ):

            reasons.append(
                f"Page {page_number}: {reason}"
            )

    return {
        "score": min(
            100,
            highest_score,
        ),

        "reasons": reasons,

        "checked": True,
    }


# ============================================================
# ELA
# ============================================================

def _run_ela(
    file_path: str,
) -> Dict[str, Any]:
    """
    Run ELA on raster image uploads and on rasterized PDF pages.

    For PDFs we never analyse the PDF container itself. Each page is
    rendered to JPEG first, then passed through the existing ELA checker.
    Page results are aggregated into one document-level signal.
    """

    ext = os.path.splitext(file_path)[1].lower()

    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    if ext in (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
    ):
        return ela_check.check(file_path)

    # ========================================================
    # UNSUPPORTED NON-PDF
    # ========================================================

    if ext != ".pdf":
        return {
            "score": 0,
            "reasons": ["ELA is not applicable to this file type."],
            "checked": False,
            "status": "not_applicable",
        }

    # ========================================================
    # PDF -> RASTERIZE EACH PAGE -> ELA
    # ========================================================

    try:
        import fitz
        import tempfile
    except ImportError as exc:
        return {
            "score": 0,
            "reasons": [f"PDF-page ELA unavailable: {exc}"],
            "checked": False,
            "status": "unavailable",
        }

    page_results: List[Dict[str, Any]] = []

    try:
        with fitz.open(file_path) as doc:
            if doc.page_count == 0:
                return {
                    "score": 0,
                    "reasons": ["PDF has no pages available for ELA."],
                    "checked": False,
                    "status": "unavailable",
                }

            with tempfile.TemporaryDirectory(prefix="doc_verify_pdf_ela_") as tmp_dir:
                # 2x rendering gives ELA enough pixel detail without making
                # verification unnecessarily expensive.
                matrix = fitz.Matrix(2.0, 2.0)

                for page_index in range(doc.page_count):
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(
                        matrix=matrix,
                        alpha=False,
                    )

                    page_path = os.path.join(
                        tmp_dir,
                        f"page_{page_index + 1}.jpg",
                    )

                    # Render to a real JPEG so the existing ELA algorithm
                    # compares JPEG compression history rather than a PNG
                    # raster against a newly-created JPEG.
                    pix.save(page_path, jpg_quality=95)

                    page_output_dir = os.path.join(
                        tmp_dir,
                        f"ela_page_{page_index + 1}",
                    )

                    page_result = ela_check.check(
                        page_path,
                        output_dir=page_output_dir,
                    )

                    page_results.append({
                        "page": page_index + 1,
                        **page_result,
                    })

    except Exception as exc:
        return {
            "score": 0,
            "reasons": [f"PDF-page ELA failed: {exc}"],
            "checked": False,
            "status": "failed",
        }

    checked_pages = [
        result
        for result in page_results
        if result.get("checked") is True
    ]

    if not checked_pages:
        return {
            "score": 0,
            "reasons": ["ELA could not be completed on any PDF page."],
            "checked": False,
            "status": "unavailable",
            "details": {"pages": page_results},
        }

    scores = [float(result.get("score", 0)) for result in checked_pages]
    highest_score = max(scores)
    average_score = sum(scores) / len(scores)

    # Use a conservative blend: a single anomalous page matters, but one
    # noisy page should not completely dominate a multi-page document.
    document_score = round(
        (0.70 * highest_score) + (0.30 * average_score),
        1,
    )

    reasons = []
    suspicious_pages = []

    for result in checked_pages:
        page_number = result.get("page")
        page_score = float(result.get("score", 0))

        if page_score > 0:
            suspicious_pages.append(page_number)
            for reason in result.get("reasons", []) or []:
                reasons.append(f"Page {page_number}: {reason}")

    if not reasons:
        reasons.append(
            f"ELA completed on {len(checked_pages)} PDF page(s); "
            "no significant compression-level anomalies were detected."
        )

    return {
        "score": min(100.0, document_score),
        "checked": True,
        "status": "checked",
        "reasons": reasons,
        "details": {
            "pages_checked": len(checked_pages),
            "highest_page_score": highest_score,
            "average_page_score": round(average_score, 1),
            "suspicious_pages": suspicious_pages,
            "pages": [
                {
                    "page": result.get("page"),
                    "score": result.get("score", 0),
                    "global_mean": (result.get("details", {}) or {}).get("global_mean"),
                    "global_std": (result.get("details", {}) or {}).get("global_std"),
                    "max_difference": (result.get("details", {}) or {}).get("max_difference"),
                    "p95_difference": (result.get("details", {}) or {}).get("p95_difference"),
                    "p99_difference": (result.get("details", {}) or {}).get("p99_difference"),
                    "suspicious_patch_ratio": (result.get("details", {}) or {}).get("suspicious_patch_ratio"),
                    "maximum_patch_mean": (result.get("details", {}) or {}).get("maximum_patch_mean"),
                    "median_patch_mean": (result.get("details", {}) or {}).get("median_patch_mean"),
                    "checked": result.get("checked", False),
                    "status": result.get("status", "unknown"),
                    "reasons": result.get("reasons", []),
                }
                for result in page_results
            ],
        },
    }

