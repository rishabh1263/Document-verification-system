"""
PDF structural verification signal.

Inspects the internal structure of a PDF for indicators that may deserve
additional review.

This checker does NOT prove authenticity or forgery. It looks for structural
anomalies such as:

1. Excessive font diversity
2. Very small / invisible-looking text
3. Overlapping text spans
4. Mixed fonts inside numeric / salary-heavy regions
5. Large numbers of embedded images
6. Suspiciously fragmented text
7. PDF page / object characteristics

Higher score = more suspicious.

The thresholds are starting points and should eventually be calibrated
against a labelled dataset of genuine and tampered salary slips.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

MAX_NORMAL_FONTS = 8

VERY_HIGH_FONT_COUNT = 15

MIN_VISIBLE_FONT_SIZE = 4.0

MAX_SMALL_TEXT_RATIO = 0.05

MAX_FRAGMENTATION_RATIO = 0.40

MAX_IMAGES_PER_PAGE = 8

OVERLAP_RATIO_THRESHOLD = 0.08


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def check(file_path: str) -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "score": 0,
        "checked": False,
        "status": "not_applicable",
        "reasons": [],
        "details": {},
    }

    # ========================================================
    # PDF ONLY
    # ========================================================

    if not file_path.lower().endswith(".pdf"):

        result["reasons"].append(
            "PDF structural verification is not applicable "
            "to this image upload."
        )

        return result

    # ========================================================
    # PYMUPDF
    # ========================================================

    try:
        import fitz

    except ImportError:

        result["status"] = "unavailable"

        result["reasons"].append(
            "PyMuPDF is not installed, so PDF structural "
            "verification could not be performed."
        )

        return result

    # ========================================================
    # OPEN PDF
    # ========================================================

    try:
        doc = fitz.open(file_path)

    except Exception as exc:

        result["status"] = "failed"

        result["reasons"].append(
            f"Could not open PDF for structural inspection: {exc}"
        )

        return result

    score = 0
    reasons: List[str] = []

    font_counter: Counter = Counter()

    total_spans = 0
    small_spans = 0

    total_text_characters = 0
    short_spans = 0

    total_images = 0

    numeric_font_counter: Counter = Counter()

    overlapping_pairs = 0
    overlap_candidates = 0

    page_details = []

    # ========================================================
    # PROCESS PAGES
    # ========================================================

    try:

        page_count = len(doc)

        for page_number, page in enumerate(doc):

            page_fonts = Counter()

            page_spans = []

            page_numeric_fonts = Counter()

            page_small_spans = 0

            page_short_spans = 0

            page_total_spans = 0

            # =================================================
            # TEXT STRUCTURE
            # =================================================

            try:

                text_dict = page.get_text(
                    "dict"
                )

            except Exception:

                text_dict = {
                    "blocks": []
                }

            for block in text_dict.get(
                "blocks",
                [],
            ):

                # Type 0 = text block.
                if block.get("type") != 0:
                    continue

                for line in block.get(
                    "lines",
                    [],
                ):

                    for span in line.get(
                        "spans",
                        [],
                    ):

                        text = str(
                            span.get(
                                "text",
                                "",
                            )
                        ).strip()

                        if not text:
                            continue

                        font = str(
                            span.get(
                                "font",
                                "unknown",
                            )
                        )

                        try:
                            size = float(
                                span.get(
                                    "size",
                                    0,
                                )
                            )

                        except (
                            ValueError,
                            TypeError,
                        ):
                            size = 0.0

                        bbox = span.get(
                            "bbox"
                        )

                        page_total_spans += 1
                        total_spans += 1

                        total_text_characters += len(
                            text
                        )

                        font_counter[
                            font
                        ] += 1

                        page_fonts[
                            font
                        ] += 1

                        # =====================================
                        # VERY SMALL TEXT
                        # =====================================

                        if (
                            size > 0
                            and size < MIN_VISIBLE_FONT_SIZE
                        ):

                            small_spans += 1

                            page_small_spans += 1

                        # =====================================
                        # FRAGMENTED TEXT
                        # =====================================

                        if len(text) <= 2:

                            short_spans += 1

                            page_short_spans += 1

                        # =====================================
                        # NUMERIC / MONEY TEXT
                        # =====================================

                        if _looks_numeric_or_money(
                            text
                        ):

                            numeric_font_counter[
                                font
                            ] += 1

                            page_numeric_fonts[
                                font
                            ] += 1

                        # =====================================
                        # STORE BBOX FOR OVERLAP ANALYSIS
                        # =====================================

                        if (
                            bbox
                            and len(bbox) == 4
                        ):

                            try:

                                normalized_bbox = tuple(
                                    float(v)
                                    for v in bbox
                                )

                                page_spans.append(
                                    {
                                        "text": text,
                                        "font": font,
                                        "size": size,
                                        "bbox": normalized_bbox,
                                    }
                                )

                            except (
                                ValueError,
                                TypeError,
                            ):
                                pass

            # =================================================
            # IMAGES
            # =================================================

            try:

                page_images = page.get_images(
                    full=True
                )

            except Exception:

                page_images = []

            image_count = len(
                page_images
            )

            total_images += image_count

            # =================================================
            # OVERLAPPING TEXT
            # =================================================

            page_overlaps, page_overlap_candidates = (
                _count_suspicious_overlaps(
                    page_spans
                )
            )

            overlapping_pairs += (
                page_overlaps
            )

            overlap_candidates += (
                page_overlap_candidates
            )

            # =================================================
            # PAGE DETAILS
            # =================================================

            page_details.append(
                {
                    "page": page_number + 1,

                    "font_count": len(
                        page_fonts
                    ),

                    "fonts": dict(
                        page_fonts
                    ),

                    "span_count": (
                        page_total_spans
                    ),

                    "small_text_spans": (
                        page_small_spans
                    ),

                    "short_text_spans": (
                        page_short_spans
                    ),

                    "image_count": (
                        image_count
                    ),

                    "numeric_fonts": dict(
                        page_numeric_fonts
                    ),

                    "overlapping_text_pairs": (
                        page_overlaps
                    ),
                }
            )

    finally:

        doc.close()

    # ========================================================
    # CALCULATED METRICS
    # ========================================================

    unique_fonts = len(
        font_counter
    )

    numeric_font_count = len(
        numeric_font_counter
    )

    small_text_ratio = (
        small_spans / total_spans
        if total_spans
        else 0.0
    )

    fragmentation_ratio = (
        short_spans / total_spans
        if total_spans
        else 0.0
    )

    images_per_page = (
        total_images / page_count
        if page_count
        else 0.0
    )

    overlap_ratio = (
        overlapping_pairs
        / overlap_candidates
        if overlap_candidates
        else 0.0
    )

    # ========================================================
    # STORE DETAILS
    # ========================================================

    result["details"] = {

        "page_count": page_count,

        "total_text_spans": total_spans,

        "total_text_characters": (
            total_text_characters
        ),

        "unique_font_count": (
            unique_fonts
        ),

        "fonts": dict(
            font_counter
        ),

        "numeric_font_count": (
            numeric_font_count
        ),

        "numeric_fonts": dict(
            numeric_font_counter
        ),

        "small_text_spans": (
            small_spans
        ),

        "small_text_ratio": round(
            small_text_ratio,
            4,
        ),

        "short_text_spans": (
            short_spans
        ),

        "fragmentation_ratio": round(
            fragmentation_ratio,
            4,
        ),

        "total_images": (
            total_images
        ),

        "images_per_page": round(
            images_per_page,
            2,
        ),

        "overlapping_text_pairs": (
            overlapping_pairs
        ),

        "overlap_ratio": round(
            overlap_ratio,
            4,
        ),

        "pages": page_details,
    }

    # ========================================================
    # SIGNAL 1 â€” FONT DIVERSITY
    # ========================================================

    if unique_fonts > VERY_HIGH_FONT_COUNT:

        score += 25

        reasons.append(
            f"The PDF uses {unique_fonts} different fonts, "
            "which is unusually high for a salary slip."
        )

    elif unique_fonts > MAX_NORMAL_FONTS:

        score += 10

        reasons.append(
            f"The PDF uses {unique_fonts} different fonts. "
            "High font diversity may deserve manual review."
        )

    else:

        reasons.append(
            f"Font usage is within the expected range "
            f"({unique_fonts} unique font(s))."
        )

    # ========================================================
    # SIGNAL 2 â€” NUMERIC FONT DIVERSITY
    # ========================================================

    if numeric_font_count >= 5:

        score += 20

        reasons.append(
            f"Numeric values use {numeric_font_count} different "
            "fonts. Salary amounts with inconsistent typography "
            "can indicate later editing."
        )

    elif numeric_font_count >= 3:

        score += 8

        reasons.append(
            f"Numeric values use {numeric_font_count} different "
            "fonts. This is worth reviewing."
        )

    # ========================================================
    # SIGNAL 3 â€” VERY SMALL TEXT
    # ========================================================

    if (
        total_spans > 0
        and small_text_ratio
        > MAX_SMALL_TEXT_RATIO
    ):

        score += 15

        reasons.append(
            f"{small_text_ratio * 100:.1f}% of text spans use "
            "very small font sizes. Hidden or overlay text may "
            "require review."
        )

    # ========================================================
    # SIGNAL 4 â€” TEXT FRAGMENTATION
    # ========================================================

    if (
        total_spans >= 20
        and fragmentation_ratio
        > MAX_FRAGMENTATION_RATIO
    ):

        score += 10

        reasons.append(
            f"{fragmentation_ratio * 100:.1f}% of text spans are "
            "very short. The PDF text layer is unusually fragmented."
        )

    # ========================================================
    # SIGNAL 5 â€” EXCESSIVE IMAGES
    # ========================================================

    if (
        page_count > 0
        and images_per_page
        > MAX_IMAGES_PER_PAGE
    ):

        score += 10

        reasons.append(
            f"The PDF contains an average of "
            f"{images_per_page:.1f} embedded images per page."
        )

    # ========================================================
    # SIGNAL 6 â€” OVERLAPPING TEXT
    # ========================================================

    if (
        overlap_candidates > 0
        and overlap_ratio
        > OVERLAP_RATIO_THRESHOLD
    ):

        score += 25

        reasons.append(
            f"{overlap_ratio * 100:.1f}% of nearby text-span "
            "comparisons showed substantial overlap. "
            "Text overlays can be associated with PDF editing."
        )

    # ========================================================
    # EMPTY / IMAGE-ONLY PDF
    # ========================================================

    if total_spans == 0:

        reasons.append(
            "No native text spans were found. The PDF may be "
            "image-based, so structural text analysis is limited."
        )

    # ========================================================
    # FINAL
    # ========================================================

    result["score"] = min(
        100,
        round(
            score,
            1,
        ),
    )

    result["checked"] = True
    result["status"] = "checked"

    if score == 0:

        reasons.append(
            "No significant PDF structural anomalies were detected."
        )

    result["reasons"] = reasons

    return result


# ============================================================
# NUMERIC / MONEY DETECTION
# ============================================================

def _looks_numeric_or_money(
    text: str,
) -> bool:

    if not text:
        return False

    value = text.strip()

    # Examples:
    #
    # 31,533.00
    # 16667
    # â‚¹29,508
    # Rs. 2025.00

    value = re.sub(
        r"^(?:â‚¹|Rs\.?|INR)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    return bool(
        re.fullmatch(
            r"[\d,]+(?:\.\d{1,2})?",
            value,
        )
    )


# ============================================================
# TEXT OVERLAP ANALYSIS
# ============================================================

def _count_suspicious_overlaps(
    spans,
) -> Tuple[int, int]:
    """
    Compare nearby text spans and count substantial geometric overlaps.

    We intentionally ignore exact/near-exact duplicate text because some
    PDF generators legitimately create duplicated rendering layers.
    """

    suspicious = 0
    comparisons = 0

    # Avoid quadratic explosion on pathological PDFs.
    maximum_spans = 500

    spans = spans[
        :maximum_spans
    ]

    for i in range(
        len(spans)
    ):

        first = spans[i]

        for j in range(
            i + 1,
            len(spans)
        ):

            second = spans[j]

            # ------------------------------------------------
            # Fast vertical-distance rejection.
            # ------------------------------------------------

            if abs(
                first["bbox"][1]
                - second["bbox"][1]
            ) > 30:

                continue

            comparisons += 1

            overlap = _bbox_overlap_ratio(
                first["bbox"],
                second["bbox"],
            )

            if overlap < 0.50:
                continue

            # Same text rendered twice can occur legitimately.
            if (
                first["text"].strip().lower()
                == second["text"].strip().lower()
            ):
                continue

            suspicious += 1

    return (
        suspicious,
        comparisons,
    )


# ============================================================
# BOUNDING BOX OVERLAP
# ============================================================

def _bbox_overlap_ratio(
    first,
    second,
) -> float:

    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second

    intersection_x0 = max(
        ax0,
        bx0,
    )

    intersection_y0 = max(
        ay0,
        by0,
    )

    intersection_x1 = min(
        ax1,
        bx1,
    )

    intersection_y1 = min(
        ay1,
        by1,
    )

    intersection_width = max(
        0.0,
        intersection_x1
        - intersection_x0,
    )

    intersection_height = max(
        0.0,
        intersection_y1
        - intersection_y0,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    if intersection_area <= 0:
        return 0.0

    first_area = max(
        0.0,
        ax1 - ax0,
    ) * max(
        0.0,
        ay1 - ay0,
    )

    second_area = max(
        0.0,
        bx1 - bx0,
    ) * max(
        0.0,
        by1 - by0,
    )

    smaller_area = min(
        first_area,
        second_area,
    )

    if smaller_area <= 0:
        return 0.0

    return (
        intersection_area
        / smaller_area
    )
