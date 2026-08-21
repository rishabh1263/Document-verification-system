"""
PDF metadata verification signal.

Purpose:
    Inspect PDF metadata for suspicious indicators without treating
    metadata as proof that a document is genuine or fake.

Checks:
    1. PDF readability / encryption
    2. Producer and creator metadata
    3. Known image/design editing software
    4. Missing producer/creator metadata
    5. Creation and modification timestamps
    6. Invalid or suspicious PDF dates
    7. Future timestamps
    8. Modification significantly after creation
    9. Basic PDF structural information

Important:
    Metadata is only one verification signal.

    A PDF created by Microsoft Word, ReportLab, SAP, etc. is NOT
    automatically genuine.

    Likewise, missing metadata alone is NOT enough to call a
    document fake.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import re


# ============================================================
# SOFTWARE CLASSIFICATION
# ============================================================

# Stronger suspicion for software primarily associated with
# image manipulation/design rather than payroll generation.
SUSPICIOUS_PRODUCERS = [
    "photoshop",
    "adobe photoshop",
    "gimp",
    "paint.net",
    "mspaint",
    "snapseed",
    "pixlr",
    "photopea",
]


# These aren't necessarily fraudulent, but deserve attention
# because documents can easily be manually designed/edited.
REVIEW_PRODUCERS = [
    "canva",
    "illustrator",
    "adobe illustrator",
    "inkscape",
    "coreldraw",
]


# Common PDF/document-generation software.
#
# IMPORTANT:
# We do NOT subtract risk simply because one of these appears.
KNOWN_DOCUMENT_PRODUCERS = [
    "microsoft",
    "word",
    "excel",
    "libreoffice",
    "google docs",
    "wkhtmltopdf",
    "reportlab",
    "itext",
    "crystal reports",
    "sap",
    "oracle",
    "adobe pdf library",
    "acrobat distiller",
    "pdfium",
    "chromium",
    "chrome",
]


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def check(file_path: str) -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "score": 0,
        "reasons": [],
        "checked": False,
        "status": "not_applicable",
        "raw_metadata": {},
        "details": {},
    }

    # ========================================================
    # NON-PDF
    # ========================================================

    if not file_path.lower().endswith(".pdf"):

        result["reasons"].append(
            "Metadata verification is not applicable to this image upload."
        )

        return result

    # ========================================================
    # IMPORT PYMUPDF
    # ========================================================

    try:
        import fitz

    except ImportError:

        result["status"] = "unavailable"

        result["reasons"].append(
            "PyMuPDF is not installed, so PDF metadata could not be inspected."
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
            f"PDF metadata inspection failed because the document "
            f"could not be opened: {exc}"
        )

        return result

    # ========================================================
    # COLLECT BASIC PDF INFORMATION
    # ========================================================

    try:

        metadata = doc.metadata or {}

        page_count = len(doc)

        is_encrypted = bool(
            getattr(doc, "is_encrypted", False)
        )

        needs_pass = bool(
            getattr(doc, "needs_pass", False)
        )

    finally:
        doc.close()

    result["checked"] = True
    result["status"] = "checked"
    result["raw_metadata"] = metadata

    result["details"] = {
        "page_count": page_count,
        "is_encrypted": is_encrypted,
        "needs_password": needs_pass,
    }

    score = 0
    reasons = []

    # ========================================================
    # PRODUCER / CREATOR
    # ========================================================

    producer_raw = (
        metadata.get("producer") or ""
    ).strip()

    creator_raw = (
        metadata.get("creator") or ""
    ).strip()

    producer = producer_raw.lower()
    creator = creator_raw.lower()

    software_text = (
        producer + " " + creator
    ).strip()

    result["details"]["producer"] = (
        producer_raw or None
    )

    result["details"]["creator"] = (
        creator_raw or None
    )

    # ========================================================
    # SUSPICIOUS EDITING SOFTWARE
    # ========================================================

    suspicious_matches = [
        software
        for software in SUSPICIOUS_PRODUCERS
        if software in software_text
    ]

    if suspicious_matches:

        score += 55

        reasons.append(
            "PDF metadata references image-editing software "
            f"({', '.join(sorted(set(suspicious_matches)))}). "
            "This is unusual for a direct payroll-system export."
        )

    # ========================================================
    # DESIGN / MANUAL EDITING SOFTWARE
    # ========================================================

    review_matches = [
        software
        for software in REVIEW_PRODUCERS
        if software in software_text
    ]

    if review_matches:

        score += 30

        reasons.append(
            "PDF metadata references design/editing software "
            f"({', '.join(sorted(set(review_matches)))}). "
            "The document should receive additional verification."
        )

    # ========================================================
    # KNOWN DOCUMENT PRODUCER
    # ========================================================

    known_matches = [
        software
        for software in KNOWN_DOCUMENT_PRODUCERS
        if software in software_text
    ]

    if known_matches:

        reasons.append(
            "Producer/creator metadata references common "
            "document-generation software "
            f"({', '.join(sorted(set(known_matches)))}). "
            "This is informational and does not prove authenticity."
        )

    # ========================================================
    # MISSING PRODUCER / CREATOR
    # ========================================================

    if not producer_raw and not creator_raw:

        score += 10

        reasons.append(
            "Producer and creator metadata are missing. "
            "This reduces metadata-based verification confidence "
            "but is not proof of tampering."
        )

    elif not producer_raw:

        score += 3

        reasons.append(
            "PDF producer metadata is missing."
        )

    elif not creator_raw:

        score += 3

        reasons.append(
            "PDF creator metadata is missing."
        )

    # ========================================================
    # PDF CREATION / MODIFICATION DATES
    # ========================================================

    creation_raw = (
        metadata.get("creationDate") or ""
    ).strip()

    modification_raw = (
        metadata.get("modDate") or ""
    ).strip()

    result["details"]["creation_date_raw"] = (
        creation_raw or None
    )

    result["details"]["modification_date_raw"] = (
        modification_raw or None
    )

    creation_date = _parse_pdf_date(
        creation_raw
    )

    modification_date = _parse_pdf_date(
        modification_raw
    )

    result["details"]["creation_date_parsed"] = (
        creation_date.isoformat()
        if creation_date
        else None
    )

    result["details"]["modification_date_parsed"] = (
        modification_date.isoformat()
        if modification_date
        else None
    )

    # ========================================================
    # INVALID DATE METADATA
    # ========================================================

    if creation_raw and creation_date is None:

        score += 5

        reasons.append(
            "PDF creation timestamp exists but could not be parsed."
        )

    if modification_raw and modification_date is None:

        score += 5

        reasons.append(
            "PDF modification timestamp exists but could not be parsed."
        )

    # ========================================================
    # FUTURE TIMESTAMPS
    # ========================================================

    now = datetime.now(
        timezone.utc
    )

    if creation_date:

        normalized_creation = _ensure_timezone(
            creation_date
        )

        if normalized_creation > now:

            score += 25

            reasons.append(
                "PDF creation timestamp is in the future, "
                "which is suspicious."
            )

    if modification_date:

        normalized_modification = _ensure_timezone(
            modification_date
        )

        if normalized_modification > now:

            score += 25

            reasons.append(
                "PDF modification timestamp is in the future, "
                "which is suspicious."
            )

    # ========================================================
    # CREATION VS MODIFICATION
    # ========================================================

    if creation_date and modification_date:

        creation_compare = _ensure_timezone(
            creation_date
        )

        modification_compare = _ensure_timezone(
            modification_date
        )

        difference_seconds = (
            modification_compare
            - creation_compare
        ).total_seconds()

        result["details"][
            "modification_delay_seconds"
        ] = difference_seconds

        # ----------------------------------------------------
        # Modification BEFORE creation
        # ----------------------------------------------------

        if difference_seconds < -60:

            score += 30

            reasons.append(
                "PDF modification timestamp occurs before its "
                "creation timestamp, which is inconsistent."
            )

        # ----------------------------------------------------
        # Modified more than 24 hours after creation
        # ----------------------------------------------------

        elif difference_seconds > 86400:

            score += 15

            reasons.append(
                "PDF was modified more than 24 hours after "
                "its recorded creation time."
            )

        # ----------------------------------------------------
        # Modified after creation, but close in time
        # ----------------------------------------------------

        elif difference_seconds > 60:

            score += 5

            reasons.append(
                "PDF modification time differs from its creation "
                "time. This may be normal, but indicates the file "
                "was changed after initial creation."
            )

        else:

            reasons.append(
                "PDF creation and modification timestamps are "
                "consistent."
            )

    elif not creation_raw and not modification_raw:

        reasons.append(
            "Creation and modification timestamps are unavailable, "
            "so timestamp-based verification could not be performed."
        )

    # ========================================================
    # ENCRYPTION
    # ========================================================

    if is_encrypted or needs_pass:

        score += 5

        reasons.append(
            "The PDF uses encryption or password protection. "
            "This is not evidence of forgery but limits inspection."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result["score"] = min(
        100,
        round(score, 1),
    )

    if not reasons:

        reasons.append(
            "No suspicious metadata indicators were detected."
        )

    result["reasons"] = reasons

    return result


# ============================================================
# PDF DATE PARSER
# ============================================================

def _parse_pdf_date(
    value: str,
) -> Optional[datetime]:
    """
    Parse common PDF date formats.

    Examples:

        D:20260804101530+05'30'
        D:20260804101530Z
        D:20260804101530
        20260804101530
    """

    if not value:
        return None

    value = value.strip()

    if value.startswith("D:"):
        value = value[2:]

    # --------------------------------------------------------
    # Extract:
    #
    # YYYY MM DD HH MM SS
    # timezone optional
    # --------------------------------------------------------

    match = re.match(
        r"^"
        r"(\d{4})"
        r"(\d{2})?"
        r"(\d{2})?"
        r"(\d{2})?"
        r"(\d{2})?"
        r"(\d{2})?"
        r"(.*)"
        r"$",
        value,
    )

    if not match:
        return None

    try:

        year = int(
            match.group(1)
        )

        month = int(
            match.group(2) or 1
        )

        day = int(
            match.group(3) or 1
        )

        hour = int(
            match.group(4) or 0
        )

        minute = int(
            match.group(5) or 0
        )

        second = int(
            match.group(6) or 0
        )

        timezone_part = (
            match.group(7) or ""
        ).strip()

        parsed = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
        )

        # ----------------------------------------------------
        # UTC
        # ----------------------------------------------------

        if timezone_part.startswith("Z"):

            return parsed.replace(
                tzinfo=timezone.utc
            )

        # ----------------------------------------------------
        # +05'30'
        # -04'00'
        # ----------------------------------------------------

        timezone_match = re.search(
            r"([+-])"
            r"(\d{2})"
            r"'?"
            r"(\d{2})?"
            r"'?",
            timezone_part,
        )

        if timezone_match:

            from datetime import timedelta

            sign = (
                1
                if timezone_match.group(1) == "+"
                else -1
            )

            hours = int(
                timezone_match.group(2)
            )

            minutes = int(
                timezone_match.group(3) or 0
            )

            offset = timedelta(
                hours=hours,
                minutes=minutes,
            )

            offset *= sign

            return parsed.replace(
                tzinfo=timezone(offset)
            )

        # No timezone information.
        return parsed.replace(
            tzinfo=timezone.utc
        )

    except (
        ValueError,
        TypeError,
    ):

        return None


# ============================================================
# NORMALIZE DATETIME
# ============================================================

def _ensure_timezone(
    value: datetime,
) -> datetime:

    if value.tzinfo is None:

        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )
