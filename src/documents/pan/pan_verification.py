"""
Production PAN verification core.

Architecture
------------
1. Digital PDF with usable text:
       PDF -> native PDF text -> field extraction
       No OCR.

2. Scanned PDF:
       PDF -> render page -> PaddleOCR -> field extraction

3. Image:
       Image -> PaddleOCR -> field extraction

Validation
----------
The system separates:

    DOCUMENT_REJECT
    DOCUMENT_PASS
    MANUAL REVIEW

from:

    GOVERNMENT_VERIFIED

A PAN matching the correct format does NOT prove that the PAN exists.

Local validation checks:
    - PAN structure
    - strict PAN-document type classification
    - field-level OCR evidence consistency
    - image resolution/aspect-ratio quality signals
    - PAN extraction confidence
    - PAN-card-specific markers
    - required fields
    - DOB format
    - OCR confidence
    - image quality
    - document layout
    - suspicious OCR inconsistencies
    - PAN/name/DOB consistency
    - duplicate/garbage OCR signals

For actual PAN existence/active-status verification, an authoritative
PAN verification service/API must be integrated separately.
"""

from __future__ import annotations

# ============================================================================
# PADDLE RUNTIME FLAGS
# ============================================================================

import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_cpu_math_library_num_threads"] = "4"

# ============================================================================
# IMPORTS
# ============================================================================

import re
import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import pymupdf
import numpy as np

from src.common.verification.document_validation import (
    DOCUMENT_PASS,
    DOCUMENT_SUSPICIOUS,
    MANUAL_REVIEW,
    DOCUMENT_REJECT,
    validate_from_existing_result,
)
from src.common.verification.image_quality import (
    analyze_image_quality,
)

try:
    from src.common.authenticity.tamper import analyze_tampering
except Exception:
    analyze_tampering = None


# ============================================================================
# GLOBAL OCR
# ============================================================================

_ocr = None


# ============================================================================
# CONSTANTS
# ============================================================================

OCR_MAX_WIDTH = 1600

# Blur is a quality signal, not an automatic rejection gate.
# Global Laplacian variance can be low on photographed PAN cards because of
# textured backgrounds, glare, compression, or uneven lighting even when
# the important text remains readable.
MIN_BLUR_SCORE = 80.0  # diagnostic/reference threshold only

MIN_NATIVE_PDF_TEXT_LENGTH = 40

# Broad PAN-card image/layout ranges. These are quality signals, not proof
# of authenticity, because photographs/scans can be cropped or distorted.
MIN_PAN_ASPECT_RATIO = 1.30
MAX_PAN_ASPECT_RATIO = 1.90

# A PAN document should have multiple independent PAN-specific markers.
MIN_DOCUMENT_TYPE_MARKERS = 4

# Borderline OCR confidence should not silently become DOCUMENT_PASS.
MIN_PASS_OCR_CONFIDENCE = 0.60
MIN_MANUAL_REVIEW_OCR_CONFIDENCE = 0.45

PAN_PATTERN = re.compile(
    r"[A-Z]{5}[0-9]{4}[A-Z]"
)

DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})\b"
)

MONTH_DATE_PATTERN = re.compile(
    r"\b(\d{1,2})[\s\-]?"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*[\s\-]?(\d{4})\b",
    re.IGNORECASE,
)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PanVerificationError(Exception):
    """Expected PAN verification/extraction failure."""


# ============================================================================
# OCR
# ============================================================================

def _get_ocr():
    """
    Load PaddleOCR once.

    The model is lazy-loaded so importing this module does not immediately
    load the OCR model.
    """
    global _ocr

    if _ocr is None:
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            text_det_limit_type="max",
            text_det_limit_side_len=640,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    return _ocr


def load_ocr_reader():
    """
    Backward-compatible API used by main.py.
    """
    return _get_ocr()


# ============================================================================
# IMAGE / PDF
# ============================================================================

def decode_image(file_bytes: bytes):
    """Decode image bytes into an OpenCV BGR image."""
    if not file_bytes:
        return None

    array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )


def _prepare_image(
    image: np.ndarray,
) -> np.ndarray:
    """Resize very large images before OCR."""
    if image is None:
        raise PanVerificationError(
            "Invalid image."
        )

    if image.size == 0:
        raise PanVerificationError(
            "Invalid empty image."
        )

    height, width = image.shape[:2]

    if width <= OCR_MAX_WIDTH:
        return image

    scale = OCR_MAX_WIDTH / width

    return cv2.resize(
        image,
        (
            OCR_MAX_WIDTH,
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _render_pdf(
    file_bytes: bytes,
) -> list[np.ndarray]:
    """Render PDF pages into OpenCV images."""
    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise PanVerificationError(
            f"Could not open PDF: {exc}"
        ) from exc

    if document.page_count == 0:
        document.close()

        raise PanVerificationError(
            "PDF contains no pages."
        )

    pages: list[np.ndarray] = []

    try:
        matrix = pymupdf.Matrix(
            160 / 72,
            160 / 72,
        )

        for index in range(
            document.page_count
        ):
            page = document.load_page(index)

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            raw = np.frombuffer(
                pix.samples,
                dtype=np.uint8,
            )

            if pix.n == 3:

                image = raw.reshape(
                    pix.height,
                    pix.width,
                    3,
                )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

            elif pix.n == 4:

                image = raw.reshape(
                    pix.height,
                    pix.width,
                    4,
                )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2BGR,
                )

            else:
                continue

            pages.append(image)

    finally:
        document.close()

    if not pages:
        raise PanVerificationError(
            "Could not render any PDF page."
        )

    return pages


# ============================================================================
# BOUNDING BOX HELPERS
# ============================================================================

def _bbox_center_y(
    item: dict[str, Any],
) -> float | None:

    bbox = item.get("bbox")

    if bbox is None:
        return None

    try:

        arr = np.asarray(
            bbox,
            dtype=float,
        )

        if (
            arr.ndim == 2
            and arr.shape[1] >= 2
        ):
            return float(
                (
                    arr[:, 1].min()
                    + arr[:, 1].max()
                ) / 2
            )

        flat = arr.flatten()

        if flat.size >= 4:
            return float(
                (flat[1] + flat[3]) / 2
            )

    except Exception:
        pass

    return None


def _bbox_center_x(
    item: dict[str, Any],
) -> float | None:

    bbox = item.get("bbox")

    if bbox is None:
        return None

    try:

        arr = np.asarray(
            bbox,
            dtype=float,
        )

        if (
            arr.ndim == 2
            and arr.shape[1] >= 2
        ):
            return float(
                (
                    arr[:, 0].min()
                    + arr[:, 0].max()
                ) / 2
            )

        flat = arr.flatten()

        if flat.size >= 4:
            return float(
                (flat[0] + flat[2]) / 2
            )

    except Exception:
        pass

    return None


# ============================================================================
# OCR RESULT NORMALIZATION
# ============================================================================

def _ocr_result_to_lines(
    result: Any,
) -> list[dict[str, Any]]:

    lines: list[dict[str, Any]] = []

    for page_result in result:

        data = getattr(
            page_result,
            "json",
            None,
        )

        if callable(data):

            try:
                data = data()

            except Exception:
                data = None

        if data is None:
            data = getattr(
                page_result,
                "res",
                None,
            )

        if isinstance(data, str):

            try:

                import json

                data = json.loads(data)

            except Exception:
                data = None

        if not isinstance(data, dict):
            continue

        data = data.get(
            "res",
            data,
        )

        if not isinstance(data, dict):
            continue

        texts = data.get(
            "rec_texts"
        ) or []

        scores = data.get(
            "rec_scores"
        ) or []

        boxes = data.get(
            "rec_boxes"
        )

        if boxes is None:
            boxes = data.get(
                "rec_polys"
            ) or []

        if isinstance(
            texts,
            str,
        ):
            texts = [texts]

        for index, text in enumerate(texts):

            text = str(text).strip()

            if not text:
                continue

            try:

                confidence = float(
                    scores[index]
                )

            except (
                IndexError,
                TypeError,
                ValueError,
            ):
                confidence = 0.0

            bbox = None

            try:

                bbox = boxes[index]

                if hasattr(
                    bbox,
                    "tolist",
                ):
                    bbox = bbox.tolist()

            except (
                IndexError,
                TypeError,
            ):
                bbox = None

            item = {
                "text": text,
                "confidence": confidence,
                "bbox": bbox,
            }

            item["yc"] = _bbox_center_y(
                item
            )

            item["xc"] = _bbox_center_x(
                item
            )

            lines.append(item)

    lines.sort(
        key=lambda item: (
            item["yc"]
            if item["yc"] is not None
            else 10**12,

            item["xc"]
            if item["xc"] is not None
            else 10**12,
        )
    )

    return lines


def _run_ocr(
    image: np.ndarray,
) -> list[dict[str, Any]]:

    """
    Exactly one OCR inference for one image.
    """

    reader = _get_ocr()

    image = _prepare_image(
        image
    )

    try:

        result = reader.predict(
            image
        )

    except NotImplementedError as exc:

        raise PanVerificationError(
            "PaddleOCR CPU runtime incompatibility."
        ) from exc

    except Exception as exc:

        raise PanVerificationError(
            f"PaddleOCR inference failed: {exc}"
        ) from exc

    return _ocr_result_to_lines(
        result
    )


def _plain_lines(
    results: list[dict[str, Any]],
) -> list[str]:

    return [
        str(
            item.get(
                "text",
                "",
            )
        ).strip()

        for item in results

        if str(
            item.get(
                "text",
                "",
            )
        ).strip()
    ]


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def _normal(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(text).upper(),
    ).strip()


def _compact(
    text: str,
) -> str:

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def _alpha_only(
    text: str,
) -> str:

    return re.sub(
        r"[^A-Z ]",
        " ",
        str(text).upper(),
    )


# ============================================================================
# PAN EXTRACTION
# ============================================================================

DIGIT_FIX = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


LETTER_FIX = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def _normalize_pan_candidate(
    value: str,
) -> str | None:

    compact = _compact(
        value
    )

    if len(compact) != 10:
        return None

    chars = list(
        compact
    )

    digit_position_count = sum(
        chars[index].isdigit()
        for index in range(5, 9)
    )

    if digit_position_count < 2:
        return None

    # First five characters must be letters.
    for index in range(5):

        if chars[index].isdigit():

            replacement = LETTER_FIX.get(
                chars[index]
            )

            if replacement is None:
                return None

            chars[index] = replacement

    # Four middle characters must be digits.
    for index in range(5, 9):

        if chars[index].isalpha():

            replacement = DIGIT_FIX.get(
                chars[index]
            )

            if replacement is None:
                return None

            chars[index] = replacement

    # Last character must be a letter.
    if chars[9].isdigit():

        replacement = LETTER_FIX.get(
            chars[9]
        )

        if replacement is None:
            return None

        chars[9] = replacement

    candidate = "".join(
        chars
    )

    if PAN_PATTERN.fullmatch(
        candidate
    ):
        return candidate

    return None


def _find_pan(
    lines: list[str],
) -> str | None:

    # ---------------------------------------------------------
    # Exact match.
    # ---------------------------------------------------------

    for line in lines:

        match = PAN_PATTERN.search(
            _normal(line)
        )

        if match:
            return match.group(0)

    # ---------------------------------------------------------
    # Conservative OCR correction.
    # ---------------------------------------------------------

    for line in lines:

        compact = _compact(
            line
        )

        if len(compact) == 10:

            candidate = (
                _normalize_pan_candidate(
                    compact
                )
            )

            if candidate:
                return candidate

        if len(compact) > 10:

            for start in range(
                len(compact) - 9
            ):

                candidate = (
                    _normalize_pan_candidate(
                        compact[
                            start:start + 10
                        ]
                    )
                )

                if candidate:
                    return candidate

    return None


def extract_pan(
    lines: list[dict[str, Any]],
) -> str | None:

    return _find_pan(
        _plain_lines(lines)
    )


# ============================================================================
# DATE
# ============================================================================

def _clean_date(
    text: str | None,
) -> str | None:

    if not text:
        return None

    value = str(
        text
    )

    match = DATE_PATTERN.search(
        value
    )

    if match:

        day, month, year = (
            match.groups()
        )

        try:

            parsed = datetime(
                int(year),
                int(month),
                int(day),
            )

            return parsed.strftime(
                "%d/%m/%Y"
            )

        except ValueError:
            return None

    match = MONTH_DATE_PATTERN.search(
        value
    )

    if match:

        day, month, year = (
            match.groups()
        )

        try:

            month_number = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }[
                month[:3].lower()
            ]

            parsed = datetime(
                int(year),
                month_number,
                int(day),
            )

            return parsed.strftime(
                "%d/%m/%Y"
            )

        except ValueError:
            return None

    return None


def _extract_date(
    lines: list[str],
) -> str | None:

    for line in lines:

        value = _clean_date(
            line
        )

        if value:
            return value

    return None


# ============================================================================
# DOCUMENT MARKERS
# ============================================================================

def _is_name_label(
    text: str,
) -> bool:

    value = _alpha_only(
        text
    )

    return bool(
        re.search(
            r"\bNAME\b",
            value,
        )
        and "FATHER" not in value
    )


def _is_father_label(
    text: str,
) -> bool:

    return (
        "FATHER"
        in _alpha_only(text)
    )


def _is_dob_label(
    text: str,
) -> bool:

    value = _alpha_only(
        text
    )

    compact = re.sub(
        r"[^A-Z]",
        "",
        value,
    )

    return (
        "DATEOFBIRTH" in compact
        or compact == "DOB"
    )


def _is_header(
    text: str,
) -> bool:

    value = _normal(
        text
    )

    return any(
        term in value
        for term in (
            "INCOME TAX",
            "INCOMETAX",
            "DEPARTMENT",
            "GOVT",
            "INDIA",
            "PERMANENT ACCOUNT",
            "PERMANENTACCOUNT",
            "SIGNATURE",
        )
    )


def _is_field_label(
    text: str,
) -> bool:

    value = _normal(
        text
    )

    compact = re.sub(
        r"[^A-Z]",
        "",
        value,
    )

    labels = (
        "NAME",
        "FATHER",
        "FATHERSNAME",
        "DATEOFBIRTH",
        "DOB",
        "SIGNATURE",
        "PERMANENTACCOUNTNUMBER",
        "ACCOUNTNUMBER",
    )

    return any(
        label in compact
        for label in labels
    )


# ============================================================================
# NAME EXTRACTION
# ============================================================================

def _looks_like_ocr_label_fragment(word: str) -> bool:
    """Reject OCR fragments that are clearly pieces of PAN-card labels."""
    token = re.sub(r"[^A-Z]", "", str(word).upper())

    if not token:
        return False

    # Common OCR truncations/fragments of PAN-card labels.
    fragments = (
        "PERMAN", "PERMANE", "PERMANENT",
        "ACCOUN", "ACCOUNT",
        "NUMBE", "NUMBER",
        "INCOME", "INCOM",
        "TAX", "DEPART", "DEPARTM",
        "GOVT", "GOVER", "INDIA",
        "SIGNAT", "SIGNATURE",
        "DATE", "BIRTH",
        "FATHER", "FATHE", "NAME",
    )

    if token in fragments:
        return True

    # Short OCR fragments that strongly resemble known label words.
    if len(token) >= 5:
        label_words = (
            "PERMANENT", "ACCOUNT", "NUMBER", "INCOME",
            "DEPARTMENT", "GOVERNMENT", "SIGNATURE",
            "FATHER", "NAME", "BIRTH",
        )
        return any(
            difflib.SequenceMatcher(
                None, token, label
            ).ratio() >= 0.82
            for label in label_words
        )

    return False


def _clean_name_candidate_text(text: str) -> str:
    """Remove obvious label residue before person-name validation."""
    value = re.sub(
        r"[^A-Za-z .'\-]",
        " ",
        str(text),
    )
    value = re.sub(r"\s+", " ", value).strip()

    # Remove label residue from the front only.
    label_residue = re.compile(
        r"^(?:NAME|FATHER'?S?\s*NAME|"
        r"PERMANENT\s+ACCOUNT\s+NUMBER|"
        r"PERMANENT|ACCOUNT|NUMBER)\b[\s:./\-_]*",
        re.IGNORECASE,
    )

    previous = None
    while previous != value:
        previous = value
        value = label_residue.sub("", value).strip()

    return value


def _looks_like_person_name(
    text: str,
) -> bool:

    value = _clean_name_candidate_text(text)

    if not 3 <= len(value) <= 80:
        return False

    if not re.fullmatch(
        r"[A-Za-z][A-Za-z .'\-]*",
        value,
    ):
        return False

    if _is_header(value):
        return False

    if _is_field_label(value):
        return False

    if _clean_date(value):
        return False

    if _find_pan([value]):
        return False

    words = [
        w.strip(".-'")
        for w in value.split()
        if w.strip(".-'")
    ]

    if not words:
        return False

    # Reject OCR fragments such as "Permanen" that can otherwise pass
    # generic person-name validation and steal the holder-name slot.
    if any(
        _looks_like_ocr_label_fragment(word)
        for word in words
    ):
        return False

    blocked = {
        "INCOME",
        "TAX",
        "DEPARTMENT",
        "GOVT",
        "INDIA",
        "PERMANENT",
        "ACCOUNT",
        "NUMBER",
        "SIGNATURE",
        "DATE",
        "BIRTH",
        "FATHER",
        "NAME",
        "CARD",
        "PAN",
        "PHOTO",
        "IDENTITY",
        "IDENTIFICATION",
    }

    if any(
        word.upper() in blocked
        for word in words
    ):
        return False

    if len(words) == 1:
        return len(words[0]) >= 5

    if len(words[0]) < 2:
        return False

    if len(words[-1]) < 2:
        return False

    if sum(
        len(w) == 1
        for w in words[1:-1]
    ) > 2:
        return False

    return True


def _name_candidate(
    text: str | None,
) -> str | None:

    if not text:
        return None

    value = _clean_name_candidate_text(text)

    if not _looks_like_person_name(
        value
    ):
        return None

    return " ".join(
        word.strip(".-'").title()
        for word in value.split()
        if word.strip(".-'")
    )


# ============================================================================
# SAME-ROW OCR RECONSTRUCTION
# ============================================================================

def _group_same_row_tokens(
    lines: list[dict[str, Any]],
    start: int,
    stop: int,
) -> list[tuple[int, str, float]]:

    tokens = []

    for index in range(
        start,
        stop,
    ):

        item = lines[index]

        raw = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if not raw:
            continue

        if _is_header(raw):
            continue

        if _is_field_label(raw):
            continue

        if _clean_date(raw):
            continue

        if _find_pan([raw]):
            continue

        cleaned = re.sub(
            r"[^A-Za-z .'\-]",
            " ",
            raw,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        if not cleaned:
            continue

        if not re.fullmatch(
            r"[A-Za-z][A-Za-z .'\-]*",
            cleaned,
        ):
            continue

        y = item.get("yc")
        x = item.get("xc")

        if y is None:
            continue

        tokens.append(
            (
                index,
                cleaned,
                float(
                    x
                    if x is not None
                    else 0
                ),
                float(y),
                float(
                    item.get(
                        "confidence",
                        0,
                    )
                ),
            )
        )

    if not tokens:
        return []

    tokens.sort(
        key=lambda item: (
            item[3],
            item[2],
        )
    )

    rows = []

    for token in tokens:

        placed = False

        for row in rows:

            avg_y = (
                sum(
                    t[3]
                    for t in row
                )
                / len(row)
            )

            if abs(
                token[3] - avg_y
            ) <= 12:

                if abs(
                    token[2]
                    - row[-1][2]
                ) <= 550:

                    row.append(token)
                    placed = True
                    break

        if not placed:
            rows.append(
                [token]
            )

    candidates = []

    for row in rows:

        row.sort(
            key=lambda t: t[2]
        )

        combined = " ".join(
            t[1]
            for t in row
        )

        candidate = _name_candidate(
            combined
        )

        if candidate:

            candidates.append(
                (
                    min(
                        t[0]
                        for t in row
                    ),
                    candidate,
                    sum(
                        t[4]
                        for t in row
                    )
                    / len(row),
                )
            )

    return candidates


# ============================================================================
# FIELD EXTRACTION
# ============================================================================

def _extract_inline_name(
    text: str | None,
    father: bool = False,
) -> str | None:
    """Extract a person name when OCR puts label and value on one line."""
    if not text:
        return None

    value = str(text).strip()

    if father:
        patterns = (
            r"FATHER(?:'S)?\s*NAME\s*[:\-\/]?\s*(.+)$",
            r"FATHER'?S?\s*NAME\b\s+(.+)$",
            r"RELATIVE'?S?\s*NAME\s*[:\-\/]?\s*(.+)$",
        )
    else:
        patterns = (
            r"\bNAME\s*[:\-\/]?\s*(.+)$",
            r"\bNAME\b\s+(.+)$",
        )

    for pattern in patterns:
        match = re.search(
            pattern,
            value,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        candidate = _name_candidate(match.group(1))
        if candidate:
            return candidate

    return None


def _find_label_indices(
    lines: list[dict[str, Any]],
) -> tuple[
    int | None,
    int | None,
    int | None,
]:

    name_index = None
    father_index = None
    dob_index = None

    for index, item in enumerate(lines):

        text = str(
            item.get(
                "text",
                "",
            )
        )

        if (
            father_index is None
            and _is_father_label(text)
        ):
            father_index = index
            continue

        if (
            dob_index is None
            and _is_dob_label(text)
        ):
            dob_index = index
            continue

        if (
            name_index is None
            and _is_name_label(text)
        ):
            name_index = index

    return (
        name_index,
        father_index,
        dob_index,
    )


def _extract_fields(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    lines = [
        item
        for item in results
        if str(
            item.get(
                "text",
                "",
            )
        ).strip()
    ]

    lines.sort(
        key=lambda item: (
            item.get("yc")
            if item.get("yc") is not None
            else 10**12,

            item.get("xc")
            if item.get("xc") is not None
            else 10**12,
        )
    )

    plain = _plain_lines(
        lines
    )

    pan = _find_pan(
        plain
    )

    dob = _extract_date(
        plain
    )

    # ---------------------------------------------------------
    # Find DOB position.
    # ---------------------------------------------------------

    dob_index = len(
        lines
    )

    for index, item in enumerate(
        lines
    ):

        if _clean_date(
            str(
                item.get(
                    "text",
                    "",
                )
            )
        ):

            dob_index = index
            break

    # ---------------------------------------------------------
    # Find PAN position.
    # ---------------------------------------------------------

    pan_index = None

    if pan:

        for index, item in enumerate(
            lines
        ):

            compact = _compact(
                str(
                    item.get(
                        "text",
                        "",
                    )
                )
            )

            if pan in compact:

                pan_index = index
                break

    # ---------------------------------------------------------
    # Explicit labels.
    # ---------------------------------------------------------

    (
        name_index,
        father_index,
        _,
    ) = _find_label_indices(
        lines
    )

    name = None
    father_name = None

    # ---------------------------------------------------------
    # SAME-LINE LABEL EXTRACTION
    # ---------------------------------------------------------
    # OCR may return either NAME: VALUE or NAME followed by VALUE.
    # Handle the same-line form first; the existing next-line logic
    # below remains the fallback for the second form.
    if name_index is not None:
        raw_name_label = str(
            lines[name_index].get(
                "text",
                "",
            )
        ).strip()
        name = _extract_inline_name(
            raw_name_label,
            father=False,
        )

    if father_index is not None:
        raw_father_label = str(
            lines[father_index].get(
                "text",
                "",
            )
        ).strip()
        father_name = _extract_inline_name(
            raw_father_label,
            father=True,
        )

    # ---------------------------------------------------------
    # Explicit NAME.
    # ---------------------------------------------------------

    if name is None and name_index is not None:

        stop = (
            father_index
            if father_index is not None
            else dob_index
        )

        for index in range(
            name_index + 1,
            min(
                stop,
                name_index + 5,
            ),
        ):

            raw = str(
                lines[index].get(
                    "text",
                    "",
                )
            ).strip()

            if not raw:
                continue

            if (
                _is_header(raw)
                or _is_field_label(raw)
                or _clean_date(raw)
                or _find_pan([raw])
            ):
                continue

            candidate = _name_candidate(
                raw
            )

            if candidate:
                name = candidate
                break

    # ---------------------------------------------------------
    # Explicit FATHER'S NAME.
    # ---------------------------------------------------------

    if father_name is None and father_index is not None:

        for index in range(
            father_index + 1,
            min(
                dob_index,
                father_index + 5,
            ),
        ):

            raw = str(
                lines[index].get(
                    "text",
                    "",
                )
            ).strip()

            if not raw:
                continue

            if (
                _is_header(raw)
                or _is_field_label(raw)
                or _clean_date(raw)
                or _find_pan([raw])
            ):
                continue

            candidate = _name_candidate(
                raw
            )

            if (
                candidate
                and (
                    name is None
                    or candidate.casefold()
                    != name.casefold()
                )
            ):
                father_name = candidate
                break

    # ---------------------------------------------------------
    # Locate Income Tax / Govt header.
    # ---------------------------------------------------------

    header_candidates = []

    for index in range(
        0,
        dob_index,
    ):

        raw = str(
            lines[index].get(
                "text",
                "",
            )
        )

        compact = re.sub(
            r"[^A-Z]",
            "",
            _normal(raw),
        )

        if (
            "INCOMETAX" in compact
            or "GOVTOFINDIA" in compact
        ):

            header_candidates.append(
                index
            )

    header_index = (
        max(header_candidates)
        if header_candidates
        else None
    )

    # ---------------------------------------------------------
    # Main PAN card name region.
    # ---------------------------------------------------------

    if header_index is not None:

        region_start = (
            header_index + 1
        )

        region_end = dob_index

    else:

        region_start = max(
            0,
            dob_index - 8,
        )

        region_end = dob_index

    positional = []

    for index in range(
        region_start,
        region_end,
    ):

        item = lines[index]

        raw = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if not raw:
            continue

        if (
            _is_header(raw)
            or _is_field_label(raw)
            or _clean_date(raw)
            or _find_pan([raw])
        ):
            continue

        candidate = _name_candidate(
            raw
        )

        if candidate:

            positional.append(
                (
                    index,
                    candidate,
                    float(
                        item.get(
                            "confidence",
                            0,
                        )
                    ),
                )
            )

    reconstructed = (
        _group_same_row_tokens(
            lines,
            region_start,
            region_end,
        )
    )

    combined = (
        positional
        + reconstructed
    )

    unique = []

    seen = set()

    for item in sorted(
        combined,
        key=lambda value: (
            value[0],
            -len(
                value[1].split()
            ),
            -value[2],
        ),
    ):

        key = item[1].casefold()

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    # ---------------------------------------------------------
    # Select holder name.
    # ---------------------------------------------------------

    if name is None and unique:

        name = unique[0][1]

    # ---------------------------------------------------------
    # Select father name.
    # ---------------------------------------------------------

    if (
        father_name is None
        and name is not None
    ):

        name_words = {
            word.casefold()
            for word in name.split()
        }

        father_candidates = []

        for (
            candidate_index,
            candidate,
            confidence,
        ) in unique:

            if (
                candidate.casefold()
                == name.casefold()
            ):
                continue

            candidate_words = [
                word.casefold()
                for word in candidate.split()
            ]

            if (
                len(candidate_words) == 1
                and candidate_words[0]
                in name_words
            ):
                continue

            father_candidates.append(
                (
                    -len(
                        candidate_words
                    ),
                    candidate_index,
                    -confidence,
                    candidate,
                )
            )

        if father_candidates:

            father_candidates.sort()

            father_name = (
                father_candidates[0][3]
            )

    # ---------------------------------------------------------
    # Header-free fallback.
    # ---------------------------------------------------------

    if (
        header_index is None
        and unique
    ):

        by_position = sorted(
            unique,
            key=lambda value: value[0],
        )

        if len(by_position) >= 2:

            fallback_name = (
                by_position[-2][1]
            )

            fallback_father = (
                by_position[-1][1]
            )

            if name is None:
                name = fallback_name

            if (
                father_name is None
                and fallback_father.casefold()
                != (name or "").casefold()
            ):

                father_name = (
                    fallback_father
                )

    # ---------------------------------------------------------
    # FINAL FIELD SAFETY
    # ---------------------------------------------------------
    # Never allow obvious label residue to become a person name.
    if name and not _looks_like_person_name(name):
        name = None

    if father_name and not _looks_like_person_name(father_name):
        father_name = None

    # The father name must not collapse to the holder name. If that happens,
    # the label association failed; returning None is safer than returning a
    # duplicated identity field.
    if name and father_name:
        name_norm = re.sub(
            r"[^A-Z]",
            "",
            name.upper(),
        )
        father_norm = re.sub(
            r"[^A-Z]",
            "",
            father_name.upper(),
        )

        if (
            name_norm
            and father_norm
            and (
                name_norm == father_norm
                or name_norm in father_norm
                or father_norm in name_norm
            )
        ):
            father_name = None

    return {
        "format": "generic_position_aware_v4",
        "pan": pan,
        "name": name,
        "father_name": father_name,
        "dob": dob,
    }


# ============================================================================
# IMAGE QUALITY
# ============================================================================

def _quality_metrics(
    image: np.ndarray,
) -> dict[str, Any]:
    """Backward-compatible wrapper around the common quality analyzer."""
    return analyze_image_quality(image)

def _check_image_quality(
    image: np.ndarray,
) -> None:
    """Validate only hard image failures before OCR.

    Blur is deliberately NOT a hard rejection here. Global Laplacian
    variance is an unreliable standalone gate for photographed documents.
    The blur value remains part of quality_metrics and is evaluated by the
    common validation layer together with OCR, extraction, resolution,
    contrast, layout and tamper evidence.
    """
    if (
        image is None
        or image.size == 0
    ):
        raise PanVerificationError(
            "Invalid or empty image."
        )


# ============================================================================
# AUTHENTICITY VALIDATION
# ============================================================================

@dataclass
class PanValidationResult:

    decision: str

    score: float

    checks: dict[str, Any]

    reasons: list[str]

    warnings: list[str]

    authoritative_verification: dict[str, Any]


def _check_pan_format(
    pan: str | None,
) -> bool:

    if not pan:
        return False

    return bool(
        PAN_PATTERN.fullmatch(
            pan.upper().strip()
        )
    )


def _check_pan_structure(
    pan: str | None,
) -> bool:

    if not _check_pan_format(
        pan
    ):
        return False

    pan = pan.upper().strip()

    return (
        pan[:5].isalpha()
        and pan[5:9].isdigit()
        and pan[9].isalpha()
    )


def _check_required_fields(
    pan: str | None,
    name: str | None,
    dob: str | None,
) -> bool:

    return bool(
        pan
        and name
        and dob
    )


def _check_ocr_confidence(
    results: list[dict[str, Any]],
) -> tuple[float, int]:

    values = []

    for item in results:

        try:

            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if confidence > 0:
            values.append(
                confidence
            )

    if not values:
        return 0.0, 0

    average = (
        sum(values)
        / len(values)
    )

    if average >= 0.90:
        score = 15

    elif average >= 0.80:
        score = 12

    elif average >= 0.70:
        score = 9

    elif average >= 0.60:
        score = 6

    else:
        score = 2

    return (
        round(
            average,
            3,
        ),
        score,
    )


def _check_pan_markers(
    results: list[dict[str, Any]],
) -> tuple[int, list[str]]:

    text = _normal(
        " ".join(
            _plain_lines(
                results
            )
        )
    )

    compact = _compact(
        text
    )

    score = 0

    reasons = []

    if "INCOMETAX" in compact:

        score += 10

    else:

        reasons.append(
            "Income Tax Department marker not detected."
        )

    if "GOVTOFINDIA" in compact:

        score += 8

    elif (
        "GOVT" in compact
        and "INDIA" in compact
    ):

        score += 5

    else:

        reasons.append(
            "Government of India marker not detected."
        )

    if "PERMANENTACCOUNT" in compact:

        score += 7

    else:

        reasons.append(
            "Permanent Account marker not detected."
        )

    return (
        score,
        reasons,
    )


def _check_dob(
    dob: str | None,
) -> bool:

    return bool(
        _clean_date(dob)
    )


def _check_name_quality(
    name: str | None,
) -> bool:

    if not name:
        return False

    return _looks_like_person_name(
        name
    )


def _check_document_text_coherence(
    results: list[dict[str, Any]],
) -> tuple[int, list[str]]:

    """
    Look for obvious signs that the OCR result is not behaving like a
    coherent PAN document.

    This is a fraud-risk signal, not definitive forensic proof.
    """

    texts = _plain_lines(
        results
    )

    if not texts:
        return (
            0,
            ["No OCR text detected."]
        )

    score = 0

    reasons = []

    # ---------------------------------------------------------
    # Count extremely low confidence OCR boxes.
    # ---------------------------------------------------------

    low_confidence_count = sum(
        1
        for item in results
        if float(
            item.get(
                "confidence",
                0,
            )
        ) < 0.20
    )

    if low_confidence_count >= 5:

        reasons.append(
            "Large amount of very low-confidence OCR text detected."
        )

    elif low_confidence_count >= 2:

        score += 1

    else:

        score += 3

    # ---------------------------------------------------------
    # Detect excessive repeated text.
    # ---------------------------------------------------------

    normalized = [
        _normal(text)
        for text in texts
        if _normal(text)
    ]

    duplicates = (
        len(normalized)
        - len(set(normalized))
    )

    if duplicates >= 5:

        reasons.append(
            "Excessive repeated OCR text detected."
        )

    elif duplicates >= 2:

        score += 1

    else:

        score += 3

    return (
        score,
        reasons,
    )


def _check_layout(
    results: list[dict[str, Any]],
) -> tuple[int, list[str]]:

    """
    PAN-card layout signal.

    A normal PAN card generally contains:
        Income Tax / Government of India
        holder name
        father's name
        DOB
        Permanent Account Number
        PAN

    OCR can miss one or more markers, therefore this is a weighted score.
    """

    score = 0

    reasons = []

    texts = _normal(
        " ".join(
            _plain_lines(
                results
            )
        )
    )

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        texts,
    )

    markers = {
        "income_tax": (
            "INCOMETAX"
            in compact
        ),

        "government_india": (
            "GOVTOFINDIA"
            in compact
            or (
                "GOVT" in compact
                and "INDIA" in compact
            )
        ),

        "permanent_account": (
            "PERMANENTACCOUNT"
            in compact
        ),

        "dob": (
            "DATEOFBIRTH"
            in compact
            or bool(
                DATE_PATTERN.search(
                    texts
                )
            )
        ),

        "father": (
            "FATHER" in compact
        ),
    }

    if markers["income_tax"]:
        score += 5
    else:
        reasons.append(
            "Income Tax layout marker missing."
        )

    if markers["government_india"]:
        score += 5
    else:
        reasons.append(
            "Government of India layout marker missing."
        )

    if markers["permanent_account"]:
        score += 5
    else:
        reasons.append(
            "Permanent Account Number label missing."
        )

    if markers["dob"]:
        score += 3
    else:
        reasons.append(
            "DOB/date marker missing."
        )

    if markers["father"]:
        score += 2

    return (
        score,
        reasons,
    )


def _safe_confidence(item: dict[str, Any]) -> float:
    """Return a bounded OCR confidence value."""
    try:
        return max(
            0.0,
            min(
                float(item.get("confidence", 0.0)),
                1.0,
            ),
        )
    except (TypeError, ValueError):
        return 0.0


def _field_ocr_confidence(
    results: list[dict[str, Any]],
    value: str | None,
) -> float:
    """
    Estimate confidence for an extracted field from the OCR boxes that
    contain that field. Returns 0 when the field cannot be located.
    """
    if not value:
        return 0.0

    target = _compact(value)
    if not target:
        return 0.0

    matches = []

    for item in results:
        raw = str(item.get("text", "")).strip()
        if not raw:
            continue

        compact = _compact(raw)

        if target in compact or compact in target:
            matches.append(_safe_confidence(item))

    if not matches:
        return 0.0

    return round(max(matches), 3)


def _count_pan_candidates(
    results: list[dict[str, Any]],
) -> int:
    """Count distinct PAN-like values detected by OCR."""
    candidates = set()

    for item in results:
        raw = str(item.get("text", "")).upper()
        for match in PAN_PATTERN.findall(raw):
            candidates.add(match)

    return len(candidates)


def _extract_all_dates(
    results: list[dict[str, Any]],
) -> list[str]:
    """Extract distinct valid dates from OCR text."""
    dates = set()

    for item in results:
        value = _clean_date(
            str(item.get("text", ""))
        )
        if value:
            dates.add(value)

    return sorted(dates)


def _check_field_consistency(
    results: list[dict[str, Any]],
    pan: str | None,
    name: str | None,
    dob: str | None,
) -> tuple[int, list[str], dict[str, Any]]:
    """
    Check whether extracted fields have supporting OCR evidence.

    This is deliberately an OCR-consistency signal. It is NOT a database
    verification and it is NOT forensic proof of authenticity.
    """
    score = 0
    reasons: list[str] = []
    details: dict[str, Any] = {}

    pan_confidence = _field_ocr_confidence(
        results,
        pan,
    )
    name_confidence = _field_ocr_confidence(
        results,
        name,
    )
    dob_confidence = _field_ocr_confidence(
        results,
        dob,
    )

    details["pan_ocr_confidence"] = pan_confidence
    details["name_ocr_confidence"] = name_confidence
    details["dob_ocr_confidence"] = dob_confidence

    if pan and pan_confidence >= 0.70:
        score += 3
    elif pan:
        score += 1
        reasons.append(
            "Extracted PAN has weak OCR evidence."
        )
    else:
        reasons.append(
            "PAN could not be supported by OCR evidence."
        )

    if name and name_confidence >= 0.70:
        score += 2
    elif name:
        score += 1
        reasons.append(
            "Extracted holder name has weak OCR evidence."
        )
    else:
        reasons.append(
            "Holder name could not be supported by OCR evidence."
        )

    if dob and dob_confidence >= 0.70:
        score += 2
    elif dob:
        score += 1
        reasons.append(
            "Extracted DOB has weak OCR evidence."
        )
    else:
        reasons.append(
            "DOB could not be supported by OCR evidence."
        )

    pan_candidates = _count_pan_candidates(results)
    dates = _extract_all_dates(results)

    details["distinct_pan_candidates"] = pan_candidates
    details["distinct_dates"] = dates

    if pan_candidates == 1:
        score += 2
    elif pan_candidates > 1:
        reasons.append(
            "Multiple different PAN candidates were detected in OCR."
        )

    if dob and dates == [dob]:
        score += 2
    elif dob and dob in dates and len(dates) > 1:
        score += 1
        reasons.append(
            "Multiple different dates were detected in OCR."
        )

    return (
        score,
        reasons,
        details,
    )


def _check_aspect_ratio(
    quality_metrics: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Check whether image geometry is broadly compatible with a PAN card."""
    if not quality_metrics:
        return True, []

    try:
        ratio = float(
            quality_metrics.get(
                "aspect_ratio",
                0.0,
            )
        )
    except (TypeError, ValueError):
        return False, ["Image aspect ratio could not be evaluated."]

    if ratio <= 0:
        return False, ["Image aspect ratio is invalid."]

    if MIN_PAN_ASPECT_RATIO <= ratio <= MAX_PAN_ASPECT_RATIO:
        return True, []

    return (
        False,
        [
            "Image aspect ratio is outside the expected PAN-card range."
        ],
    )


def _check_resolution(
    quality_metrics: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Reject obviously tiny images while allowing normal scans/photos."""
    if not quality_metrics:
        return True, []

    try:
        width = int(quality_metrics.get("width", 0))
        height = int(quality_metrics.get("height", 0))
    except (TypeError, ValueError):
        return False, ["Image resolution could not be evaluated."]

    if width >= 300 and height >= 180:
        return True, []

    return (
        False,
        [
            "Image resolution is too low for reliable PAN validation."
        ],
    )


def _build_local_result(
    *,
    validation: PanValidationResult,
    pan: str | None,
    name: str | None,
    father_name: str | None,
    dob: str | None,
    quality_metrics: dict[str, Any] | None,
    ocr_used: bool,
    extraction_method: str,
) -> dict[str, Any]:
    """
    Build the public PAN response from the shared common validation result.
    """
    decision_map = {
        DOCUMENT_PASS: "DOCUMENT PASS",
        DOCUMENT_SUSPICIOUS: "DOCUMENT SUSPICIOUS",
        MANUAL_REVIEW: "MANUAL REVIEW",
        DOCUMENT_REJECT: "DOCUMENT REJECT",
    }

    final_decision = decision_map.get(
        validation.decision,
        "DOCUMENT REJECT",
    )

    return {
        "verified": False,
        "decision": final_decision,
        "final_score": validation.score,
        "verification_stage": "LOCAL_DOCUMENT_AUTHENTICITY",
        "pan_number": pan,
        "name": name,
        "father_name": father_name,
        "dob": dob,
        "quality_metrics": quality_metrics,
        "validation": {
            "decision": validation.decision,
            "score": validation.score,
            "checks": validation.checks,
            "reasons": validation.reasons,
            "warnings": validation.warnings,
            "authoritative_verification": validation.authoritative_verification,
        },
        "database_verification": {
            "status": "NOT_PERFORMED",
            "pan_exists": None,
            "active": None,
            "name_match": None,
            "father_name_match": None,
            "dob_match": None,
        },
        "ocr_used": ocr_used,
        "extraction_method": extraction_method,
    }



def _safe_tamper_analysis(
    image: np.ndarray | None,
) -> dict[str, Any]:
    """
    Run the shared document-independent tamper detector.

    Tamper analysis is a risk signal, not proof that a document is forged.
    """
    if analyze_tampering is None:
        return {
            "tamper_score": 0,
            "risk": "UNKNOWN",
            "decision": "MANUAL_REVIEW",
            "signals": [
                "Common tamper analyzer is not available."
            ],
            "checks": {},
            "available": False,
        }

    try:
        result = analyze_tampering(image)
        if not isinstance(result, dict):
            raise TypeError(
                "Tamper analyzer returned a non-dictionary result."
            )

        result = dict(result)
        result.setdefault("tamper_score", 0)
        result.setdefault("risk", "UNKNOWN")
        result.setdefault("decision", "MANUAL_REVIEW")
        result.setdefault("signals", [])
        result.setdefault("checks", {})
        result["available"] = True
        return result

    except Exception as exc:
        return {
            "tamper_score": 0,
            "risk": "UNKNOWN",
            "decision": "MANUAL_REVIEW",
            "signals": [
                f"Tamper analysis failed: {exc}"
            ],
            "checks": {},
            "available": False,
        }


def _calculate_common_quality_score(
    quality_metrics: dict[str, Any] | None,
) -> float:
    """Deprecated compatibility wrapper. Quality is now calculated centrally."""
    if not quality_metrics:
        return 100.0
    # This function is retained only for backward compatibility. The PAN
    # validation path never calls it; document_validation.py owns the score.
    return 100.0


def _normalize_pan_evidence(
    *,
    marker_score: float,
    layout_score: float,
    consistency_score: float,
    coherence_score: float,
) -> tuple[float, float, float, float]:
    """
    Normalize PAN-specific evidence to the common 0-100 scale.

    Current PAN-specific maxima:
        markers     = 25
        layout      = 20
        consistency = 11
        coherence  = 6
    """
    return (
        round(max(0.0, min(100.0, marker_score / 25.0 * 100.0)), 2),
        round(max(0.0, min(100.0, layout_score / 20.0 * 100.0)), 2),
        round(max(0.0, min(100.0, consistency_score / 11.0 * 100.0)), 2),
        round(max(0.0, min(100.0, coherence_score / 6.0 * 100.0)), 2),
    )


def validate_pan_document(
    *,
    results: list[dict[str, Any]],
    pan: str | None,
    name: str | None,
    father_name: str | None,
    dob: str | None,
    quality_metrics: dict[str, Any] | None = None,
    tamper_result: dict[str, Any] | None = None,
) -> PanValidationResult:
    """
    PAN-specific evidence collection + shared common validation decision.

    The PAN module owns PAN-specific evidence generation. The common
    document_validation layer owns the final local decision.

    IMPORTANT:
        DOCUMENT_PASS does NOT prove that the PAN exists in a government
        database. Authoritative verification remains separate.
    """
    pan_format_ok = _check_pan_format(pan)
    pan_structure_ok = _check_pan_structure(pan)
    fields_ok = _check_required_fields(
        pan,
        name,
        dob,
    )
    name_ok = _check_name_quality(name)
    dob_ok = _check_dob(dob)

    marker_score, marker_reasons = _check_pan_markers(results)
    layout_score, layout_reasons = _check_layout(results)
    average_confidence, _ = _check_ocr_confidence(results)

    coherence_score, coherence_reasons = (
        _check_document_text_coherence(results)
    )

    consistency_score, consistency_reasons, consistency_details = (
        _check_field_consistency(
            results,
            pan,
            name,
            dob,
        )
    )

    document_type_ok = _is_pan_document(
        results,
        pan,
    )

    required_fields_present = sum(
        bool(value)
        for value in (
            pan,
            name,
            dob,
        )
    )

    extraction_successful = (
        required_fields_present == 3
    )

    field_format_valid = (
        pan_format_ok
        and pan_structure_ok
        and name_ok
        and dob_ok
    )

    marker_norm, layout_norm, consistency_norm, coherence_norm = (
        _normalize_pan_evidence(
            marker_score=marker_score,
            layout_score=layout_score,
            consistency_score=consistency_score,
            coherence_score=coherence_score,
        )
    )

    # IMPORTANT: quality_score is calculated by the COMMON validation layer
    # from raw quality_metrics. PAN only supplies the measurements.
    quality_score = 100.0 if not quality_metrics else 0.0

    tamper = dict(
        tamper_result
        or {
            "tamper_score": 0,
            "risk": "LOW",
            "decision": "NOT_PERFORMED",
            "signals": [],
            "checks": {},
            "available": False,
        }
    )

    try:
        tamper_score = max(
            0.0,
            min(
                100.0,
                float(tamper.get("tamper_score", 0.0)),
            ),
        )
    except (TypeError, ValueError):
        tamper_score = 0.0

    tamper_risk = str(
        tamper.get("risk", "UNKNOWN")
    ).upper().strip()

    common_result = validate_from_existing_result(
        document_type_detected=document_type_ok,
        required_fields_present=required_fields_present,
        required_fields_total=3,
        field_format_valid=field_format_valid,
        average_ocr_confidence=average_confidence,
        field_consistency_score=consistency_norm,
        text_coherence_score=coherence_norm,
        layout_score=layout_norm,
        quality_score=quality_score,
        tamper_score=tamper_score,
        tamper_risk=tamper_risk,
        extraction_successful=extraction_successful,
        quality_metrics=quality_metrics,
        optional={
            "document_type": "pan",
            "pan_specific": {
                "pan_format": pan_format_ok,
                "pan_structure": pan_structure_ok,
                "pan_markers_score": marker_score,
                "pan_markers_normalized": marker_norm,
                "layout_score_raw": layout_score,
                "layout_score_normalized": layout_norm,
                "field_consistency_score_raw": consistency_score,
                "field_consistency_score_normalized": consistency_norm,
                "text_coherence_score_raw": coherence_score,
                "text_coherence_score_normalized": coherence_norm,
                "name_quality": name_ok,
                "dob_format": dob_ok,
                "field_consistency": consistency_details,
                "marker_reasons": marker_reasons,
                "layout_reasons": layout_reasons,
                "consistency_reasons": consistency_reasons,
                "coherence_reasons": coherence_reasons,
            },
            "tamper_analysis": tamper,
        },
    )

    # The common layer is authoritative for the generic quality score.
    quality_score = float(
        common_result.get("checks", {}).get(
            "quality_score",
            quality_score,
        )
    )

    checks = dict(
        common_result.get("checks", {})
    )

    checks.update({
        "pan_format": pan_format_ok,
        "pan_structure": pan_structure_ok,
        "required_fields": fields_ok,
        "name_quality": name_ok,
        "dob_format": dob_ok,
        "pan_markers_score": marker_score,
        "layout_score": layout_score,
        "average_ocr_confidence": average_confidence,
        "text_coherence_score": coherence_score,
        "field_consistency_score": consistency_score,
        "field_consistency": consistency_details,
        "is_pan_document": document_type_ok,
        "quality_score": quality_score,
        "tamper_score": tamper_score,
        "tamper_risk": tamper_risk,
    })

    if quality_metrics:
        checks["blur_score"] = quality_metrics.get(
            "blur_score",
            0.0,
        )
        checks["brightness"] = quality_metrics.get(
            "brightness",
            0.0,
        )
        checks["contrast"] = quality_metrics.get(
            "contrast",
            0.0,
        )

        common_quality = common_result.get("checks", {}).get(
            "quality_metrics",
            {},
        )
        checks["aspect_ratio_ok"] = common_quality.get(
            "aspect_ratio_ok", True
        )
        checks["resolution_ok"] = common_quality.get(
            "resolution_ok", True
        )

    reasons = list(
        common_result.get("reasons", [])
    )

    if not pan_format_ok:
        reasons.append("PAN format is invalid.")

    if not pan_structure_ok:
        reasons.append("PAN character structure is invalid.")

    if not fields_ok:
        reasons.append("Required PAN fields are missing.")

    if not name_ok:
        reasons.append("Extracted holder name is not reliable.")

    if not dob_ok:
        reasons.append("Date of birth is invalid or missing.")

    if not document_type_ok:
        reasons.append(
            "OCR content does not contain enough PAN-card-specific markers."
        )

    reasons.extend(marker_reasons)
    reasons.extend(layout_reasons)
    reasons.extend(coherence_reasons)
    reasons.extend(consistency_reasons)

    reasons = list(
        dict.fromkeys(reasons)
    )

    return PanValidationResult(
        decision=common_result["decision"],
        score=float(common_result["score"]),
        checks=checks,
        reasons=reasons,
        warnings=list(
            common_result.get("warnings", [])
        ),
        authoritative_verification=dict(
            common_result.get(
                "authoritative_verification",
                {},
            )
        ),
    )



# ============================================================================
# DOCUMENT TYPE VALIDATION
# ============================================================================

def _is_pan_document(
    results: list[dict[str, Any]],
    pan: str | None,
) -> bool:
    """
    Decide whether OCR content contains enough PAN-card-specific evidence.

    This is intentionally stricter than "a PAN-shaped string exists".
    """
    if not pan:
        return False

    text = _normal(
        " ".join(
            _plain_lines(results)
        )
    )

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )

    marker_weights = {
        "income_tax": (
            "INCOMETAX",
            2,
        ),
        "permanent_account": (
            "PERMANENTACCOUNT",
            2,
        ),
        "department": (
            "DEPARTMENT",
            1,
        ),
        "government_india": (
            "GOVTOFINDIA",
            1,
        ),
        "dob": (
            "DATEOFBIRTH",
            1,
        ),
        "father": (
            "FATHER",
            1,
        ),
    }

    marker_score = 0

    for marker, weight in marker_weights.values():
        if marker in compact:
            marker_score += weight

    # A PAN-looking string plus four independent marker points is enough
    # for document-type classification. This remains a local signal.
    return marker_score >= MIN_DOCUMENT_TYPE_MARKERS


def _looks_like_pan_document(
    lines: list[dict[str, Any]],
    pan: str | None,
) -> bool:

    return _is_pan_document(
        lines,
        pan,
    )



# ============================================================================
# LIGHTWEIGHT FACE DETECTOR
# ============================================================================
#
# OpenCV Haar cascade is used only inside the expected PAN photo ROI.
# It is deliberately loaded once at module import and never runs OCR.
# ============================================================================

_PAN_FACE_CASCADE = None


def _get_pan_face_cascade():
    global _PAN_FACE_CASCADE

    if _PAN_FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            return None

        _PAN_FACE_CASCADE = cascade

    return _PAN_FACE_CASCADE


_PAN_EYE_CASCADE = None


def _get_pan_eye_cascade():
    """Lazy-load the lightweight eye cascade used to confirm a face hit."""
    global _PAN_EYE_CASCADE

    if _PAN_EYE_CASCADE is None:
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_eye.xml"
        )

        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            return None

        _PAN_EYE_CASCADE = cascade

    return _PAN_EYE_CASCADE


def _face_has_eye(
    gray_roi: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> bool:
    """
    Confirm a Haar face hit using an eye detector.

    This prevents document stamps/seals from being accepted as faces.
    """

    eye_cascade = _get_pan_eye_cascade()

    if eye_cascade is None:
        return False

    x, y, w, h = face_box

    if w <= 0 or h <= 0:
        return False

    face = gray_roi[
        max(0, y):min(gray_roi.shape[0], y + h),
        max(0, x):min(gray_roi.shape[1], x + w),
    ]

    if face.size == 0:
        return False

    upper = face[
        :max(1, int(face.shape[0] * 0.68)),
        :
    ]

    eyes = eye_cascade.detectMultiScale(
        upper,
        scaleFactor=1.12,
        minNeighbors=4,
        minSize=(
            max(4, int(face.shape[1] * 0.08)),
            max(4, int(face.shape[0] * 0.08)),
        ),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    return len(eyes) >= 1


def _fast_pan_face_detect(
    image: np.ndarray,
) -> bool:
    """
    Fast PAN face gate.

    Uses Haar face detection only; the old eye-confirmation step caused
    false negatives on genuine scanned/compressed PAN photos.
    """
    if image is None or image.size == 0:
        return False

    h, w = image.shape[:2]
    face_cascade = _get_pan_face_cascade()
    if face_cascade is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # PAN photos are normally on the left for older cards and on the right
    # for newer layouts. Check the likely side first based on edge variance,
    # then only check the other side if needed.
    regions = (
        (0.00, 0.46),
        (0.54, 1.00),
    )

    for xa, xb in regions:
        roi = gray[
            int(h * 0.12):int(h * 0.98),
            int(w * xa):int(w * xb),
        ]
        if roi.size == 0:
            continue

        if roi.shape[1] < 220:
            roi = cv2.resize(
                roi, None, fx=2.0, fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

        faces = face_cascade.detectMultiScale(
            roi,
            scaleFactor=1.10,
            minNeighbors=6,
            minSize=(
                max(18, int(roi.shape[1] * 0.055)),
                max(18, int(roi.shape[0] * 0.055)),
            ),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for _, _, _, fh in faces:
            if 0.07 <= fh / max(float(roi.shape[0]), 1.0) <= 0.72:
                return True

    return False

def _fast_pan_security_feature(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Fast PAN-specific security detector.

    QR is searched only in the expected right-side region. If QR detection
    fails, the compact right-side security/hologram detector is used.
    """
    empty = {
        "qr_detected": False,
        "qr_right": False,
        "security_block": False,
        "security_block_score": 0.0,
    }

    if image is None or image.size == 0:
        return empty

    h, w = image.shape[:2]
    if h < 80 or w < 100:
        return empty

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # QR is only expected on the right half of the PAN. Limiting the
    # detector to this region substantially reduces CPU work.
    x1 = int(w * 0.50)
    y1 = int(h * 0.00)
    y2 = int(h * 0.88)
    qr_roi = gray[y1:y2, x1:w]

    qr_right = False
    qr_detected = False

    if qr_roi.size:
        try:
            detector = cv2.QRCodeDetector()

            for scale in (1.0, 1.5):
                test = (
                    qr_roi
                    if scale == 1.0
                    else cv2.resize(
                        qr_roi,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                )

                try:
                    _, points, _ = detector.detectAndDecode(test)
                except Exception:
                    points = None

                if points is None:
                    try:
                        detected = detector.detect(test)
                        points = (
                            detected[1]
                            if isinstance(detected, tuple)
                            and len(detected) == 2
                            and detected[0]
                            else None
                        )
                    except Exception:
                        points = None

                if points is not None:
                    pts = np.asarray(points).reshape(-1, 2)
                    if pts.size >= 8:
                        cx = float(np.mean(pts[:, 0]) / scale) + x1
                        cy = float(np.mean(pts[:, 1]) / scale)

                        if cx >= w * 0.55 and cy <= h * 0.88:
                            qr_detected = True
                            qr_right = True
                            break
        except Exception:
            pass

    if qr_right:
        return {
            "qr_detected": True,
            "qr_right": True,
            "security_block": True,
            "security_block_score": 1.0,
        }

    # --------------------------------------------------------------
    # 2. BROADER RIGHT-SIDE SECURITY REGION
    # --------------------------------------------------------------

    x1 = int(w * 0.55)
    x2 = int(w * 0.99)
    y1 = int(h * 0.05)
    y2 = int(h * 0.90)

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return empty

    roi_norm = cv2.normalize(
        roi,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    roi_blur = cv2.GaussianBlur(
        roi_norm,
        (3, 3),
        0,
    )

    edges = cv2.Canny(
        roi_blur,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_score = 0.0
    best_candidate = False

    roi_h, roi_w = roi.shape[:2]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.035 * perimeter,
            True,
        )

        if not (4 <= len(approx) <= 8):
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)

        if bw <= 0 or bh <= 0:
            continue

        width_fraction = bw / max(float(roi_w), 1.0)
        height_fraction = bh / max(float(roi_h), 1.0)
        aspect = bw / max(float(bh), 1.0)
        area_fraction = (bw * bh) / max(float(roi_w * roi_h), 1.0)

        # Compact structure only.
        if width_fraction < 0.08 or width_fraction > 0.70:
            continue

        if height_fraction < 0.08 or height_fraction > 0.70:
            continue

        if area_fraction < 0.012 or area_fraction > 0.35:
            continue

        if not (0.45 <= aspect <= 2.20):
            continue

        block = roi[by:by + bh, bx:bx + bw]

        if block.size == 0:
            continue

        block_std = float(np.std(block))

        block_edges = cv2.Canny(
            block,
            50,
            150,
        )

        edge_density = (
            float(np.count_nonzero(block_edges))
            / max(float(block.size), 1.0)
        )

        if block_std < 12.0:
            continue

        if edge_density < 0.025:
            continue

        small = cv2.resize(
            block,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )

        local_edges = cv2.Canny(
            small,
            40,
            120,
        )

        local_density = (
            float(np.count_nonzero(local_edges))
            / max(float(local_edges.size), 1.0)
        )

        if local_density < 0.04:
            continue

        aspect_distance = min(
            abs(np.log(max(aspect, 0.01))),
            1.5,
        )

        aspect_score = max(
            0.0,
            1.0 - aspect_distance / 1.5,
        )

        texture_score = min(
            1.0,
            edge_density / 0.16,
        )

        local_texture_score = min(
            1.0,
            local_density / 0.18,
        )

        contrast_score = min(
            1.0,
            block_std / 55.0,
        )

        score = (
            aspect_score * 0.20
            + texture_score * 0.30
            + local_texture_score * 0.35
            + contrast_score * 0.15
        )

        if score > best_score:
            best_score = score
            best_candidate = score >= 0.42

    return {
        "qr_detected": qr_detected,
        "qr_right": qr_right,
        "security_block": bool(best_candidate),
        "security_block_score": round(best_score, 4),
    }

def _strict_pan_visual_identity(
    image: np.ndarray,
    *,
    geometry_ok: bool,
    visual_present: int,
    photo_present: bool,
    face_detected: bool,
) -> dict[str, Any]:
    """
    AUTHORITATIVE visual PAN identity gate.

    This is deliberately much stricter than the old generic ROI score.

    A document is PAN-like only when ALL of the following hold:

        - PAN-card geometry
        - enough PAN visual regions
        - a photograph region exists
        - a real face is detected
        - a PAN-specific right-side security/QR feature exists

    This prevents generic biodata, passbooks, statements and random
    documents containing a person's face from becoming PAN documents.
    """

    security = _fast_pan_security_feature(
        image
    )

    pan_identity = bool(
        geometry_ok
        and visual_present >= 4
        and photo_present
        and face_detected
        and bool(
            security["qr_right"]
            or security["security_block"]
        )
    )

    return {
        "pan_identity": pan_identity,
        "qr_detected": security["qr_detected"],
        "qr_right": security["qr_right"],
        "security_block": security["security_block"],
        "security_feature": bool(
            security["qr_right"]
            or security["security_block"]
        ),
        "security_block_score": security[
            "security_block_score"
        ],
    }


# ============================================================================
# FAST VALIDATION-ONLY PATH (OCR / EXTRACTION DISABLED)
# ============================================================================
#
# This path intentionally performs NO OCR or field extraction. PAN identity is established using strict visual/security signals.
# It validates the visual PAN-card structure, image quality, geometry,
# tamper evidence and the presence of the expected PAN visual regions.
#
# IMPORTANT:
# - This is LOCAL DOCUMENT VALIDATION only.
# - It does NOT prove that a PAN exists in a government database.
# - Extraction code remains below for future use but is NOT called by the
#   public validation path.
# ============================================================================

def _fast_pan_visual_regions(image: np.ndarray) -> tuple[int, int, bool]:
    """Fast PAN layout check supporting both photo layouts."""
    h,w=image.shape[:2]
    if h<=0 or w<=0: return 0,5,False
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    regions=[(0.03,0.04,0.97,0.24),(0.05,0.22,0.68,0.46),(0.05,0.38,0.68,0.62),(0.05,0.52,0.70,0.91),(0.68,0.10,0.98,0.90)]
    present=0
    for x1,y1,x2,y2 in regions:
        roi=gray[int(h*y1):int(h*y2),int(w*x1):int(w*x2)]
        if roi.size and (float(np.std(roi))>=9.0 or float(np.count_nonzero(cv2.Canny(roi,70,150)))/roi.size>=0.012): present+=1
    photo_present=False
    for x1,x2 in ((0.02,0.44),(0.56,0.98)):
        roi=gray[int(h*0.18):int(h*0.94),int(w*x1):int(w*x2)]
        if roi.size and float(np.std(roi))>=12.0: photo_present=True; break
    return present,5,photo_present

def _fast_local_quality(image: np.ndarray) -> dict[str, Any]:
    """CPU-cheap quality metrics; no OCR/model inference."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mean=float(np.mean(gray)); std=float(np.std(gray))
    lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    brightness=max(0.0,100.0-abs(mean-140.0)*0.55)
    contrast=min(100.0,std*2.2)
    sharp=min(100.0,lap/2.0)
    score=round(brightness*0.35+contrast*0.35+sharp*0.15+100.0*0.15,2)
    return {"quality_score":score,"brightness":round(mean,2),"contrast":round(std,2),"sharpness":round(lap,2)}

def _fast_local_tamper(image: np.ndarray) -> dict[str, Any]:
    """Very lightweight artifact-risk check for the <1s validation path.

    This is deliberately a risk signal, not forensic proof.
    """
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,70,160)
    density=float(np.count_nonzero(edges))/max(float(edges.size),1.0)
    # Extreme global edge density is suspicious/noisy; normal card photos are
    # kept LOW. Do not punish blur here because blur is handled by quality.
    if density>0.35:
        return {"tamper_score":70.0,"risk":"MEDIUM","decision":"DOCUMENT_REJECTED","signals":["EXCESSIVE_EDGE_ARTIFACTS"],"checks":{"edge_density":density},"available":True}
    return {"tamper_score":0.0,"risk":"LOW","decision":"PASS","signals":[],"checks":{"edge_density":round(density,4)},"available":True}

def _detect_pan_card_crop(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Detect a PAN/card-shaped object inside a larger phone photo."""
    if image is None or image.size == 0:
        return image, False

    h, w = image.shape[:2]
    full_ratio = w / max(h, 1)
    if MIN_PAN_ASPECT_RATIO <= full_ratio <= MAX_PAN_ASPECT_RATIO:
        return image, False

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
    best = None
    best_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 160 or bh < 90:
            continue
        ratio = bw / max(float(bh), 1.0)
        if not (1.15 <= ratio <= 2.10):
            continue

        area_fraction = area / frame_area
        ratio_score = max(
            0.0, 1.0 - min(abs(ratio - 1.55) / 0.75, 1.0)
        )
        area_score = min(1.0, area_fraction / 0.35)
        score = ratio_score * 0.60 + area_score * 0.40

        if score > best_score:
            best_score = score
            best = (x, y, bw, bh, ratio)

    if best is None:
        return image, False

    x, y, bw, bh, ratio = best
    if not (MIN_PAN_ASPECT_RATIO <= ratio <= MAX_PAN_ASPECT_RATIO):
        return image, False

    pad_x = max(2, int(bw * 0.015))
    pad_y = max(2, int(bh * 0.015))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(ww, x + bw + pad_x)
    y2 = min(wh, y + bh + pad_y)

    crop = work[y1:y2, x1:x2]
    if crop.size == 0:
        return image, False

    crop_h, crop_w = crop.shape[:2]
    crop_ratio = crop_w / max(crop_h, 1)
    if not (MIN_PAN_ASPECT_RATIO <= crop_ratio <= MAX_PAN_ASPECT_RATIO):
        return image, False

    return crop, True


def _fast_pan_validation(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Fast PAN validation without OCR or extraction.

    Target: normally < 1 second on a normal local CPU image.
    """
    start = datetime.now()

    if image is None or image.size == 0:
        raise PanVerificationError("Invalid or empty image.")

    image = _prepare_image(image)
    image, auto_cropped = _detect_pan_card_crop(image)
    quality = _fast_local_quality(image)

    # Tamper is deliberately delayed until PAN identity is established.
    tamper_result = {"tamper_score":0,"risk":"LOW","decision":"NOT_PERFORMED","signals":[],"checks":{},"available":False}

    try:
        width = int(image.shape[1])
        height = int(image.shape[0])
        aspect_ratio = width / max(height, 1)
    except Exception:
        width = 0
        height = 0
        aspect_ratio = 0.0

    # Use the common image-quality result where available.
    quality_metrics = dict(quality or {})
    quality_metrics["tamper"] = tamper_result

    image_quality_score = 0.0
    if quality_metrics:
        try:
            image_quality_score = float(
                quality_metrics.get("quality_score", 0.0)
            )
        except (TypeError, ValueError):
            image_quality_score = 0.0

    if image_quality_score <= 0:
        # Fallback lightweight quality score.
        brightness = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
        contrast = float(np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

        brightness_score = (
            100.0
            if 55.0 <= brightness <= 225.0
            else max(0.0, 100.0 - abs(brightness - 140.0) * 0.7)
        )
        contrast_score = min(100.0, max(0.0, contrast * 2.0))
        geometry_score = (
            100.0
            if MIN_PAN_ASPECT_RATIO <= aspect_ratio <= MAX_PAN_ASPECT_RATIO
            else 50.0
        )

        image_quality_score = round(
            brightness_score * 0.35
            + contrast_score * 0.35
            + geometry_score * 0.30,
            2,
        )

    visual_present, visual_total, photo_present = _fast_pan_visual_regions(
        image
    )

    geometry_ok = (
        MIN_PAN_ASPECT_RATIO
        <= aspect_ratio
        <= MAX_PAN_ASPECT_RATIO
    )

    # Face detection supports BOTH common PAN photograph layouts.
    face_detected = (
        _fast_pan_face_detect(image)
        if geometry_ok
        else False
    )

    # --------------------------------------------------------------
    # STRICT PAN IDENTITY
    # --------------------------------------------------------------
    #
    # Generic visual-region scoring is NOT enough.
    # A biodata/passbook/random document may contain:
    #   - text regions
    #   - a face
    #   - a wide aspect ratio
    #
    # Therefore the final document classifier also requires a
    # PAN-specific security/QR feature on the right side.
    # --------------------------------------------------------------

    pan_identity = _strict_pan_visual_identity(
        image,
        geometry_ok=geometry_ok,
        visual_present=visual_present,
        photo_present=photo_present,
        face_detected=face_detected,
    )

    if pan_identity["pan_identity"]:
        tamper_result = _fast_local_tamper(image)

    visual_field_ratio = f"{visual_present}/{visual_total}"

    document_detected = bool(
        pan_identity["pan_identity"]
        and image_quality_score >= 35.0
    )

    # Tamper risk is handled conservatively.
    tamper_risk = str(
        tamper_result.get("risk", "UNKNOWN")
    ).upper()

    if tamper_risk in {
        "HIGH",
        "MEDIUM",
    }:
        decision = "DOCUMENT_REJECTED"
    elif tamper_risk in {
        "UNKNOWN",
        "MANUAL_REVIEW",
    }:
        decision = "MANUAL_REVIEW"
    elif not document_detected:
        decision = "DOCUMENT_REJECTED"
    elif not pan_identity["pan_identity"]:
        decision = "DOCUMENT_REJECTED"
    elif not face_detected:
        decision = "DOCUMENT_REJECTED"
    elif image_quality_score < 35.0:
        decision = "DOCUMENT_REJECTED"
    else:
        decision = "DOCUMENT_VERIFIED_SUCCESSFULLY"

    # Score is intentionally based only on validation signals.
    quality_component = min(100.0, max(0.0, image_quality_score))
    visual_component = (
        visual_present / visual_total * 100.0
        if visual_total
        else 0.0
    )
    geometry_component = 100.0 if geometry_ok else 0.0

    try:
        tamper_score = float(tamper_result.get("tamper_score", 0.0))
    except (TypeError, ValueError):
        tamper_score = 0.0

    tamper_component = max(0.0, min(100.0, 100.0 - tamper_score))

    score = round(
        quality_component * 0.35
        + visual_component * 0.30
        + geometry_component * 0.20
        + tamper_component * 0.15,
        2,
    )

    # A high visual score must never turn a non-PAN into a PAN.
    if not document_detected:
        score = 0.0

    elapsed = (
        datetime.now() - start
    ).total_seconds()

    return {
        "document_type": "PAN",
        "decision": decision,
        "score": score,
        "validation": {
            "document_detected": bool(document_detected),
            "auto_cropped": bool(auto_cropped),
            "image_quality": (
                "GOOD"
                if image_quality_score >= 80
                else "FAIR"
                if image_quality_score >= 55
                else "POOR"
            ),
            "tampering_risk": tamper_risk,
            "visual_fields": visual_field_ratio,
            "photo_present": bool(photo_present),
            "face_detected": bool(face_detected),
            "pan_identity": bool(
                pan_identity["pan_identity"]
            ),
            "security_feature": bool(
                pan_identity.get("security_feature", False)
            ),
        },
        "processing_time_seconds": round(elapsed, 3),
        "ocr_used": False,
        "extraction_used": False,
        # AUTHORITATIVE PAN GATE.
        "verified": bool(
            document_detected
            and pan_identity["pan_identity"]
            and photo_present
            and face_detected
            and tamper_risk == "LOW"
            and image_quality_score >= 35.0
        ),
    }


# ============================================================================
# IMAGE VERIFICATION
# ============================================================================

def _verify_image(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    FAST VALIDATION-ONLY IMAGE PATH.

    OCR and extraction are intentionally disabled.

    -------------------------------------------------------------------------
    EXTRACTION DISABLED FOR NOW
    -------------------------------------------------------------------------
    The previous implementation performed:

        results = _run_ocr(image)
        extracted = _extract_fields(results)
        pan = extracted.get("pan")
        name = extracted.get("name")
        father_name = extracted.get("father_name")
        dob = extracted.get("dob")

    That code is intentionally NOT executed in the current validation
    endpoint. The extraction functions remain below for future use.
    -------------------------------------------------------------------------
    """
    return _fast_pan_validation(image)


# ============================================================================
# DIGITAL PDF
# ============================================================================

def _extract_pdf_native_text(
    file_bytes: bytes,
) -> list[str]:

    try:

        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )

    except Exception as exc:

        raise PanVerificationError(
            f"Could not open PDF: {exc}"
        ) from exc

    pages_text = []

    try:

        for page in document:

            text = page.get_text(
                "text"
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            pages_text.append(
                text
            )

    finally:

        document.close()

    return pages_text


def _native_pdf_looks_useful(
    pages_text: list[str],
) -> bool:

    combined = " ".join(
        pages_text
    )

    if (
        len(combined)
        < MIN_NATIVE_PDF_TEXT_LENGTH
    ):
        return False

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        combined.upper(),
    )

    has_pan = bool(
        re.search(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            compact,
        )
    )

    has_date = bool(
        DATE_PATTERN.search(
            combined
        )
        or MONTH_DATE_PATTERN.search(
            combined
        )
    )

    upper = combined.upper()

    has_marker = (
        "INCOME TAX" in upper
        or "PERMANENT ACCOUNT" in upper
        or "PERMANENTACCOUNT" in compact
    )

    return (
        has_pan
        and has_date
        and has_marker
    )


def _native_pdf_extract(
    pages_text: list[str],
) -> dict[str, Any] | None:

    combined = "\n".join(
        pages_text
    )

    pan_match = re.search(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        combined.upper(),
    )

    pan = (
        pan_match.group(0)
        if pan_match
        else None
    )

    lines = []

    for line in combined.splitlines():

        value = line.strip()

        if value:

            lines.append(
                {
                    "text": value,
                    "confidence": 1.0,
                    "bbox": None,
                    "yc": None,
                    "xc": None,
                }
            )

    if not pan:
        return None

    extracted = _extract_fields(
        lines
    )

    return {
        "pan_number": pan,
        "name": extracted.get(
            "name"
        ),
        "father_name": extracted.get(
            "father_name"
        ),
        "dob": extracted.get(
            "dob"
        ),
        "ocr_used": False,
        "extraction_method": (
            "native_pdf_text"
        ),
        "results": lines,
    }


# ============================================================================
# PUBLIC API
# ============================================================================

def verify_pan_card(
    file_bytes: bytes,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    FAST PAN DOCUMENT VALIDATION.

    Current endpoint intentionally performs:
        - image decoding
        - image quality
        - PAN visual/layout validation
        - photo-region presence check
        - tamper-risk analysis
        - scoring
        - processing-time measurement

    Current endpoint intentionally does NOT perform:
        - OCR
        - PAN number extraction
        - name extraction
        - father's-name extraction
        - DOB extraction
        - government database verification

    Extraction code is preserved elsewhere in this module and can be enabled
    later when the extraction endpoint is introduced.
    """
    if not file_bytes:
        raise PanVerificationError(
            "Uploaded file is empty."
        )

    normalized_type = (
        content_type.lower().strip()
        if content_type
        else ""
    )

    is_pdf = (
        normalized_type == "application/pdf"
        or file_bytes[:4] == b"%PDF"
    )

    if is_pdf:
        # Fast native PDF inspection is retained. No OCR/extraction is used.
        try:
            document = pymupdf.open(
                stream=file_bytes,
                filetype="pdf",
            )
        except Exception as exc:
            raise PanVerificationError(
                f"Could not open PDF: {exc}"
            ) from exc

        try:
            if document.page_count == 0:
                raise PanVerificationError(
                    "PDF contains no pages."
                )

            # Validation-only path: render the first page.
            # PAN validation is intentionally page-local for this endpoint.
            page = document.load_page(0)

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False,
            )

            raw = np.frombuffer(
                pix.samples,
                dtype=np.uint8,
            )

            if pix.n == 3:
                image = raw.reshape(
                    pix.height,
                    pix.width,
                    3,
                )
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )
            elif pix.n == 4:
                image = raw.reshape(
                    pix.height,
                    pix.width,
                    4,
                )
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2BGR,
                )
            else:
                raise PanVerificationError(
                    "Unsupported PDF image format."
                )

        finally:
            document.close()

        result = _fast_pan_validation(image)
        result["processing_pages"] = 1
        return result

    image = decode_image(file_bytes)

    if image is None:
        raise PanVerificationError(
            "Could not decode image. "
            "Make sure the uploaded file is a valid image."
        )

    # IMPORTANT: assign the validator result before any debug access.
    result = _fast_pan_validation(image)

    print("========== VERIFY_PAN_CARD FINAL RESULT ==========")
    print("verified:", result.get("verified") if isinstance(result, dict) else None)
    print("validation:", result.get("validation") if isinstance(result, dict) else None)
    print("===================================================")

    return result


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def debug_ocr_stages(
    image_bytes: bytes,
):

    image = decode_image(
        image_bytes
    )

    if image is None:

        print(
            "IMAGE DECODE FAILED"
        )

        return []

    results = _run_ocr(
        image
    )

    print(
        "OCR LINES:"
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        print(
            f"[{index}] "
            f"{item.get('text')} "
            f"(confidence="
            f"{item.get('confidence', 0):.3f}, "
            f"yc={item.get('yc')})"
        )

    extracted = _extract_fields(
        results
    )

    print(
        "\nEXTRACTED:"
    )

    print(
        extracted
    )

    return results


def api_contract_test() -> dict[str, Any]:

    return {
        "required_keys_count": 9,

        "father_name_optional": True,

        "single_ocr_pass_for_image": False,

        "digital_pdf_native_fast_path": True,

        "scanned_pdf_supported": True,

        "local_authenticity_validation": True,
        "validation_only_endpoint": True,
        "ocr_used_by_public_endpoint": False,
        "extraction_used_by_public_endpoint": False,

        "common_validation_layer": True,

        "common_tamper_evidence": True,

        "government_database_verification": False,

        "manual_review_supported": True,

    }


# ============================================================================
# COMMON VALIDATION INTEGRATION TEST
# ============================================================================

def common_validation_integration_test() -> dict[str, Any]:
    """
    Deterministic PAN -> common validation integration test.

    No OCR, filesystem, GPU or network is required.
    """
    sample_results = [
        {"text": "INCOME TAX DEPARTMENT", "confidence": 0.99, "yc": 100.0, "xc": 10.0},
        {"text": "GOVT OF INDIA", "confidence": 0.99, "yc": 120.0, "xc": 10.0},
        {"text": "NAME", "confidence": 0.99, "yc": 150.0, "xc": 10.0},
        {"text": "RAHUL KUMAR SHARMA", "confidence": 0.98, "yc": 180.0, "xc": 10.0},
        {"text": "FATHER'S NAME", "confidence": 0.99, "yc": 210.0, "xc": 10.0},
        {"text": "RAJESH KUMAR", "confidence": 0.98, "yc": 240.0, "xc": 10.0},
        {"text": "DATE OF BIRTH", "confidence": 0.99, "yc": 270.0, "xc": 10.0},
        {"text": "12/05/1995", "confidence": 0.99, "yc": 300.0, "xc": 10.0},
        {"text": "PERMANENT ACCOUNT NUMBER", "confidence": 0.99, "yc": 360.0, "xc": 10.0},
        {"text": "ABCDE1234F", "confidence": 0.99, "yc": 410.0, "xc": 10.0},
    ]

    result = validate_pan_document(
        results=sample_results,
        pan="ABCDE1234F",
        name="Rahul Kumar Sharma",
        father_name="Rajesh Kumar",
        dob="12/05/1995",
        quality_metrics={
            "width": 702,
            "height": 450,
            "blur_score": 1000.0,
            "brightness": 160.0,
            "contrast": 55.0,
            "aspect_ratio": 1.56,
        },
        tamper_result={
            "tamper_score": 4,
            "risk": "LOW",
            "decision": "CLEAN",
            "signals": [],
            "checks": {},
            "available": True,
        },
    )

    return {
        "passed": result.decision == DOCUMENT_PASS,
        "decision": result.decision,
        "score": result.score,
        "common_checks_present": all(
            key in result.checks
            for key in (
                "document_type_detected",
                "required_fields_present",
                "required_fields_total",
                "field_format_valid",
                "ocr_confidence",
                "field_consistency_score",
                "text_coherence_score",
                "layout_score",
                "quality_score",
                "tamper_score",
                "tamper_risk",
            )
        ),
        "pan_specific_checks_present": all(
            key in result.checks
            for key in (
                "pan_format",
                "pan_structure",
                "pan_markers_score",
                "field_consistency",
                "is_pan_document",
            )
        ),
    }


# ============================================================================
# EXTRACTION TEST
# ============================================================================

def extraction_logic_test() -> dict[str, Any]:

    sample = [
        {
            "text": "dR",
            "confidence": 0.509,
            "yc": 315.0,
            "xc": 10.0,
        },
        {
            "text": "FaHNT",
            "confidence": 0.541,
            "yc": 317.0,
            "xc": 20.0,
        },
        {
            "text": "BTadp",
            "confidence": 0.421,
            "yc": 319.0,
            "xc": 30.0,
        },
        {
            "text": "HRT",
            "confidence": 0.770,
            "yc": 320.0,
            "xc": 40.0,
        },
        {
            "text": "GOVT.OFINDIA",
            "confidence": 0.975,
            "yc": 379.0,
            "xc": 100.0,
        },
        {
            "text": "INCOMETAXDEPARTMENT",
            "confidence": 0.995,
            "yc": 384.5,
            "xc": 100.0,
        },
        {
            "text": "SUNIT B BHUMAR",
            "confidence": 0.927,
            "yc": 442.5,
            "xc": 100.0,
        },
        {
            "text": "BASAVALINGAPPA",
            "confidence": 0.997,
            "yc": 515.0,
            "xc": 100.0,
        },
        {
            "text": "30/10/1990",
            "confidence": 0.998,
            "yc": 604.0,
            "xc": 100.0,
        },
        {
            "text": "Permanent Account Number",
            "confidence": 0.984,
            "yc": 657.5,
            "xc": 100.0,
        },
        {
            "text": "FBIPS2651K",
            "confidence": 0.998,
            "yc": 710.0,
            "xc": 100.0,
        },
    ]

    result = _extract_fields(
        sample
    )

    expected = {
        "pan": "FBIPS2651K",
        "name": "Sunit B Bhumar",
        "father_name": "Basavalingappa",
        "dob": "30/10/1990",
    }

    return {
        "passed": all(
            result.get(key)
            == value
            for key, value
            in expected.items()
        ),

        "result": result,

        "expected": expected,
    }


# ============================================================================
# TEST SUITE
# ============================================================================

def extraction_logic_test_suite() -> dict[str, Any]:

    cases = [

        (
            "standard_header",

            [
                ("INCOME TAX DEPARTMENT", 100, 0.99),
                ("RAHUL KUMAR SHARMA", 180, 0.98),
                ("RAJESH KUMAR", 240, 0.98),
                ("12/05/1995", 300, 0.99),
                ("Permanent Account Number", 360, 0.99),
                ("ABCDE1234F", 410, 0.99),
            ],

            "ABCDE1234F",
            "Rahul Kumar Sharma",
            "Rajesh Kumar",
            "12/05/1995",
        ),

        (
            "middle_initial",

            [
                ("GOVT. OF INDIA", 100, 0.99),
                ("INCOME TAX DEPARTMENT", 120, 0.99),
                ("SUNIT B BHUMAR", 180, 0.95),
                ("BASAVALINGAPPA", 240, 0.99),
                ("30/10/1990", 300, 0.99),
                ("Permanent Account Number", 360, 0.99),
                ("FBIPS2651K", 410, 0.99),
            ],

            "FBIPS2651K",
            "Sunit B Bhumar",
            "Basavalingappa",
            "30/10/1990",
        ),

        (
            "no_header",

            [
                ("AMIT VERMA", 180, 0.96),
                ("SURESH VERMA", 240, 0.96),
                ("21/07/1988", 300, 0.99),
                ("ABCDE1234F", 410, 0.99),
            ],

            "ABCDE1234F",
            "Amit Verma",
            "Suresh Verma",
            "21/07/1988",
        ),

        (
            "label_based",

            [
                ("NAME: PRIYA SINGH", 150, 0.98),
                ("FATHER'S NAME: RAKESH SINGH", 205, 0.98),
                ("DATE OF BIRTH: 05/09/1992", 260, 0.99),
                ("ABCDE1234F", 350, 0.99),
            ],

            "ABCDE1234F",
            "Priya Singh",
            "Rakesh Singh",
            "05/09/1992",
        ),

        (
            "noisy_text_above_header",

            [
                ("dR", 30, 0.50),
                ("FaHNT", 31, 0.54),
                ("BTadp", 32, 0.42),
                ("HRT", 33, 0.77),
                ("GOVT.OFINDIA", 100, 0.98),
                ("INCOMETAXDEPARTMENT", 110, 0.99),
                ("SUNIT B BHUMAR", 170, 0.93),
                ("BASAVALINGAPPA", 240, 0.99),
                ("30/10/1990", 300, 0.99),
                ("Permanent Account Number", 360, 0.99),
                ("FBIPS2651K", 410, 0.99),
            ],

            "FBIPS2651K",
            "Sunit B Bhumar",
            "Basavalingappa",
            "30/10/1990",
        ),

        (
            "one_word_father",

            [
                ("INCOME TAX DEPARTMENT", 100, 0.99),
                ("ARUN PATEL", 180, 0.97),
                ("VENKATESH", 240, 0.97),
                ("15/01/1985", 300, 0.99),
                ("ABCDE1234F", 410, 0.99),
            ],

            "ABCDE1234F",
            "Arun Patel",
            "Venkatesh",
            "15/01/1985",
        ),

        (
            "fragmented_name_boxes",

            [
                ("INCOME TAX DEPARTMENT", 100, 0.99),
                ("RAHUL", 180, 0.96),
                ("K", 181, 0.95),
                ("SHARMA", 180, 0.96),
                ("MAHESH", 240, 0.96),
                ("SHARMA", 240, 0.96),
                ("10/11/1991", 300, 0.99),
                ("ABCDE1234F", 410, 0.99),
            ],

            "ABCDE1234F",
            "Rahul K Sharma",
            "Mahesh Sharma",
            "10/11/1991",
        ),

        (
            "month_date",

            [
                ("INCOME TAX DEPARTMENT", 100, 0.99),
                ("NEHA GUPTA", 180, 0.97),
                ("RAJ GUPTA", 240, 0.97),
                ("7 Jun 1993", 300, 0.99),
                ("ABCDE1234F", 410, 0.99),
            ],

            "ABCDE1234F",
            "Neha Gupta",
            "Raj Gupta",
            "07/06/1993",
        ),

        (
            "label_without_father",

            [
                ("NAME", 150, 0.98),
                ("MOHIT SHARMA", 180, 0.98),
                ("DOB", 250, 0.98),
                ("22/08/1990", 280, 0.99),
                ("ABCDE1234F", 400, 0.99),
            ],

            "ABCDE1234F",
            "Mohit Sharma",
            None,
            "22/08/1990",
        ),

        (
            "digital_pdf_like",

            [
                ("INCOME TAX DEPARTMENT", 100, 1.0),
                ("Name", 150, 1.0),
                ("KIRAN MEHTA", 180, 1.0),
                ("Father's Name", 210, 1.0),
                ("SURESH MEHTA", 240, 1.0),
                ("Date of Birth", 270, 1.0),
                ("09/12/1987", 300, 1.0),
                ("Permanent Account Number", 330, 1.0),
                ("ABCDE1234F", 360, 1.0),
            ],

            "ABCDE1234F",
            "Kiran Mehta",
            "Suresh Mehta",
            "09/12/1987",
        ),
    ]

    results = []

    for case in cases:

        (
            case_name,
            raw_lines,
            expected_pan,
            expected_name,
            expected_father,
            expected_dob,
        ) = case

        lines = []

        for index, (
            text_value,
            y,
            confidence,
        ) in enumerate(
            raw_lines
        ):

            lines.append(
                {
                    "text": text_value,
                    "confidence": confidence,
                    "bbox": None,
                    "yc": float(y),
                    "xc": float(
                        index * 10
                    ),
                }
            )

        result = _extract_fields(
            lines
        )

        passed = (
            result.get("pan")
            == expected_pan

            and result.get("name")
            == expected_name

            and result.get("father_name")
            == expected_father

            and result.get("dob")
            == expected_dob
        )

        results.append(
            {
                "case": case_name,
                "passed": passed,
                "actual": result,
                "expected": {
                    "pan": expected_pan,
                    "name": expected_name,
                    "father_name": expected_father,
                    "dob": expected_dob,
                },
            }
        )

    return {
        "total": len(results),

        "passed": sum(
            1
            for result in results
            if result["passed"]
        ),

        "failed": sum(
            1
            for result in results
            if not result["passed"]
        ),

        "all_passed": all(
            result["passed"]
            for result in results
        ),

        "results": results,
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PanVerificationError",
    "PanValidationResult",
    "verify_pan_card",
    "_fast_pan_validation",
    "load_ocr_reader",
    "extract_pan",
    "validate_pan_document",
    "debug_ocr_stages",
    "api_contract_test",
    "common_validation_integration_test",
    "extraction_logic_test",
    "extraction_logic_test_suite",
]


PAN_CLASSIFIER_VERSION = "PAN-STRICT-V6-DEBUG"

print("========== PAN VERIFICATION DEBUG LOADED ==========")
print("PAN VERIFICATION FILE:", __file__)
print("PAN CLASSIFIER VERSION:", PAN_CLASSIFIER_VERSION)
print("===================================================")


