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


def _fast_pan_face_detect(
    image: np.ndarray,
) -> bool:
    """
    Fast PAN photo-face check.

    No OCR and no extraction.

    The previous implementation cropped too tightly from x=70% onward.
    That cut off the left side of the photograph on camera-captured PANs.
    This version checks the two plausible PAN photo zones at low resolution:
        - right-side photo zone
        - left-side photo zone

    The input to Haar is capped at 320 px wide, so this remains CPU-cheap.
    """

    if image is None or image.size == 0:
        return False

    cascade = _get_pan_face_cascade()

    if cascade is None:
        return False

    height, width = image.shape[:2]

    # PAN layouts commonly place the portrait on either side depending
    # on the card generation/layout. Check both without scanning the
    # entire photograph at full resolution.
    zones = (
        (0.52, 0.22, 0.99, 0.98),  # right-side portrait
        (0.01, 0.18, 0.48, 0.82),  # left-side portrait
    )

    for x1f, y1f, x2f, y2f in zones:

        x1 = max(0, int(width * x1f))
        y1 = max(0, int(height * y1f))
        x2 = min(width, int(width * x2f))
        y2 = min(height, int(height * y2f))

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        # Keep detector input small.
        if gray.shape[1] > 320:
            scale = 320.0 / float(gray.shape[1])
            gray = cv2.resize(
                gray,
                (
                    320,
                    max(
                        1,
                        int(gray.shape[0] * scale),
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.equalizeHist(gray)

        try:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except Exception:
            continue

        if len(faces) == 0:
            continue

        # Reject extremely tiny detections.
        roi_area = float(
            gray.shape[0] * gray.shape[1]
        )

        for _, _, fw, fh in faces:
            face_area = float(fw * fh)

            if (
                face_area / max(roi_area, 1.0)
                >= 0.010
            ):
                return True

    return False


# ============================================================================
# FAST VALIDATION-ONLY PATH (OCR / EXTRACTION DISABLED)
# ============================================================================
#
# This path intentionally performs NO OCR and NO field extraction.
# It validates the visual PAN-card structure, image quality, geometry,
# tamper evidence and the presence of the expected PAN visual regions.
#
# IMPORTANT:
# - This is LOCAL DOCUMENT VALIDATION only.
# - It does NOT prove that a PAN exists in a government database.
# - Extraction code remains below for future use but is NOT called by the
#   public validation path.
# ============================================================================

def _fast_pan_visual_regions(
    image: np.ndarray,
) -> tuple[int, int, bool]:
    """
    Detect broad PAN-card visual regions using inexpensive OpenCV operations.

    Returns:
        present_count, total_expected, photo_present
    """
    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return 0, 5, False

    # Normalize only for analysis; do not run OCR.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # PAN cards normally have a wide rectangular layout.
    ratio = width / max(height, 1)

    # Broad geometry gate. Photographs/scans may be cropped, so keep this
    # deliberately tolerant.
    geometry_ok = 1.25 <= ratio <= 2.05

    # Edge map is inexpensive and gives a useful signal that the image has
    # structured card content rather than being an empty/flat image.
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # Horizontal/vertical structure provides a cheap layout signal.
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    structure_score = (
        min(1.0, float(np.mean(np.abs(horizontal))) / 35.0)
        + min(1.0, float(np.mean(np.abs(vertical))) / 35.0)
    ) / 2.0

    # Expected broad regions:
    # 1. government/header area
    # 2. holder-name area
    # 3. father's-name area
    # 4. DOB/PAN information area
    # 5. photo/signature side
    #
    # We use ROI variance/edge activity instead of OCR text.
    regions = [
        (0.04, 0.05, 0.96, 0.24),
        (0.10, 0.25, 0.72, 0.48),
        (0.10, 0.40, 0.72, 0.62),
        (0.10, 0.55, 0.90, 0.92),
        (0.72, 0.18, 0.98, 0.78),
    ]

    present = 0

    for x1, y1, x2, y2 in regions:
        xa = max(0, int(width * x1))
        ya = max(0, int(height * y1))
        xb = min(width, int(width * x2))
        yb = min(height, int(height * y2))

        roi = gray[ya:yb, xa:xb]

        if roi.size == 0:
            continue

        roi_std = float(np.std(roi))
        roi_edges = cv2.Canny(roi, 50, 150)
        roi_edge_density = (
            float(np.count_nonzero(roi_edges)) / float(roi_edges.size)
            if roi_edges.size
            else 0.0
        )

        if roi_std >= 10.0 or roi_edge_density >= 0.015:
            present += 1

    # Photo presence is a visual signal only. Detect a compact textured
    # rectangular region on the right side; no face model is loaded here,
    # keeping the endpoint fast.
    photo_x1 = int(width * 0.68)
    photo_y1 = int(height * 0.18)
    photo_x2 = int(width * 0.98)
    photo_y2 = int(height * 0.78)

    photo_roi = gray[photo_y1:photo_y2, photo_x1:photo_x2]

    photo_present = False
    if photo_roi.size:
        photo_std = float(np.std(photo_roi))
        photo_edges = cv2.Canny(photo_roi, 50, 150)
        photo_edge_density = (
            float(np.count_nonzero(photo_edges)) / float(photo_edges.size)
        )

        # Avoid requiring a face detector. The purpose here is to detect
        # whether the expected photo region contains meaningful visual data.
        photo_present = (
            photo_std >= 12.0
            and photo_edge_density >= 0.01
        )

    # If the image has strong card geometry/structure, tolerate one weak ROI.
    if geometry_ok and structure_score >= 0.12 and edge_density >= 0.01:
        present = max(present, 4)

    return present, 5, photo_present


def _fast_crop_pan_card(
    image: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """
    Locate a PAN-shaped card inside a larger camera photograph.

    One bounded contour pass. No OCR.
    """
    if image is None or image.size == 0:
        return image, False

    h, w = image.shape[:2]
    full_ratio = w / max(float(h), 1.0)

    if MIN_PAN_ASPECT_RATIO <= full_ratio <= MAX_PAN_ASPECT_RATIO:
        return image, False

    scale = min(1.0, 900.0 / max(float(w), 1.0))
    work = image
    if scale < 1.0:
        work = cv2.resize(
            image,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        140,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    wh = work.shape[0]
    ww = work.shape[1]
    image_area = float(wh * ww)
    best = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True,
        )
        if not 4 <= len(approx) <= 8:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < ww * 0.25 or ch < wh * 0.20:
            continue

        ratio = cw / max(float(ch), 1.0)
        if not (
            MIN_PAN_ASPECT_RATIO * 0.82
            <= ratio
            <= MAX_PAN_ASPECT_RATIO * 1.18
        ):
            continue

        fill = area / max(float(cw * ch), 1.0)
        if fill < 0.45:
            continue

        ratio_score = max(
            0.0,
            1.0 - min(abs(ratio - 1.58) / 1.58, 1.0),
        )
        area_score = min(
            1.0,
            area / max(image_area * 0.60, 1.0),
        )

        score = (
            ratio_score * 0.55
            + fill * 0.25
            + area_score * 0.20
        )

        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None or best_score < 0.45:
        return image, False

    x, y, cw, ch = best
    inv = 1.0 / max(scale, 1e-6)

    x1 = max(0, int(x * inv))
    y1 = max(0, int(y * inv))
    x2 = min(w, int((x + cw) * inv))
    y2 = min(h, int((y + ch) * inv))

    mx = int((x2 - x1) * 0.025)
    my = int((y2 - y1) * 0.025)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image, False

    crop_ratio = crop.shape[1] / max(float(crop.shape[0]), 1.0)

    if not (
        MIN_PAN_ASPECT_RATIO * 0.85
        <= crop_ratio
        <= MAX_PAN_ASPECT_RATIO * 1.15
    ):
        return image, False

    return crop, True


def _fast_pan_layout_identity(
    image: np.ndarray,
    *,
    visual_present: int,
    photo_present: bool,
    security_feature: bool,
) -> bool:
    """
    Final cheap PAN visual-layout gate.

    The detailed edge-row heuristic used here previously was redundant:
    visual_regions + photo + face + security are already evaluated by the
    authoritative validator.  Keeping another texture heuristic created
    false negatives on photographed/scanned PAN cards.

    No OCR and no extraction.
    """
    return bool(
        image is not None
        and image.size > 0
        and visual_present >= 4
        and photo_present
        and security_feature
    )



# ============================================================================
# FAST LOCAL QUALITY / SECURITY / TAMPER HELPERS
# ============================================================================
# These helpers are intentionally lightweight. The public verification path
# below is validation-only: it does not run OCR or field extraction.
# ============================================================================

def _fast_local_quality(
    image: np.ndarray,
) -> dict[str, Any]:
    """Calculate a bounded, CPU-cheap image-quality score."""
    if image is None or image.size == 0:
        return {
            "image_quality": "BAD",
            "score": 0.0,
            "blur_score": 0.0,
            "contrast": 0.0,
            "brightness": 0.0,
        }

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return {
            "image_quality": "BAD",
            "score": 0.0,
            "blur_score": 0.0,
            "contrast": 0.0,
            "brightness": 0.0,
        }

    # Bound CPU work for very large phone photos.
    work = image
    if w > 900:
        scale = 900.0 / float(w)
        work = cv2.resize(
            image,
            (
                900,
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(
        work,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # Brightness: strongest around a normal mid-range exposure.
    brightness_component = max(
        0.0,
        100.0 - abs(brightness - 140.0) * 0.55,
    )

    # Contrast and sharpness are capped so one metric cannot dominate.
    contrast_component = min(
        100.0,
        contrast * 2.2,
    )

    sharpness_component = min(
        100.0,
        blur_score / 2.0,
    )

    score = round(
        brightness_component * 0.35
        + contrast_component * 0.35
        + sharpness_component * 0.15
        + 15.0,
        2,
    )

    score = max(
        0.0,
        min(100.0, score),
    )

    if score >= 70.0:
        quality = "GOOD"
    elif score >= 45.0:
        quality = "FAIR"
    else:
        quality = "POOR"

    return {
        "image_quality": quality,
        "score": score,
        "blur_score": round(blur_score, 2),
        "contrast": round(contrast, 2),
        "brightness": round(brightness, 2),
    }


def _fast_pan_security_feature(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Detect PAN-specific right-side security evidence without OCR.

    Signals:
      1. QR detection at multiple scales.
      2. Compact textured rectangular structure in the right-side PAN area.

    This is a local visual signal, not proof of government authenticity.
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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------------
    # 1. MULTI-SCALE QR DETECTION
    # --------------------------------------------------------------
    qr_detected = False
    qr_right = False

    try:
        detector = cv2.QRCodeDetector()

        for scale in (1.0, 1.5, 2.0):
            if scale == 1.0:
                test = gray
            else:
                test = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            try:
                decoded, points, _ = detector.detectAndDecode(test)

                if points is not None:
                    pts = np.asarray(points).reshape(-1, 2)
                    if pts.size >= 8:
                        cx = float(np.mean(pts[:, 0]) / scale)
                        cy = float(np.mean(pts[:, 1]) / scale)

                        if (
                            cx >= w * 0.55
                            and cy <= h * 0.85
                        ):
                            qr_detected = True
                            qr_right = True
                            break
            except Exception:
                pass

            try:
                detected = detector.detect(test)

                if isinstance(detected, tuple):
                    ok, points = detected

                    if ok and points is not None:
                        pts = np.asarray(points).reshape(-1, 2)
                        if pts.size >= 8:
                            cx = float(np.mean(pts[:, 0]) / scale)
                            cy = float(np.mean(pts[:, 1]) / scale)

                            if (
                                cx >= w * 0.55
                                and cy <= h * 0.85
                            ):
                                qr_detected = True
                                qr_right = True
                                break
            except Exception:
                pass

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
        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.035 * perimeter,
            True,
        )

        if not 4 <= len(approx) <= 8:
            continue

        bx, by, bw, bh = cv2.boundingRect(
            contour,
        )

        if bw <= 0 or bh <= 0:
            continue

        width_fraction = bw / max(
            float(roi_w),
            1.0,
        )
        height_fraction = bh / max(
            float(roi_h),
            1.0,
        )
        aspect = bw / max(
            float(bh),
            1.0,
        )
        area_fraction = (
            bw * bh
        ) / max(
            float(roi_w * roi_h),
            1.0,
        )

        if width_fraction < 0.08 or width_fraction > 0.70:
            continue
        if height_fraction < 0.08 or height_fraction > 0.70:
            continue
        if area_fraction < 0.012 or area_fraction > 0.35:
            continue
        if not 0.45 <= aspect <= 2.20:
            continue

        block = roi[
            by:by + bh,
            bx:bx + bw,
        ]

        if block.size == 0:
            continue

        block_std = float(
            np.std(block)
        )

        block_edges = cv2.Canny(
            block,
            50,
            150,
        )

        edge_density = (
            float(np.count_nonzero(block_edges))
            / max(float(block.size), 1.0)
        )

        if block_std < 12.0 or edge_density < 0.025:
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
        "security_block_score": round(
            best_score,
            4,
        ),
    }


def _fast_local_tamper(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Very lightweight artifact-risk check.

    This is a fraud-risk signal only; it is not forensic proof of forgery.
    """
    if image is None or image.size == 0:
        return {
            "tamper_score": 100.0,
            "risk": "HIGH",
            "decision": "DOCUMENT_REJECTED",
            "signals": ["INVALID_IMAGE"],
            "checks": {},
            "available": True,
        }

    h, w = image.shape[:2]
    work = image

    if w > 800:
        scale = 800.0 / float(w)
        work = cv2.resize(
            image,
            (
                800,
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(
        work,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        70,
        160,
    )

    density = (
        float(np.count_nonzero(edges))
        / max(float(edges.size), 1.0)
    )

    if density > 0.35:
        return {
            "tamper_score": 70.0,
            "risk": "MEDIUM",
            "decision": "DOCUMENT_REJECTED",
            "signals": ["EXCESSIVE_EDGE_ARTIFACTS"],
            "checks": {
                "edge_density": round(density, 4),
            },
            "available": True,
        }

    return {
        "tamper_score": 0.0,
        "risk": "LOW",
        "decision": "PASS",
        "signals": [],
        "checks": {
            "edge_density": round(density, 4),
        },
        "available": True,
    }


def _fast_pan_validation(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    VERIFICATION-ONLY FAST PATH.

    No OCR and no field extraction.

    Authoritative local PAN-document gates:
      1. PAN geometry / card crop
      2. PAN visual-region structure
      3. photograph region
      4. face in photograph region
      5. PAN-specific security/QR evidence
      6. conservative image quality
      7. low tamper risk

    This does NOT verify PAN existence with a government database.
    """
    started = datetime.now()

    if image is None or image.size == 0:
        raise PanVerificationError("Invalid image.")

    image = _prepare_image(image)

    image, auto_cropped = _fast_crop_pan_card(image)

    # Bound all downstream CPU work.  The validator does not need the
    # original multi-megapixel resolution for geometry, face, QR, quality,
    # or tamper checks. Keep aspect ratio intact.
    max_width = 1200
    if image.shape[1] > max_width:
        scale = max_width / float(image.shape[1])
        image = cv2.resize(
            image,
            (
                max_width,
                max(1, int(image.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    height, width = image.shape[:2]
    ratio = width / max(float(height), 1.0)

    geometry_ok = (
        MIN_PAN_ASPECT_RATIO
        <= ratio
        <= MAX_PAN_ASPECT_RATIO
    )

    quality_result = _fast_local_quality(image)
    image_quality = str(
        quality_result.get("image_quality", "UNKNOWN")
    ).upper()
    image_quality_score = float(
        quality_result.get("score", 0.0)
    )

    visual_present, visual_total, photo_present = (
        _fast_pan_visual_regions(image)
    )

    face_detected = _fast_pan_face_detect(image)

    security = _fast_pan_security_feature(image)
    security_feature = bool(
        security.get("qr_right")
        or security.get("security_block")
    )

    tamper_result = _fast_local_tamper(image)
    tamper_risk = str(
        tamper_result.get("risk", "UNKNOWN")
    ).upper()

    pan_identity = bool(
        geometry_ok
        and visual_present >= 4
        and photo_present
        and face_detected
        and security_feature
        and _fast_pan_layout_identity(
            image,
            visual_present=visual_present,
            photo_present=photo_present,
            security_feature=security_feature,
        )
    )

    document_detected = bool(
        pan_identity
        and image_quality_score >= 35.0
    )

    if not geometry_ok:
        decision = "DOCUMENT_REJECTED"
    elif not pan_identity:
        decision = "DOCUMENT_REJECTED"
    elif not document_detected:
        decision = "DOCUMENT_REJECTED"
    elif not face_detected:
        decision = "DOCUMENT_REJECTED"
    elif image_quality_score < 35.0:
        decision = "DOCUMENT_REJECTED"
    elif tamper_risk in {"HIGH", "MEDIUM"}:
        decision = "DOCUMENT_REJECTED"
    elif tamper_risk in {"UNKNOWN", "MANUAL_REVIEW"}:
        decision = "MANUAL_REVIEW"
    else:
        decision = "DOCUMENT_VERIFIED_SUCCESSFULLY"

    quality_component = min(
        100.0,
        max(0.0, image_quality_score),
    )
    visual_component = (
        visual_present / visual_total * 100.0
        if visual_total
        else 0.0
    )
    geometry_component = 100.0 if geometry_ok else 0.0
    face_component = 100.0 if face_detected else 0.0
    security_component = 100.0 if security_feature else 0.0

    score = round(
        (
            quality_component * 0.20
            + visual_component * 0.20
            + geometry_component * 0.20
            + face_component * 0.15
            + security_component * 0.25
        ),
        2,
    )

    verified = bool(
        decision == "DOCUMENT_VERIFIED_SUCCESSFULLY"
        and document_detected
        and geometry_ok
        and pan_identity
        and photo_present
        and face_detected
        and security_feature
        and tamper_risk == "LOW"
    )

    elapsed = (
        datetime.now() - started
    ).total_seconds()

    return {
        "document_type": "PAN",
        "decision": (
            "DOCUMENT_VERIFIED_SUCCESSFULLY"
            if verified
            else decision
        ),
        "score": score if verified else 0.0,
        "validation": {
            "document_detected": document_detected,
            "auto_cropped": auto_cropped,
            "image_quality": image_quality,
            "tampering_risk": tamper_risk,
            "visual_fields": f"{visual_present}/{visual_total}",
            "photo_present": bool(photo_present),
            "face_detected": bool(face_detected),
            "pan_identity": bool(pan_identity),
            "security_feature": bool(security_feature),
            "qr_right": bool(security.get("qr_right", False)),
            "security_block": bool(
                security.get("security_block", False)
            ),
            "security_block_score": float(
                security.get("security_block_score", 0.0)
            ),
        },
        "processing_time_seconds": round(elapsed, 3),
        "ocr_used": False,
        "extraction_used": False,
        "verified": verified,
    }



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

    return _fast_pan_validation(image)


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