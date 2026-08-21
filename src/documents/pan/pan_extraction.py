"""
PAN EXTRACTION - NAME ONLY

This file is ONLY responsible for extracting the PAN holder name.

It does NOT:
- validate the PAN
- make PASS/REJECT decisions
- perform tamper detection
- extract father name
- extract DOB
- extract PAN number

Pipeline:
    Digital PDF -> native PDF text -> name scoring -> name
    Scanned PDF/JPG/PNG -> ONE PaddleOCR pass -> name scoring -> name

PaddleOCR is loaded lazily and reused.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_enable_pir_api", "0")

MAX_IMAGE_WIDTH = 1600
ROW_Y_TOLERANCE = 18
ROW_X_GAP = 650
NAME_LOOKAHEAD_ROWS = 4
MIN_NAME_SCORE = 0.52

_ocr = None
_ocr_lock = threading.Lock()
_cv2 = None
_np = None


# ============================================================
# LAZY DEPENDENCIES
# ============================================================

def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_ocr():
    """Create PaddleOCR once, only when scanned extraction is needed."""
    global _ocr

    if _ocr is not None:
        return _ocr

    with _ocr_lock:
        if _ocr is None:
            from paddleocr import PaddleOCR

            _ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )

    return _ocr


# ============================================================
# IMAGE / PDF
# ============================================================

def _decode_image(file_bytes: bytes):
    np = _get_numpy()
    cv2 = _get_cv2()

    image = cv2.imdecode(
        np.frombuffer(file_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("Unable to decode PAN image.")

    return image


def _prepare_image(image):
    cv2 = _get_cv2()

    height, width = image.shape[:2]

    if width <= MAX_IMAGE_WIDTH:
        return image

    scale = MAX_IMAGE_WIDTH / float(width)

    return cv2.resize(
        image,
        (
            MAX_IMAGE_WIDTH,
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _pdf_native_text(file_bytes: bytes) -> list[dict[str, Any]]:
    """
    Read selectable PDF text without OCR.

    The returned structure is intentionally compatible with OCR rows.
    """
    try:
        import fitz
    except ImportError:
        return []

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf",
    )

    try:
        if document.page_count == 0:
            return []

        page = document.load_page(0)
        words = page.get_text("words")

        rows = []

        for word in words:
            if len(word) < 5:
                continue

            x0, y0, x1, y1, text = word[:5]
            text = str(text).strip()

            if not text:
                continue

            rows.append(
                {
                    "text": text,
                    "confidence": 1.0,
                    "x1": float(x0),
                    "y1": float(y0),
                    "x2": float(x1),
                    "y2": float(y1),
                    "xc": (float(x0) + float(x1)) / 2,
                    "yc": (float(y0) + float(y1)) / 2,
                    "source": "PDF_TEXT",
                }
            )

        return rows

    finally:
        document.close()


def _pdf_has_text(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 3:
        return False

    joined = " ".join(row["text"] for row in rows)

    return sum(char.isalpha() for char in joined) >= 15


def _render_pdf(file_bytes: bytes):
    try:
        import fitz
    except ImportError as exc:
        raise ValueError(
            "PyMuPDF is required for scanned PDF extraction."
        ) from exc

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf",
    )

    try:
        if document.page_count == 0:
            raise ValueError("PDF contains no pages.")

        page = document.load_page(0)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(1.5, 1.5),
            alpha=False,
        )

        np = _get_numpy()
        cv2 = _get_cv2()

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
            pix.n,
        )

        if pix.n == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    finally:
        document.close()


# ============================================================
# ONE OCR PASS
# ============================================================

def _run_ocr(image) -> list[dict[str, Any]]:
    reader = _get_ocr()
    image = _prepare_image(image)

    result = reader.predict(image)
    rows = []

    for page in result:
        data = page.json
        if callable(data):
            data = data()

        if not isinstance(data, dict):
            continue

        data = data.get("res", data)

        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        boxes = (
            data.get("rec_boxes")
            or data.get("rec_polys")
            or []
        )

        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue

            confidence = (
                float(scores[i])
                if i < len(scores)
                else 0.0
            )

            bbox = boxes[i] if i < len(boxes) else None

            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()

            x1 = y1 = x2 = y2 = 0.0

            if bbox:
                if (
                    len(bbox) == 4
                    and isinstance(bbox[0], (int, float))
                ):
                    x1, y1, x2, y2 = map(float, bbox)
                else:
                    points = [
                        p for p in bbox
                        if len(p) >= 2
                    ]
                    if points:
                        x1 = min(float(p[0]) for p in points)
                        y1 = min(float(p[1]) for p in points)
                        x2 = max(float(p[0]) for p in points)
                        y2 = max(float(p[1]) for p in points)

            rows.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "xc": (x1 + x2) / 2,
                    "yc": (y1 + y2) / 2,
                    "source": "OCR",
                }
            )

    return rows


# ============================================================
# TEXT RULES
# ============================================================

def _alpha(text: str) -> str:
    return re.sub(
        r"[^A-Z]",
        "",
        str(text).upper(),
    )


def _compact(text: str) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def _clean(text: str) -> str:
    text = re.sub(
        r"[^A-Za-z .'\-]",
        " ",
        str(text),
    )
    return re.sub(r"\s+", " ", text).strip()


def _is_pan(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            _compact(text),
        )
    )


def _is_date(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:0?[1-9]|[12][0-9]|3[01])"
            r"[\/\-.]"
            r"(?:0?[1-9]|1[0-2])"
            r"[\/\-.]"
            r"(?:19|20)\d{2}\b",
            str(text),
        )
    )


def _is_name_label(text: str) -> bool:
    value = _alpha(text)

    return (
        "NAME" in value
        and "FATHER" not in value
    )


def _is_other_label(text: str) -> bool:
    value = _alpha(text)

    labels = (
        "FATHER",
        "DATE",
        "BIRTH",
        "DOB",
        "PERMANENT",
        "ACCOUNT",
        "NUMBER",
        "INCOME",
        "TAX",
        "DEPARTMENT",
        "GOVT",
        "INDIA",
        "SIGNATURE",
        "PHOTO",
        "CARD",
    )

    return any(label in value for label in labels)


def _looks_like_name(text: str) -> bool:
    value = _clean(text)

    if not 3 <= len(value) <= 80:
        return False

    if not re.fullmatch(
        r"[A-Za-z][A-Za-z .'\-]*",
        value,
    ):
        return False

    if _is_pan(value) or _is_date(value):
        return False

    if _is_other_label(value):
        return False

    blocked = {
        "NAME",
        "FATHER",
        "DATE",
        "BIRTH",
        "DOB",
        "PERMANENT",
        "ACCOUNT",
        "NUMBER",
        "INCOME",
        "TAX",
        "DEPARTMENT",
        "GOVT",
        "INDIA",
        "SIGNATURE",
        "PAN",
        "CARD",
    }

    words = [
        word.strip(".-'")
        for word in value.split()
        if word.strip(".-'")
    ]

    if not words:
        return False

    if any(word.upper() in blocked for word in words):
        return False

    if len(words) == 1:
        return len(words[0]) >= 5

    if any(len(word) < 2 for word in words):
        return False

    return True


def _format_name(text: str) -> str:
    return " ".join(
        word.strip(".-'").title()
        for word in _clean(text).split()
        if word.strip(".-'")
    )


# ============================================================
# ROW RECONSTRUCTION
# ============================================================

def _group_rows(
    boxes: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:

    boxes = [
        item
        for item in boxes
        if str(item.get("text", "")).strip()
    ]

    boxes.sort(
        key=lambda item: (
            float(item.get("yc", 0)),
            float(item.get("xc", 0)),
        )
    )

    rows: list[list[dict[str, Any]]] = []

    for item in boxes:
        placed = False

        for row in rows:
            avg_y = sum(
                float(token.get("yc", 0))
                for token in row
            ) / len(row)

            if abs(
                float(item.get("yc", 0)) - avg_y
            ) <= ROW_Y_TOLERANCE:

                last_x = max(
                    float(
                        token.get(
                            "x2",
                            token.get("xc", 0),
                        )
                    )
                    for token in row
                )

                current_x = float(
                    item.get(
                        "x1",
                        item.get("xc", 0),
                    )
                )

                if current_x - last_x <= ROW_X_GAP:
                    row.append(item)
                    placed = True
                    break

        if not placed:
            rows.append([item])

    for row in rows:
        row.sort(
            key=lambda item: float(
                item.get("xc", 0)
            )
        )

    return rows


# ============================================================
# NAME CANDIDATES
# ============================================================

def _candidate_score(
    name: str,
    confidence: float,
    label_relation: bool,
    distance: float,
    context: float,
) -> float:

    label_score = 1.0 if label_relation else 0.0

    geometry_score = max(
        0.0,
        1.0 - min(distance / 250.0, 1.0),
    )

    confidence_score = max(
        0.0,
        min(confidence, 1.0),
    )

    words = name.split()

    shape_score = (
        1.0
        if len(words) >= 2
        else 0.65
    )

    return min(
        1.0,
        (
            label_score * 0.42
            + geometry_score * 0.18
            + confidence_score * 0.18
            + shape_score * 0.14
            + context * 0.08
        ),
    )


def _generate_candidates(
    boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    rows = _group_rows(boxes)
    candidates = []

    for row_index, row in enumerate(rows):

        row_text = " ".join(
            str(item["text"])
            for item in row
        )

        # ----------------------------------------------------
        # SAME LINE
        # NAME: RISHABH SINGH
        # ----------------------------------------------------

        if _is_name_label(row_text):

            match = re.search(
                r"\bNAME\b\s*[:\-]?\s*(.+)$",
                row_text,
                flags=re.IGNORECASE,
            )

            if match:
                value = _clean(
                    match.group(1)
                )

                if _looks_like_name(value):

                    confidence = sum(
                        float(
                            item.get(
                                "confidence",
                                0,
                            )
                        )
                        for item in row
                    ) / len(row)

                    candidates.append(
                        {
                            "name": _format_name(value),
                            "confidence": confidence,
                            "label_relation": True,
                            "distance": 0.0,
                            "context": 1.0,
                            "method": "same_line_label",
                        }
                    )

        # ----------------------------------------------------
        # NEXT LINE
        # NAME
        # RISHABH SINGH
        # ----------------------------------------------------

        if _is_name_label(row_text):

            label_y = sum(
                float(item.get("yc", 0))
                for item in row
            ) / len(row)

            for next_index in range(
                row_index + 1,
                min(
                    len(rows),
                    row_index + 1 + NAME_LOOKAHEAD_ROWS,
                ),
            ):

                next_row = rows[next_index]

                value = _clean(
                    " ".join(
                        str(item["text"])
                        for item in next_row
                    )
                )

                if not _looks_like_name(value):
                    continue

                candidate_y = sum(
                    float(item.get("yc", 0))
                    for item in next_row
                ) / len(next_row)

                confidence = sum(
                    float(
                        item.get(
                            "confidence",
                            0,
                        )
                    )
                    for item in next_row
                ) / len(next_row)

                candidates.append(
                    {
                        "name": _format_name(value),
                        "confidence": confidence,
                        "label_relation": True,
                        "distance": abs(
                            candidate_y - label_y
                        ),
                        "context": max(
                            0.0,
                            1.0
                            - (
                                next_index
                                - row_index
                            ) / 5.0,
                        ),
                        "method": "next_line_label",
                    }
                )

    # --------------------------------------------------------
    # GENERIC FALLBACK
    #
    # Used only when OCR failed to identify the NAME label.
    # It is scored rather than taking the first alphabetic line.
    # --------------------------------------------------------

    for index, row in enumerate(rows):

        value = _clean(
            " ".join(
                str(item["text"])
                for item in row
            )
        )

        if not _looks_like_name(value):
            continue

        confidence = sum(
            float(
                item.get(
                    "confidence",
                    0,
                )
            )
            for item in row
        ) / len(row)

        context = (
            1.0
            if index <= max(2, len(rows) // 2)
            else 0.55
        )

        candidates.append(
            {
                "name": _format_name(value),
                "confidence": confidence,
                "label_relation": False,
                "distance": 180.0,
                "context": context,
                "method": "generic_row",
            }
        )

    return candidates


def _select_best(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:

    ranked = []

    for item in candidates:

        score = _candidate_score(
            name=item["name"],
            confidence=float(
                item.get(
                    "confidence",
                    0,
                )
            ),
            label_relation=bool(
                item.get(
                    "label_relation",
                    False,
                )
            ),
            distance=float(
                item.get(
                    "distance",
                    999,
                )
            ),
            context=float(
                item.get(
                    "context",
                    0,
                )
            ),
        )

        ranked.append(
            {
                **item,
                "score": round(score, 4),
            }
        )

    if not ranked:
        return None

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["confidence"],
        ),
        reverse=True,
    )

    if ranked[0]["score"] < MIN_NAME_SCORE:
        return None

    return ranked[0]


# ============================================================
# PUBLIC FUNCTION - NAME ONLY
# ============================================================

def extract_pan_name(
    file_bytes: bytes,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    Extract ONLY the PAN holder name.

    No validation.
    No authenticity decision.
    No other field extraction.
    """

    if not file_bytes:
        raise ValueError(
            "Uploaded PAN file is empty."
        )

    started = time.perf_counter()

    content_type = (
        content_type.lower().strip()
        if content_type
        else ""
    )

    # --------------------------------------------------------
    # DIGITAL PDF
    # --------------------------------------------------------

    if (
        content_type == "application/pdf"
        or file_bytes.startswith(b"%PDF")
    ):

        native_rows = _pdf_native_text(
            file_bytes
        )

        if _pdf_has_text(native_rows):

            best = _select_best(
                _generate_candidates(
                    native_rows
                )
            )

            if best:
                return {
                    "name": best["name"],
                    "confidence": best["score"],
                    "method": best["method"],
                    "ocr_used": False,
                    "processing_time_seconds": round(
                        time.perf_counter()
                        - started,
                        3,
                    ),
                }

            return {
                "name": None,
                "confidence": 0.0,
                "method": "PDF_TEXT_NAME_NOT_FOUND",
                "ocr_used": False,
                "processing_time_seconds": round(
                    time.perf_counter()
                    - started,
                    3,
                ),
            }

        # Scanned PDF.
        image = _render_pdf(
            file_bytes
        )

    else:
        # JPG/JPEG/PNG.
        image = _decode_image(
            file_bytes
        )

    # --------------------------------------------------------
    # SCANNED IMAGE / PDF -> ONE OCR PASS
    # --------------------------------------------------------

    rows = _run_ocr(
        image
    )

    best = _select_best(
        _generate_candidates(
            rows
        )
    )

    elapsed = round(
        time.perf_counter()
        - started,
        3,
    )

    if not best:
        return {
            "name": None,
            "confidence": 0.0,
            "method": "OCR_NAME_NOT_FOUND",
            "ocr_used": True,
            "processing_time_seconds": elapsed,
        }

    return {
        "name": best["name"],
        "confidence": best["score"],
        "ocr_confidence": round(
            float(
                best.get(
                    "confidence",
                    0.0,
                )
            ),
            4,
        ),
        "method": best["method"],
        "ocr_used": True,
        "processing_time_seconds": elapsed,
    }


# ============================================================
# LOCAL TEST
# ============================================================

def extract_pan_name_from_path(
    file_path: str | Path,
) -> dict[str, Any]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }.get(
        path.suffix.lower(),
        "",
    )

    return extract_pan_name(
        path.read_bytes(),
        content_type,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ONLY PAN holder name."
    )

    parser.add_argument(
        "file",
        help="JPG/JPEG/PNG/PDF PAN file",
    )

    args = parser.parse_args()

    print(
        extract_pan_name_from_path(
            args.file
        )
    )