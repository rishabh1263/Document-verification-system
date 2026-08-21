"""
src/paddle_ocr_engine.py

PaddleOCR wrapper for Indian Driving Licence extraction.

Environment tested for:
    PaddleOCR    3.7.x
    PaddlePaddle 3.2.x
    Device       CPU

Goals:
    - Load PaddleOCR only once
    - Optimize CPU inference
    - Disable unnecessary document-processing models
    - Limit detector image size
    - Batch text recognition
    - Return standard OCR format used by field_extractor.py

Output format:

[
    {
        "text": "DL No: MH03 20220045390",
        "confidence": 0.9744,
        "bbox": [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ],
        "ocr_engine": "paddleocr"
    }
]
"""

import re
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ============================================================
# SETTINGS
# ============================================================

# CPU threads used by Paddle.
#
# You can benchmark 4 / 6 / 8 / 10 later.
CPU_THREADS = 10


# Detection-side resolution limit.
#
# The preprocessor may give Paddle a fairly large image.
# Paddle's detector does not necessarily need the entire
# resolution for a DL.
TEXT_DET_LIMIT_SIDE_LEN = 1280


# Number of detected text regions sent through recognition
# together.
TEXT_RECOGNITION_BATCH_SIZE = 8


# ============================================================
# GLOBAL PADDLE OCR INSTANCE
# ============================================================

_reader = None


# ============================================================
# LOAD PADDLE OCR
# ============================================================

def get_reader():
    """
    Create PaddleOCR once and reuse it for future API requests.

    Model initialization should never happen for every uploaded
    document.
    """

    global _reader

    if _reader is not None:
        return _reader

    print()
    print("=" * 70)
    print("INITIALIZING PADDLE OCR")
    print("=" * 70)

    start = time.perf_counter()

    try:

        _reader = PaddleOCR(

            # ------------------------------------------------
            # DEVICE
            # ------------------------------------------------

            device="cpu",

            # ------------------------------------------------
            # DISABLE EXTRA DOCUMENT MODELS
            # ------------------------------------------------
            #
            # DL scans normally don't need these.
            #
            # This prevents Paddle from running additional
            # document-processing stages.
            # ------------------------------------------------

            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,

            # ------------------------------------------------
            # CPU ACCELERATION
            # ------------------------------------------------

            enable_mkldnn=False,

            cpu_threads=CPU_THREADS,

            # ------------------------------------------------
            # TEXT DETECTION
            # ------------------------------------------------
            #
            # Limit detector working resolution.
            #
            # This can significantly reduce detector cost for
            # large PDF-rendered images.
            # ------------------------------------------------

            text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,

            text_det_limit_type="max",

            # ------------------------------------------------
            # TEXT RECOGNITION
            # ------------------------------------------------

            text_recognition_batch_size=(
                TEXT_RECOGNITION_BATCH_SIZE
            ),
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to initialize PaddleOCR: "
            f"{exc}"
        ) from exc

    elapsed = round(
        time.perf_counter() - start,
        3,
    )

    print(
        f"[PaddleOCR] Model loaded in "
        f"{elapsed:.3f} sec"
    )

    print(
        f"[PaddleOCR] Device: CPU"
    )

    print(
        f"[PaddleOCR] MKL-DNN: disabled"
    )

    print(
        f"[PaddleOCR] CPU threads: "
        f"{CPU_THREADS}"
    )

    print(
        f"[PaddleOCR] Detection limit: "
        f"{TEXT_DET_LIMIT_SIDE_LEN}px"
    )

    print(
        f"[PaddleOCR] Recognition batch: "
        f"{TEXT_RECOGNITION_BATCH_SIZE}"
    )

    print("=" * 70)

    return _reader


# ============================================================
# SAFE FLOAT
# ============================================================

def _safe_float(
    value,
    default=0.0,
):
    """
    Convert Paddle/numpy numeric values safely to Python float.
    """

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def _clean_text(text):
    """
    Perform conservative OCR text cleanup.

    IMPORTANT:

    Do NOT perform spelling corrections here.

    For example:

        RISHABK -> RISHABH

    must not be hardcoded because OCR should not invent
    information.
    """

    if text is None:
        return ""

    text = str(
        text
    )

    # Remove repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# IMAGE PREPARATION
# ============================================================

def _prepare_image(image):
    """
    Convert incoming OpenCV image into PaddleOCR-safe BGR uint8.

    Supported input:

        grayscale:
            H x W

        grayscale:
            H x W x 1

        BGR:
            H x W x 3

        BGRA:
            H x W x 4

    Output:

        H x W x 3
        uint8
        contiguous
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if image is None:

        raise ValueError(
            "Cannot run PaddleOCR: image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):

        raise TypeError(
            "PaddleOCR image must be a numpy array."
        )

    if image.size == 0:

        raise ValueError(
            "Cannot run PaddleOCR on an empty image."
        )

    # ========================================================
    # CHANNEL CONVERSION
    # ========================================================

    if image.ndim == 2:

        # Grayscale -> BGR

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 1
    ):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        # BGRA -> BGR

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):

        # Already BGR.
        pass

    else:

        raise ValueError(
            "Unsupported image shape for PaddleOCR: "
            f"{image.shape}"
        )

    # ========================================================
    # UINT8
    # ========================================================

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        )

        image = image.astype(
            np.uint8
        )

    # ========================================================
    # CONTIGUOUS MEMORY
    # ========================================================

    image = np.ascontiguousarray(
        image
    )

    return image


# ============================================================
# BOUNDING BOX CONVERSION
# ============================================================

def _convert_bbox(
    polygon,
):
    """
    Convert PaddleOCR polygon into normal Python coordinates.

    Output:

        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ]
    """

    if polygon is None:
        return None

    try:

        array = np.asarray(
            polygon
        )

        if array.ndim != 2:
            return None

        if array.shape[1] < 2:
            return None

        points = []

        for point in array:

            points.append(
                [
                    float(
                        point[0]
                    ),
                    float(
                        point[1]
                    ),
                ]
            )

        if len(points) < 4:
            return None

        return points

    except Exception:

        return None


# ============================================================
# SORT OCR LINES
# ============================================================

def _sort_lines(lines):
    """
    Sort OCR output:

        top -> bottom
        left -> right
    """

    def sort_key(line):

        bbox = line.get(
            "bbox"
        )

        if not bbox:

            return (
                0.0,
                0.0,
            )

        try:

            xs = [
                float(
                    point[0]
                )
                for point in bbox
            ]

            ys = [
                float(
                    point[1]
                )
                for point in bbox
            ]

            return (
                min(ys),
                min(xs),
            )

        except Exception:

            return (
                0.0,
                0.0,
            )

    return sorted(
        lines,
        key=sort_key,
    )


# ============================================================
# PADDLE RESULT -> DICTIONARY
# ============================================================

def _result_to_dict(result):
    """
    Convert PaddleOCR 3.x result object into dictionary.

    PaddleOCR 3.x normally exposes prediction data through:

        result.json

    Some versions expose it as a callable.
    """

    try:

        data = result.json

        if callable(
            data
        ):

            data = data()

    except Exception as exc:

        print(
            "[PaddleOCR] Could not read result.json:",
            exc,
        )

        return None

    if not isinstance(
        data,
        dict,
    ):

        return None

    # Some versions wrap OCR result in "res".

    if (
        "res" in data
        and isinstance(
            data["res"],
            dict,
        )
    ):

        data = data[
            "res"
        ]

    return data


# ============================================================
# PARSE ONE PADDLE RESULT
# ============================================================

def _parse_result(
    result,
    min_confidence=0.0,
):
    """
    Parse one PaddleOCR page result into the project's
    common OCR format.
    """

    data = _result_to_dict(
        result
    )

    if data is None:
        return []

    # ========================================================
    # TEXT
    # ========================================================

    texts = data.get(
        "rec_texts",
        [],
    )

    if texts is None:
        texts = []

    # ========================================================
    # CONFIDENCE
    # ========================================================

    scores = data.get(
        "rec_scores",
        [],
    )

    if scores is None:
        scores = []

    # ========================================================
    # POLYGONS
    # ========================================================

    polygons = data.get(
        "rec_polys"
    )

    if polygons is None:

        polygons = data.get(
            "dt_polys",
            [],
        )

    if polygons is None:
        polygons = []

    # ========================================================
    # SAFE RESULT COUNT
    # ========================================================

    total = min(
        len(texts),
        len(scores),
        len(polygons),
    )

    print(
        "[PaddleOCR] Parsed result:"
    )

    print(
        f"  texts    = {len(texts)}"
    )

    print(
        f"  scores   = {len(scores)}"
    )

    print(
        f"  polygons = {len(polygons)}"
    )

    lines = []

    # ========================================================
    # BUILD OCR LINES
    # ========================================================

    for index in range(
        total
    ):

        text = _clean_text(
            texts[index]
        )

        if not text:
            continue

        confidence = _safe_float(
            scores[index]
        )

        if (
            confidence
            < min_confidence
        ):

            continue

        bbox = _convert_bbox(
            polygons[index]
        )

        if bbox is None:

            print(
                "[PaddleOCR] Invalid bbox skipped:",
                text,
            )

            continue

        lines.append(
            {
                "text":
                    text,

                "confidence":
                    round(
                        confidence,
                        4,
                    ),

                "bbox":
                    bbox,

                "ocr_engine":
                    "paddleocr",
            }
        )

    return lines


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def run_ocr(
    image,
    languages=None,
    gpu=False,
    min_confidence=0.0,
):
    """
    Run PaddleOCR.

    Parameters
    ----------
    image:
        OpenCV/numpy image.

    languages:
        Kept for compatibility with the old EasyOCR interface.

    gpu:
        Kept for compatibility with the old OCR interface.
        Current Paddle configuration explicitly uses CPU.

    min_confidence:
        OCR lines below this confidence are discarded.

    Returns
    -------
    list[dict]
    """

    total_start = (
        time.perf_counter()
    )

    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    prepare_start = (
        time.perf_counter()
    )

    image = _prepare_image(
        image
    )

    prepare_seconds = round(
        time.perf_counter()
        - prepare_start,
        3,
    )

    height, width = (
        image.shape[:2]
    )

    megapixels = (
        height
        * width
        / 1_000_000
    )

    print()
    print("=" * 70)

    print(
        "PADDLE OCR INPUT"
    )

    print("=" * 70)

    print(
        f"Shape       : {image.shape}"
    )

    print(
        f"Resolution  : {width} x {height}"
    )

    print(
        f"Megapixels  : {megapixels:.2f} MP"
    )

    print(
        f"Dtype       : {image.dtype}"
    )

    print(
        f"Prepare     : {prepare_seconds:.3f} sec"
    )

    # ========================================================
    # LOAD / GET READER
    # ========================================================

    reader_start = (
        time.perf_counter()
    )

    reader = get_reader()

    reader_seconds = round(
        time.perf_counter()
        - reader_start,
        3,
    )

    print(
        f"Reader      : {reader_seconds:.3f} sec"
    )

    # ========================================================
    # RUN PREDICTION
    # ========================================================

    print()
    print(
        "[PaddleOCR] Running prediction..."
    )

    prediction_start = (
        time.perf_counter()
    )

    try:

        raw_results = (
            reader.predict(
                image
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "PaddleOCR prediction failed: "
            f"{exc}"
        ) from exc

    prediction_seconds = round(
        time.perf_counter()
        - prediction_start,
        3,
    )

    print(
        f"[PaddleOCR] Prediction completed in "
        f"{prediction_seconds:.3f} sec"
    )

    # ========================================================
    # PARSE RESULTS
    # ========================================================

    parse_start = (
        time.perf_counter()
    )

    lines = []

    result_count = 0

    try:

        for result in raw_results:

            result_count += 1

            parsed_lines = (
                _parse_result(
                    result,
                    min_confidence=(
                        min_confidence
                    ),
                )
            )

            lines.extend(
                parsed_lines
            )

    except Exception as exc:

        raise RuntimeError(
            "Failed while parsing PaddleOCR output: "
            f"{exc}"
        ) from exc

    parse_seconds = round(
        time.perf_counter()
        - parse_start,
        3,
    )

    # ========================================================
    # SORT
    # ========================================================

    lines = _sort_lines(
        lines
    )

    total_seconds = round(
        time.perf_counter()
        - total_start,
        3,
    )

    # ========================================================
    # PERFORMANCE OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "PADDLE OCR PERFORMANCE"
    )

    print("=" * 70)

    print(
        f"Image preparation : "
        f"{prepare_seconds:.3f} sec"
    )

    print(
        f"Reader/model       : "
        f"{reader_seconds:.3f} sec"
    )

    print(
        f"Prediction         : "
        f"{prediction_seconds:.3f} sec"
    )

    print(
        f"Result parsing     : "
        f"{parse_seconds:.3f} sec"
    )

    print(
        f"TOTAL OCR          : "
        f"{total_seconds:.3f} sec"
    )

    print(
        f"Result objects     : "
        f"{result_count}"
    )

    print(
        f"OCR lines          : "
        f"{len(lines)}"
    )

    print("=" * 70)

    return lines


# ============================================================
# CONVERT OCR LINES TO TEXT
# ============================================================

def lines_to_text(
    lines,
):
    """
    Convert OCR lines into plain text.
    """

    return "\n".join(
        line.get(
            "text",
            "",
        )
        for line in lines
        if line.get(
            "text"
        )
    )


# ============================================================
# PRINT OCR RESULTS
# ============================================================

def print_ocr_results(
    lines,
):
    """
    Pretty-print OCR results for debugging.
    """

    print()
    print("=" * 70)

    print(
        "PADDLE OCR RESULTS"
    )

    print("=" * 70)

    if not lines:

        print(
            "No OCR text detected."
        )

        print("=" * 70)

        return

    for line in lines:

        text = line.get(
            "text",
            "",
        )

        confidence = line.get(
            "confidence",
            0.0,
        )

        print(
            f"{confidence:.4f} | {text}"
        )

    print("=" * 70)