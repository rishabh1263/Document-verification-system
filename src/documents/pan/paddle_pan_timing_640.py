"""
PAN OCR latency benchmark - PP-OCRv5 mobile, optimized detector.

Test:
    PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec
    Detection side limited to 640 px.

Goal:
    One OCR pass, prioritize accuracy first and latency second.

Run from pan_api:
    .\.venv-paddle\Scripts\python.exe paddle_pan_timing_640.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

# Windows CPU/PIR compatibility workaround.
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
from paddleocr import PaddleOCR


IMAGE_PATH = r"C:\Users\ET0002183\Downloads\npan.jpg"

# Detection optimization.
# For this PAN image, 640 should be enough while reducing CPU work.
DET_LIMIT_TYPE = "max"
DET_LIMIT_SIDE_LEN = 640

MAX_WIDTH = 1600

EXPECTED = {
    "pan": "MHYPS5862D",
    "name": "NIDHI AJIT SINGH",
    "father_name": "AJIT SINGH",
    "dob": "12/09/1999",
}


def load_image(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not decode image: {path}"
        )

    return image


def prepare_image(image):
    """
    Minimal preprocessing only.

    No grayscale.
    No threshold.
    No sharpening.
    No rotation.
    No multiple OCR passes.
    """

    height, width = image.shape[:2]

    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width

        image = cv2.resize(
            image,
            (
                MAX_WIDTH,
                int(height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


def create_ocr():

    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",

        # Important latency settings.
        text_det_limit_type=DET_LIMIT_TYPE,
        text_det_limit_side_len=DET_LIMIT_SIDE_LEN,

        # PAN is already a normal, correctly oriented image.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # Avoid Windows oneDNN/PIR issue.
        enable_mkldnn=False,
    )


def parse_result(result):

    rows = []

    start = time.perf_counter()

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

            bbox = None

            if i < len(boxes):

                bbox = boxes[i]

                if hasattr(
                    bbox,
                    "tolist",
                ):
                    bbox = bbox.tolist()

            rows.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )

    parse_ms = (
        time.perf_counter() - start
    ) * 1000

    return rows, parse_ms


def normalize(value):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper(),
    )


def check_accuracy(rows):

    texts = [
        normalize(row["text"])
        for row in rows
    ]

    checks = {
        "PAN": normalize(EXPECTED["pan"]) in texts,
        "NAME": normalize(EXPECTED["name"]) in texts,
        "FATHER": normalize(EXPECTED["father_name"]) in texts,
        "DOB": normalize(EXPECTED["dob"]) in texts,
    }

    return checks


def main():

    print("=" * 72)
    print("PAN OCR LATENCY BENCHMARK - OPTIMIZED DETECTOR")
    print("=" * 72)

    print(
        f"Python: {os.sys.executable}"
    )

    print(
        f"PIR: {os.environ.get('FLAGS_enable_pir_api')}"
    )

    print(
        f"Detection limit type: {DET_LIMIT_TYPE}"
    )

    print(
        f"Detection limit side: {DET_LIMIT_SIDE_LEN}"
    )

    print(
        f"Image: {IMAGE_PATH}"
    )

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------

    start = time.perf_counter()

    image = load_image(
        IMAGE_PATH
    )

    image_load_ms = (
        time.perf_counter() - start
    ) * 1000

    print()
    print(
        f"Image load: "
        f"{image_load_ms:.1f} ms"
    )

    print(
        f"Original shape: "
        f"{image.shape}"
    )

    # ---------------------------------------------------------
    # PREPARATION
    # ---------------------------------------------------------

    start = time.perf_counter()

    image = prepare_image(
        image
    )

    preparation_ms = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Image preparation: "
        f"{preparation_ms:.1f} ms"
    )

    print(
        f"OCR shape: "
        f"{image.shape}"
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    print()
    print("Loading PP-OCRv5 mobile...")

    start = time.perf_counter()

    ocr = create_ocr()

    model_ms = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Model initialization: "
        f"{model_ms:.1f} ms"
    )

    # ---------------------------------------------------------
    # SINGLE MEASURED PASS
    # ---------------------------------------------------------

    print()
    print("Running ONE OCR pass...")

    start = time.perf_counter()

    result = ocr.predict(
        image
    )

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    # ---------------------------------------------------------
    # PARSE
    # ---------------------------------------------------------

    rows, parse_ms = parse_result(
        result
    )

    checks = check_accuracy(
        rows
    )

    correct = sum(
        checks.values()
    )

    total = len(checks)

    # ---------------------------------------------------------
    # TIMING
    # ---------------------------------------------------------

    print()
    print("-" * 72)
    print("TIMING")
    print("-" * 72)

    print(
        f"Image load             : "
        f"{image_load_ms:.1f} ms"
    )

    print(
        f"Image preparation      : "
        f"{preparation_ms:.1f} ms"
    )

    print(
        f"Model initialization   : "
        f"{model_ms:.1f} ms"
    )

    print(
        f"ONE OCR inference      : "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"Result parsing         : "
        f"{parse_ms:.1f} ms"
    )

    # ---------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------

    print()
    print("-" * 72)
    print("ACCURACY")
    print("-" * 72)

    for field, passed in checks.items():

        print(
            f"{field:<10}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()
    print(
        f"Field accuracy: "
        f"{correct}/{total}"
    )

    # ---------------------------------------------------------
    # OCR
    # ---------------------------------------------------------

    print()
    print("-" * 72)
    print("OCR RESULT")
    print("-" * 72)

    for i, row in enumerate(
        rows,
        start=1,
    ):

        print()
        print(
            f"[{i}]"
        )

        print(
            f"Text       : "
            f"{row['text']}"
        )

        print(
            f"Confidence : "
            f"{row['confidence']:.4f}"
        )

        print(
            f"BBox       : "
            f"{row['bbox']}"
        )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()
    print("-" * 72)
    print("FINAL BENCHMARK")
    print("-" * 72)

    print(
        f"OCR latency : "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"Accuracy    : "
        f"{correct}/{total}"
    )

    if correct == 4 and inference_ms <= 3000:
        print(
            "Decision    : GOOD - proceed to production integration."
        )
    elif correct == 4 and inference_ms <= 4000:
        print(
            "Decision    : ACCEPTABLE - one final optimization "
            "is optional."
        )
    elif correct == 4:
        print(
            "Decision    : ACCURATE but still slow."
        )
    else:
        print(
            "Decision    : ACCURACY REGRESSION - do not use "
            "this configuration."
        )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()