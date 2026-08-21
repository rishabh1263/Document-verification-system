"""
PAN OCR latency benchmark - PaddleOCR PP-OCRv5 mobile.

Goal:
    Measure where the time is spent while keeping ONE OCR pass.

This benchmark does NOT change the production API.

It measures:
    1. Model initialization
    2. Total pipeline inference
    3. Number of detected text boxes
    4. OCR accuracy against the known test PAN

Current test image:
    C:\\Users\\ET0002183\\Downloads\\npan.jpg

Run:
    .\\.venv-paddle\\Scripts\\python.exe paddle_pan_timing.py

NOTE:
PaddleOCR's high-level predict() pipeline does not expose a reliable,
separate detector/recognizer wall-clock measurement through the public
pipeline API. Therefore this benchmark measures the real end-to-end
inference time and separately reports the extraction/result-processing
time. We should not fake detector/recognizer timings.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

# Windows CPU compatibility workaround.
# MUST happen before importing paddle/paddleocr.
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
from paddleocr import PaddleOCR


IMAGE_PATH = r"C:\Users\ET0002183\Downloads\npan.jpg"
MAX_WIDTH = 1600

EXPECTED = {
    "pan": "MHYPS5862D",
    "name": "NIDHI AJIT SINGH",
    "father_name": "AJIT SINGH",
    "dob": "12/09/1999",
}


def load_image(path: str):
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"Image not found: {path_obj}"
        )

    image = cv2.imread(str(path_obj))

    if image is None:
        raise ValueError(
            f"Could not decode image: {path_obj}"
        )

    return image


def prepare_image(image):
    """
    Minimal preprocessing.

    One image -> one OCR pass.
    No grayscale, thresholding, sharpening, rotation, or variants.
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
    """
    Explicit PP-OCRv5 mobile models.

    Optional document modules are disabled for speed because the input
    is one normal PAN image.
    """

    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def parse_result(result):
    """
    Convert PaddleOCR structured output into simple rows.
    """

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
    """
    Simple benchmark only.

    This does NOT represent the final production extractor.
    """

    texts = [
        row["text"]
        for row in rows
    ]

    normalized_texts = [
        normalize(text)
        for text in texts
    ]

    expected_pan = normalize(
        EXPECTED["pan"]
    )

    expected_name = normalize(
        EXPECTED["name"]
    )

    expected_father = normalize(
        EXPECTED["father_name"]
    )

    expected_dob = normalize(
        EXPECTED["dob"]
    )

    pan_ok = any(
        expected_pan == text
        for text in normalized_texts
    )

    name_ok = any(
        expected_name == text
        for text in normalized_texts
    )

    father_ok = any(
        expected_father == text
        for text in normalized_texts
    )

    dob_ok = any(
        expected_dob == text
        for text in normalized_texts
    )

    checks = {
        "PAN": pan_ok,
        "NAME": name_ok,
        "FATHER": father_ok,
        "DOB": dob_ok,
    }

    return checks


def main():

    print("=" * 72)
    print("PAN OCR LATENCY BENCHMARK")
    print("PP-OCRv5 MOBILE / ONE OCR PASS")
    print("=" * 72)

    print(
        f"Python: {os.sys.executable}"
    )

    print(
        f"PIR: {os.environ.get('FLAGS_enable_pir_api')}"
    )

    print(
        f"Image: {IMAGE_PATH}"
    )

    # ---------------------------------------------------------------
    # Image loading
    # ---------------------------------------------------------------

    image_start = time.perf_counter()

    image = load_image(
        IMAGE_PATH
    )

    image_load_ms = (
        time.perf_counter()
        - image_start
    ) * 1000

    print(
        f"\nImage load: "
        f"{image_load_ms:.1f} ms"
    )

    print(
        f"Original shape: "
        f"{image.shape}"
    )

    # ---------------------------------------------------------------
    # Minimal preprocessing
    # ---------------------------------------------------------------

    prep_start = time.perf_counter()

    image = prepare_image(
        image
    )

    prep_ms = (
        time.perf_counter()
        - prep_start
    ) * 1000

    print(
        f"Image preparation: "
        f"{prep_ms:.1f} ms"
    )

    print(
        f"OCR shape: "
        f"{image.shape}"
    )

    # ---------------------------------------------------------------
    # Model initialization
    # ---------------------------------------------------------------

    print(
        "\nLoading PP-OCRv5 mobile..."
    )

    model_start = time.perf_counter()

    ocr = create_ocr()

    model_load_ms = (
        time.perf_counter()
        - model_start
    ) * 1000

    print(
        f"Model initialization: "
        f"{model_load_ms:.1f} ms"
    )

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    print(
        "\nRunning warm-up inference..."
    )

    warmup_start = time.perf_counter()

    _ = ocr.predict(
        image
    )

    warmup_ms = (
        time.perf_counter()
        - warmup_start
    ) * 1000

    print(
        f"Warm-up inference: "
        f"{warmup_ms:.1f} ms"
    )

    # ---------------------------------------------------------------
    # Actual benchmark
    # ---------------------------------------------------------------

    print(
        "\nRunning ONE measured OCR pass..."
    )

    start = time.perf_counter()

    result = ocr.predict(
        image
    )

    inference_ms = (
        time.perf_counter()
        - start
    ) * 1000

    # ---------------------------------------------------------------
    # Result parsing
    # ---------------------------------------------------------------

    rows, parse_ms = parse_result(
        result
    )

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------

    checks = check_accuracy(
        rows
    )

    correct = sum(
        checks.values()
    )

    total = len(checks)

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

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
        f"{prep_ms:.1f} ms"
    )

    print(
        f"Model initialization   : "
        f"{model_load_ms:.1f} ms"
    )

    print(
        f"Warm-up inference      : "
        f"{warmup_ms:.1f} ms"
    )

    print(
        f"Measured OCR inference : "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"Result parsing         : "
        f"{parse_ms:.1f} ms"
    )

    print()
    print("-" * 72)
    print("ACCURACY")
    print("-" * 72)

    for field, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{field:<10}: "
            f"{status}"
        )

    print()
    print(
        f"Field accuracy: "
        f"{correct}/{total}"
    )

    print()
    print("-" * 72)
    print("OCR TEXT")
    print("-" * 72)

    for i, row in enumerate(
        rows,
        start=1,
    ):

        print(
            f"[{i}] "
            f"{row['text']} "
            f"(conf={row['confidence']:.4f})"
        )

    print()
    print("-" * 72)
    print("FINAL BENCHMARK")
    print("-" * 72)

    print(
        f"ONE OCR pass : "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"Accuracy     : "
        f"{correct}/{total}"
    )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()