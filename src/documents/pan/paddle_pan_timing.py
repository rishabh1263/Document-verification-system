import os
import re
import time
from pathlib import Path

# Windows CPU compatibility
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


def load_image(path):
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

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

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
        row["text"]
        for row in rows
    ]

    normalized_texts = [
        normalize(text)
        for text in texts
    ]

    checks = {}

    checks["PAN"] = any(
        normalize(EXPECTED["pan"]) == text
        for text in normalized_texts
    )

    checks["NAME"] = any(
        normalize(EXPECTED["name"]) == text
        for text in normalized_texts
    )

    checks["FATHER"] = any(
        normalize(EXPECTED["father_name"]) == text
        for text in normalized_texts
    )

    checks["DOB"] = any(
        normalize(EXPECTED["dob"]) == text
        for text in normalized_texts
    )

    return checks


def main():

    print("=" * 70)
    print("PAN OCR LATENCY BENCHMARK")
    print("PP-OCRv5 MOBILE")
    print("=" * 70)

    print(
        f"Python: {os.sys.executable}"
    )

    print(
        f"PIR: {os.environ.get('FLAGS_enable_pir_api')}"
    )

    print(
        f"Image: {IMAGE_PATH}"
    )

    # ---------------------------------------------------------
    # IMAGE LOAD
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
        f"Image load: {image_load_ms:.1f} ms"
    )

    print(
        f"Original shape: {image.shape}"
    )

    # ---------------------------------------------------------
    # PREPARE
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
        f"OCR shape: {image.shape}"
    )

    # ---------------------------------------------------------
    # MODEL LOAD
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
    # WARMUP
    # ---------------------------------------------------------

    print()
    print("Running warm-up inference...")

    start = time.perf_counter()

    _ = ocr.predict(
        image
    )

    warmup_ms = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Warm-up inference: "
        f"{warmup_ms:.1f} ms"
    )

    # ---------------------------------------------------------
    # ACTUAL TEST
    # ---------------------------------------------------------

    print()
    print("Running ONE measured OCR pass...")

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

    # ---------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------

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
    print("-" * 70)
    print("TIMING")
    print("-" * 70)

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

    # ---------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("ACCURACY")
    print("-" * 70)

    for field, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{field:<10}: {status}"
        )

    print()
    print(
        f"Field accuracy: "
        f"{correct}/{total}"
    )

    # ---------------------------------------------------------
    # OCR RESULT
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("OCR RESULT")
    print("-" * 70)

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
    print("-" * 70)
    print("FINAL BENCHMARK")
    print("-" * 70)

    print(
        f"OCR latency : "
        f"{inference_ms:.1f} ms"
    )

    print(
        f"Accuracy    : "
        f"{correct}/{total}"
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()