from pathlib import Path
from time import perf_counter

from PIL import Image
from transformers import pipeline


MODEL_NAME = "mdefrance/yolos-tiny-signature-detection"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGES = [
    PROJECT_ROOT / "test_signature.jpg",
    PROJECT_ROOT / "samples" / "view.jpg",
    PROJECT_ROOT / "samples" / "upan.webp",
    PROJECT_ROOT / "samples" / "adhar1.webp",
]


print("=" * 60)
print("YOLOS SIGNATURE DETECTOR TEST")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print("Device: CPU")
print()

print("Loading model...")

load_start = perf_counter()

detector = pipeline(
    task="object-detection",
    model=MODEL_NAME,
    device=-1,
)

load_time = perf_counter() - load_start

print(
    f"Model load time: {load_time:.3f} seconds"
)

print()


for image_path in TEST_IMAGES:

    print("=" * 60)
    print(f"IMAGE: {image_path.name}")
    print("=" * 60)

    if not image_path.exists():

        print(
            f"SKIPPED: {image_path}"
        )

        continue

    image = Image.open(
        image_path
    ).convert("RGB")

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    detector(
        image
    )

    # -----------------------------------------------------
    # Actual benchmark
    # -----------------------------------------------------

    start = perf_counter()

    predictions = detector(
        image
    )

    inference_time = (
        perf_counter() -
        start
    )

    print(
        f"Inference time: "
        f"{inference_time:.3f} seconds"
    )

    if not predictions:

        print(
            "No signature detected."
        )

        continue

    print(
        f"Detections: "
        f"{len(predictions)}"
    )

    for detection in predictions:

        print(
            f"Label : "
            f"{detection['label']}"
        )

        print(
            f"Score : "
            f"{detection['score']:.4f}"
        )

        print(
            f"Box   : "
            f"{detection['box']}"
        )

    print()