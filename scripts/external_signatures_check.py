"""
Test the trained signature classifier against external images.

Expected folder:

    data/
        external_test/
            signatures/
                *.jpg
                *.jpeg
                *.png
                *.webp

This script only tests the existing model.
It does NOT train or modify anything.
"""

from __future__ import annotations

import sys
from pathlib import Path


# =========================================================
# Add project root to Python path
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =========================================================
# Imports
# =========================================================

import cv2

from src.documents.signature.services.signature_classifier import (
    classify_signature,
)


# =========================================================
# Configuration
# =========================================================

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_test"
    / "signatures"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# =========================================================
# Find images
# =========================================================

def find_images() -> list[Path]:

    if not TEST_DIR.exists():

        print(
            "Test directory does not exist:"
        )

        print(
            TEST_DIR
        )

        print(
            "\nCreate it with:"
        )

        print(
            r"New-Item '.\data\external_test\signatures' -ItemType Directory -Force"
        )

        return []

    images = []

    for path in sorted(
        TEST_DIR.iterdir()
    ):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(
                path
            )

    return images


# =========================================================
# Test one image
# =========================================================

def test_image(
    image_path: Path,
) -> dict:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return {
            "file": image_path.name,
            "prediction": "ERROR",
            "confidence": 0.0,
            "signature_probability": 0.0,
            "passed": False,
        }

    try:

        result = classify_signature(
            image
        )

    except Exception as exc:

        print(
            f"\nERROR: {image_path.name}"
        )

        print(
            f"       {exc}"
        )

        return {
            "file": image_path.name,
            "prediction": "ERROR",
            "confidence": 0.0,
            "signature_probability": 0.0,
            "passed": False,
        }

    passed = bool(
        result.is_signature
    )

    return {
        "file": image_path.name,
        "prediction": (
            result.predicted_class
        ),
        "confidence": (
            result.confidence
        ),
        "signature_probability": (
            result.signature_probability
        ),
        "passed": passed,
    }


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXTERNAL SIGNATURE GENERALIZATION TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nTest directory:"
    )

    print(
        TEST_DIR
    )

    images = find_images()

    if not images:

        print(
            "\nNo external signature images found."
        )

        print(
            "\nPut images here:"
        )

        print(
            TEST_DIR
        )

        return

    print(
        f"\nImages found: {len(images)}"
    )

    print(
        "\nExpected class: signature"
    )

    print(
        "\n"
        + "=" * 70
    )

    results = []

    for image_path in images:

        result = test_image(
            image_path
        )

        results.append(
            result
        )

        print(
            f"\n{result['file']}"
        )

        print(
            "  Prediction            : "
            f"{result['prediction']}"
        )

        print(
            "  Confidence            : "
            f"{result['confidence']:.4f}"
        )

        print(
            "  Signature probability : "
            f"{result['signature_probability']:.4f}"
        )

        print(
            "  Result                : "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    # =====================================================
    # Summary
    # =====================================================

    total = len(
        results
    )

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = (
        total - passed
    )

    accuracy = (
        passed / total
        if total
        else 0.0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXTERNAL TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal images : {total}"
    )

    print(
        f"Passed       : {passed}"
    )

    print(
        f"Failed       : {failed}"
    )

    print(
        f"Accuracy     : {accuracy:.2%}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 70
    )

    if total < 5:

        print(
            "\nToo few external images."
        )

        print(
            "Use at least 5 different signatures."
        )

    elif accuracy >= 0.80:

        print(
            "\nGood external generalization."
        )

        print(
            "The classifier is performing reasonably "
            "well on unseen signatures."
        )

    elif accuracy >= 0.50:

        print(
            "\nWeak external generalization."
        )

        print(
            "The classifier recognizes some external "
            "signatures but needs more diverse training data."
        )

    else:

        print(
            "\nPoor external generalization."
        )

        print(
            "The classifier is likely overfitting "
            "to the original training dataset."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These images were not used for training."
    )

    print(
        "Do not add them to the training dataset yet."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()