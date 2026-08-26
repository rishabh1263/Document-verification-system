from pathlib import Path

from PIL import Image

from src.documents.signature.services.signature_detector import (
    detect_signatures
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEST_IMAGES = {
    "real_signature": (
        PROJECT_ROOT /
        "test_signature.jpg"
    ),

    "bank_document": (
        PROJECT_ROOT /
        "samples" /
        "view.jpg"
    ),

    "pan": (
        PROJECT_ROOT /
        "samples" /
        "upan.webp"
    ),

    "aadhaar": (
        PROJECT_ROOT /
        "samples" /
        "adhar1.webp"
    ),
}


def test_real_signature_detection():

    image_path = TEST_IMAGES[
        "real_signature"
    ]

    assert image_path.exists()

    image = Image.open(
        image_path
    )

    result = detect_signatures(
        image
    )

    print(
        "\n========== REAL SIGNATURE =========="
    )

    print(
        f"Detected       : "
        f"{result.detected}"
    )

    print(
        f"Count          : "
        f"{result.detection_count}"
    )

    print(
        f"Highest score  : "
        f"{result.highest_score}"
    )

    print(
        f"Largest ratio  : "
        f"{result.largest_area_ratio}"
    )

    assert result.detected is True

    assert result.highest_score > 0.90


def test_bank_document_detection():

    image_path = TEST_IMAGES[
        "bank_document"
    ]

    assert image_path.exists()

    image = Image.open(
        image_path
    )

    result = detect_signatures(
        image
    )

    print(
        "\n========== BANK DOCUMENT =========="
    )

    print(
        f"Detected       : "
        f"{result.detected}"
    )

    print(
        f"Count          : "
        f"{result.detection_count}"
    )

    print(
        f"Highest score  : "
        f"{result.highest_score}"
    )

    print(
        f"Multiple       : "
        f"{result.multiple_signatures}"
    )

    assert result.detected is True


def test_pan_detection():

    image_path = TEST_IMAGES[
        "pan"
    ]

    assert image_path.exists()

    image = Image.open(
        image_path
    )

    result = detect_signatures(
        image
    )

    print(
        "\n========== PAN =========="
    )

    print(
        f"Detected       : "
        f"{result.detected}"
    )

    print(
        f"Count          : "
        f"{result.detection_count}"
    )

    print(
        f"Highest score  : "
        f"{result.highest_score}"
    )

    print(
        f"Multiple       : "
        f"{result.multiple_signatures}"
    )

    assert result.detected is True


def test_aadhaar_detection():

    image_path = TEST_IMAGES[
        "aadhaar"
    ]

    assert image_path.exists()

    image = Image.open(
        image_path
    )

    result = detect_signatures(
        image
    )

    print(
        "\n========== AADHAAR =========="
    )

    print(
        f"Detected       : "
        f"{result.detected}"
    )

    print(
        f"Count          : "
        f"{result.detection_count}"
    )

    print(
        f"Highest score  : "
        f"{result.highest_score}"
    )

    assert result.detected is True


def test_detection_structure():

    image_path = TEST_IMAGES[
        "real_signature"
    ]

    image = Image.open(
        image_path
    )

    result = detect_signatures(
        image
    )

    if result.detected:

        detection = result.detections[0]

        assert detection.score > 0

        assert detection.width > 0

        assert detection.height > 0

        assert detection.area > 0

        assert (
            0 <
            detection.area_ratio
            <= 1
        )

        assert (
            0 <
            detection.width_ratio
            <= 1
        )

        assert (
            0 <
            detection.height_ratio
            <= 1
        )