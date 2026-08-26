from pathlib import Path

import cv2
from PIL import Image

from src.documents.signature.services.signature_detector import (
    detect_signatures
)

from src.documents.signature.services.image_context import (
    analyze_image_context
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


def load_cv_image(
    path: Path
):

    image = cv2.imread(
        str(path)
    )

    assert image is not None, (
        f"Could not read image: {path}"
    )

    return image


def print_context(
    name,
    context,
):

    print(
        "\n========================================"
    )

    print(
        f"IMAGE: {name}"
    )

    print(
        f"Dimensions       : "
        f"{context.image_width} x "
        f"{context.image_height}"
    )

    print(
        f"Foreground       : "
        f"{context.foreground_density}"
    )

    print(
        f"Edge density     : "
        f"{context.edge_density}"
    )

    print(
        f"Components       : "
        f"{context.connected_components}"
    )

    print(
        f"Large rectangles : "
        f"{context.large_rectangle_count}"
    )

    print(
        f"Largest rectangle: "
        f"{context.largest_rectangle_area_ratio}"
    )

    print(
        f"Signature area   : "
        f"{context.signature_area_ratio}"
    )

    print(
        f"Signature count  : "
        f"{context.signature_count}"
    )

    print(
        f"Multiple signs   : "
        f"{context.multiple_signatures}"
    )

    print(
        f"Document-like    : "
        f"{context.document_like}"
    )

    print(
        "========================================"
    )


def analyze(
    path: Path
):

    image = load_cv_image(
        path
    )

    pil_image = Image.open(
        path
    ).convert(
        "RGB"
    )

    signature_result = detect_signatures(
        pil_image
    )

    context = analyze_image_context(
        image,
        signature_result
    )

    return context


def test_real_signature_context():

    path = TEST_IMAGES[
        "real_signature"
    ]

    assert path.exists()

    context = analyze(
        path
    )

    print_context(
        "REAL SIGNATURE",
        context
    )

    assert context.image_width > 0
    assert context.image_height > 0


def test_bank_document_context():

    path = TEST_IMAGES[
        "bank_document"
    ]

    assert path.exists()

    context = analyze(
        path
    )

    print_context(
        "BANK DOCUMENT",
        context
    )

    assert context.image_width > 0
    assert context.image_height > 0


def test_pan_context():

    path = TEST_IMAGES[
        "pan"
    ]

    assert path.exists()

    context = analyze(
        path
    )

    print_context(
        "PAN",
        context
    )

    assert context.image_width > 0
    assert context.image_height > 0


def test_aadhaar_context():

    path = TEST_IMAGES[
        "aadhaar"
    ]

    assert path.exists()

    context = analyze(
        path
    )

    print_context(
        "AADHAAR",
        context
    )

    assert context.image_width > 0
    assert context.image_height > 0