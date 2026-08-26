from pathlib import Path

import cv2

from src.documents.signature.services.signature_validator import (
    validate_signature,
    SignatureDecision,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[4]


# =========================================================
# Negative document samples
# =========================================================

NEGATIVE_IMAGES = [

    PROJECT_ROOT /
    "samples" /
    "view.jpg",

    PROJECT_ROOT /
    "samples" /
    "upan.webp",

    PROJECT_ROOT /
    "samples" /
    "adhar1.webp",

    PROJECT_ROOT /
    "samples" /
    "aadhar 2.jpg",

    PROJECT_ROOT /
    "samples" /
    "aadhar3.jpg",

    PROJECT_ROOT /
    "samples" /
    "jpan.jpg",

]


# =========================================================
# Test
# =========================================================

def test_negative_documents():

    for image_path in NEGATIVE_IMAGES:

        assert image_path.exists(), (
            f"Missing test image: "
            f"{image_path}"
        )

        image = cv2.imread(
            str(image_path)
        )

        assert image is not None, (
            f"Could not read image: "
            f"{image_path}"
        )

        result = validate_signature(
            image
        )

        print(
            "\n================================"
        )

        print(
            f"Image      : "
            f"{image_path.name}"
        )

        print(
            f"Decision   : "
            f"{result.decision.value}"
        )

        print(
            f"Confidence : "
            f"{result.confidence}"
        )

        print(
            f"Reason     : "
            f"{result.reason_code}"
        )

        if result.features:

            print(
                f"Density    : "
                f"{result.features.foreground_density}"
            )

            print(
                f"BBox       : "
                f"{result.features.bbox_width} x "
                f"{result.features.bbox_height}"
            )

            print(
                f"Aspect     : "
                f"{result.features.aspect_ratio}"
            )

            print(
                f"Occupancy  : "
                f"{result.features.occupancy_ratio}"
            )

            print(
                f"Components : "
                f"{result.features.connected_components}"
            )

        print(
            "================================"
        )

        # -------------------------------------------------
        # Security requirement:
        #
        # A known non-signature document must NEVER be
        # automatically accepted.
        # -------------------------------------------------

        assert result.decision != (
            SignatureDecision.ACCEPT
        ), (
            f"FALSE ACCEPT: "
            f"{image_path.name}"
        )