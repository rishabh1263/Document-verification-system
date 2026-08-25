from pathlib import Path

import cv2

from src.documents.signature.services.signature_validator import (
    SignatureDecision,
    validate_signature
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

SAMPLES_DIR = PROJECT_ROOT / "samples"


NEGATIVE_IMAGES = [
    SAMPLES_DIR / "view.jpg",
    SAMPLES_DIR / "upan.webp",
    SAMPLES_DIR / "aadhar1.webp",
]


def test_negative_documents():

    for image_path in NEGATIVE_IMAGES:

        assert image_path.exists(), (
            f"Missing test image: {image_path}"
        )

        image = cv2.imread(
            str(image_path)
        )

        assert image is not None, (
            f"Could not read image: {image_path}"
        )

        result = validate_signature(
            image
        )

        print("\n================================")
        print(
            f"Image      : {image_path.name}"
        )
        print(
            f"Decision   : {result.decision.value}"
        )
        print(
            f"Confidence : {result.confidence}"
        )
        print(
            f"Reason     : {result.reason_code}"
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

        print("================================")

        # A non-signature document must NEVER be
        # automatically accepted.
        assert result.decision != (
            SignatureDecision.ACCEPT
        ), (
            f"FALSE ACCEPT: {image_path.name}"
        )