from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.documents.signature.services.signature_classifier import (
    classify_signature,
)


IMAGE_PATH = Path(
    "data/signature_classifier/test_signature_google.jpg"
)


def test_image(
    name: str,
    image: np.ndarray,
):
    result = classify_signature(image)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(result)


def main():

    image = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not load: {IMAGE_PATH}"
        )

    print("Original")
    print("Shape:", image.shape)

    test_image(
        "ORIGINAL",
        image,
    )

    # --------------------------------------------------
    # Grayscale
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------
    # CLAHE
    # --------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_gray = clahe.apply(
        gray
    )

    enhanced = cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2BGR,
    )

    test_image(
        "CLAHE",
        enhanced,
    )

    # --------------------------------------------------
    # Contrast stretch
    # --------------------------------------------------

    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    normalized = cv2.cvtColor(
        normalized,
        cv2.COLOR_GRAY2BGR,
    )

    test_image(
        "CONTRAST_NORMALIZED",
        normalized,
    )


if __name__ == "__main__":
    main()