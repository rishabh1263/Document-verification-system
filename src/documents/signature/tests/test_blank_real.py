from pathlib import Path

import cv2

from src.documents.signature.validators.blank_detector import detect_blank


def test_real_signature_image():
    image_path = Path("test_signature.jpg")

    image = cv2.imread(str(image_path))

    assert image is not None, f"Could not read image: {image_path}"

    result = detect_blank(image)

    print("\n========== REAL IMAGE TEST ==========")
    print(f"Is blank           : {result.is_blank}")
    print(f"Reason             : {result.reason_code}")
    print(f"Message            : {result.message}")
    print(f"Foreground density : {result.foreground_density}")

    assert result.is_blank is False