from pathlib import Path

import cv2

from src.documents.signature.validators.signature_features import (
    extract_signature_features
)


def test_real_signature_features():

    image_path = Path("test_signature.jpg")

    image = cv2.imread(
        str(image_path)
    )

    assert image is not None, (
        f"Could not read image: {image_path}"
    )

    features = extract_signature_features(
        image
    )

    print(
        "\n========== REAL SIGNATURE FEATURES =========="
    )

    print(
        f"Image width           : "
        f"{features.image_width}"
    )

    print(
        f"Image height          : "
        f"{features.image_height}"
    )

    print(
        f"Foreground density    : "
        f"{features.foreground_density}"
    )

    print(
        f"Bounding box X        : "
        f"{features.bbox_x}"
    )

    print(
        f"Bounding box Y        : "
        f"{features.bbox_y}"
    )

    print(
        f"Bounding box width    : "
        f"{features.bbox_width}"
    )

    print(
        f"Bounding box height   : "
        f"{features.bbox_height}"
    )

    print(
        f"Aspect ratio          : "
        f"{features.aspect_ratio}"
    )

    print(
        f"Occupancy ratio       : "
        f"{features.occupancy_ratio}"
    )

    print(
        f"Connected components  : "
        f"{features.connected_components}"
    )

    print(
        f"Contour count         : "
        f"{features.contour_count}"
    )

    print(
        f"Largest contour area  : "
        f"{features.largest_contour_area}"
    )

    print(
        f"Total contour area    : "
        f"{features.total_contour_area}"
    )

    assert features.foreground_density > 0
    assert features.bbox_width is not None
    assert features.bbox_height is not None
    assert features.aspect_ratio is not None