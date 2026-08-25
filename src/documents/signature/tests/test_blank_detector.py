import cv2
import numpy as np

from src.documents.signature.validators.blank_detector import (
    detect_blank
)


def test_none_image():
    result = detect_blank(None)

    assert result.is_blank is True
    assert result.reason_code == "NO_IMAGE"


def test_empty_image():
    image = np.array(
        [],
        dtype=np.uint8
    )

    result = detect_blank(image)

    assert result.is_blank is True
    assert result.reason_code == "EMPTY_IMAGE"


def test_white_image_is_blank():
    image = np.full(
        (500, 1000, 3),
        255,
        dtype=np.uint8
    )

    result = detect_blank(image)

    assert result.is_blank is True
    assert result.reason_code == "BLANK_IMAGE"


def test_solid_black_image():
    image = np.zeros(
        (500, 1000, 3),
        dtype=np.uint8
    )

    result = detect_blank(image)

    assert result.is_blank is False
    assert result.reason_code == "SOLID_DARK_IMAGE"


def test_image_with_foreground_is_not_blank():
    image = np.full(
        (500, 1000, 3),
        255,
        dtype=np.uint8
    )

    # Simulate a signature-like foreground mark.
    cv2.line(
        image,
        (300, 250),
        (700, 250),
        (0, 0, 0),
        5
    )

    result = detect_blank(image)

    assert result.is_blank is False
    assert result.reason_code == "NON_BLANK_IMAGE"
    assert result.foreground_density > 0