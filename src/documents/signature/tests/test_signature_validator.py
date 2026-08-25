import cv2
import numpy as np

from src.documents.signature.services.signature_validator import (
    SignatureDecision,
    validate_signature
)


def create_signature_like_image():

    image = np.full(
        (500, 1000, 3),
        255,
        dtype=np.uint8
    )

    points = np.array(
        [
            [250, 280],
            [300, 220],
            [350, 300],
            [400, 210],
            [450, 290],
            [520, 230],
            [600, 270],
            [680, 210],
        ],
        dtype=np.int32
    )

    cv2.polylines(
        image,
        [points],
        False,
        (0, 0, 0),
        5
    )

    cv2.line(
        image,
        (300, 330),
        (650, 300),
        (0, 0, 0),
        4
    )

    return image


def test_none_image_rejected():

    result = validate_signature(None)

    assert result.decision == SignatureDecision.REJECT
    assert result.reason_code == "INVALID_IMAGE"


def test_empty_image_rejected():

    image = np.array(
        [],
        dtype=np.uint8
    )

    result = validate_signature(image)

    assert result.decision == SignatureDecision.REJECT


def test_white_image_rejected():

    image = np.full(
        (500, 1000, 3),
        255,
        dtype=np.uint8
    )

    result = validate_signature(image)

    assert result.decision == SignatureDecision.REJECT


def test_signature_like_image():

    image = create_signature_like_image()

    result = validate_signature(image)

    print("\n========== SIGNATURE VALIDATOR ==========")
    print(
        f"Decision   : {result.decision.value}"
    )
    print(
        f"Confidence : {result.confidence}"
    )
    print(
        f"Reason     : {result.reason_code}"
    )
    print(
        f"Message    : {result.message}"
    )

    assert result.features is not None

    assert result.decision in (
        SignatureDecision.ACCEPT,
        SignatureDecision.REVIEW
    )


def test_solid_black_image_not_accepted():

    image = np.zeros(
        (500, 1000, 3),
        dtype=np.uint8
    )

    result = validate_signature(image)

    assert result.decision != SignatureDecision.ACCEPT