import cv2
import numpy as np

from src.documents.signature.validators.signature_features import (
    create_foreground_mask,
    extract_signature_features
)


def test_foreground_mask_creation():

    image = np.full(
        (500, 1000, 3),
        255,
        dtype=np.uint8
    )

    # Curved signature-like stroke.
    points = np.array(
        [
            [250, 280],
            [300, 230],
            [350, 300],
            [400, 220],
            [450, 290],
            [520, 240],
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

    # Add a diagonal underline-like stroke.
    cv2.line(
        image,
        (300, 330),
        (650, 300),
        (0, 0, 0),
        4
    )

    mask = create_foreground_mask(
        image
    )

    assert mask is not None
    assert mask.shape == (
        500,
        1000
    )

    assert mask.dtype == np.uint8

    assert cv2.countNonZero(
        mask
    ) > 0


def test_feature_extraction():

    image = np.full(
        (500, 1000, 3),
        255,
        dtype=np.uint8
    )

    # Main signature-like stroke.
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

    # Signature underline.
    cv2.line(
        image,
        (300, 330),
        (650, 300),
        (0, 0, 0),
        4
    )

    features = extract_signature_features(
        image
    )

    assert features.image_width == 1000
    assert features.image_height == 500

    assert features.foreground_density > 0

    assert features.bbox_x is not None
    assert features.bbox_y is not None

    assert features.bbox_width is not None
    assert features.bbox_height is not None

    assert features.bbox_width > 0
    assert features.bbox_height > 0

    assert features.aspect_ratio is not None

    assert features.occupancy_ratio is not None

    assert (
        0 <
        features.occupancy_ratio
        <= 1
    )

    assert features.connected_components > 0

    assert features.contour_count > 0


def test_empty_image_rejected():

    image = np.array(
        [],
        dtype=np.uint8
    )

    try:

        extract_signature_features(
            image
        )

        assert False, (
            "Expected ValueError"
        )

    except ValueError:
        assert True