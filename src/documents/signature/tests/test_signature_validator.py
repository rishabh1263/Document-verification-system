"""
Tests for the signature validation pipeline.

The validator now uses the trained MobileNetV3 classifier
as the primary signature detector.
"""

import cv2
import numpy as np

from src.documents.signature.services.signature_validator import (
    SignatureDecision,
    validate_signature,
)


# =========================================================
# Helpers
# =========================================================

REAL_SIGNATURE_IMAGE = (
    "data/signature_classifier/"
    "val/signature/signature_00001.jpg"
)


def create_white_image():

    return np.ones(
        (
            500,
            1000,
            3,
        ),
        dtype=np.uint8,
    ) * 255


def create_black_image():

    return np.zeros(
        (
            500,
            1000,
            3,
        ),
        dtype=np.uint8,
    )


# =========================================================
# None image
# =========================================================

def test_none_image_rejected():

    result = validate_signature(
        None
    )

    assert (
        result.decision
        == SignatureDecision.REJECT
    )

    assert (
        result.reason_code
        == "INVALID_IMAGE"
    )

    assert result.features is None


# =========================================================
# Empty image
# =========================================================

def test_empty_image_rejected():

    image = np.empty(
        (
            0,
            0,
            3,
        ),
        dtype=np.uint8,
    )

    result = validate_signature(
        image
    )

    assert (
        result.decision
        == SignatureDecision.REJECT
    )

    assert (
        result.reason_code
        == "INVALID_IMAGE"
    )


# =========================================================
# White image
# =========================================================

def test_white_image_rejected():

    image = create_white_image()

    result = validate_signature(
        image
    )

    assert (
        result.decision
        == SignatureDecision.REJECT
    )

    assert result.features is not None

    assert (
        result.reason_code
        == "BLANK_IMAGE"
    )


# =========================================================
# Solid black image
# =========================================================

def test_solid_black_image_not_accepted():

    image = create_black_image()

    result = validate_signature(
        image
    )

    assert (
        result.decision
        != SignatureDecision.ACCEPT
    )


# =========================================================
# Real signature image
# =========================================================

def test_signature_like_image():

    image = cv2.imread(
        REAL_SIGNATURE_IMAGE
    )

    assert image is not None

    result = validate_signature(
        image
    )

    print(
        "\n========== SIGNATURE VALIDATOR =========="
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

    print(
        f"Message    : {result.message}"
    )

    if result.classifier:

        print(
            "Signature probability : "
            f"{result.classifier.signature_probability}"
        )

        print(
            "Non-signature probability : "
            f"{result.classifier.non_signature_probability}"
        )

        print(
            "Classifier prediction : "
            f"{result.classifier.predicted_class}"
        )

    # -----------------------------------------------------
    # Features must exist
    # -----------------------------------------------------

    assert result.features is not None

    # -----------------------------------------------------
    # Classifier must exist
    # -----------------------------------------------------

    assert result.classifier is not None

    # -----------------------------------------------------
    # Real validation image must be classified
    # as a signature.
    # -----------------------------------------------------

    assert (
        result.classifier.is_signature
        is True
    )

    assert (
        result.classifier.signature_probability
        >= 0.90
    )

    # -----------------------------------------------------
    # A genuine signature should never be rejected
    # by the signature classifier pipeline.
    # -----------------------------------------------------

    assert result.decision in (
        SignatureDecision.ACCEPT,
        SignatureDecision.REVIEW,
    )