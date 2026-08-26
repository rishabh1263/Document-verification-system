from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.documents.signature.services.signature_classifier import (
    get_signature_classifier,
)
from src.documents.signature.validators.signature_features import (
    create_foreground_mask,
)


IMAGE_PATH = Path(
    r"C:\Users\ET0002183\Downloads\document_verify\Document-verification-system\data\signature_classifier\test_signature_google.jpg"
)


def crop_signature(image: np.ndarray) -> np.ndarray:
    mask = create_foreground_mask(image)

    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError("No foreground detected.")

    x, y, w, h = cv2.boundingRect(points)

    # Add 20% padding around detected foreground.
    pad_x = max(5, int(w * 0.20))
    pad_y = max(5, int(h * 0.20))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    return image[y1:y2, x1:x2]


classifier = get_signature_classifier()

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError("Could not read image.")

print("=" * 60)
print("ORIGINAL IMAGE")
print("=" * 60)

original_result = classifier.predict(image)

print(original_result)


cropped = crop_signature(image)

print()
print("=" * 60)
print("CROPPED IMAGE")
print("=" * 60)

print(
    "Original:",
    image.shape[1],
    "x",
    image.shape[0],
)

print(
    "Cropped:",
    cropped.shape[1],
    "x",
    cropped.shape[0],
)

cropped_result = classifier.predict(cropped)

print(cropped_result)