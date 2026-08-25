from pathlib import Path

import cv2

from src.documents.signature.validators.signature_features import (
    create_foreground_mask
)


IMAGE_PATH = Path("test_signature.jpg")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )

mask = create_foreground_mask(image)

output_path = Path("debug_foreground_mask.png")

cv2.imwrite(
    str(output_path),
    mask
)

print(f"Mask saved to: {output_path.resolve()}")
print(f"Mask shape: {mask.shape}")