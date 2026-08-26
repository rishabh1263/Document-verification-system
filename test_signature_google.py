from pathlib import Path

from PIL import Image

from src.documents.signature.services.signature_classifier import (
    classify_signature,
)


IMAGE_PATH = Path(
    "data/signature_classifier/test_signature_google.jpg"
)


print("=" * 60)
print("GOOGLE SIGNATURE CLASSIFIER TEST")
print("=" * 60)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = Image.open(
    IMAGE_PATH
)

print("Path :", IMAGE_PATH)
print("Mode :", image.mode)
print("Size :", image.size)

print()
print("Running classifier...")

result = classify_signature(
    image
)

print()
print("RESULT")
print("-" * 60)
print(result)
print("-" * 60)

print(
    "Signature probability     :",
    result.signature_probability
)

print(
    "Non-signature probability :",
    result.non_signature_probability
)

print(
    "Predicted class           :",
    result.predicted_class
)

print(
    "Is signature              :",
    result.is_signature
)