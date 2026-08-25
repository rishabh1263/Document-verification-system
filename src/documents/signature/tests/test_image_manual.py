from pathlib import Path

from src.documents.signature.validators.image_validator import validate_image


IMAGE_PATH = Path("test_signature.jpg")


with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


result = validate_image(image_bytes)

print("\n========== IMAGE VALIDATION ==========")
print(f"Valid       : {result.valid}")
print(f"Reason      : {result.reason_code}")
print(f"Message     : {result.message}")
print(f"Width       : {result.width}")
print(f"Height      : {result.height}")
print(f"Channels    : {result.channels}")