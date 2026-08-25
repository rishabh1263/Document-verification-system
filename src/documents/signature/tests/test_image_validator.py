from src.documents.signature.validators.image_validator import validate_image


def test_empty_file():
    result = validate_image(b"")

    assert result.valid is False
    assert result.reason_code == "EMPTY_FILE"


def test_invalid_file():
    result = validate_image(b"this is not an image")

    assert result.valid is False
    assert result.reason_code == "INVALID_IMAGE"
