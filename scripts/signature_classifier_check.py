from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path(
    "src/documents/signature/models/signature_classifier.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# =========================================================
# Load model
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

classes = checkpoint["classes"]

model = models.mobilenet_v3_small(
    weights=None
)

in_features = model.classifier[-1].in_features

model.classifier[-1] = torch.nn.Linear(
    in_features,
    len(classes),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# =========================================================
# Prediction
# =========================================================

def predict(image_path: Path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0,
    )

    return (
        classes[index.item()],
        float(confidence.item()),
    )


# =========================================================
# Test class
# =========================================================

def test_directory(
    directory: Path,
    expected_class: str,
    limit: int = 10,
):

    images = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".webp"}
    ][:limit]

    correct = 0

    print()
    print("=" * 60)
    print(f"EXPECTED: {expected_class}")
    print("=" * 60)

    for image_path in images:

        try:

            prediction, confidence = predict(
                image_path
            )

            is_correct = (
                prediction == expected_class
            )

            if is_correct:
                correct += 1

            status = "PASS" if is_correct else "FAIL"

            print()
            print(image_path.name)
            print(f"  Expected   : {expected_class}")
            print(f"  Prediction : {prediction}")
            print(f"  Confidence : {confidence:.4f}")
            print(f"  Result     : {status}")

        except Exception as exc:

            print()
            print(image_path.name)
            print(f"  ERROR: {exc}")

    print()
    print(
        f"Accuracy: "
        f"{correct}/{len(images)}"
    )

    return correct, len(images)


# =========================================================
# Main
# =========================================================

print("=" * 60)
print("SIGNATURE CLASSIFIER REAL-WORLD TEST")
print("=" * 60)

print()
print(f"Device: {DEVICE}")
print(f"Classes: {classes}")


# ---------------------------------------------------------
# Positive samples
# ---------------------------------------------------------

signature_dir = Path(
    "data/signature_classifier/val/signature"
)

signature_correct, signature_total = (
    test_directory(
        signature_dir,
        "signature",
        limit=10,
    )
)


# ---------------------------------------------------------
# Negative samples
# ---------------------------------------------------------

non_signature_dir = Path(
    "data/signature_classifier/val/non_signature"
)

non_signature_correct, non_signature_total = (
    test_directory(
        non_signature_dir,
        "non_signature",
        limit=10,
    )
)


# ---------------------------------------------------------
# Existing real documents
# ---------------------------------------------------------

samples_dir = Path("samples")

print()
print("=" * 60)
print("EXISTING DOCUMENT SAMPLES")
print("=" * 60)

sample_images = [
    path
    for path in sorted(samples_dir.iterdir())
    if path.is_file()
    and path.suffix.lower()
    in {".jpg", ".jpeg", ".png", ".webp"}
]

sample_correct = 0

for image_path in sample_images:

    prediction, confidence = predict(
        image_path
    )

    passed = (
        prediction == "non_signature"
    )

    if passed:
        sample_correct += 1

    status = "PASS" if passed else "FAIL"

    print()
    print(image_path.name)
    print(f"  Prediction : {prediction}")
    print(f"  Confidence : {confidence:.4f}")
    print(f"  Result     : {status}")


# =========================================================
# Summary
# =========================================================

print()
print("=" * 60)
print("FINAL TEST SUMMARY")
print("=" * 60)

print(
    f"Signature validation samples: "
    f"{signature_correct}/{signature_total}"
)

print(
    f"Non-signature validation samples: "
    f"{non_signature_correct}/{non_signature_total}"
)

print(
    f"Existing document samples: "
    f"{sample_correct}/{len(sample_images)}"
)

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)