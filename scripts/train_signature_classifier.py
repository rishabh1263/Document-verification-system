from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# =========================================================
# Configuration
# =========================================================

DATA_DIR = Path(
    "data/signature_classifier"
)

OUTPUT_DIR = Path(
    "src/documents/signature/models"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "signature_classifier.pth"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_CLASSES = 2

SEED = 42


# =========================================================
# Reproducibility
# =========================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("SIGNATURE CLASSIFIER TRAINING")
print("=" * 60)

print()
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================================================
# Transforms
# =========================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomRotation(
            degrees=8
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.90, 1.10),
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# =========================================================
# Dataset validation
# =========================================================

train_path = DATA_DIR / "train"

val_path = DATA_DIR / "val"

if not train_path.exists():
    raise FileNotFoundError(
        f"Training directory not found: {train_path}"
    )

if not val_path.exists():
    raise FileNotFoundError(
        f"Validation directory not found: {val_path}"
    )


# =========================================================
# Datasets
# =========================================================

train_dataset = datasets.ImageFolder(
    train_path,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    val_path,
    transform=val_transform,
)


print()
print(
    f"Training images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Classes: "
    f"{train_dataset.classes}"
)

print(
    f"Class mapping: "
    f"{train_dataset.class_to_idx}"
)


# =========================================================
# Verify expected classes
# =========================================================

expected_classes = {
    "non_signature",
    "signature",
}

actual_classes = set(
    train_dataset.classes
)

if actual_classes != expected_classes:

    raise RuntimeError(
        "Unexpected classes.\n"
        f"Expected: {expected_classes}\n"
        f"Found: {actual_classes}"
    )


# =========================================================
# DataLoaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


# =========================================================
# Model
# =========================================================

print()
print(
    "Loading MobileNetV3-Small "
    "without pretrained weights..."
)

model = models.mobilenet_v3_small(
    weights=None
)


# =========================================================
# Replace classifier
# =========================================================

in_features = (
    model.classifier[-1]
    .in_features
)

model.classifier[-1] = nn.Linear(
    in_features,
    NUM_CLASSES,
)

model = model.to(
    DEVICE
)


# =========================================================
# Loss
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# Optimizer
# =========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


# =========================================================
# Training
# =========================================================

best_accuracy = 0.0


for epoch in range(
    EPOCHS
):

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss /
        total
    )

    train_accuracy = (
        correct /
        total
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            val_loss_total += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += (
                labels.size(0)
            )

    val_loss = (
        val_loss_total /
        val_total
    )

    val_accuracy = (
        val_correct /
        val_total
    )


    # =====================================================
    # Epoch output
    # =====================================================

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy:.4f}"
    )


    # =====================================================
    # Save best model
    # =====================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "image_size":
                    IMAGE_SIZE,

                "best_accuracy":
                    best_accuracy,
            },
            MODEL_PATH,
        )

        print(
            "  [BEST MODEL SAVED]"
        )


# =========================================================
# Final
# =========================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.4f}"
)

print()

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)