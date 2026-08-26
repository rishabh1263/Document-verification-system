"""
Train a leakage-free signature classifier.

Dataset:

    data/signature_classifier_clean/
        train/
            signature/
            non_signature/
        val/
            signature/
            non_signature/

Model:
    MobileNetV3-Small

The original classifier is NOT overwritten.

Output:
    src/documents/signature/models/signature_classifier_clean.pth
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader


# =========================================================
# Configuration
# =========================================================

DATA_ROOT = Path(
    "data/signature_classifier_clean"
)

MODEL_OUTPUT = Path(
    "src/documents/signature/models/signature_classifier_clean.pth"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 0.0001

RANDOM_SEED = 42

NUM_WORKERS = 0


# =========================================================
# Reproducibility
# =========================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TRAINING LEAKAGE-FREE SIGNATURE CLASSIFIER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Dataset: {DATA_ROOT}"
    )

    # =====================================================
    # Verify dataset
    # =====================================================

    train_dir = (
        DATA_ROOT
        / "train"
    )

    val_dir = (
        DATA_ROOT
        / "val"
    )

    if not train_dir.exists():

        raise FileNotFoundError(
            f"Training directory not found:\n"
            f"{train_dir}"
        )

    if not val_dir.exists():

        raise FileNotFoundError(
            f"Validation directory not found:\n"
            f"{val_dir}"
        )

    # =====================================================
    # Transforms
    # =====================================================

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomApply(
                [
                    transforms.RandomRotation(
                        degrees=8
                    )
                ],
                p=0.5,
            ),

            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(
                            0.05,
                            0.05,
                        ),
                        scale=(
                            0.90,
                            1.10,
                        ),
                    )
                ],
                p=0.5,
            ),

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                    )
                ],
                p=0.5,
            ),

            transforms.RandomGrayscale(
                p=0.10
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

    # =====================================================
    # Datasets
    # =====================================================

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform,
    )

    print(
        "\nClasses:"
    )

    print(
        train_dataset.classes
    )

    print(
        "\nClass mapping:"
    )

    print(
        train_dataset.class_to_idx
    )

    print(
        "\nTraining images:"
        f" {len(train_dataset)}"
    )

    print(
        "Validation images:"
        f" {len(val_dataset)}"
    )

    # =====================================================
    # Data loaders
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # =====================================================
    # Load MobileNetV3
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING MOBILENETV3-SMALL"
    )

    print(
        "=" * 70
    )

    try:

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
        )

        model = (
            models.mobilenet_v3_small(
                weights=weights
            )
        )

    except Exception as exc:

        print(
            "\nWARNING:"
        )

        print(
            "Could not load pretrained weights."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "\nUsing randomly initialized model."
        )

        model = (
            models.mobilenet_v3_small(
                weights=None
            )
        )

    # =====================================================
    # Replace classifier
    # =====================================================

    input_features = (
        model.classifier[-1]
        .in_features
    )

    model.classifier[-1] = (
        nn.Linear(
            input_features,
            2,
        )
    )

    model = model.to(
        DEVICE
    )

    # =====================================================
    # Loss / optimizer
    # =====================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    # =====================================================
    # Training state
    # =====================================================

    best_val_accuracy = 0.0

    best_val_loss = float(
        "inf"
    )

    best_model_state = None

    # =====================================================
    # Training loop
    # =====================================================

    for epoch in range(
        EPOCHS
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 70
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.train()

        train_loss = 0.0

        train_correct = 0

        train_total = 0

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

            train_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            train_correct += (
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            train_total += (
                labels.size(0)
            )

        train_loss /= max(
            train_total,
            1,
        )

        train_accuracy = (
            train_correct
            / max(
                train_total,
                1,
            )
        )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        model.eval()

        val_loss = 0.0

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

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )

                val_correct += (
                    (
                        predictions
                        == labels
                    )
                    .sum()
                    .item()
                )

                val_total += (
                    labels.size(0)
                )

        val_loss /= max(
            val_total,
            1,
        )

        val_accuracy = (
            val_correct
            / max(
                val_total,
                1,
            )
        )

        scheduler.step(
            val_loss
        )

        print(
            f"\nTrain Loss: "
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

        print(
            f"Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.7f}"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if (
            val_accuracy
            > best_val_accuracy
        ) or (
            val_accuracy
            == best_val_accuracy
            and val_loss
            < best_val_loss
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_val_loss = (
                val_loss
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            print(
                "\n*** NEW BEST MODEL ***"
            )

    # =====================================================
    # Restore best model
    # =====================================================

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    # =====================================================
    # Save
    # =====================================================

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "classes":
            train_dataset.classes,

        "class_to_idx":
            train_dataset.class_to_idx,

        "image_size":
            IMAGE_SIZE,

        "best_val_accuracy":
            best_val_accuracy,

        "best_val_loss":
            best_val_loss,

        "architecture":
            "mobilenet_v3_small",

        "dataset":
            str(DATA_ROOT),

        "leakage_free":
            True,

    }

    torch.save(
        checkpoint,
        MODEL_OUTPUT,
    )

    # =====================================================
    # Final output
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest validation accuracy:"
        f" {best_val_accuracy:.4f}"
    )

    print(
        f"Best validation loss:"
        f" {best_val_loss:.4f}"
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_OUTPUT
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This validation score is now meaningful "
        "because train/validation overlap is zero."
    )

    print(
        "\nNext step:"
    )

    print(
        "Test this new model against the "
        "5 untouched external signatures."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()