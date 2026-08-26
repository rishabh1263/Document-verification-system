"""
Production-oriented MobileNetV3 signature classifier training.

Dataset structure:

data/signature_classifier/
    train/
        signature/
        non_signature/
    val/
        signature/
        non_signature/

The training pipeline uses realistic augmentation so the model
does not learn only clean/high-contrast signatures.

Output:
    src/documents/signature/models/signature_classifier.pth
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# =========================================================
# Configuration
# =========================================================

DATASET_DIR = Path(
    "data/signature_classifier"
)

MODEL_PATH = Path(
    "src/documents/signature/models/signature_classifier.pth"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Transform
# =========================================================

NORMALIZE = transforms.Normalize(
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
)


def create_train_transform():

    return transforms.Compose(
        [

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),

            # Small geometric variation.
            transforms.RandomRotation(
                degrees=8,
                fill=255,
            ),

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
                shear=5,
                fill=255,
            ),

            # Simulate different scanning/camera conditions.
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.35,
                        contrast=0.45,
                        saturation=0.10,
                    )
                ],
                p=0.8,
            ),

            # Occasionally convert to grayscale.
            transforms.RandomGrayscale(
                p=0.25
            ),

            # Simulate camera/scanning blur.
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(
                            0.1,
                            1.5,
                        ),
                    )
                ],
                p=0.20,
            ),

            # Slightly different crops/scales.
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(
                    0.85,
                    1.0,
                ),
                ratio=(
                    0.75,
                    1.35,
                ),
            ),

            transforms.ToTensor(),

            # Tensor-level brightness/contrast variation.
            transforms.RandomApply(
                [
                    transforms.Lambda(
                        lambda x: torch.clamp(
                            x * random.uniform(
                                0.75,
                                1.10,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                ],
                p=0.35,
            ),

            NORMALIZE,
        ]
    )


def create_validation_transform():

    return transforms.Compose(
        [

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),

            transforms.ToTensor(),

            NORMALIZE,
        ]
    )


# =========================================================
# Dataset
# =========================================================

def create_datasets():

    train_dir = (
        DATASET_DIR
        / "train"
    )

    val_dir = (
        DATASET_DIR
        / "val"
    )

    if not train_dir.exists():

        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not val_dir.exists():

        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}"
        )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=create_train_transform(),
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=create_validation_transform(),
    )

    return (
        train_dataset,
        val_dataset,
    )


# =========================================================
# Model
# =========================================================

def create_model(
    number_of_classes: int,
):

    model = models.mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model
        .classifier[-1]
        .in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        number_of_classes,
    )

    return model


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    model,
    loader,
    criterion,
    class_names,
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    class_correct = {
        index: 0
        for index in range(
            len(class_names)
        )
    }

    class_total = {
        index: 0
        for index in range(
            len(class_names)
        )
    }

    with torch.inference_mode():

        for images, labels in loader:

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

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += (
                labels.size(0)
            )

            for class_index in range(
                len(class_names)
            ):

                mask = (
                    labels
                    == class_index
                )

                class_total[
                    class_index
                ] += mask.sum().item()

                class_correct[
                    class_index
                ] += (
                    (
                        predictions[mask]
                        == labels[mask]
                    )
                    .sum()
                    .item()
                )

    average_loss = (
        total_loss /
        max(total_samples, 1)
    )

    accuracy = (
        total_correct /
        max(total_samples, 1)
    )

    class_accuracy = {}

    for index, name in enumerate(
        class_names
    ):

        class_accuracy[name] = (
            class_correct[index]
            /
            max(
                class_total[index],
                1,
            )
        )

    return (
        average_loss,
        accuracy,
        class_accuracy,
    )


# =========================================================
# Training
# =========================================================

def main():

    set_seed()

    print("=" * 70)
    print("SIGNATURE CLASSIFIER TRAINING")
    print("=" * 70)

    print()
    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Dataset: {DATASET_DIR}"
    )

    print()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    (
        train_dataset,
        val_dataset,
    ) = create_datasets()

    print(
        f"Training samples   : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(val_dataset)}"
    )

    print(
        f"Classes            : {train_dataset.classes}"
    )

    # -----------------------------------------------------
    # Verify classes
    # -----------------------------------------------------

    required_classes = {
        "non_signature",
        "signature",
    }

    if set(
        train_dataset.classes
    ) != required_classes:

        raise RuntimeError(
            "Dataset must contain exactly "
            "'non_signature' and 'signature'. "
            f"Found: {train_dataset.classes}"
        )

    if (
        train_dataset.classes
        != val_dataset.classes
    ):

        raise RuntimeError(
            "Training and validation classes "
            "do not match."
        )

    # -----------------------------------------------------
    # Data loaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = create_model(
        len(
            train_dataset.classes
        )
    )

    model = model.to(
        DEVICE
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Training state
    # -----------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_val_accuracy = 0.0

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Epochs
    # =====================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        model.train()

        running_loss = 0.0

        running_correct = 0

        running_samples = 0

        for images, labels in (
            train_loader
        ):

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

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
                * labels.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            running_correct += (
                predictions == labels
            ).sum().item()

            running_samples += (
                labels.size(0)
            )

        train_loss = (
            running_loss /
            max(running_samples, 1)
        )

        train_accuracy = (
            running_correct /
            max(running_samples, 1)
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        (
            val_loss,
            val_accuracy,
            class_accuracy,
        ) = calculate_metrics(
            model,
            val_loader,
            criterion,
            train_dataset.classes,
        )

        signature_accuracy = (
            class_accuracy[
                "signature"
            ]
        )

        non_signature_accuracy = (
            class_accuracy[
                "non_signature"
            ]
        )

        # -------------------------------------------------
        # Print
        # -------------------------------------------------

        print(
            f"Train loss          : "
            f"{train_loss:.6f}"
        )

        print(
            f"Train accuracy      : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation loss     : "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation accuracy : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Signature accuracy  : "
            f"{signature_accuracy:.4f}"
        )

        print(
            f"Non-signature acc   : "
            f"{non_signature_accuracy:.4f}"
        )

        # -------------------------------------------------
        # Save best model
        #
        # Use validation LOSS, not just accuracy.
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            best_val_accuracy = (
                val_accuracy
            )

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "best_val_loss":
                    best_val_loss,

                "best_val_accuracy":
                    best_val_accuracy,

                "architecture":
                    "mobilenet_v3_small",

                "image_size":
                    IMAGE_SIZE,

                "dataset":
                    str(DATASET_DIR),

                "augmentation":
                    "real_world_signature_v1",
            }

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

            print()
            print(
                ">>> BEST MODEL SAVED"
            )

            print(
                f">>> {MODEL_PATH}"
            )

    # =====================================================
    # Complete
    # =====================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation accuracy : "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best validation loss     : "
        f"{best_val_loss:.6f}"
    )

    print()
    print(
        f"Model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        f"Classes:"
    )

    print(
        train_dataset.classes
    )

    print()
    print(
        "Next:"
    )

    print(
        "Run the Google signature test."
    )


if __name__ == "__main__":

    main()