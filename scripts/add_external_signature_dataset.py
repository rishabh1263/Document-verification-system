"""
Add an external signature dataset to the local classifier dataset.

Source:
    rakshitdabral/Signature-Verification-Dataset

The dataset contains handwritten signatures organized by
signer identity. Every image is converted into the positive
"signature" class.

IMPORTANT:
    The external dataset is NOT used as a non-signature class.

It is used only to increase signature diversity.
"""

from pathlib import Path
import random
import shutil

from datasets import load_dataset


# =========================================================
# Configuration
# =========================================================

DATASET_NAME = (
    "rakshitdabral/Signature-Verification-Dataset"
)

OUTPUT_ROOT = Path(
    "data/signature_classifier_external"
)

TRAIN_DIR = (
    OUTPUT_ROOT
    / "train"
    / "signature"
)

VAL_DIR = (
    OUTPUT_ROOT
    / "val"
    / "signature"
)

# Keep the external validation set separate.
# This allows us to test generalization.
TRAIN_LIMIT = 4000
VAL_LIMIT = 1000

SEED = 42


# =========================================================
# Helpers
# =========================================================

def save_image(
    image,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path
    )


# =========================================================
# Main
# =========================================================

def main():

    random.seed(
        SEED
    )

    print(
        "=" * 60
    )

    print(
        "DOWNLOADING EXTERNAL SIGNATURE DATASET"
    )

    print(
        "=" * 60
    )

    print(
        f"Dataset: {DATASET_NAME}"
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME
    )

    print(
        "\nAvailable splits:"
    )

    print(
        list(dataset.keys())
    )

    # -----------------------------------------------------
    # Use train split as source.
    # The dataset itself is already organized by signer.
    # -----------------------------------------------------

    source = dataset[
        "train"
    ]

    print(
        f"\nSource samples: {len(source)}"
    )

    # -----------------------------------------------------
    # Shuffle indices.
    # -----------------------------------------------------

    indices = list(
        range(
            len(source)
        )
    )

    random.shuffle(
        indices
    )

    # -----------------------------------------------------
    # Select samples.
    # -----------------------------------------------------

    total_required = (
        TRAIN_LIMIT
        +
        VAL_LIMIT
    )

    if len(indices) < total_required:

        total_required = len(
            indices
        )

    selected = indices[
        :total_required
    ]

    train_indices = selected[
        :TRAIN_LIMIT
    ]

    val_indices = selected[
        TRAIN_LIMIT:
        TRAIN_LIMIT + VAL_LIMIT
    ]

    # -----------------------------------------------------
    # Prepare directories.
    # -----------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous external dataset..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save training images.
    # -----------------------------------------------------

    print(
        "\nPreparing external training signatures..."
    )

    saved_train = 0

    for position, index in enumerate(
        train_indices,
        start=1,
    ):

        sample = source[
            index
        ]

        image = sample[
            "image"
        ]

        output_path = (
            TRAIN_DIR
            / f"external_train_{position:05d}.png"
        )

        save_image(
            image,
            output_path,
        )

        saved_train += 1

        if saved_train % 500 == 0:

            print(
                f"  Saved train: "
                f"{saved_train}/{len(train_indices)}"
            )

    # -----------------------------------------------------
    # Save validation images.
    # -----------------------------------------------------

    print(
        "\nPreparing external validation signatures..."
    )

    saved_val = 0

    for position, index in enumerate(
        val_indices,
        start=1,
    ):

        sample = source[
            index
        ]

        image = sample[
            "image"
        ]

        output_path = (
            VAL_DIR
            / f"external_val_{position:05d}.png"
        )

        save_image(
            image,
            output_path,
        )

        saved_val += 1

        if saved_val % 250 == 0:

            print(
                f"  Saved validation: "
                f"{saved_val}/{len(val_indices)}"
            )

    # -----------------------------------------------------
    # Summary.
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EXTERNAL DATASET PREPARATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTraining signatures: "
        f"{saved_train}"
    )

    print(
        f"Validation signatures: "
        f"{saved_val}"
    )

    print(
        "\nTraining directory:"
    )

    print(
        TRAIN_DIR
    )

    print(
        "\nValidation directory:"
    )

    print(
        VAL_DIR
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are POSITIVE signature samples only."
    )

    print(
        "They have NOT been merged into the classifier yet."
    )

    print(
        "The next step is to combine them with the existing"
    )

    print(
        "training dataset and retrain MobileNetV3."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()