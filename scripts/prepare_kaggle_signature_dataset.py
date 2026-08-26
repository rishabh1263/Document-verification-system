"""
Prepare an external handwritten-signature dataset.

Source:
    Kaggle - Handwritten Signature Verification
    https://www.kaggle.com/datasets/tienen/handwritten-signature-verification

Expected downloaded structure:

    handwritten-signature-verification/
        real/
        fake/

Only REAL signatures are copied into the external
positive-signature dataset.

The fake class is deliberately not used here because
our current MobileNet model is a signature-vs-document
classifier, not a genuine-vs-forged signature verifier.
"""

from pathlib import Path
import random
import shutil


# =========================================================
# Configuration
# =========================================================

# Change this only if you download the Kaggle dataset
# somewhere else.
SOURCE_ROOT = Path(
    "data/external/kaggle_signature"
)

REAL_ROOT = (
    SOURCE_ROOT / "real"
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

TRAIN_LIMIT = 3000

VAL_LIMIT = 1000

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# Find images
# =========================================================

def find_images(
    root: Path,
) -> list[Path]:

    if not root.exists():

        raise FileNotFoundError(
            f"Real signature directory not found:\n"
            f"{root}\n\n"
            "Download the Kaggle dataset and extract "
            "the 'real' folder there first."
        )

    images = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS:

            images.append(path)

    return images


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
        "PREPARING EXTERNAL SIGNATURE DATASET"
    )

    print(
        "=" * 60
    )

    print(
        f"Source:\n{REAL_ROOT}"
    )

    # -----------------------------------------------------
    # Find real signatures
    # -----------------------------------------------------

    images = find_images(
        REAL_ROOT
    )

    print(
        f"\nFound real signature images: "
        f"{len(images)}"
    )

    if not images:

        raise RuntimeError(
            "No signature images were found."
        )

    # -----------------------------------------------------
    # Shuffle
    # -----------------------------------------------------

    random.shuffle(
        images
    )

    required = (
        TRAIN_LIMIT
        +
        VAL_LIMIT
    )

    if len(images) < required:

        required = len(
            images
        )

    selected = images[
        :required
    ]

    train_images = selected[
        :TRAIN_LIMIT
    ]

    val_images = selected[
        TRAIN_LIMIT:
        TRAIN_LIMIT + VAL_LIMIT
    ]

    # -----------------------------------------------------
    # Clean previous output
    # -----------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous output..."
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
    # Copy training signatures
    # -----------------------------------------------------

    print(
        "\nCopying training signatures..."
    )

    for index, source in enumerate(
        train_images,
        start=1,
    ):

        destination = (
            TRAIN_DIR
            / f"external_train_{index:05d}"
            f"{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            destination,
        )

        if index % 500 == 0:

            print(
                f"  Copied "
                f"{index}/{len(train_images)}"
            )

    # -----------------------------------------------------
    # Copy validation signatures
    # -----------------------------------------------------

    print(
        "\nCopying validation signatures..."
    )

    for index, source in enumerate(
        val_images,
        start=1,
    ):

        destination = (
            VAL_DIR
            / f"external_val_{index:05d}"
            f"{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            destination,
        )

        if index % 250 == 0:

            print(
                f"  Copied "
                f"{index}/{len(val_images)}"
            )

    # -----------------------------------------------------
    # Summary
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
        f"\nExternal training signatures: "
        f"{len(train_images)}"
    )

    print(
        f"External validation signatures: "
        f"{len(val_images)}"
    )

    print(
        "\nTraining:"
    )

    print(
        TRAIN_DIR
    )

    print(
        "\nValidation:"
    )

    print(
        VAL_DIR
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Only REAL signatures were copied."
    )

    print(
        "FAKE signatures were intentionally excluded."
    )

    print(
        "\nDo NOT retrain yet."
    )

    print(
        "First verify that the files were copied correctly."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()