"""
Build a clean signature classifier dataset.

Purpose
-------
Create a new training dataset from the existing prepared dataset
without modifying the original dataset.

Existing source:

    data/signature_classifier/
        train/
            signature/
            non_signature/
        val/
            signature/
            non_signature/

External test images are intentionally NOT included.

Output:

    data/signature_classifier_v2/
        train/
            signature/
            non_signature/
        val/
            signature/
            non_signature/

Design
------
1. Keep original training data.
2. Keep original validation data separate.
3. Do NOT use external_test images.
4. Remove duplicate filenames.
5. Create a balanced dataset.
6. Copy files instead of moving them.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path


# =========================================================
# Configuration
# =========================================================

SOURCE_ROOT = Path(
    "data/signature_classifier"
)

OUTPUT_ROOT = Path(
    "data/signature_classifier_v2"
)

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# =========================================================
# Dataset paths
# =========================================================

SOURCE_TRAIN_SIGNATURE = (
    SOURCE_ROOT
    / "train"
    / "signature"
)

SOURCE_TRAIN_NON_SIGNATURE = (
    SOURCE_ROOT
    / "train"
    / "non_signature"
)

SOURCE_VAL_SIGNATURE = (
    SOURCE_ROOT
    / "val"
    / "signature"
)

SOURCE_VAL_NON_SIGNATURE = (
    SOURCE_ROOT
    / "val"
    / "non_signature"
)


OUTPUT_TRAIN_SIGNATURE = (
    OUTPUT_ROOT
    / "train"
    / "signature"
)

OUTPUT_TRAIN_NON_SIGNATURE = (
    OUTPUT_ROOT
    / "train"
    / "non_signature"
)

OUTPUT_VAL_SIGNATURE = (
    OUTPUT_ROOT
    / "val"
    / "signature"
)

OUTPUT_VAL_NON_SIGNATURE = (
    OUTPUT_ROOT
    / "val"
    / "non_signature"
)


# =========================================================
# Helpers
# =========================================================

def find_images(
    directory: Path,
) -> list[Path]:

    if not directory.exists():

        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )

    images = []

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(path)

    return sorted(
        images
    )


def file_hash(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def remove_duplicate_files(
    images: list[Path],
) -> list[Path]:

    unique = []

    hashes = set()

    for image in images:

        try:

            digest = file_hash(
                image
            )

        except Exception:

            continue

        if digest in hashes:
            continue

        hashes.add(
            digest
        )

        unique.append(
            image
        )

    return unique


def prepare_directory(
    directory: Path,
):

    if directory.exists():

        shutil.rmtree(
            directory
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_images(
    images: list[Path],
    destination: Path,
    prefix: str,
):

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, source in enumerate(
        images,
        start=1,
    ):

        extension = (
            source.suffix.lower()
        )

        target = (
            destination
            / f"{prefix}_{index:05d}{extension}"
        )

        shutil.copy2(
            source,
            target,
        )


def print_count(
    label: str,
    images: list[Path],
):

    print(
        f"{label:<35}: {len(images)}"
    )


# =========================================================
# Main
# =========================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    print(
        "=" * 70
    )

    print(
        "BUILDING SIGNATURE CLASSIFIER DATASET V2"
    )

    print(
        "=" * 70
    )

    print(
        "\nSource:"
    )

    print(
        SOURCE_ROOT
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )

    # =====================================================
    # 1. Load existing training data
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING TRAINING DATA"
    )

    print(
        "=" * 70
    )

    train_signature = (
        find_images(
            SOURCE_TRAIN_SIGNATURE
        )
    )

    train_non_signature = (
        find_images(
            SOURCE_TRAIN_NON_SIGNATURE
        )
    )

    print_count(
        "Signature training images",
        train_signature,
    )

    print_count(
        "Non-signature training images",
        train_non_signature,
    )

    # =====================================================
    # 2. Remove duplicates
    # =====================================================

    print(
        "\nRemoving duplicate files..."
    )

    train_signature = (
        remove_duplicate_files(
            train_signature
        )
    )

    train_non_signature = (
        remove_duplicate_files(
            train_non_signature
        )
    )

    print_count(
        "Unique signature images",
        train_signature,
    )

    print_count(
        "Unique non-signature images",
        train_non_signature,
    )

    # =====================================================
    # 3. Balance training classes
    # =====================================================

    train_count = min(
        len(train_signature),
        len(train_non_signature),
    )

    random.shuffle(
        train_signature
    )

    random.shuffle(
        train_non_signature
    )

    train_signature = (
        train_signature[
            :train_count
        ]
    )

    train_non_signature = (
        train_non_signature[
            :train_count
        ]
    )

    # =====================================================
    # 4. Load validation data
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING VALIDATION DATA"
    )

    print(
        "=" * 70
    )

    val_signature = (
        find_images(
            SOURCE_VAL_SIGNATURE
        )
    )

    val_non_signature = (
        find_images(
            SOURCE_VAL_NON_SIGNATURE
        )
    )

    print_count(
        "Signature validation images",
        val_signature,
    )

    print_count(
        "Non-signature validation images",
        val_non_signature,
    )

    # =====================================================
    # 5. Remove duplicates from validation
    # =====================================================

    val_signature = (
        remove_duplicate_files(
            val_signature
        )
    )

    val_non_signature = (
        remove_duplicate_files(
            val_non_signature
        )
    )

    # =====================================================
    # 6. Balance validation
    # =====================================================

    val_count = min(
        len(val_signature),
        len(val_non_signature),
    )

    random.shuffle(
        val_signature
    )

    random.shuffle(
        val_non_signature
    )

    val_signature = (
        val_signature[
            :val_count
        ]
    )

    val_non_signature = (
        val_non_signature[
            :val_count
        ]
    )

    # =====================================================
    # 7. Clean output
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PREPARING OUTPUT DIRECTORIES"
    )

    print(
        "=" * 70
    )

    prepare_directory(
        OUTPUT_ROOT
    )

    # =====================================================
    # 8. Copy training data
    # =====================================================

    print(
        "\nCopying training signatures..."
    )

    copy_images(
        train_signature,
        OUTPUT_TRAIN_SIGNATURE,
        "signature",
    )

    print(
        "Copying training non-signatures..."
    )

    copy_images(
        train_non_signature,
        OUTPUT_TRAIN_NON_SIGNATURE,
        "non_signature",
    )

    # =====================================================
    # 9. Copy validation data
    # =====================================================

    print(
        "Copying validation signatures..."
    )

    copy_images(
        val_signature,
        OUTPUT_VAL_SIGNATURE,
        "signature",
    )

    print(
        "Copying validation non-signatures..."
    )

    copy_images(
        val_non_signature,
        OUTPUT_VAL_NON_SIGNATURE,
        "non_signature",
    )

    # =====================================================
    # 10. Final summary
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATASET V2 READY"
    )

    print(
        "=" * 70
    )

    print(
        "\nTRAIN"
    )

    print(
        f"  Signature     : "
        f"{len(train_signature)}"
    )

    print(
        f"  Non-signature : "
        f"{len(train_non_signature)}"
    )

    print(
        "\nVALIDATION"
    )

    print(
        f"  Signature     : "
        f"{len(val_signature)}"
    )

    print(
        f"  Non-signature : "
        f"{len(val_non_signature)}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nExternal test images remain untouched:"
    )

    print(
        "data/external_test/signatures"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script only reorganizes the existing dataset."
    )

    print(
        "It does NOT add the failed Google signatures."
    )

    print(
        "It does NOT train the model."
    )

    print(
        "Do NOT retrain until the dataset counts are verified."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()