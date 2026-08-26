"""
Build a leakage-free signature classifier dataset.

Problem being fixed
-------------------
The original dataset contains many duplicate files.

More importantly, the same signature images exist in both:

    train/signature
    val/signature

This creates train/validation leakage.

This script:

1. Reads the existing dataset.
2. Removes duplicate image content using SHA-256.
3. Combines train + validation before splitting.
4. Creates a NEW 80/20 split.
5. Keeps signature/non-signature classes balanced.
6. Guarantees no identical image exists in both splits.
7. Does NOT touch external_test.
8. Does NOT train the model.

Output:

    data/signature_classifier_clean/
        train/
            signature/
            non_signature/
        val/
            signature/
            non_signature/
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
    "data/signature_classifier_clean"
)

TRAIN_RATIO = 0.80

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# =========================================================
# Source directories
# =========================================================

SOURCE_TRAIN_SIGNATURE = (
    SOURCE_ROOT
    / "train"
    / "signature"
)

SOURCE_VAL_SIGNATURE = (
    SOURCE_ROOT
    / "val"
    / "signature"
)

SOURCE_TRAIN_NON_SIGNATURE = (
    SOURCE_ROOT
    / "train"
    / "non_signature"
)

SOURCE_VAL_NON_SIGNATURE = (
    SOURCE_ROOT
    / "val"
    / "non_signature"
)


# =========================================================
# Output directories
# =========================================================

OUTPUT_TRAIN_SIGNATURE = (
    OUTPUT_ROOT
    / "train"
    / "signature"
)

OUTPUT_VAL_SIGNATURE = (
    OUTPUT_ROOT
    / "val"
    / "signature"
)

OUTPUT_TRAIN_NON_SIGNATURE = (
    OUTPUT_ROOT
    / "train"
    / "non_signature"
)

OUTPUT_VAL_NON_SIGNATURE = (
    OUTPUT_ROOT
    / "val"
    / "non_signature"
)


# =========================================================
# Find images
# =========================================================

def find_images(
    directory: Path,
) -> list[Path]:

    if not directory.exists():

        return []

    images = []

    for path in directory.rglob("*"):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(
                path
            )

    return images


# =========================================================
# Hash image
# =========================================================

def calculate_hash(
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


# =========================================================
# Deduplicate
# =========================================================

def deduplicate_images(
    images: list[Path],
) -> list[Path]:

    unique = []

    seen_hashes = set()

    for image in images:

        try:

            image_hash = (
                calculate_hash(
                    image
                )
            )

        except Exception as exc:

            print(
                f"WARNING: Could not hash "
                f"{image}: {exc}"
            )

            continue

        if image_hash in seen_hashes:

            continue

        seen_hashes.add(
            image_hash
        )

        unique.append(
            image
        )

    return unique


# =========================================================
# Prepare output directory
# =========================================================

def prepare_output():

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous clean dataset..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_TRAIN_SIGNATURE.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_VAL_SIGNATURE.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_TRAIN_NON_SIGNATURE.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_VAL_NON_SIGNATURE.mkdir(
        parents=True,
        exist_ok=True,
    )


# =========================================================
# Split images
# =========================================================

def split_images(
    images: list[Path],
) -> tuple[
    list[Path],
    list[Path],
]:

    images = list(
        images
    )

    random.shuffle(
        images
    )

    split_index = int(
        len(images)
        * TRAIN_RATIO
    )

    train_images = images[
        :split_index
    ]

    val_images = images[
        split_index:
    ]

    return (
        train_images,
        val_images,
    )


# =========================================================
# Copy images
# =========================================================

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
            / f"{prefix}_{index:05d}"
            f"{extension}"
        )

        shutil.copy2(
            source,
            target,
        )


# =========================================================
# Verify no overlap
# =========================================================

def get_directory_hashes(
    directory: Path,
) -> set[str]:

    hashes = set()

    for image in find_images(
        directory
    ):

        try:

            hashes.add(
                calculate_hash(
                    image
                )
            )

        except Exception:

            continue

    return hashes


def verify_no_overlap(
    train_directory: Path,
    val_directory: Path,
):

    train_hashes = (
        get_directory_hashes(
            train_directory
        )
    )

    val_hashes = (
        get_directory_hashes(
            val_directory
        )
    )

    overlap = (
        train_hashes
        & val_hashes
    )

    print(
        "\nLeakage verification:"
    )

    print(
        f"Train unique hashes      : "
        f"{len(train_hashes)}"
    )

    print(
        f"Validation unique hashes : "
        f"{len(val_hashes)}"
    )

    print(
        f"Train/validation overlap : "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "identical images exist in "
            "both train and validation."
        )

    print(
        "NO TRAIN/VALIDATION IMAGE OVERLAP"
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
        "BUILDING LEAKAGE-FREE SIGNATURE DATASET"
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
    # 1. Load ALL signatures
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING SIGNATURE IMAGES"
    )

    print(
        "=" * 70
    )

    train_signature = (
        find_images(
            SOURCE_TRAIN_SIGNATURE
        )
    )

    val_signature = (
        find_images(
            SOURCE_VAL_SIGNATURE
        )
    )

    all_signature = (
        train_signature
        + val_signature
    )

    print(
        f"\nRaw signature files:"
    )

    print(
        f"  Train: {len(train_signature)}"
    )

    print(
        f"  Val  : {len(val_signature)}"
    )

    print(
        f"  Total: {len(all_signature)}"
    )

    # =====================================================
    # 2. Deduplicate signatures
    # =====================================================

    print(
        "\nRemoving duplicate signatures..."
    )

    all_signature = (
        deduplicate_images(
            all_signature
        )
    )

    print(
        f"Unique signatures: "
        f"{len(all_signature)}"
    )

    # =====================================================
    # 3. Load ALL non-signatures
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING NON-SIGNATURE IMAGES"
    )

    print(
        "=" * 70
    )

    train_non_signature = (
        find_images(
            SOURCE_TRAIN_NON_SIGNATURE
        )
    )

    val_non_signature = (
        find_images(
            SOURCE_VAL_NON_SIGNATURE
        )
    )

    all_non_signature = (
        train_non_signature
        + val_non_signature
    )

    print(
        f"\nRaw non-signature files:"
    )

    print(
        f"  Train: "
        f"{len(train_non_signature)}"
    )

    print(
        f"  Val  : "
        f"{len(val_non_signature)}"
    )

    print(
        f"  Total: "
        f"{len(all_non_signature)}"
    )

    # =====================================================
    # 4. Deduplicate non-signatures
    # =====================================================

    print(
        "\nRemoving duplicate non-signatures..."
    )

    all_non_signature = (
        deduplicate_images(
            all_non_signature
        )
    )

    print(
        f"Unique non-signatures: "
        f"{len(all_non_signature)}"
    )

    # =====================================================
    # 5. Balance classes BEFORE splitting
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BALANCING CLASSES"
    )

    print(
        "=" * 70
    )

    class_count = min(
        len(all_signature),
        len(all_non_signature),
    )

    random.shuffle(
        all_signature
    )

    random.shuffle(
        all_non_signature
    )

    all_signature = (
        all_signature[
            :class_count
        ]
    )

    all_non_signature = (
        all_non_signature[
            :class_count
        ]
    )

    print(
        f"Balanced signature count    : "
        f"{len(all_signature)}"
    )

    print(
        f"Balanced non-signature count: "
        f"{len(all_non_signature)}"
    )

    # =====================================================
    # 6. Split AFTER deduplication
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CREATING 80/20 SPLIT"
    )

    print(
        "=" * 70
    )

    (
        train_signature,
        val_signature,
    ) = split_images(
        all_signature
    )

    (
        train_non_signature,
        val_non_signature,
    ) = split_images(
        all_non_signature
    )

    print(
        "\nSignature:"
    )

    print(
        f"  Train: "
        f"{len(train_signature)}"
    )

    print(
        f"  Val  : "
        f"{len(val_signature)}"
    )

    print(
        "\nNon-signature:"
    )

    print(
        f"  Train: "
        f"{len(train_non_signature)}"
    )

    print(
        f"  Val  : "
        f"{len(val_non_signature)}"
    )

    # =====================================================
    # 7. Prepare output
    # =====================================================

    prepare_output()

    # =====================================================
    # 8. Copy training data
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COPYING TRAINING DATA"
    )

    print(
        "=" * 70
    )

    copy_images(
        train_signature,
        OUTPUT_TRAIN_SIGNATURE,
        "signature",
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
        "\n"
        + "=" * 70
    )

    print(
        "COPYING VALIDATION DATA"
    )

    print(
        "=" * 70
    )

    copy_images(
        val_signature,
        OUTPUT_VAL_SIGNATURE,
        "signature",
    )

    copy_images(
        val_non_signature,
        OUTPUT_VAL_NON_SIGNATURE,
        "non_signature",
    )

    # =====================================================
    # 10. Verify leakage
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VERIFYING DATASET"
    )

    print(
        "=" * 70
    )

    verify_no_overlap(
        OUTPUT_TRAIN_SIGNATURE,
        OUTPUT_VAL_SIGNATURE,
    )

    verify_no_overlap(
        OUTPUT_TRAIN_NON_SIGNATURE,
        OUTPUT_VAL_NON_SIGNATURE,
    )

    # =====================================================
    # 11. Final summary
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LEAKAGE-FREE DATASET READY"
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
        "\nExternal test remains untouched:"
    )

    print(
        "data\\external_test\\signatures"
    )

    print(
        "\nDO NOT TRAIN YET."
    )

    print(
        "Send the output above first."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()