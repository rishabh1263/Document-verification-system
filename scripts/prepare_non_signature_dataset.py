from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io
import random
import shutil


# =========================================================
# Configuration
# =========================================================

DATASET_NAME = "Mels22/SigDetectVerifyFlow"

OUTPUT_DIR = Path(
    "data/signature_classifier"
)

TOTAL_SAMPLES = 3000

TRAIN_RATIO = 0.80

SEED = 42


# =========================================================
# Save image
# =========================================================

def save_image(
    image,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not isinstance(
        image,
        Image.Image,
    ):
        image = Image.open(
            io.BytesIO(image)
        )

    image = image.convert(
        "RGB"
    )

    image.save(
        path,
        format="JPEG",
        quality=90,
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("NON-SIGNATURE DATASET PREPARATION")
    print("=" * 60)

    print()
    print(
        f"Loading dataset: {DATASET_NAME}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    # =====================================================
    # Collect document images
    # =====================================================

    samples = []

    print()
    print(
        "Finding document images..."
    )

    for index, row in enumerate(
        dataset
    ):

        document = row.get(
            "document"
        )

        if document is None:
            continue

        samples.append(
            (
                index,
                document,
            )
        )

        if len(samples) >= TOTAL_SAMPLES:
            break

    print()
    print(
        f"Documents selected: "
        f"{len(samples)}"
    )

    if not samples:

        raise RuntimeError(
            "No document images found."
        )

    # =====================================================
    # Shuffle
    # =====================================================

    random.seed(
        SEED
    )

    random.shuffle(
        samples
    )

    # =====================================================
    # Train / validation
    # =====================================================

    split_index = int(
        len(samples)
        * TRAIN_RATIO
    )

    train_samples = (
        samples[:split_index]
    )

    val_samples = (
        samples[split_index:]
    )

    print(
        f"Train samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_samples)}"
    )

    # =====================================================
    # Directories
    # =====================================================

    train_dir = (
        OUTPUT_DIR
        / "train"
        / "non_signature"
    )

    val_dir = (
        OUTPUT_DIR
        / "val"
        / "non_signature"
    )

    # Remove old negative dataset only
    if train_dir.exists():
        shutil.rmtree(train_dir)

    if val_dir.exists():
        shutil.rmtree(val_dir)

    train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Save training documents
    # =====================================================

    print()
    print(
        "Saving non-signature training images..."
    )

    for count, (
        index,
        image,
    ) in enumerate(
        train_samples,
        start=1,
    ):

        save_image(
            image,
            train_dir
            / f"document_{count:05d}.jpg",
        )

        if count % 500 == 0:

            print(
                f"  Saved {count}/"
                f"{len(train_samples)}"
            )

    # =====================================================
    # Save validation documents
    # =====================================================

    print()
    print(
        "Saving non-signature validation images..."
    )

    for count, (
        index,
        image,
    ) in enumerate(
        val_samples,
        start=1,
    ):

        save_image(
            image,
            val_dir
            / f"document_{count:05d}.jpg",
        )

        if count % 500 == 0:

            print(
                f"  Saved {count}/"
                f"{len(val_samples)}"
            )

    # =====================================================
    # Result
    # =====================================================

    print()
    print("=" * 60)
    print("NON-SIGNATURE DATASET COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Train: {train_dir}"
    )

    print(
        f"Validation: {val_dir}"
    )

    print()
    print(
        f"Train non-signatures: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation non-signatures: "
        f"{len(val_samples)}"
    )

    print()
    print(
        "Dataset is now balanced:"
    )

    print(
        "2400 signature + 2400 non-signature"
    )

    print(
        "600 signature + 600 non-signature"
    )

    print()
    print(
        "Next step: train the classifier."
    )


if __name__ == "__main__":
    main()