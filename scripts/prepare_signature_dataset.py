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
        quality=95,
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("SIGNATURE DATASET PREPARATION")
    print("=" * 60)

    print()
    print(
        f"Dataset: {DATASET_NAME}"
    )

    print(
        f"Target samples: {TOTAL_SAMPLES}"
    )

    # =====================================================
    # Clean previous generated dataset
    # =====================================================

    if OUTPUT_DIR.exists():

        print()
        print(
            "Removing previous generated dataset..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    # =====================================================
    # Download dataset
    # =====================================================

    print()
    print(
        "Loading Hugging Face dataset..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print()
    print(
        f"Dataset size: {len(dataset)}"
    )

    print(
        f"Columns: {dataset.column_names}"
    )

    # =====================================================
    # Collect signature samples
    # =====================================================

    samples = []

    print()
    print(
        "Finding usable signature samples..."
    )

    for index, row in enumerate(
        dataset
    ):

        signature = row.get(
            "to_verify_signature"
        )

        if signature is None:
            continue

        samples.append(
            (
                index,
                signature,
            )
        )

        if len(samples) >= TOTAL_SAMPLES:
            break

    # =====================================================
    # Validate
    # =====================================================

    print()

    print(
        f"Usable samples selected: "
        f"{len(samples)}"
    )

    if not samples:

        raise RuntimeError(
            "No signature samples found."
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
    # Train / validation split
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
        / "signature"
    )

    val_dir = (
        OUTPUT_DIR
        / "val"
        / "signature"
    )

    train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Save training images
    # =====================================================

    print()
    print(
        "Saving training images..."
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
            / f"signature_{count:05d}.jpg",
        )

        if count % 500 == 0:

            print(
                f"  Saved {count}/"
                f"{len(train_samples)}"
            )

    # =====================================================
    # Save validation images
    # =====================================================

    print()
    print(
        "Saving validation images..."
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
            / f"signature_{count:05d}.jpg",
        )

        if count % 500 == 0:

            print(
                f"  Saved {count}/"
                f"{len(val_samples)}"
            )

    # =====================================================
    # Final result
    # =====================================================

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Train directory:"
    )

    print(
        train_dir
    )

    print()
    print(
        f"Validation directory:"
    )

    print(
        val_dir
    )

    print()
    print(
        "Train signatures:",
        len(train_samples),
    )

    print(
        "Validation signatures:",
        len(val_samples),
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are POSITIVE signature samples."
    )

    print(
        "Do not train the classifier yet."
    )

    print(
        "Next we will create the NON-SIGNATURE class."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()