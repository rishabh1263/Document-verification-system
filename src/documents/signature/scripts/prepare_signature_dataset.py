from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io
import random


DATASET_NAME = "Mels22/SigDetectVerifyFlow"

OUTPUT_DIR = Path("data/signature_classifier")

TRAIN_RATIO = 0.8
SEED = 42


def save_image(image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image))

    image = image.convert("RGB")
    image.save(path, format="JPEG", quality=95)


def main():

    print("=" * 60)
    print("Downloading signature dataset")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    samples = []

    for index, row in enumerate(dataset):

        signature = row.get("to_verify_signature")

        if signature is None:
            continue

        samples.append(
            (
                index,
                signature,
            )
        )

    print(f"Usable signature samples: {len(samples)}")

    if not samples:
        raise RuntimeError(
            "No signature images found in dataset."
        )

    random.seed(SEED)
    random.shuffle(samples)

    split_index = int(
        len(samples) * TRAIN_RATIO
    )

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

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

    for index, image in train_samples:

        save_image(
            image,
            train_dir / f"signature_{index:06d}.jpg",
        )

    for index, image in val_samples:

        save_image(
            image,
            val_dir / f"signature_{index:06d}.jpg",
        )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Train: {train_dir}")
    print(f"Val:   {val_dir}")

    print()
    print(
        "IMPORTANT: These are POSITIVE signature samples."
    )
    print(
        "We still need NON-SIGNATURE samples before training."
    )


if __name__ == "__main__":
    main()