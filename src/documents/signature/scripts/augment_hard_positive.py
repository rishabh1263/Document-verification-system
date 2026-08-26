from pathlib import Path
import random

import cv2
import numpy as np


SOURCE = Path(
    "data/signature_classifier/test_signature_google.jpg"
)

OUTPUT_DIR = Path(
    "data/signature_classifier/hard_positive/google"
)

COUNT = 100

SEED = 42


def random_variant(image: np.ndarray) -> np.ndarray:

    result = image.copy()

    # --------------------------------------------------
    # Brightness / contrast
    # --------------------------------------------------

    alpha = random.uniform(
        0.75,
        1.25,
    )

    beta = random.randint(
        -25,
        15,
    )

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=beta,
    )

    # --------------------------------------------------
    # Small rotation
    # --------------------------------------------------

    height, width = result.shape[:2]

    angle = random.uniform(
        -6,
        6,
    )

    matrix = cv2.getRotationMatrix2D(
        (
            width / 2,
            height / 2,
        ),
        angle,
        random.uniform(
            0.90,
            1.10,
        ),
    )

    result = cv2.warpAffine(
        result,
        matrix,
        (
            width,
            height,
        ),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # --------------------------------------------------
    # Slight blur
    # --------------------------------------------------

    if random.random() < 0.35:

        kernel = random.choice(
            [
                3,
                5,
            ]
        )

        result = cv2.GaussianBlur(
            result,
            (
                kernel,
                kernel,
            ),
            random.uniform(
                0.2,
                1.2,
            ),
        )

    # --------------------------------------------------
    # Slight resize variation
    # --------------------------------------------------

    scale = random.uniform(
        0.85,
        1.15,
    )

    new_width = max(
        32,
        int(width * scale),
    )

    new_height = max(
        32,
        int(height * scale),
    )

    result = cv2.resize(
        result,
        (
            new_width,
            new_height,
        ),
        interpolation=random.choice(
            [
                cv2.INTER_AREA,
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
            ]
        ),
    )

    # Return to original canvas size.

    canvas = np.full(
        (
            height,
            width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    copy_height = min(
        height,
        new_height,
    )

    copy_width = min(
        width,
        new_width,
    )

    y = random.randint(
        0,
        height - copy_height,
    )

    x = random.randint(
        0,
        width - copy_width,
    )

    canvas[
        y:y + copy_height,
        x:x + copy_width
    ] = result[
        :copy_height,
        :copy_width
    ]

    return canvas


def main():

    random.seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(SOURCE),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"Could not load: {SOURCE}"
        )

    print(
        f"Source: {SOURCE}"
    )

    print(
        f"Generating {COUNT} hard-positive images..."
    )

    for index in range(
        COUNT
    ):

        variant = random_variant(
            image
        )

        output = (
            OUTPUT_DIR
            / f"google_hard_{index + 1:04d}.jpg"
        )

        cv2.imwrite(
            str(output),
            variant,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                random.randint(
                    70,
                    95,
                ),
            ],
        )

    print()
    print(
        "DONE"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Files: {COUNT}"
    )


if __name__ == "__main__":
    main()