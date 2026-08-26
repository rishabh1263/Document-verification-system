from pathlib import Path

import cv2
import numpy as np


IMAGES = {
    "known_good": Path(
        r".\data\signature_classifier\real_signature.jpg"
    ),
    "google": Path(
        r".\data\signature_classifier\test_signature_google.jpg"
    ),
}


for name, path in IMAGES.items():

    image = cv2.imread(str(path))

    if image is None:
        print(f"{name}: FAILED TO LOAD")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Path       :", path)
    print("Width      :", image.shape[1])
    print("Height     :", image.shape[0])
    print("Channels   :", image.shape[2])

    print(
        "Mean pixel :",
        round(float(np.mean(gray)), 2),
    )

    print(
        "Std pixel  :",
        round(float(np.std(gray)), 2),
    )

    print(
        "Min pixel  :",
        int(np.min(gray)),
    )

    print(
        "Max pixel  :",
        int(np.max(gray)),
    )

    dark_pixels = np.sum(
        gray < 128
    )

    total_pixels = gray.size

    print(
        "Dark ratio :",
        round(
            float(dark_pixels / total_pixels),
            6,
        ),
    )