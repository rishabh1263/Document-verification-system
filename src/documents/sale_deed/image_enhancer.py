"""
High Performance Image Enhancer
Optimized for OCR
"""

from pathlib import Path

import cv2
import numpy as np

from src.documents.sale_deed.config.paths import ENHANCED_DIR


class ImageEnhancer:

    def __init__(self, image):

        self.image = image

    # ----------------------------------------------------

    def deskew(self, image):

        return image

    # ----------------------------------------------------
    # Fast Noise Removal
    # ----------------------------------------------------

    def remove_noise(self, image):

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )

        noise = cv2.Laplacian(

            gray,

            cv2.CV_64F

        ).var()

        # Skip denoising if image is already sharp

        if noise > 300:

            return image

        return cv2.GaussianBlur(

            image,

            (3, 3),

            0

        )

    # ----------------------------------------------------

    def enhance_contrast(self, image):

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )

        clahe = cv2.createCLAHE(

            clipLimit=2.0,

            tileGridSize=(8, 8)

        )

        gray = clahe.apply(gray)

        return cv2.cvtColor(

            gray,

            cv2.COLOR_GRAY2BGR

        )

    # ----------------------------------------------------

    def sharpen(self, image):

        kernel = np.array(

            [

                [0, -1, 0],

                [-1, 5, -1],

                [0, -1, 0]

            ],

            dtype=np.float32

        )

        return cv2.filter2D(

            image,

            -1,

            kernel

        )

    # ----------------------------------------------------

    def binarize(self, image):

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )

        return cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY,

            25,

            10

        )

    # ----------------------------------------------------

    def save_image(self, image, filename):

        ENHANCED_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        output = ENHANCED_DIR / filename

        cv2.imwrite(

            str(output),

            image

        )

        return output

    # ----------------------------------------------------

    def process(

        self,

        denoise=True,

        contrast=True,

        sharpen=True,

        binarize=False

    ):

        image = self.image

        image = self.deskew(image)

        if denoise:

            image = self.remove_noise(image)

        if contrast:

            image = self.enhance_contrast(image)

        if sharpen:

            image = self.sharpen(image)

        if binarize:

            image = self.binarize(image)

        return image
