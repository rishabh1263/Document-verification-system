"""
Optimized Quality Checker
"""

import cv2
import numpy as np

from src.documents.sale_deed.config.quality_thresholds import (
    MIN_BLUR_SCORE,
    MIN_CONTRAST,
)


class QualityChecker:

    def __init__(self, image):

        self.image = image

        self.gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        self.height, self.width = self.gray.shape

    # -----------------------------------------------------

    def _result(self, check, score, status, remarks):

        return {

            "check": check,

            "score": score,

            "status": "PASS" if status else "FAIL",

            "remarks": remarks

        }

    # -----------------------------------------------------

    def check_blur(self):

        score = cv2.Laplacian(

            self.gray,

            cv2.CV_64F

        ).var()

        return self._result(

            "Blur",

            round(float(score), 2),

            score >= MIN_BLUR_SCORE,

            "Sharp Image"

            if score >= MIN_BLUR_SCORE

            else "Blur Detected"

        )

    # -----------------------------------------------------

    def check_contrast(self):

        score = float(

            self.gray.std()

        )

        return self._result(

            "Contrast",

            round(score, 2),

            score >= MIN_CONTRAST,

            "Good Contrast"

            if score >= MIN_CONTRAST

            else "Low Contrast"

        )

    # -----------------------------------------------------

    def check_resolution(self):

        status = (

            self.height >= 3000

            and

            self.width >= 2000

        )

        return self._result(

            "Resolution",

            f"{self.width} x {self.height}",

            status,

            "Good Resolution"

            if status

            else "Low Resolution"

        )

    # -----------------------------------------------------

    def check_blank_page(self):

        dark_pixels = np.count_nonzero(

            self.gray < 240

        )

        return self._result(

            "Blank Page",

            int(dark_pixels),

            dark_pixels > 1000,

            "Content Found"

            if dark_pixels > 1000

            else "Blank Page"

        )

    # -----------------------------------------------------

    def check_half_page(self):

        top = np.mean(

            self.gray[:50]

        )

        bottom = np.mean(

            self.gray[self.height - 50:]

        )

        diff = abs(top - bottom)

        return self._result(

            "Half Page",

            round(diff, 2),

            diff < 80,

            "Page Complete"

            if diff < 80

            else "Possible Cropped Scan"

        )

    # -----------------------------------------------------

    def check_rotation(self):

        return self._result(

            "Rotation",

            "0Â°",

            True,

            "Rotation Detection (V2)"

        )

    # -----------------------------------------------------

    def run_all_checks(self):

        checks = {

            "Blur": self.check_blur(),

            "Contrast": self.check_contrast(),

            "Resolution": self.check_resolution(),

            "Blank Page": self.check_blank_page(),

            "Half Page": self.check_half_page(),

            "Rotation": self.check_rotation()

        }

        passed = sum(

            1

            for value in checks.values()

            if value["status"] == "PASS"

        )

        score = round(

            passed * 100 / len(checks),

            2

        )

        return {

            "quality_score": score,

            "recommendation":

                "Ready for OCR"

                if score >= 80

                else "Image Enhancement Required",

            "checks": checks

        }
