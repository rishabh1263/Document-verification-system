import cv2
import numpy as np

from src.documents.agent_customer_verification.services.embedding_service import EmbeddingService
from src.documents.agent_customer_verification.services.verification_service import VerificationService


class FaceService:

    BRIGHTNESS_MIN = 50
    BRIGHTNESS_MAX = 220

    # Lower threshold for webcam images
    BLUR_THRESHOLD = 10

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.verification_service = VerificationService()

    def check_brightness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))

        print("\n========== Brightness Check ==========")
        print(f"Image Shape      : {image.shape}")
        print(f"Brightness Score : {brightness:.2f}")
        print("======================================")

        if brightness < self.BRIGHTNESS_MIN:
            return (
                False,
                f"Image is too dark. Brightness={brightness:.2f}",
                brightness,
            )

        if brightness > self.BRIGHTNESS_MAX:
            return (
                False,
                f"Image is overexposed. Brightness={brightness:.2f}",
                brightness,
            )

        return True, "", brightness

    def check_blur(self, image):

        # Save the exact image received by the backend
        cv2.imwrite("debug_input.jpg", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        print("\n============= Blur Check =============")
        print(f"Image Shape    : {image.shape}")
        print(f"Blur Threshold : {self.BLUR_THRESHOLD}")
        print(f"Blur Score     : {blur_score:.2f}")
        print("Saved Image    : debug_input.jpg")
        print("======================================")

        if blur_score < self.BLUR_THRESHOLD:
            return (
                False,
                f"Image is blurry. Blur Score={blur_score:.2f}",
                blur_score,
            )

        return True, "", blur_score

    def validate_image(self, image):

        ok, message, brightness = self.check_brightness(image)

        if not ok:
            return {
                "success": False,
                "message": message,
                "metrics": {
                    "brightness": round(brightness, 2),
                },
            }

        ok, message, blur = self.check_blur(image)

        if not ok:
            return {
                "success": False,
                "message": message,
                "metrics": {
                    "brightness": round(brightness, 2),
                    "blur": round(blur, 2),
                },
            }

        return {
            "success": True,
            "message": "Image validation successful.",
            "metrics": {
                "brightness": round(brightness, 2),
                "blur": round(blur, 2),
            },
        }
