import cv2


class LayoutVerifier:

    @classmethod
    def verify(cls, image_path: str):

        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        orientation = (
            "Landscape"
            if width > height
            else "Portrait"
        )

        return {
            "passed": True,
            "page_width": width,
            "page_height": height,
            "aspect_ratio": round(width / height, 3),
            "orientation": orientation,
            "reason": "Page dimensions recorded. Passport region detection will be applied in a later phase."
        }
