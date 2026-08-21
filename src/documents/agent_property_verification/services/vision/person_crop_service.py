from PIL import Image

from src.documents.agent_property_verification.services.vision.yolo_service import yolo_service


class PropertyCropService:

    def crop(self, image_path: str) -> Image.Image:
        """
        Crop the image to keep the property while removing
        as much of the selfie person as possible.
        """

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        detections = yolo_service.detect(image_path)

        persons = [
            obj
            for obj in detections
            if obj["label"] == "person"
        ]

        # No person detected
        if not persons:
            return image

        # Use the largest detected person
        person = max(
            persons,
            key=lambda p: (
                (p["bbox"][2] - p["bbox"][0]) *
                (p["bbox"][3] - p["bbox"][1])
            )
        )

        x1, y1, x2, y2 = map(int, person["bbox"])

        left_space = x1
        right_space = width - x2

        top_space = y1
        bottom_space = height - y2

        crops = []

        # Left crop
        if left_space > width * 0.25:
            crops.append(
                image.crop(
                    (
                        0,
                        0,
                        x1,
                        height
                    )
                )
            )

        # Right crop
        if right_space > width * 0.25:
            crops.append(
                image.crop(
                    (
                        x2,
                        0,
                        width,
                        height
                    )
                )
            )

        # Top crop
        if top_space > height * 0.25:
            crops.append(
                image.crop(
                    (
                        0,
                        0,
                        width,
                        y1
                    )
                )
            )

        # Bottom crop
        if bottom_space > height * 0.25:
            crops.append(
                image.crop(
                    (
                        0,
                        y2,
                        width,
                        height
                    )
                )
            )

        # If no suitable crop found
        if not crops:
            return image

        # Return the largest crop by area
        largest_crop = max(
            crops,
            key=lambda img: img.size[0] * img.size[1]
        )

        return largest_crop


property_crop_service = PropertyCropService()
