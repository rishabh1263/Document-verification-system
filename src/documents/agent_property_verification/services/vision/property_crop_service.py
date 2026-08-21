from PIL import Image, ImageDraw

from src.documents.agent_property_verification.services.vision.yolo_service import yolo_service


class PropertyCropService:

    def crop(self, image_path: str) -> Image.Image:

        # ==========================================
        # 1. LOAD IMAGE
        # ==========================================

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        # ==========================================
        # 2. DETECT OBJECTS
        # ==========================================

        detections = yolo_service.detect(
            image_path
        )

        persons = [
            obj
            for obj in detections
            if obj.get("label") == "person"
        ]

        # ==========================================
        # 3. NO PERSON
        # ==========================================
        #
        # For normal property photos, simply use
        # the complete image.
        # ==========================================

        if not persons:
            return image

        # ==========================================
        # 4. FIND MAIN PERSON
        # ==========================================
        #
        # Usually the field agent will be the
        # largest person in the selfie.
        # ==========================================

        person = max(
            persons,
            key=lambda p: (
                (
                    p["bbox"][2]
                    -
                    p["bbox"][0]
                )
                *
                (
                    p["bbox"][3]
                    -
                    p["bbox"][1]
                )
            )
        )

        x1, y1, x2, y2 = map(
            int,
            person["bbox"]
        )

        # Clamp bounding box
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        person_width = max(
            0,
            x2 - x1
        )

        person_height = max(
            0,
            y2 - y1
        )

        person_area = (
            person_width
            *
            person_height
        )

        image_area = (
            width
            *
            height
        )

        person_ratio = (
            person_area
            /
            image_area
            if image_area
            else 0
        )

        # ==========================================
        # 5. PERSON IS SMALL
        # ==========================================
        #
        # If the person occupies little of the
        # image, masking is unnecessary.
        # ==========================================

        if person_ratio < 0.20:
            return image

        # ==========================================
        # 6. MASK PERSON
        # ==========================================
        #
        # Do NOT crop away half the image.
        #
        # Property may exist:
        # - behind the person
        # - above the person
        # - on both sides
        #
        # Instead mask the main person while
        # preserving the entire scene.
        # ==========================================

        property_image = image.copy()

        draw = ImageDraw.Draw(
            property_image
        )

        # Slightly expand the mask so clothing,
        # arms, etc. don't dominate CLIP.
        padding_x = int(
            person_width * 0.05
        )

        padding_y = int(
            person_height * 0.03
        )

        mask_x1 = max(
            0,
            x1 - padding_x
        )

        mask_y1 = max(
            0,
            y1 - padding_y
        )

        mask_x2 = min(
            width,
            x2 + padding_x
        )

        mask_y2 = min(
            height,
            y2 + padding_y
        )

        # Neutral gray avoids introducing a strong
        # artificial semantic signal.
        draw.rectangle(
            (
                mask_x1,
                mask_y1,
                mask_x2,
                mask_y2
            ),
            fill=(127, 127, 127)
        )

        # ==========================================
        # 7. SAFETY CHECK
        # ==========================================
        #
        # If the person occupies almost the whole
        # image, there isn't enough background
        # evidence to reliably verify a property.
        #
        # Return original so downstream detection
        # can make the final decision.
        # ==========================================

        if person_ratio > 0.75:
            return image

        return property_image


property_crop_service = PropertyCropService()
