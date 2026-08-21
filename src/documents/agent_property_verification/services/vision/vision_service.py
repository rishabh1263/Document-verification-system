from src.documents.agent_property_verification.services.vision.yolo_service import yolo_service
from src.documents.agent_property_verification.services.vision.property_analysis_service import property_analysis_service


class VisionService:

    def analyze(self, image_path: str) -> dict:
        """
        Analyze the image using all vision services.

        Returns:
        {
            "property": {...},
            "objects": [...]
        }
        """

        # Detect all objects
        objects = yolo_service.detect(image_path)

        # Analyze property
        property_result = property_analysis_service.analyze(image_path)

        return {
            "property": property_result,
            "objects": objects
        }


vision_service = VisionService()
