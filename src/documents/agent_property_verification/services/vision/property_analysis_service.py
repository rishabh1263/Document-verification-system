from src.documents.agent_property_verification.services.vision.property_service import property_service
from src.documents.agent_property_verification.services.vision.yolo_service import yolo_service
from src.documents.agent_property_verification.services.vision.object_mapper import map_objects
from src.documents.agent_property_verification.services.vision.property_reasoning_service import (
    property_reasoning_service
)


class PropertyAnalysisService:

    def analyze(self, image_path: str) -> dict:

        # ==========================================
        # 1. PROPERTY CLASSIFICATION
        # ==========================================

        property_result = property_service.classify(
            image_path
        )

        property_detected = bool(
            property_result.get(
                "detected",
                False
            )
        )

        property_type = property_result.get(
            "type"
        )

        # Keep classifier debug information
        # temporarily while tuning the system.
        property_debug = property_result.get(
            "debug",
            {}
        )

        # ==========================================
        # 2. YOLO OBJECT DETECTION
        # ==========================================

        objects = yolo_service.detect(
            image_path
        )

        # ==========================================
        # 3. MAP OBJECTS TO FEATURES
        # ==========================================

        features = map_objects(
            objects
        )

        # ==========================================
        # 4. NO PROPERTY
        # ==========================================
        #
        # YOLO objects must NOT independently prove
        # that a property exists.
        #
        # Examples:
        # person + car + plant != property
        #
        # Property existence is determined by the
        # property classifier.
        # ==========================================

        if not property_detected:

            return {

                "detected": False,

                "property_type": None,

                "features": features,

                "reason": [
                    "No clearly visible property or building detected."
                ],

                "debug": property_debug
            }

        # ==========================================
        # 5. PROPERTY DETECTED
        # ==========================================

        result = property_reasoning_service.generate(
            property_type=property_type,
            features=features
        )

        # ==========================================
        # 6. ADD DEBUG INFORMATION
        # ==========================================

        result["debug"] = property_debug

        return result


property_analysis_service = PropertyAnalysisService()
