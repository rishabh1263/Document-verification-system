from src.documents.agent_property_verification.services.face.face_service import face_service
from src.documents.agent_property_verification.services.face.embedding_service import embedding_service
from src.documents.agent_property_verification.services.face.liveness_service import liveness_service

from src.documents.agent_property_verification.services.vision.vision_service import vision_service

from src.documents.agent_property_verification.services.image_integrity_service import (
    image_integrity_service
)


class VerificationService:

    def verify(self, image_path: str):

        try:

            # ==========================================
            # 1. FACE DETECTION
            # ==========================================

            face_result = face_service.detect_face(
                image_path
            )

            image = face_result["image"]
            face = face_result["face"]

            # ==========================================
            # 2. PASSIVE ANTI-SPOOF
            # ==========================================

            liveness = liveness_service.check(
                image=image,
                face=face
            )

            anti_spoof_passed = bool(
                liveness["is_live"]
            )

            # ==========================================
            # 3. IDENTITY VERIFICATION
            # ==========================================

            identity = embedding_service.compare(
                face.embedding
            )

            identity_verified = bool(
                identity["verified"]
            )

            # ==========================================
            # 4. AI / SYNTHETIC IMAGE DETECTION
            # ==========================================

            image_integrity = (
                image_integrity_service.analyze(
                    image_path
                )
            )

            synthetic_detection_available = bool(
                image_integrity.get(
                    "synthetic_detection_available",
                    False
                )
            )

            synthetic_suspected = (
                image_integrity.get(
                    "synthetic_suspected"
                )
            )

            if synthetic_suspected is not None:

                synthetic_suspected = bool(
                    synthetic_suspected
                )

            # ==========================================
            # 5. IMAGE INTEGRITY DECISION
            # ==========================================

            image_integrity_passed = True

            if synthetic_detection_available:

                image_integrity_passed = bool(
                    synthetic_suspected is False
                )

            # ==========================================
            # 6. PROPERTY ANALYSIS
            # ==========================================

            vision_result = vision_service.analyze(
                image_path
            )

            property_result = vision_result[
                "property"
            ]

            objects = vision_result[
                "objects"
            ]

            property_detected = bool(
                property_result.get(
                    "detected",
                    False
                )
            )

            property_type = property_result.get(
                "property_type"
            )

            property_reason = property_result.get(
                "reason",
                []
            )

            if isinstance(property_reason, str):

                property_reason = [
                    property_reason
                ]

            # ==========================================
            # 7. PERSON + PROPERTY CHECK
            # ==========================================

            person_detected = any(

                obj.get("label") == "person"

                for obj in objects
            )

            agent_and_property_visible = bool(
                person_detected
                and property_detected
            )

            # ==========================================
            # 8. FINAL VERIFICATION
            # ==========================================

            verification_passed = bool(

                identity_verified

                and anti_spoof_passed

                and image_integrity_passed

                and property_detected

                and agent_and_property_visible
            )

            # ==========================================
            # 9. FAILED CHECKS
            # ==========================================

            failed_checks = []

            if not identity_verified:

                failed_checks.append(
                    "identity"
                )

            if not anti_spoof_passed:

                failed_checks.append(
                    "anti_spoof"
                )

            if not image_integrity_passed:

                failed_checks.append(
                    "image_integrity"
                )

            if not property_detected:

                failed_checks.append(
                    "property_visible"
                )

            if not agent_and_property_visible:

                failed_checks.append(
                    "agent_and_property_visible"
                )

            # ==========================================
            # 10. API RESPONSE
            # ==========================================

            return {

                "success": True,

                "agent": {

                    "identity_verified": (
                        identity_verified
                    ),

                    "anti_spoof_passed": (
                        anti_spoof_passed
                    )
                },

                # ======================================
                # TEMPORARY AI DETECTOR DEBUG
                # ======================================

                "image_integrity": {

                    "synthetic_suspected": (
                        synthetic_suspected
                    ),

                    "classification": (
                        image_integrity.get(
                            "classification"
                        )
                    ),

                    "synthetic_score": (
                        image_integrity.get(
                            "synthetic_score"
                        )
                    ),

                    "real_score": (
                        image_integrity.get(
                            "real_score"
                        )
                    ),

                    "threshold": (
                        image_integrity.get(
                            "threshold"
                        )
                    )
                },

                "property": {

                    "detected": (
                        property_detected
                    ),

                    "property_type": (
                        property_type
                    ),

                    "reason": (
                        property_reason
                    )
                },

                "onsite_verified": (
                    agent_and_property_visible
                ),

                "verification": {

                    "passed": (
                        verification_passed
                    ),

                    "failed_checks": (
                        failed_checks
                    )
                }
            }

        except Exception as e:

            return {

                "success": False,

                "message": str(e),

                "verification": {

                    "passed": False
                }
            }


verification_service = VerificationService()
