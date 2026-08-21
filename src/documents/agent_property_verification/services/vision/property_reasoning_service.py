class PropertyReasoningService:

    def generate(
        self,
        property_type: str,
        features: dict
    ) -> dict:

        reasons = []

        # ==========================================
        # 1. PROPERTY CLASSIFICATION
        # ==========================================

        if property_type:

            reasons.append(
                f"Property classified as '{property_type}'."
            )

        # ==========================================
        # 2. RELEVANT SUPPORTING FEATURES
        # ==========================================

        if features.get("parking"):

            reasons.append(
                "Vehicle or parking-related feature detected."
            )

        if features.get("garden"):

            reasons.append(
                "Outdoor garden-related feature detected."
            )

        if features.get("person"):

            reasons.append(
                "Person and property are visible in the image."
            )

        # ==========================================
        # 3. RESULT
        # ==========================================

        return {

            "detected": True,

            "property_type": property_type,

            "features": features,

            "reason": reasons
        }


property_reasoning_service = PropertyReasoningService()
