OBJECT_MAPPING = {

    # ==========================================
    # PARKING / VEHICLES
    # ==========================================

    "car": "parking",
    "motorcycle": "parking",
    "truck": "parking",
    "bus": "parking",
    "bicycle": "parking",

    # ==========================================
    # GARDEN / OUTDOOR FEATURES
    # ==========================================

    "potted plant": "garden",
    "bench": "garden",

    # ==========================================
    # PERSON
    # ==========================================

    "person": "person"
}


def map_objects(objects: list) -> dict:

    # ==========================================
    # DEFAULT FEATURES
    # ==========================================

    features = {

        "parking": False,

        "garden": False,

        "person": False,

        "detected_objects": []
    }

    # ==========================================
    # MAP YOLO OBJECTS
    # ==========================================

    for obj in objects:

        label = obj.get("label")

        if not label:
            continue

        confidence = obj.get(
            "confidence",
            0
        )

        # Store useful detection information
        features["detected_objects"].append({

            "label": label,

            "confidence": confidence

        })

        feature = OBJECT_MAPPING.get(
            label
        )

        if feature:

            features[feature] = True

    return features
