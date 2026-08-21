"""
YOLO Object Detection Service.

Loads the local YOLOv8 model used by the
Agent + Property Verification pipeline.
"""

from pathlib import Path
from typing import Any

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

# Current file:
#
# src/
#   documents/
#     agent_property_verification/
#       services/
#         vision/
#           yolo_service.py
#
# parents[2] = agent_property_verification
MODULE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    MODULE_ROOT
    / "models"
    / "yolov8m.pt"
)


# ============================================================
# YOLO SERVICE
# ============================================================

class YOLOService:

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
    ):

        self.model_path = Path(
            model_path
            if model_path is not None
            else DEFAULT_MODEL_PATH
        ).resolve()

        # ----------------------------------------------------
        # Validate model file
        # ----------------------------------------------------

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "YOLO model not found: "
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # Validate threshold
        # ----------------------------------------------------

        self.confidence_threshold = float(
            confidence_threshold
        )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "YOLO confidence threshold must be "
                "between 0.0 and 1.0."
            )

        # ----------------------------------------------------
        # Load YOLO
        # ----------------------------------------------------

        try:

            self.model = YOLO(
                str(self.model_path)
            )

        except Exception as exc:

            raise RuntimeError(
                "Unable to load YOLO model from "
                f"{self.model_path}: {exc}"
            ) from exc

    # ========================================================
    # DETECTION
    # ========================================================

    def detect(
        self,
        image_path: str | Path,
        confidence: float | None = None,
    ) -> list[dict[str, Any]]:

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if not image_path:
            return []

        image_path = Path(
            image_path
        ).resolve()

        if not image_path.is_file():
            return []

        # ----------------------------------------------------
        # Confidence threshold
        # ----------------------------------------------------

        conf = (
            float(confidence)
            if confidence is not None
            else self.confidence_threshold
        )

        conf = max(
            0.0,
            min(
                1.0,
                conf,
            ),
        )

        # ----------------------------------------------------
        # Run YOLO
        # ----------------------------------------------------

        try:

            results = self.model.predict(
                source=str(image_path),
                save=False,
                verbose=False,
                conf=conf,
            )

        except Exception:
            return []

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        # ----------------------------------------------------
        # Build detections
        # ----------------------------------------------------

        detections: list[
            dict[str, Any]
        ] = []

        for box in result.boxes:

            try:

                class_id = int(
                    box.cls[0].item()
                )

                confidence_score = float(
                    box.conf[0].item()
                )

                bbox = [
                    round(
                        float(value),
                        2,
                    )
                    for value
                    in box.xyxy[0].tolist()
                ]

                label = self.model.names.get(
                    class_id,
                    str(class_id),
                )

                detections.append(
                    {
                        "label": str(label),

                        "class_id": class_id,

                        "confidence": round(
                            confidence_score,
                            3,
                        ),

                        "bbox": bbox,
                    }
                )

            except Exception:
                continue

        # ----------------------------------------------------
        # Highest confidence first
        # ----------------------------------------------------

        detections.sort(
            key=lambda item: item[
                "confidence"
            ],
            reverse=True,
        )

        return detections


# ============================================================
# SHARED SERVICE INSTANCE
# ============================================================

yolo_service = YOLOService()