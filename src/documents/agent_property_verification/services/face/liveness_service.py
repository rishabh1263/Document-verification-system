"""
Liveness / Anti-Spoofing Service.

Uses MiniFASNetV2 to determine whether the detected face
belongs to a real/live person or a spoofed presentation.
"""

from pathlib import Path

import numpy as np
import torch

from src.documents.agent_property_verification.services.face.models.fastnet import (
    MiniFASNetV2,
)
from src.documents.agent_property_verification.services.face.utils import (
    crop_face,
    to_tensor,
    xyxy2xywh,
)


# ============================================================
# PATHS
# ============================================================

# Current file:
#
# agent_property_verification/
#   services/
#     face/
#       liveness_service.py
#       weights/
#         MiniFASNetV2.pth
#
FACE_SERVICE_DIR = Path(__file__).resolve().parent

WEIGHTS_DIR = FACE_SERVICE_DIR / "weights"

DEFAULT_WEIGHTS_PATH = (
    WEIGHTS_DIR
    / "MiniFASNetV2.pth"
)


# ============================================================
# LIVENESS SERVICE
# ============================================================

class LivenessService:

    def __init__(
        self,
        weights_path: str | Path | None = None,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.weights_path = Path(
            weights_path
            if weights_path is not None
            else DEFAULT_WEIGHTS_PATH
        ).resolve()

        # ----------------------------------------------------
        # Validate model file
        # ----------------------------------------------------

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "MiniFASNetV2 weights not found: "
                f"{self.weights_path}"
            )

        # ----------------------------------------------------
        # Initialize model
        # ----------------------------------------------------

        self.model = MiniFASNetV2()

        # ----------------------------------------------------
        # Load model weights
        # ----------------------------------------------------

        try:

            state = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=True,
            )

        except TypeError:
            # Compatibility with older PyTorch versions
            # where weights_only is unavailable.
            state = torch.load(
                self.weights_path,
                map_location=self.device,
            )

        except Exception as exc:
            raise RuntimeError(
                "Unable to load MiniFASNetV2 weights from "
                f"{self.weights_path}: {exc}"
            ) from exc

        # Some checkpoints may wrap the actual state dict.
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(
                state["state_dict"],
                dict,
            )
        ):
            state = state["state_dict"]

        try:
            self.model.load_state_dict(
                state
            )

        except Exception as exc:
            raise RuntimeError(
                "MiniFASNetV2 weights are incompatible "
                f"with the model architecture: {exc}"
            ) from exc

        self.model.to(
            self.device
        )

        self.model.eval()

    # ========================================================
    # CHECK LIVENESS
    # ========================================================

    def check(
        self,
        image,
        face,
    ) -> dict:

        if image is None:
            return {
                "is_live": False,
                "status": "Unable to Verify",
                "score": 0.0,
            }

        if face is None:
            return {
                "is_live": False,
                "status": "Unable to Verify",
                "score": 0.0,
            }

        if not hasattr(
            face,
            "bbox",
        ):
            return {
                "is_live": False,
                "status": "Unable to Verify",
                "score": 0.0,
            }

        try:

            # ------------------------------------------------
            # Face bounding box
            # ------------------------------------------------

            bbox = np.asarray(
                face.bbox,
                dtype=np.float32,
            )

            if bbox.size != 4:
                return {
                    "is_live": False,
                    "status": "Unable to Verify",
                    "score": 0.0,
                }

            bbox = xyxy2xywh(
                bbox
            ).astype(
                int
            ).tolist()

            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            face_crop = crop_face(
                image=image,
                bbox=bbox,
                scale=2.7,
                out_w=80,
                out_h=80,
            )

            if face_crop is None:
                return {
                    "is_live": False,
                    "status": "Unable to Verify",
                    "score": 0.0,
                }

            if getattr(
                face_crop,
                "size",
                0,
            ) == 0:
                return {
                    "is_live": False,
                    "status": "Unable to Verify",
                    "score": 0.0,
                }

            # ------------------------------------------------
            # Convert to model tensor
            # ------------------------------------------------

            tensor = to_tensor(
                face_crop
            )

            tensor = tensor.unsqueeze(
                0
            ).to(
                self.device
            )

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            with torch.no_grad():

                output = self.model(
                    tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()[0]
            )

            if len(probabilities) < 2:
                return {
                    "is_live": False,
                    "status": "Unable to Verify",
                    "score": 0.0,
                }

            predicted_class = int(
                probabilities.argmax()
            )

            live_score = float(
                probabilities[1]
            )

            is_live = bool(
                predicted_class == 1
            )

            return {
                "is_live": is_live,

                "status": (
                    "Real"
                    if is_live
                    else "Fake"
                ),

                "score": round(
                    live_score,
                    3,
                ),
            }

        except Exception as exc:

            return {
                "is_live": False,
                "status": "Unable to Verify",
                "score": 0.0,
                "error": str(exc),
            }


# ============================================================
# SHARED INSTANCE
# ============================================================

liveness_service = LivenessService()